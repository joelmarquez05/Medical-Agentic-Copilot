"""Provider access, request budgets, and private usage accounting."""
import json
import math
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

from medical_copilot.config import resolve_credential
from medical_copilot.files import write_json

ACCOUNT_LIMIT = 6.0
UNCERTAINTY_HEADROOM = 0.15

class BudgetLimitExceeded(RuntimeError):
    """No provider request was sent because the local allowance was too small."""


class RequestBudget:
    def __init__(self, limit_usd: float, spent_usd: float = 0.0):
        if not math.isfinite(limit_usd) or limit_usd <= 0 or not math.isfinite(spent_usd) or spent_usd < 0:
            raise ValueError("Budget values must be finite and nonnegative, with a positive limit.")
        self.limit_usd, self.spent_usd = limit_usd, spent_usd
        self.unreported_charge_reserve_usd = 0.0

    def check(self, estimated_maximum: float):
        if self.spent_usd + self.unreported_charge_reserve_usd + estimated_maximum > self.limit_usd:
            raise BudgetLimitExceeded("The local paid-request allowance is exhausted.")

    def public_dict(self):
        return {"limit_usd": self.limit_usd, "reported_or_estimated_spend_usd": round(self.spent_usd, 8), "unreported_charge_reserve_usd": round(self.unreported_charge_reserve_usd, 8), "policy": "Before each paid request, reserve a conservative prompt-byte estimate plus the full output cap at configured provider price ceilings. Failed requests without usage retain that reserve. This is a local guard, not a provider billing guarantee."}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite_number(value):
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def public_error(exc):
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return {"error_type": type(exc).__name__, "http_status": code if type(code) is int else None}


def request_json(url, credential=None):
    headers = {"User-Agent": "MedicalCopilotBaseline/1.0"}
    if credential:
        headers["Authorization"] = "Bearer " + credential
    with urlopen(Request(url, headers=headers), timeout=25) as response:
        return json.load(response)


def observe_ledger(config, root):
    credential, source = resolve_credential(config, root)
    data = request_json("https://openrouter.ai/api/v1/credits", credential)["data"]
    if not all(finite_number(data.get(key)) for key in ["total_usage", "total_credits"]):
        raise ValueError("Credit ledger lacks valid numeric totals.")
    return {"checked_at_utc": utc_now(), "credential_source": source,
            "total_usage": data["total_usage"], "total_credits": data["total_credits"],
            "interpretation": "Account-wide observation; delayed billing or other account activity remains possible."}


def preflight(config, root):
    public = request_json("https://openrouter.ai/api/v1/models/" + config.model + "/endpoints")["data"]
    provider = config.provider_options["extra_body"]["provider"]["order"][0]
    matching = [endpoint for endpoint in public.get("endpoints", [])
                if endpoint.get("provider_name", "").casefold() == provider.casefold()]
    ceilings = config.provider_options["extra_body"]["provider"]["max_price"]
    eligible = []
    for endpoint in matching:
        supported = endpoint.get("supported_parameters") or []
        prices = endpoint.get("pricing", {})
        if not {"reasoning", "response_format"}.issubset(supported):
            continue
        if any(float(prices[field]) * 1_000_000 > ceilings[field] for field in ["prompt", "completion"]):
            continue
        eligible.append({key: endpoint.get(key) for key in ["provider_name", "name", "tag", "pricing", "supported_parameters", "max_completion_tokens"]})
    if not eligible:
        raise ValueError("Pinned provider does not currently advertise the required parameters/prices.")
    return {"checked_at_utc": utc_now(), "model": config.model, "eligible_endpoints": eligible,
            "ledger": observe_ledger(config, root), "model_inference_requests": 0,
            "interpretation": "Advertised capability and account access, not completed generation."}


def maximum_request_cost(config, input_bytes):
    ceilings = config.provider_options["extra_body"]["provider"]["max_price"]
    return ((input_bytes + 1024) * ceilings["prompt"] + config.max_tokens * ceilings["completion"]) / 1_000_000


class BaselineAudit:
    """Reserve unknown charges, retain numeric usage, omit private reasoning text."""

    def __init__(self, inner, config, budget, redactor, path):
        self.inner, self.config, self.budget, self.redactor, self.path = inner, config, budget, redactor, path
        self.records = []
        self.case_id = None

    def invoke(self, messages):
        maximum = maximum_request_cost(self.config, sum(len(str(message.content).encode("utf-8")) for message in messages))
        self.budget.check(maximum)
        record = {"question_id": self.case_id, "started_at_utc": utc_now(),
                  "estimated_maximum_usd": maximum, "request_sent": True}
        started = time.perf_counter()
        try:
            message = self.inner.invoke(messages)
            metadata, usage = message.response_metadata or {}, message.usage_metadata or {}
            provider_usage = metadata.get("token_usage", {})
            # Store only known numeric fields; no provider thoughts/headers/errors.
            numeric = {key: provider_usage[key] for key in ["prompt_tokens", "completion_tokens", "total_tokens", "cost"]
                       if finite_number(provider_usage.get(key))}
            reason = (provider_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if finite_number(reason):
                numeric["reasoning_tokens"] = reason
            normalized = {key: usage[key] for key in ["input_tokens", "output_tokens", "total_tokens"] if finite_number(usage.get(key))}
            cost = numeric.get("cost")
            if cost is not None:
                basis = "Provider-reported usage.cost."
            elif normalized.get("input_tokens") and "output_tokens" in normalized:
                prices = self.config.provider_options["extra_body"]["provider"]["max_price"]
                cost = (normalized["input_tokens"] * prices["prompt"] + normalized["output_tokens"] * prices["completion"]) / 1_000_000
                basis = "Reported tokens at configured price ceilings; not an observed bill."
            else:
                self.budget.unreported_charge_reserve_usd += maximum
                basis = "No reliable usage/cost; full request allowance retained as unknown-charge reserve."
            if cost is not None:
                self.budget.spent_usd += cost
            record.update(status="ok", usage=normalized, provider_usage=numeric, cost_usd=cost,
                          cost_basis=basis, finish_reason=metadata.get("finish_reason"),
                          model_name=metadata.get("model_name"), provider_route="Configured pin; actual route not independently exposed by this response.")
            return message
        except Exception as exc:
            self.budget.unreported_charge_reserve_usd += maximum
            record.update(status="error", cost_usd=None, cost_basis="Failed request retains full unknown-charge allowance.", **public_error(exc))
            raise
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.records.append(self.redactor.object(record))
            write_json(self.path, {"records": self.records, "budget": self.budget.public_dict()})


def recoverable_error(row):
    """Recognize the connection-error alias actually observed in the first control."""
    return row.get("http_status") in {429, 500, 502, 503, 504} or row.get("error_type") in {
        "APITimeoutError", "APIConnectionError", "OpenAIConnectionError", "TimeoutError", "ConnectTimeout", "ReadTimeout"}
