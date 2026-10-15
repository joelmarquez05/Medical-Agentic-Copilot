"""Append the user-selected Ling model as a separate, identically configured cohort."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from medical_copilot.comparison_evaluation import RequestBudget, utc_now
from medical_copilot.config import ModelConfig, resolve_credential
from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.medqa_mini import AGENT_SYSTEM, ANSWER_SYSTEM, INPUT_BYTE_CAP, OUTPUT_CAP, PLAN_SYSTEM, MiniAudit, MiniCopilot, summarize
from medical_copilot.models import ModelClient, create_backend
from medical_copilot.pubmed import PubMedClient, PubMedConfig

LING_ID = "inclusionai/ling-3.0-flash-sante"
LING_PROFILE = "ling-3.0-flash-sante"


def ling_config():
    return ModelConfig(provider="openai_compatible", model=LING_ID, api_key_env="OPENROUTER_API_KEY", credential_file="open_router_key.md", base_url="https://openrouter.ai/api/v1", temperature=None, max_tokens=OUTPUT_CAP, timeout_seconds=180, max_retries=0, min_interval_seconds=1, rate_limit_retries=0, max_revisions=0, json_repair_attempts=0, provider_options={"use_responses_api": False, "extra_body": {"reasoning": {"enabled": True, "exclude": True}, "provider": {"require_parameters": True, "sort": "price", "max_price": {"prompt": 0.05, "completion": 0.15}}}})


def extension_bound():
    return 16 * ((INPUT_BYTE_CAP + 1024) * 0.05 + OUTPUT_CAP * 0.15) / 1_000_000


def read_ledger(config, root):
    credential, _ = resolve_credential(config, root)
    request = Request("https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + credential, "User-Agent": "MedicalCopilotBudgetAudit/1.0"})
    try:
        with urlopen(request, timeout=20) as response:
            data = json.load(response)["data"]
        if not isinstance(data.get("total_usage"), (int, float)):
            raise ValueError("Missing numeric ledger total.")
        return {key: data[key] for key in ("total_credits", "total_usage") if isinstance(data.get(key), (int, float))}
    except Exception as exc:
        raise RuntimeError("Read-only ledger audit failed (" + type(exc).__name__ + "); paid extension blocked.") from None


def run_extension(root: Path, primary: Path, destination: Path):
    if destination.exists():
        raise ValueError("Do not overwrite or implicitly rerun an extension.")
    original = read_json(primary / "manifest.json")
    if not original.get("completed_at_utc"):
        raise ValueError("Run the extension only after the primary cohort finishes.")
    discovery = read_json(root / "data/checks/ling-sante-discovery-20261011.json")
    item = next(model for model in discovery["models"] if model["id"] == LING_ID)
    if "reasoning" not in item["supported_parameters"] or float(item["pricing"]["prompt"]) * 1_000_000 > 0.05 or float(item["pricing"]["completion"]) * 1_000_000 > 0.15:
        raise ValueError("Ling capability/price advertisement does not fit the agreed profile.")
    config = ling_config()
    ledger = read_ledger(config, root)
    previous_budget = read_json(primary / "budget.json")
    initial = max(previous_budget["reported_or_estimated_spend_usd"], float(ledger["total_usage"]))
    budget = RequestBudget(6, initial)
    budget.unreported_charge_reserve_usd = previous_budget["unreported_charge_reserve_usd"]
    budget.check(extension_bound())
    destination.mkdir(parents=True)
    for filename in ("questions.json", "answer-key.json", "subset-manifest.json", "protocol.md"):
        shutil.copyfile(primary / filename, destination / filename)
    write_json(destination / "initial-account-audit.json", {"checked_at_utc": utc_now(), "ledger": ledger, "prior_recorded_budget": previous_budget, "new_inference_calls": 0})
    questions = read_json(destination / "questions.json")["questions"]
    if len(questions) != 10:
        raise ValueError("Extension must reuse exactly ten frozen questions.")
    order = [{"profile": LING_PROFILE, "mode": mode, "question_id": question["id"]} for mode, cases in (("direct", questions), ("agent", questions[:3])) for question in cases]
    snapshot = destination / "source-snapshot"
    snapshot.mkdir()
    hashes = {}
    for path in sorted((root / "src/medical_copilot").glob("*.py")):
        shutil.copyfile(path, snapshot / path.name)
        hashes[path.name] = file_hash(path)
    manifest = {"started_at_utc": utc_now(), "cohort": "User-requested sixth-model extension after the frozen five-model cohort", "primary_cohort": primary.relative_to(root).as_posix(), "primary_manifest_sha256": file_hash(primary / "manifest.json"), "profiles": [{"id": LING_PROFILE, "config": config.public_dict(), "native_effort": config.provider_options["extra_body"]["reasoning"]}], "case_order": order, "question_ids": original["question_ids"], "agent_question_ids": original["agent_question_ids"], "source_hashes": hashes, "question_file_sha256": file_hash(destination / "questions.json"), "answer_key_file_sha256": file_hash(destination / "answer-key.json"), "maximum_requests": 16, "maximum_new_request_allowance_usd": extension_bound(), "initial_account_usage_usd": initial, "prompts": {"direct": ANSWER_SYSTEM, "plan": PLAN_SYSTEM, "agent": AGENT_SYSTEM}, "reasoning_limitation": "Ling advertises reasoning enablement but no native effort scale; explicitly enabled, analogous to MiMo/Gemma, not a claim of native high or equal compute.", "timing_limitation": "Added at the user's request after primary generation began; sequential later cohort, not interleaved with the initial five. No primary answer is replaced."}
    write_json(destination / "manifest.json", manifest)
    (destination / "protocol-addendum.md").write_text("# Ling sixth-model extension\n\nAdded at the user's request after the primary cohort began. Reuse its unchanged ten questions, released key and first three agent cases, prompts, 6144-token output cap, 5120-byte input bound and zero retries/repairs/revisions. Preserve the primary cohort. Exact ID: inclusionai/ling-3.0-flash-sante; reasoning enabled=true, exclude=true, with no advertised native effort scale. Price ceilings USD/M: prompt 0.05, completion 0.15. Maximum 13 case records and 16 model requests, allowance USD " + str(extension_bound()) + ". Refresh the authenticated ledger and carry forward earlier consumption and uncertainty reserves under the cumulative USD 6 cap. This is a later, separately timed cohort and not evidence of equal native compute.\n", encoding="utf-8")
    write_json(destination / "budget.json", budget.public_dict())
    directory = destination / "models" / LING_PROFILE
    directory.mkdir(parents=True)
    backend, redactor, _ = create_backend(config, root)
    audited = MiniAudit(backend, config, budget, redactor, directory / "usage-audit.json")
    client = ModelClient(audited, config, redactor)
    write_json(directory / "config.json", config.public_dict())
    by_id = {question["id"]: question for question in questions}
    failures = 0
    for index, case in enumerate(order, 1):
        mode, qid = case["mode"], case["question_id"]
        print(f"[Ling {index}/13] START {mode} {qid}", flush=True)
        audited.case_id = mode + ":" + qid
        call_start, audit_start = len(client.calls), len(audited.records)
        pubmed = PubMedClient(PubMedConfig(max_retries=0, timeout_seconds=15, max_abstract_characters=500), directory / "pubmed-snapshots")
        if failures >= 2:
            record = {"status": "not_run_provider_circuit", "elapsed_seconds": 0}
        else:
            record = MiniCopilot(client, pubmed, mode).run(by_id[qid])
            failures = failures + 1 if record["status"] == "error" else 0
        record.update(case, question_sha256=by_id[qid]["question_sha256"], model_calls=client.calls[call_start:], audit_records=audited.records[audit_start:], pubmed_requests=pubmed.requests)
        write_json(directory / "answers" / (mode + "-" + qid + ".json"), record)
        write_json(destination / "budget.json", budget.public_dict())
        print(f"[Ling {index}/13] END {mode} {qid}: {record['status']} cumulative_recorded_usd={budget.spent_usd:.6f}", flush=True)
    manifest["completed_at_utc"] = utc_now()
    write_json(destination / "manifest.json", manifest)
    return summarize(destination)
