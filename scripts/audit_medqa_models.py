"""Read public model metadata and the authorized account ledger; no inference."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from medical_copilot.config import load_config, resolve_credential

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["z-ai/glm-5.3", "moonshotai/kimi-k3", "xiaomi/mimo-v2.6-pro", "qwen/qwen3.8-27b", "google/gemma-4-26b-a4b-it"]


def fetch(url: str, credential: str | None = None):
    headers = {"User-Agent": "MedicalCopilotMetadataAudit/1.0"}
    if credential:
        headers["Authorization"] = "Bearer " + credential
    with urlopen(Request(url, headers=headers), timeout=20) as response:
        return json.load(response)


def error_record(exc: Exception):
    return {"error_type": type(exc).__name__, "http_status": getattr(exc, "code", None), "reason_type": type(getattr(exc, "reason", None)).__name__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", action="store_true", help="Read credit totals with the user-provided OpenRouter key; no generation")
    args = parser.parse_args()
    record = {"checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "new_inference_calls": 0, "selected_model_ids": MODELS, "models": []}
    try:
        models = fetch("https://openrouter.ai/api/v1/models")["data"]
        for wanted in MODELS:
            item = next((model for model in models if model["id"] == wanted), None)
            if item is None:
                record["models"].append({"id": wanted, "listed": False})
                continue
            safe = {key: item.get(key) for key in ("id", "name", "pricing", "supported_parameters", "reasoning", "top_provider", "context_length")}
            safe["listed"] = True
            try:
                data = fetch("https://openrouter.ai/api/v1/models/" + wanted + "/endpoints")["data"]
                safe["endpoints"] = [{key: endpoint.get(key) for key in ("name", "provider_name", "pricing", "supported_parameters", "quantization", "max_completion_tokens", "context_length", "reasoning")} for endpoint in data.get("endpoints", [])]
            except Exception as exc:
                safe["endpoint_error"] = error_record(exc)
            record["models"].append(safe)
    except Exception as exc:
        record["catalogue_error"] = error_record(exc)
    if args.ledger:
        try:
            credential, _ = resolve_credential(load_config(ROOT / "config/models/openrouter-glm-5.3.json"), ROOT)
            data = fetch("https://openrouter.ai/api/v1/credits", credential)["data"]
            record["ledger"] = {key: data.get(key) for key in ("total_credits", "total_usage") if isinstance(data.get(key), (float, int))}
            record["ledger_interpretation"] = "Account-wide observation, not a per-request invoice; delayed billing and other account activity remain possible."
        except Exception as exc:
            record["ledger_error"] = error_record(exc)
    path = ROOT / "data/checks/medqa-selected-model-audit-20261011.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for item in record["models"]:
        print(json.dumps({key: item.get(key) for key in ("id", "listed", "pricing", "reasoning", "supported_parameters")}))
    print(json.dumps({key: record[key] for key in ("ledger", "ledger_error", "catalogue_error", "new_inference_calls") if key in record}))


if __name__ == "__main__":
    main()
