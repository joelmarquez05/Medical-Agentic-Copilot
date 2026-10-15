"""Answer new questions with the tested Gemma profile and resilient clinical graph.

Input: one ClinicalQuestion object, or {"questions": [ClinicalQuestion, ...]}.
Evaluator keys and other undeclared fields are rejected before any API access.
"""

import argparse
import json
from pathlib import Path

from medical_copilot.clinical_evaluation import (
    ACCOUNT_LIMIT, UNCERTAINTY_HEADROOM, DEFAULT_PROFILE,
    maximum_request_cost, observe_ledger, preflight, public_error,
)
from medical_copilot.clinical_graph import ClinicalQuestion, save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_literature_evaluation import PUBMED_CONFIG
from medical_copilot.clinical_resilient import RESILIENT_SYSTEM, VERSION, ResilientClinicalAgent, ResilientLimits
from medical_copilot.clinical_resilient_evaluation import POLICY, make_model
from medical_copilot.comparison_evaluation import RequestBudget
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.pubmed import PubMedClient, load_pubmed_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    if run.exists():
        raise ValueError("Choose a fresh output directory.")
    data = read_json(args.input_json.resolve())
    rows = data["questions"] if isinstance(data, dict) and set(data) == {"questions"} else [data]
    questions = [ClinicalQuestion.model_validate(row).model_dump(mode="json") for row in rows]
    if not questions or len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Provide a nonempty list with unique question IDs.")
    policy = read_json(root / POLICY)
    limits = ResilientLimits(**{k: v for k, v in policy.items() if k in ResilientLimits.model_fields})
    # This CLI uses the reference profile whose native final mode was actually
    # probed. Other backends can use PhaseBackend(native_final=False), with their
    # own transport/cost validation before hosted use.
    config = load_config(root / DEFAULT_PROFILE)
    observation = preflight(config, root)
    if not any("structured_outputs" in e["supported_parameters"] for e in observation["eligible_endpoints"]):
        raise ValueError("Native final output is not currently advertised by the pinned endpoint.")
    budget = RequestBudget(ACCOUNT_LIMIT, observation["ledger"]["total_usage"])
    budget.unreported_charge_reserve_usd = UNCERTAINTY_HEADROOM
    budget.check(len(questions) * limits.max_model_turns * maximum_request_cost(config, limits.max_input_bytes + 4096))
    run.mkdir(parents=True)
    save_trace(run / "questions.json", {"questions": questions})
    save_trace(run / "preflight.json", observation)
    save_trace(run / "configuration.json", {"version": VERSION, "model": config.public_dict(), "policy": policy,
        "system": RESILIENT_SYSTEM, "input_sha256": file_hash(args.input_json.resolve()),
        "scoring": "None; new application questions, not an evaluation rerun."})
    model, client, audit, phased, credential_source = make_model(config, root, budget, run / "usage-audit.json", policy)
    save_trace(run / "credential-resolution.json", {"source": credential_source, "value_saved": False})
    coverage, completed, application_messages, clarifications = 0, 0, 0, 0
    for index, question in enumerate(questions, 1):
        audit.case_id = question["question_id"]
        start_calls, start_audit, start_formats = len(client.calls), len(audit.records), len(phased.formats)
        tools = ClinicalLiteratureTools(PubMedClient(load_pubmed_config(root / PUBMED_CONFIG), snapshot_dir=run / "http-snapshots"))
        record = ResilientClinicalAgent(model, tools.definitions(), limits).run(question)
        record.update(model_calls=client.calls[start_calls:], response_audit=audit.records[start_audit:],
            transport_formats=phased.formats[start_formats:], literature=tools.public_record())
        # Sequential numeric paths also keep arbitrary user IDs out of paths.
        save_trace(run / "answers" / f"{index:04d}.json", record)
        write_json(run / "budget.json", budget.public_dict())
        coverage += bool(record["response"])
        completed += record["status"] in {"completed", "completed_with_citation_issues"}
        application_messages += record["output_origin"] == "application"
        clarifications += record["status"] == "awaiting_clarification"
        print(json.dumps({"question_id": question["question_id"], "status": record["status"], "response": record["response"]}, ensure_ascii=False), flush=True)
    try:
        ledger = observe_ledger(config, root)
    except Exception as exc:
        ledger = {"error": public_error(exc)}
    save_trace(run / "final-ledger.json", ledger)
    save_trace(run / "summary.json", {"questions": len(questions), "terminal_messages": coverage,
        "completed_model_answers": completed, "application_nonanswers": application_messages,
        "clarification_requests": clarifications,
        "provider_requests": len(client.calls), "budget": budget.public_dict(), "accuracy": "Not scored."})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": public_error(exc)}))
        raise SystemExit(1) from None
