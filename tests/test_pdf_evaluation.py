"""Meaningful scheduling, frozen prerequisites and cumulative-billing safeguards."""
from pathlib import Path
from unittest.mock import patch
import pytest
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import write_json
from medical_copilot.pdf_agent import CONDITIONS
from medical_copilot.pdf_evaluation import episode_order, prior_totals, request_budget, validate_prerequisite, run_stage, recoverable_failure


def test_balanced_arm_rotation():
    rows = [{"question_id": str(n)} for n in range(6)]
    order = episode_order(rows)
    assert len(set(order)) == 18
    assert [arm for identifier, arm in order if identifier == "0"] == list(CONDITIONS)
    assert [arm for identifier, arm in order if identifier == "1"] == list(CONDITIONS[1:] + CONDITIONS[:1])
    assert [arm for identifier, arm in order if identifier == "2"] == list(CONDITIONS[2:] + CONDITIONS[:2])
    assert {arm: sum(order[n][1] == arm for n in range(0, 18, 3)) for arm in CONDITIONS} == dict.fromkeys(CONDITIONS, 2)


def test_lagging_ledger_retains_cost_and_unknown_request_allowance():
    prior = {"account_anchor_usd": 4.8, "known_cost_usd": .07, "unknown_reserve_usd": .02}
    budget = request_budget(4.81, prior)
    assert budget.spent_usd == pytest.approx(4.87)
    assert budget.unreported_charge_reserve_usd == pytest.approx(.17)
    assert request_budget(4.92, prior).spent_usd == pytest.approx(4.92)


def test_unfinished_stage_counts_known_and_unknown_charges(tmp_path):
    run = tmp_path / "data/evaluation/pdf-runs/unfinished"
    save_trace(run / "manifest.json", {"requirement": 6, "account_anchor_usd": 4.8})
    save_trace(run / "C-pdf-tools/usage-audit.json", {"records": [
        {"request_sent": True, "cost_usd": .01}, {"request_sent": True, "cost_usd": None, "estimated_maximum_usd": .018}]})
    totals = prior_totals(tmp_path)
    assert totals == {"account_anchor_usd": 4.8, "known_cost_usd": .01, "unknown_reserve_usd": .018, "saved_requests": 2}


def test_prerequisite_must_match_sources_and_native_transport(tmp_path):
    expected = {"source_hashes": {"pdf_agent.py": "frozen"}, "document_sha256": "doc"}
    save_trace(tmp_path / "manifest.json", {"stage": "probe", "finished_at_utc": "saved", **expected})
    path = tmp_path / "B-full-context/answers/DOC-PILOT.json"
    row = {"status": "completed", "output_origin": "model", "transport_formats": [{"response_format": "json_schema"}]}
    save_trace(path, row)
    validate_prerequisite(tmp_path, "probe", expected)
    with pytest.raises(ValueError, match="provenance"):
        validate_prerequisite(tmp_path, "probe", {**expected, "document_sha256": "changed"})
    write_json(path, {**row, "transport_formats": [{"response_format": "json_object"}]})
    with pytest.raises(ValueError, match="transport"):
        validate_prerequisite(tmp_path, "probe", expected)


def test_existing_run_refused_before_credentials_or_preflight(tmp_path):
    run = tmp_path / "data/evaluation/pdf-runs/existing"
    run.mkdir(parents=True)
    with patch("medical_copilot.pdf_evaluation.preflight") as preflight:
        with pytest.raises(ValueError, match="fresh"):
            run_stage(tmp_path, run, "main")
        preflight.assert_not_called()


def test_evaluator_fields_rejected_before_credentials(tmp_path):
    valid = {"question_id": "Q", "question": "What is stated?", "options": {"A": "Yes", "B": "No"}}
    with patch("medical_copilot.pdf_evaluation.validate_set", return_value=([], [])), patch("medical_copilot.pdf_evaluation.preflight") as preflight:
        with pytest.raises(ValueError):
            run_stage(tmp_path, tmp_path / "data/evaluation/pdf-runs/new", "ask", custom_questions=[{**valid, "answer_idx": "A"}])
        preflight.assert_not_called()


def test_output_identifier_cannot_escape_answer_directory(tmp_path):
    question = {"question_id": "../outside", "question": "What is stated?", "options": {"A": "Yes", "B": "No"}}
    with patch("medical_copilot.pdf_evaluation.validate_set", return_value=([], [])), patch("medical_copilot.pdf_evaluation.preflight") as preflight:
        with pytest.raises(ValueError, match="identifier"):
            run_stage(tmp_path, tmp_path / "data/evaluation/pdf-runs/new", "ask", custom_questions=[question])
        preflight.assert_not_called()


@pytest.mark.parametrize("error,expected", [("LengthFinishReasonError", True), ("AuthenticationError", False), ("BadRequestError", False)])
def test_only_recoverable_transport_or_length_errors_use_reserved_finalization(error, expected):
    assert recoverable_failure({"error_type": error, "http_status": None}) is expected


def test_observed_length_error_closes_once_and_retains_unknown_cost(tmp_path):
    import json
    from langchain_core.messages import AIMessage
    from medical_copilot.config import load_config
    from medical_copilot.models import Redactor
    from medical_copilot.pdf_agent import PDFClinicalAgent
    from medical_copilot.pdf_documents import PDFLibrary
    from medical_copilot.pdf_evaluation import make_model
    from medical_copilot.evaluation import read_json

    root = Path(__file__).resolve().parents[1]
    library = PDFLibrary.load([root / "data/documents/ingested/espen-cancer-v2-20261012/document.json"])
    config = load_config(root / "config/models/openrouter-gemma-4-reference.json")
    policy = read_json(root / "config/pdf-evaluation.json")
    length_error = type("LengthFinishReasonError", (Exception,), {})

    class Backend:
        def __init__(self):
            self.requests, self.formats = [], []

        def bind(self, *, response_format):
            self.formats.append(response_format)
            return self

        def invoke(self, messages):
            request = json.loads(messages[-1].content)
            self.requests.append(request)
            if len(self.requests) == 1:
                raise length_error("Synthetic private partial output must never be saved.")
            return AIMessage(content=json.dumps({"type": "final", "answer": "Unable to verify the unavailable document.",
                "justification": "Synthetic failure-recovery test; no medical conclusion.", "selected_option": None, "citations": [], "limitations": ["No document text."]}),
                response_metadata={"token_usage": {"cost": .0001}})

    backend = Backend()
    budget = request_budget(4.8, {"account_anchor_usd": 4.8, "known_cost_usd": 0, "unknown_reserve_usd": 0})
    with patch("medical_copilot.pdf_evaluation.create_backend", return_value=(backend, Redactor(), "fixture-no-key")):
        model, client, audit, phased, _ = make_model(config, root, budget, tmp_path / "audit.json", policy)
        row = PDFClinicalAgent(model, library, "A-no-document").run({"question_id": "FIXTURE", "question": "What is in the unavailable document?"})
    assert row["status"] == "completed" and row["output_origin"] == "model"
    assert [r["phase"] for r in backend.requests] == ["research", "finalize"]
    assert row["provider_failures"] == 1 and row["final_attempts"] == 1
    assert len(client.calls) == len(audit.records) == 2 and len(backend.formats) == 1
    assert backend.requests[-1]["tools"] == [] and backend.requests[-1]["evidence_registry"] == {}
    assert budget.unreported_charge_reserve_usd == pytest.approx(.15 + audit.records[0]["estimated_maximum_usd"])
    assert budget.spent_usd == pytest.approx(4.8001)
    assert "Synthetic private partial" not in (tmp_path / "audit.json").read_text()
