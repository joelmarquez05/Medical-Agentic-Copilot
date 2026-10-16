"""Pre-request validation and cumulative-accounting checks for patient applications."""
import json
from pathlib import Path
import runpy
import sys

import pytest

from medical_copilot.patient_memory import PatientMemory, PatientScopeError
from medical_copilot.patient_runtime import prior_usage, recoverable_error

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_patient_accounting_retains_pdf_and_incomplete_patient_charges_once(tmp_path):
    pdf = tmp_path / "data/evaluation/pdf-runs/old"
    write(pdf / "manifest.json", {"requirement": 6, "account_anchor_usd": 4})
    write(pdf / "B-full-context/usage-audit.json", {"records": [{"request_sent": True, "cost_usd": .01}, {"request_sent": True, "cost_usd": None, "estimated_maximum_usd": .02}]})
    child = tmp_path / "data/evaluation/patient-runs/new-child"
    write(child / "manifest.json", {"requirement": 7, "account_anchor_usd": 4})
    write(child / "usage-audit.json", {"records": [{"request_sent": True, "cost_usd": .03}]})
    # Aggregate metadata has no duplicate billing audit.
    write(tmp_path / "data/evaluation/patient-runs/aggregate/manifest.json", {"requirement": 7, "aggregate_only": True})
    result = prior_usage(tmp_path)
    assert result["known_cost_usd"] == pytest.approx(.04)
    assert result["unknown_reserve_usd"] == pytest.approx(.02)
    assert result["saved_requests"] == 3 and result["account_anchor_usd"] == 4


@pytest.mark.parametrize("case", ["foreign_conversation", "gold_input", "unknown_patient", "invalid_identifier"])
def test_application_rejects_invalid_scope_input_before_preflight(tmp_path, monkeypatch, case):
    database = tmp_path / "memory.sqlite3"
    with PatientMemory(database) as memory:
        memory.create_patient("alpha", "Synthetic Alpha")
        memory.create_patient("beta", "Synthetic Beta")
        memory.conversation("alpha", "alpha-thread", create=True)
    question = {"question_id": "X", "question": "Synthetic input"}
    if case == "gold_input":
        question["answer_key"] = "Must not enter generation"
    if case == "invalid_identifier":
        question["question_id"] = "../escape"
    input_path = tmp_path / "input.json"
    write(input_path, question)
    module = runpy.run_path(str(ROOT / "scripts/ask_patient.py"), run_name="patient_cli_test")
    called = []
    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("Preflight/credential access must not occur.")
    monkeypatch.setitem(module["main"].__globals__, "PatientRun", forbidden)
    patient = "missing" if case == "unknown_patient" else "beta" if case == "foreign_conversation" else "alpha"
    conversation = "alpha-thread" if case == "foreign_conversation" else "new-thread"
    monkeypatch.setattr(sys, "argv", ["ask_patient.py", "--database", str(database), "--patient", patient, "--conversation", conversation,
                                     "--input-json", str(input_path), "--run-dir", str(tmp_path / "output")])
    with pytest.raises((ValueError, PatientScopeError)):
        module["main"]()
    assert called == []


def test_observed_connection_alias_uses_one_reserved_recovery():
    assert recoverable_error({"error_type": "OpenAIConnectionError", "http_status": None})
    assert not recoverable_error({"error_type": "BadRequestError", "http_status": 400})


def test_reserved_recovery_after_observed_connection_failure(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from medical_copilot.comparison_evaluation import RequestBudget
    from medical_copilot.config import load_config
    from medical_copilot.models import Redactor
    from medical_copilot.patient_agent import PatientAgent
    import medical_copilot.patient_runtime as runtime

    class OpenAIConnectionError(Exception):
        pass
    class Transport:
        def __init__(self):
            self.count, self.formats = 0, []
        def bind(self, **kwargs):
            self.formats.append(kwargs)
            return self
        def invoke(self, messages):
            self.count += 1
            if self.count == 1:
                raise OpenAIConnectionError("Synthetic transport loss; not a real provider request.")
            return AIMessage(content=json.dumps({"type": "final", "answer": '{"weight_kg": null}', "justification": "No stored data supplied.",
                                                "selected_option": None, "citations": [], "limitations": ["Unknown patient measurement"]}),
                             response_metadata={"finish_reason": "stop", "token_usage": {"prompt_tokens": 100, "completion_tokens": 40, "cost": .00002}},
                             usage_metadata={"input_tokens": 100, "output_tokens": 40, "total_tokens": 140})
    transport = Transport()
    monkeypatch.setattr(runtime, "create_backend", lambda config, root: (transport, Redactor(), "synthetic_transport"))
    budget = RequestBudget(6)
    model, client, audit, phased, _ = runtime.make_model(load_config(ROOT / "config/models/openrouter-gemma-4-reference.json"), ROOT, budget, tmp_path / "audit.json", 65536)
    with PatientMemory(tmp_path / "memory.sqlite3") as memory:
        memory.create_patient("alpha", "Synthetic Alpha")
        result = PatientAgent(model, memory, "alpha", "control", memory_enabled=False).run({"question_id": "RECOVER", "question": "What is my recorded weight?"})
    assert result["status"] == "completed" and result["model_turns"] == 2 and result["final_attempts"] == 1
    assert result["response"]["answer"] == '{"weight_kg": null}'
    assert [r["status"] for r in audit.records] == ["error", "ok"]
    assert budget.unreported_charge_reserve_usd > 0 and phased.formats[-1]["response_format"] == "json_schema"
