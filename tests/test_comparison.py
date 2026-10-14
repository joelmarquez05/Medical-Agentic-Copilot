"""Offline controls for evidence identity, bounded execution and paid auditing."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from medical_copilot.comparison import ComparisonCopilot, ComparisonPlan
from medical_copilot.comparison_evaluation import AuditedBackend, BudgetLimitExceeded, RequestBudget, freeze_evidence
from medical_copilot.config import ModelConfig
from medical_copilot.evaluation import load_questions, read_json
from medical_copilot.models import ModelClient, Redactor

ROOT = Path(__file__).resolve().parents[1]


class Backend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages)
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, AIMessage) else AIMessage(content=json.dumps(value) if isinstance(value, dict) else value)


def plan():
    return {"scope": "oncology_nutrition", "answer_focus": ["Answer the synthetic test case."], "missing_information": [], "tool_calls": [{"name": "read_question_evidence", "arguments": {}}]}


def review(supported=True):
    return {"needs_revision": False, "issues": [], "uncertainty_note": "Test fixture only.", "retrieval_fit": "direct", "paper_checks": [{"source_id": "P123", "reported_result": "Fictional fixture result, never used as clinical evidence.", "draft_claim_supported": supported, "issue": "Reverse attribution." if not supported else ""}]}


def evidence():
    return {"mode": "fixed_test_fixture", "cards": [], "sources": [{"id": "P123", "authors": "Fictional author", "year": "2000", "title": "TEST ONLY: fictional record", "url": "https://example.invalid/123", "abstract": "A fictional adverse result, for testing only."}]}


def agent(outputs, fixed=None, **options):
    backend = Backend(outputs)
    config = ModelConfig(provider="google_genai", model="test-only", **options)
    return ComparisonCopilot(ModelClient(backend, config), fixed or evidence()), backend


def test_cannot_replace_bound_evidence_with_query_or_patient_identifier():
    value = plan()
    value["tool_calls"][0]["arguments"] = {"query": "other evidence"}
    with pytest.raises(ValidationError):
        ComparisonPlan.model_validate(value)
    value["tool_calls"] = []
    with pytest.raises(ValidationError):
        ComparisonPlan.model_validate(value)


def test_fixed_graph_is_complete_without_any_live_search_and_uses_copy():
    original = evidence()
    copilot, backend = agent([plan(), "Fixture answer [P123].", review()], original)
    original["sources"][0]["abstract"] = "Mutation after construction."
    result = copilot.run("Synthetic adult cancer nutrition test.")
    assert result["status"] == "self_review_passed"
    assert result["live_pubmed_requests"] == 0
    assert result["trace"] == ["plan", "execute_tools", "draft", "review", "finalize"]
    assert "Mutation after construction" not in backend.requests[1][1].content
    assert "No successful PubMed search" not in str(result)
    assert "previously retrieved" in result["answer"]


def test_attribution_failure_forces_revision_with_identical_evidence():
    copilot, backend = agent([plan(), "Faulty [P123].", review(False), "Corrected [P123].", review()])
    result = copilot.run("Synthetic adult cancer nutrition test.")
    assert result["revision_count"] == 1
    assert result["status"] == "self_review_passed"
    draft_context = json.loads(backend.requests[1][1].content)
    revision_context = json.loads(backend.requests[3][1].content)
    assert draft_context["tool_results"] == revision_context["tool_results"]


def test_missing_paper_check_cannot_silently_pass():
    absent = dict(review(), paper_checks=[])
    copilot, _ = agent([plan(), "Fixture [P123].", absent], max_revisions=0)
    result = copilot.run("Synthetic adult cancer nutrition test.")
    assert result["status"] == "needs_human_review"
    assert "attribution check" in str(result["mechanical_issues"])


def test_nullable_positive_issue_is_accepted_and_nullable_negative_still_revises():
    positive = review()
    positive["paper_checks"][0]["issue"] = None
    negative = review(False)
    negative["paper_checks"][0]["issue"] = None
    copilot, _ = agent([plan(), "Faulty [P123].", negative, "Corrected [P123].", positive])
    result = copilot.run("Synthetic adult cancer nutrition test.")
    assert result["revision_count"] == 1
    assert result["status"] == "self_review_passed"
    assert result["reviews"][0]["paper_checks"][0]["draft_claim_supported"] is False


def test_interrupted_review_preserves_draft_but_never_a_final_answer():
    copilot, _ = agent([plan(), "Fixture [P123].", TimeoutError("test-only")])
    result = copilot.run("Synthetic adult cancer nutrition test.")
    assert result["status"] == "error" and result["answer"] == ""
    assert result["partial_answer"] == "Fixture [P123]."
    assert result["tool_results"][0]["output"] == evidence()


def test_real_frozen_union_is_identical_and_contains_observed_conflicting_trial(tmp_path):
    questions = load_questions(ROOT / "data/evaluation/oncology_nutrition_questions.json")["questions"]
    manifest = freeze_evidence(ROOT, tmp_path, questions)
    assert [len(entry["pubmed_ids"]) for entry in manifest["questions"]] == [7, 5, 6, 8, 9, 5, 10, 6, 4, 5]
    fixed = read_json(tmp_path / "ON07.json")
    assert len(fixed["cards"]) == 10
    trial = next(source for source in fixed["sources"] if source["id"] == "P41406403")
    assert "31.4%" in trial["abstract"] and "20.2%" in trial["abstract"]
    assert not any(key in str(fixed) for key in ["expected_key_points", "critical_failures", "relevance_grade"])
    with pytest.raises(ValueError):
        freeze_evidence(ROOT, tmp_path, questions)


def paid_config():
    return ModelConfig(provider="openai_compatible", model="test-only", base_url="https://openrouter.ai/api/v1", max_tokens=256, provider_options={"extra_body": {"provider": {"max_price": {"prompt": 4, "completion": 20}}}})


def test_budget_blocks_before_provider_request(tmp_path):
    backend = Backend([])
    audited = AuditedBackend(backend, paid_config(), RequestBudget(0.001), Redactor(), tmp_path / "audit.json")
    with pytest.raises(BudgetLimitExceeded):
        audited.invoke([HumanMessage(content="Test.")])
    assert not backend.requests
    assert audited.records[0]["request_sent"] is False


def test_audit_uses_reported_cost_and_excludes_private_reasoning(tmp_path):
    response = AIMessage(content="OK", additional_kwargs={"reasoning_content": "PRIVATE TEST THOUGHT"}, usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}, response_metadata={"token_usage": {"cost": 0.00005}, "model_name": "test-only"})
    budget = RequestBudget(1)
    audited = AuditedBackend(Backend([response]), paid_config(), budget, Redactor(), tmp_path / "audit.json")
    audited.invoke([HumanMessage(content="Test.")])
    assert budget.spent_usd == 0.00005
    assert "PRIVATE TEST THOUGHT" not in (tmp_path / "audit.json").read_text()


def test_failed_paid_request_keeps_unknown_charge_reserve(tmp_path):
    budget = RequestBudget(1)
    audited = AuditedBackend(Backend([TimeoutError("private error body")]), paid_config(), budget, Redactor(), tmp_path / "audit.json")
    with pytest.raises(TimeoutError):
        audited.invoke([HumanMessage(content="Test.")])
    assert budget.unreported_charge_reserve_usd > 0
    assert "private error body" not in (tmp_path / "audit.json").read_text()


def test_provider_circuit_records_unattempted_cases_separately(tmp_path, monkeypatch):
    from medical_copilot import comparison_evaluation
    backend = Backend([TimeoutError("test-only") for _ in range(3)])
    monkeypatch.setattr(comparison_evaluation, "create_backend", lambda *args: (backend, Redactor(), "test_only"))
    # Disable pacing/retry only in this offline fixture.
    original_load = comparison_evaluation.load_config
    monkeypatch.setattr(comparison_evaluation, "load_config", lambda path: original_load(path).model_copy(update={"min_interval_seconds": 0, "rate_limit_retries": 0}))
    summary = comparison_evaluation.run_comparison(ROOT, tmp_path / "run", profiles=["gemini-3.8-flash"], ids=["ON01", "ON02", "ON03", "ON04"])
    observed = summary["models"][0]
    assert observed["attempted_cases"] == 3
    assert observed["unattempted_provider_unavailable_cases"] == 1
    assert observed["completed_answers"] == 0
    assert len(backend.requests) == 3
    skipped = read_json(tmp_path / "run/models/gemini-3.8-flash/answers/ON04.json")
    assert skipped["status"] == "not_run_provider_unavailable" and skipped["model_calls"] == []


class ServerFailure(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__("Private error body for test only.")


def test_separate_recovery_retries_only_transient_server_errors(tmp_path):
    from medical_copilot.comparison_recovery import TransientRetryBackend
    backend = Backend([ServerFailure(503), ServerFailure(500), AIMessage(content="OK")])
    config = ModelConfig(provider="google_genai", model="test-only")
    audited = AuditedBackend(backend, config, RequestBudget(1), Redactor(), tmp_path / "audit.json")
    waits = []
    wrapped = TransientRetryBackend(audited, sleep=waits.append)
    messages = [HumanMessage(content="Unchanged fixture prompt.")]
    assert wrapped.invoke(messages).content == "OK"
    assert waits == [30, 30] and backend.requests == [messages, messages, messages]
    assert "Private error body" not in (tmp_path / "audit.json").read_text()


def test_recovery_never_retries_a_timeout_or_authentication_error(tmp_path):
    from medical_copilot.comparison_recovery import TransientRetryBackend
    for code in [504, 401, 403, 404, 429]:
        backend = Backend([ServerFailure(code)])
        config = ModelConfig(provider="google_genai", model="test-only")
        audited = AuditedBackend(backend, config, RequestBudget(1), Redactor(), tmp_path / f"audit-{code}.json")
        wrapped = TransientRetryBackend(audited, sleep=lambda seconds: pytest.fail("This error must not be retried here."))
        with pytest.raises(ServerFailure):
            wrapped.invoke([HumanMessage(content="Test.")])
        assert len(backend.requests) == 1


def test_transient_recovery_attempt_count_is_bounded(tmp_path):
    from medical_copilot.comparison_recovery import TransientRetryBackend
    backend = Backend([ServerFailure(503) for _ in range(3)])
    config = ModelConfig(provider="google_genai", model="test-only")
    audited = AuditedBackend(backend, config, RequestBudget(1), Redactor(), tmp_path / "audit.json")
    wrapped = TransientRetryBackend(audited, sleep=lambda seconds: None)
    with pytest.raises(ServerFailure):
        wrapped.invoke([HumanMessage(content="Test.")])
    assert len(backend.requests) == 3 and len(audited.records) == 3


def test_connection_failure_recovery_is_bounded_and_reserves_unknown_cost(tmp_path):
    from medical_copilot.comparison_recovery import TransientRetryBackend
    failure = type("OpenAIConnectionError", (Exception,), {})
    response = AIMessage(content="OK", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, response_metadata={"token_usage": {"cost": 0.00005}})
    budget = RequestBudget(1)
    backend = Backend([failure("private fixture"), response])
    audited = AuditedBackend(backend, paid_config(), budget, Redactor(), tmp_path / "audit.json")
    wrapped = TransientRetryBackend(audited, sleep=lambda seconds: None)
    assert wrapped.invoke([HumanMessage(content="Test.")]).content == "OK"
    assert len(backend.requests) == 2
    assert budget.unreported_charge_reserve_usd > 0 and budget.spent_usd == 0.00005
