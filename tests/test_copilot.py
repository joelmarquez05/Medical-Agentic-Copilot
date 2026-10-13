"""Offline engineering checks. Scripted responses are never evaluation data."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from medical_copilot.config import ConfigurationError, ModelConfig, load_config, resolve_credential
from medical_copilot.evaluation import load_questions, run_evaluation
from medical_copilot.graph import MedicalCopilot, citation_issues, cited_source_ids
from medical_copilot.models import ModelClient, ModelError, Redactor, create_backend, message_text
from medical_copilot.tools import estimate_nutrition_targets

ROOT = Path(__file__).resolve().parents[1]


class ScriptedBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        text = json.dumps(output) if isinstance(output, dict) else output
        return AIMessage(content=text, usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}, response_metadata={"finish_reason": "STOP"})


def config(**kwargs):
    return ModelConfig(provider="google_genai", model="test-only-model", **kwargs)


def plan(topics=None, calculator=False, scope="oncology_nutrition"):
    calls = [{"name": "lookup_guidance", "arguments": {"topics": topics or ["requirements"]}}] if topics is not False else []
    if calculator:
        calls.append({"name": "estimate_nutrition_targets", "arguments": {"weight_kg": 60}})
    return {"scope": scope, "answer_focus": ["Answer the question."], "missing_information": [], "tool_calls": calls}


def review(needs=False, issues=None, additional=None):
    return {"needs_revision": needs, "issues": issues or [], "additional_topics": additional or [], "uncertainty_note": "This is an internal self-review only."}


def agent(outputs, **kwargs):
    backend = ScriptedBackend(outputs)
    return MedicalCopilot(ModelClient(backend, config(**kwargs))), backend


def test_calculator_preserves_protein_inequality_and_units():
    result = estimate_nutrition_targets(60)
    assert result["energy_kcal_day"] == {"lower": 1500.0, "upper": 1800.0}
    assert result["protein_g_day"] == {"greater_than": 60.0, "up_to": 90.0}
    assert "refeeding" in result["limitation"]


@pytest.mark.parametrize("weight", [-1, 0, float("nan"), float("inf"), 501, True, "60"])
def test_calculator_rejects_invalid_weights(weight):
    with pytest.raises(ValidationError):
        estimate_nutrition_targets(weight)


def test_environment_key_takes_precedence_without_echo(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_COPILOT_KEY", "environment-value-for-unit-test")
    (tmp_path / "api_key.md").write_text("file-value-for-unit-test-123", encoding="utf-8")
    settings = config(api_key_env="TEST_COPILOT_KEY", credential_file="api_key.md")
    key, source = resolve_credential(settings, tmp_path)
    assert key == "environment-value-for-unit-test" and source == "environment"
    assert key not in json.dumps(settings.public_dict())


@pytest.mark.parametrize("options", [{"api_key": "do-not-echo-this"}, {"nested": {"authorization": "do-not-echo-this"}}, {"max_tokens": 1}])
def test_literal_credentials_or_shared_overrides_fail_safely(tmp_path, options):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"provider": "google_genai", "model": "test", "provider_options": options}), encoding="utf-8")
    with pytest.raises(ConfigurationError) as captured:
        load_config(path)
    assert "do-not-echo-this" not in str(captured.value)


def test_credential_url_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai_compatible", model="test", base_url="https://example.com/v1?key=secret")


def test_successful_graph_uses_selected_tools_and_no_extra_calls():
    copilot, backend = agent([plan(calculator=True), "General targets: 1500-1800 kcal/day; protein above 60 and up to 90 g/day [S01].", review()])
    result = copilot.run("What are general targets for a 60 kg adult with cancer?")
    assert result["status"] == "self_review_passed"
    assert result["trace"] == ["plan", "execute_tools", "draft", "review", "finalize"]
    assert len(backend.messages) == 3
    assert result["tool_results"][1]["output"]["energy_kcal_day"]["lower"] == 1500
    assert "ESPEN" in result["answer"]


def test_reviewer_can_request_guidance_and_revision():
    copilot, backend = agent([plan(), "Incomplete advice [S01].", review(True, ["Add refeeding precautions."], ["refeeding"]), "Cautious restart requires expert assessment [S04].", review()])
    result = copilot.run("Explain cancer nutrition and refeeding precautions.")
    assert result["revision_count"] == 1
    assert result["status"] == "self_review_passed"
    assert result["tool_results"][-1]["arguments"]["topics"] == ["refeeding"]
    assert len(backend.messages) == 5


def test_unresolved_review_stops_at_revision_limit():
    copilot, backend = agent([plan(), "Advice [S01].", review(True, ["Material issue."]), "Revised advice [S01].", review(True, ["Still unresolved."])], max_revisions=1)
    result = copilot.run("A question about adult cancer nutrition.")
    assert result["status"] == "needs_human_review"
    assert result["revision_count"] == 1 and len(backend.messages) == 5


def test_mechanical_unknown_citation_triggers_revision():
    copilot, _ = agent([plan(), "Targets [S99].", review(), "Targets [S01].", review()])
    result = copilot.run("Adult cancer nutrition targets.")
    assert result["revision_count"] == 1
    assert "[S99]" not in result["answer"]
    assert result["mechanical_issues"] == []


def test_grouped_citations_include_every_source_and_detect_bad_labels():
    assert cited_source_ids("Advice [S01, S08] and another [G08, S06].") == {"S01", "S08", "S06"}
    assert citation_issues("Advice [S01, S08].", {"S01", "S08"}, True) == []
    assert citation_issues("Advice [S05 is invalid; see S01].", {"S01"}, True) == ["Unknown or unavailable source marker [S05]."]
    copilot, _ = agent([plan(["food_safety"]), "Food hygiene [S05, S08].", review()])
    result = copilot.run("Explain food hygiene in adult cancer care.")
    assert {source["id"] for source in result["sources"]} == {"S05", "S08"}


def test_out_of_scope_has_no_tool_execution():
    copilot, _ = agent([plan(scope="out_of_scope"), "This prototype covers oncology nutrition.", review()])
    result = copilot.run("How can I repair my car?")
    assert result["status"] == "insufficient_evidence"
    assert result["tool_results"] == []


def test_invalid_tool_arguments_are_not_silently_fixed():
    invalid = plan(topics=False)
    invalid["tool_calls"] = [{"name": "estimate_nutrition_targets", "arguments": {"weight_kg": -60}}]
    copilot, _ = agent([invalid, "The supplied calculator input is unusable.", review()])
    result = copilot.run("What targets apply to a 60 kg adult?")
    assert result["tool_results"][0]["status"] == "error"
    assert result["status"] == "insufficient_evidence"


def test_provider_exception_does_not_leak_credentials():
    secret = "unit-test-secret-not-a-real-key"
    client = ModelClient(ScriptedBackend([RuntimeError(secret)]), config(), Redactor([secret]))
    result = MedicalCopilot(client).run("A benign question.")
    assert result["status"] == "error"
    assert secret not in json.dumps(result)
    assert result["model_calls"][0]["error_type"] == "RuntimeError"


def test_interrupted_graph_preserves_completed_evidence_and_draft():
    copilot, _ = agent([plan(calculator=True), "Partial draft [S01].", RuntimeError("test-only interrupted review")])
    result = copilot.run("General targets for a 60 kg adult with cancer?")
    assert result["status"] == "error" and result["answer"] == ""
    assert result["partial_answer"] == "Partial draft [S01]."
    assert len(result["tool_results"]) == 2
    assert result["trace"] == ["plan", "execute_tools", "draft"]


def test_agent_rejects_a_calculator_weight_invented_by_the_planner():
    fabricated = plan(topics=False, calculator=True)
    copilot, _ = agent([fabricated, "A weight is needed for calculation.", review()])
    result = copilot.run("How should general adult cancer nutrition targets be estimated?")
    assert result["tool_results"][0]["status"] == "error"
    assert result["status"] == "insufficient_evidence"


def test_json_repair_is_bounded_and_logged():
    copilot, backend = agent(["not JSON", plan(), "Advice [S01].", review()])
    result = copilot.run("Adult cancer nutrition.")
    assert result["status"] == "self_review_passed"
    assert [item["stage"] for item in result["model_calls"]] == ["plan", "plan_json_repair", "draft", "review"]
    failed, backend = agent(["not JSON", "still not JSON"])
    assert failed.run("Adult cancer nutrition.")["status"] == "error"
    assert len(backend.messages) == 2


def test_text_blocks_exclude_reasoning():
    message = AIMessage(content=[{"type": "reasoning", "reasoning": "Do not expose this."}, {"type": "text", "text": "Visible answer."}])
    assert message_text(message) == "Visible answer."


def test_openai_compatible_factory_routing(monkeypatch, tmp_path):
    recorded = {}
    def fake_factory(**kwargs):
        recorded.update(kwargs)
        return ScriptedBackend([])
    monkeypatch.setattr("medical_copilot.models.init_chat_model", fake_factory)
    monkeypatch.setenv("TEST_CF_KEY", "fake-cloud-token-for-unit-test")
    settings = ModelConfig(provider="openai_compatible", model="example-model", api_key_env="TEST_CF_KEY", base_url="https://example.com/ai/v1")
    _, redactor, source = create_backend(settings, tmp_path)
    assert recorded["model_provider"] == "openai"
    assert recorded["base_url"] == settings.base_url
    assert recorded["use_responses_api"] is False
    assert source == "environment"
    assert "fake-cloud-token-for-unit-test" not in redactor.text(json.dumps(recorded))


def test_rate_limit_retry_is_bounded_and_auditable():
    class SyntheticRateLimitError(RuntimeError):
        pass
    client = ModelClient(ScriptedBackend([SyntheticRateLimitError("unit test"), "OK"]), config(rate_limit_retries=1, rate_limit_wait_seconds=0))
    assert client.generate("Test", "Test", "test_only") == "OK"
    assert [attempt["status"] for attempt in client.calls[0]["attempts"]] == ["error", "ok"]
    client = ModelClient(ScriptedBackend([SyntheticRateLimitError("unit test"), SyntheticRateLimitError("unit test")]), config(rate_limit_retries=1, rate_limit_wait_seconds=0))
    with pytest.raises(ModelError):
        client.generate("Test", "Test", "test_only")
    assert len(client.calls[0]["attempts"]) == 2


@pytest.mark.parametrize("profile,model", [("nvidia-kimi-k3.json", "moonshotai/kimi-k3"), ("nvidia-deepseek-v4.1-flash.json", "deepseek-ai/deepseek-v4.1-flash"), ("nvidia-gpt-oss-20b.json", "openai/gpt-oss-20b")])
def test_nvidia_profiles_validate_with_verified_endpoint_ids(profile, model):
    settings = load_config(ROOT / "config/models" / profile)
    assert settings.model == model
    assert settings.provider == "openai_compatible"
    assert settings.base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.api_key_env == "NVIDIA_API_KEY"


def test_questions_are_ten_unique_synthetic_cases():
    dataset = load_questions(ROOT / "data/evaluation/oncology_nutrition_questions.json")
    assert len(dataset["questions"]) == 10
    assert len({question["topic"] for question in dataset["questions"]}) == 10


def test_evaluation_gold_points_do_not_enter_model_prompts(tmp_path, monkeypatch):
    dataset = {"dataset_id": "test-only-dataset", "questions": [{"id": "TEST01", "topic": "requirements", "question": "Adult cancer nutrition targets?", "expected_key_points": ["GOLD_CRITERIA_MUST_NOT_LEAK"]}]}
    question_path = tmp_path / "questions.json"
    question_path.write_text(json.dumps(dataset), encoding="utf-8")
    backend = ScriptedBackend([plan(), "General guidance [S01].", review()])
    monkeypatch.setattr("medical_copilot.evaluation.create_backend", lambda *args: (backend, Redactor(), "test_only"))
    run_dir = tmp_path / "offline-test-run"
    result = run_evaluation(config(), ROOT, question_path, ROOT / "data/evaluation/qualitative_rubric.json", run_dir, delay_seconds=0)
    assert result["question_count"] == 1
    assert "GOLD_CRITERIA_MUST_NOT_LEAK" not in " ".join(str(message.content) for group in backend.messages for message in group)
    assert (run_dir / "answers/TEST01.json").exists()
    with pytest.raises(ValueError):
        run_evaluation(config(), ROOT, question_path, ROOT / "data/evaluation/qualitative_rubric.json", run_dir, delay_seconds=0)
