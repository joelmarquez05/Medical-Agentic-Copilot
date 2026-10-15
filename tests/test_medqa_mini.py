"""Offline scoring, label isolation, call limits and spend safeguards."""
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from medical_copilot.comparison_evaluation import BudgetLimitExceeded, RequestBudget
from medical_copilot.medqa_mini import INPUT_BYTE_CAP, MiniAudit, MiniCopilot, cohort_bound, model_configs, prepare_subset, question_input, score_answer
from medical_copilot.models import ModelClient, Redactor

ROOT = Path(__file__).resolve().parents[1]
QUESTION = {"id": "MQ-test", "question": "An examination question.", "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"}, "answer_idx": "A", "answer": "One", "reference_rationale": "Private evaluator fixture"}


class Backend:
    def __init__(self, responses):
        self.responses, self.prompts = list(responses), []

    def invoke(self, messages):
        self.prompts.append(messages)
        return self.responses.pop(0)


class Search:
    requests = []

    def __init__(self):
        self.queries = []

    def search(self, query, max_results):
        self.queries.append((query, max_results))
        return {"sources": [{"id": "P123", "title": "Relevant fixture", "year": "2020", "abstract": "Some evidence."}], "limitation": "Abstract fixture only."}


def client_for(backend):
    config = model_configs()[0][1].model_copy(update={"min_interval_seconds": 0})
    return ModelClient(backend, config)


def test_external_key_alone_determines_correctness():
    record = {"question_id": "MQ-test", "profile": "test", "mode": "direct", "status": "completed", "answer": {"selected_option": "B", "justification": "A persuasive explanation claiming that A is right."}}
    assert score_answer(record, {"answer_idx": "A"})["correct"] is False
    record["answer"]["selected_option"] = "A"
    assert score_answer(record, {"answer_idx": "A"})["correct"] is True
    record["status"] = "error"
    assert score_answer(record, {"answer_idx": "A"})["selected_option"] is None


def test_gold_and_metadata_are_excluded_from_all_model_prompts():
    assert set(question_input(QUESTION)) == {"question", "options"}
    backend = Backend([AIMessage(content='{"use_pubmed": false, "query": null, "reason": "No search needed."}'), AIMessage(content='{"selected_option": "A", "justification": "One is the answer."}')])
    result = MiniCopilot(client_for(backend), Search(), "agent").run(QUESTION)
    assert result["status"] == "completed"
    for messages in backend.prompts:
        payload = json.loads(messages[1].content)
        assert "answer_idx" not in payload and "reference_rationale" not in payload
        assert "Private evaluator fixture" not in messages[1].content


def test_optional_search_is_one_query_and_two_model_calls():
    backend = Backend([AIMessage(content='{"use_pubmed": true, "query": "medical fixture", "reason": "Check evidence."}'), AIMessage(content='{"selected_option": "A", "justification": "Source supports the answer [P123]."}')])
    search = Search()
    result = MiniCopilot(client_for(backend), search, "agent").run(QUESTION)
    assert result["trace"] == ["decide", "retrieve", "answer"]
    assert len(backend.prompts) == 2 and search.queries == [("medical fixture", 3)]


def test_format_or_truncation_failure_has_no_paid_repair():
    for response in [AIMessage(content="not JSON"), AIMessage(content='{"selected_option": "A", "justification": "Partial"}', response_metadata={"finish_reason": "length"})]:
        backend = Backend([response])
        result = MiniCopilot(client_for(backend), Search(), "direct").run(QUESTION)
        assert result["status"] == "error" and len(backend.prompts) == 1


def test_oversized_prompt_and_budget_overrun_do_not_invoke_provider(tmp_path):
    config = model_configs()[0][1]
    backend = Backend([])
    audit = MiniAudit(backend, config, RequestBudget(6, 5.99), Redactor(), tmp_path / "audit.json")
    with pytest.raises(ValueError):
        audit.invoke([HumanMessage(content="x" * (INPUT_BYTE_CAP + 1))])
    with pytest.raises(BudgetLimitExceeded):
        audit.invoke([HumanMessage(content="Short question")])
    assert backend.prompts == []


def test_paid_profile_panel_excludes_opus_and_preserves_reasoning_decision():
    profiles = model_configs()
    assert len(profiles) == 5 and all("opus" not in config.model for _, config in profiles)
    assert [config.provider_options["extra_body"]["reasoning"].get("effort") for _, config in profiles] == ["high", "high", None, "xhigh", None]
    assert all(config.max_retries == config.rate_limit_retries == config.json_repair_attempts == config.max_revisions == 0 for _, config in profiles)
    assert 3.655788991 + 0.15 + cohort_bound() < 6


def test_fixed_selection_reproduces_original_dataset_keys(tmp_path):
    source = ROOT / "data/evaluation/medqa-mini-20261011/source-test.jsonl"
    first = prepare_subset(source, tmp_path / "a")
    second = prepare_subset(source, tmp_path / "b")
    assert first == second
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    questions = json.loads((tmp_path / "a/questions.json").read_text(encoding="utf-8"))["questions"]
    keys = json.loads((tmp_path / "a/answer-key.json").read_text(encoding="utf-8"))["keys"]
    for question, key in zip(questions, keys):
        source_row = rows[question["source_row_index"]]
        assert question_input(question) == question_input(source_row)
        assert key["answer_idx"] == source_row["answer_idx"] and key["answer"] == source_row["answer"]
    with pytest.raises(ValueError):
        prepare_subset(source, tmp_path / "a")


def test_sixth_model_extension_retains_caps_without_mutating_primary_panel():
    from medical_copilot.medqa_extension import extension_bound, ling_config
    config = ling_config()
    assert config.model == "inclusionai/ling-3.0-flash-sante"
    assert config.max_tokens == model_configs()[0][1].max_tokens
    assert config.provider_options["extra_body"]["reasoning"] == {"enabled": True, "exclude": True}
    assert len(model_configs()) == 5
    assert 3.655788991 + 0.15 + cohort_bound() + extension_bound() < 6


def test_extension_refuses_to_run_while_primary_is_active(tmp_path):
    from medical_copilot.medqa_extension import run_extension
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "manifest.json").write_text('{"started_at_utc": "fixture"}')
    with pytest.raises(ValueError, match="only after"):
        run_extension(tmp_path, primary, tmp_path / "extension")
    assert not (tmp_path / "extension").exists()


def test_option_diagnostic_extracts_explicit_json_without_a_model_judge():
    from medical_copilot.medqa_option_diagnostic import visible_option
    record = {"status": "error", "model_calls": [{"stage": "final_answer", "status": "ok", "response_metadata": {"finish_reason": "stop"}, "response": 'Extra prose. {"selected_option":"D","justification":"Follow up."}'}]}
    answer = visible_option(record)
    assert answer["selected_option"] == "D" and record["status"] == "error"
    record["model_calls"][0]["response_metadata"]["finish_reason"] = "length"
    assert visible_option(record) is None


def test_option_diagnostic_refuses_ambiguous_multiple_objects():
    from medical_copilot.medqa_option_diagnostic import visible_option
    text = '{"selected_option":"A","justification":"First"} {"selected_option":"D","justification":"Second"}'
    record = {"status": "error", "model_calls": [{"stage": "final_answer", "status": "ok", "response_metadata": {"finish_reason": "stop"}, "response": text}]}
    assert visible_option(record) is None
