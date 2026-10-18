"""Offline regressions for observed transport/closing failures and honest fallback."""

import copy
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from medical_copilot.clinical_graph import BudgetBlocked, EvidenceRecord, ToolDefinition, ToolOutput, UnusableModelResponse
from medical_copilot.clinical_resilient import (
    PhaseBackend, ResilientClinicalAgent, ResilientJSONModel, ResilientLimits,
    RESILIENT_SYSTEM, final_schema, normalize_action,
)
from medical_copilot.config import ModelConfig
from medical_copilot.models import ModelClient, ModelError

ROOT = Path(__file__).resolve().parents[1]
QUESTION = {"question_id": "SYNTHETIC", "question": "Which fixture marker is present?", "options": {"A": "BLUE", "B": "RED"}}
FINAL = {"type": "final", "answer": "BLUE", "justification": "Synthetic test only.", "selected_option": "A"}
TOOL = {"type": "tool", "name": "fixture_lookup", "arguments": {"query": "marker"}}


class Model:
    def __init__(self, responses, transient=False):
        self.responses, self.requests, self.transient = list(responses), [], transient

    def reply(self, request):
        self.requests.append(copy.deepcopy(request))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value(request) if callable(value) else value

    def may_retry_provider(self):
        return self.transient


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


def fixture_tool(executions):
    def execute(args):
        executions.append(args.query)
        return ToolOutput(content="Synthetic tool result.", sources=[EvidenceRecord(
            id="SYNTHETIC_SOURCE", title="Fixture", text="BLUE", text_kind="synthetic")])
    return ToolDefinition("fixture_lookup", "Synthetic fixture; no network.", Args, execute)


def test_nested_action_is_recovered_without_changing_arguments():
    raw = {"action": {"type": "tool", "name": "fixture_lookup", "arguments": {"query": "marker"}}}
    action, notes = normalize_action(raw)
    assert action == raw["action"]
    assert notes == ["unwrapped_single_action_object"]


@pytest.mark.parametrize("raw,note", [
    ({"action": "final", "answer": "BLUE", "justification": "Synthetic", "selected_option": "B"}, "renamed_action_discriminator_to_type"),
    ({"answer": "BLUE", "justification": "Synthetic", "selected_option": "B"}, "inferred_unique_final_discriminator"),
    ({"tool": "fixture_lookup", "arguments": {"query": "marker"}}, "canonicalized_tool_name_alias"),
])
def test_unambiguous_aliases_preserve_payload(raw, note):
    action, notes = normalize_action(raw)
    assert note in notes
    for key in ["answer", "justification", "selected_option", "arguments"]:
        if key in raw:
            assert action[key] == raw[key]


@pytest.mark.parametrize("raw", [
    {"action": FINAL, "another_action": TOOL},
    {**FINAL, "action": "tool"},
    {"action": [FINAL, TOOL]},
])
def test_ambiguous_actions_are_not_silently_selected(raw):
    with pytest.raises(ValueError):
        normalize_action(raw)


def test_null_lists_and_letter_case_do_not_rewrite_medical_text_or_choice():
    raw = {**FINAL, "selected_option": " b ", "citations": None, "limitations": None}
    result, notes = normalize_action(raw)
    assert result["selected_option"] == "B" and result["answer"] == raw["answer"]
    assert result["justification"] == raw["justification"]
    assert result["citations"] == result["limitations"] == [] and len(notes) == 3


def test_direct_answer_is_one_call_and_wrong_option_is_never_quality_retried():
    model = Model([{**FINAL, "selected_option": "B"}])
    result = ResilientClinicalAgent(model).run(QUESTION)
    assert result["status"] == "completed" and result["model_turns"] == 1
    assert result["final_attempts"] == 0 and result["tool_actions"] == 0
    assert result["response"]["selected_option"] == "B"


def test_tool_budget_routes_to_reserved_final_answer_instead_of_stopping():
    executions = []

    def close(request):
        assert request["phase"] == "finalize" and request["tools"] == []
        assert request["action_schema"]["properties"]["type"]["enum"] == ["final"]
        assert request["evidence_registry"]["SYNTHETIC_SOURCE"]["text"] == "BLUE"
        return {**FINAL, "citations": ["SYNTHETIC_SOURCE"]}

    model = Model([TOOL, TOOL, TOOL, close])
    result = ResilientClinicalAgent(model, [fixture_tool(executions)]).run(QUESTION)
    assert result["status"] == "completed" and len(executions) == 3
    assert result["model_turns"] == 4 and result["final_attempts"] == 1


def test_tool_request_during_finalization_is_not_executed_then_recovery_answers():
    executions = []
    model = Model([TOOL, TOOL, TOOL, TOOL, FINAL])
    result = ResilientClinicalAgent(model, [fixture_tool(executions)]).run(QUESTION)
    assert result["status"] == "completed" and len(executions) == 3
    assert result["model_turns"] == 5 and result["final_attempts"] == 2


def test_unrecoverable_format_enters_closing_phase_with_question_and_evidence():
    executions = []
    model = Model([TOOL, {"type": "unsupported"}, FINAL])
    result = ResilientClinicalAgent(model, [fixture_tool(executions)]).run(QUESTION)
    assert result["status"] == "completed" and result["final_attempts"] == 1
    assert model.requests[-1]["question"] == QUESTION
    assert model.requests[-1]["evidence_registry"]["SYNTHETIC_SOURCE"]["text"] == "BLUE"


def test_repeated_bad_final_has_honest_application_message_not_fake_medical_answer():
    model = Model([{"type": "bad"}, {"type": "bad"}, {"type": "bad"}])
    result = ResilientClinicalAgent(model).run(QUESTION)
    assert result["status"] == "operationally_unanswered" and result["response"]
    assert result["output_origin"] == "application" and result["response"]["selected_option"] is None
    assert len(model.requests) == 3
    assert result["response"]["selected_option"] is None


def test_transient_provider_failure_gets_one_closing_recovery_and_no_raw_error_leak():
    model = Model([ModelError("FAKE_PRIVATE_EXCEPTION"), FINAL], transient=True)
    result = ResilientClinicalAgent(model).run(QUESTION)
    assert result["status"] == "completed" and len(model.requests) == 2
    assert "FAKE_PRIVATE" not in json.dumps(result)


def test_repeated_provider_failure_stops_after_one_retry_but_returns_message():
    model = Model([ModelError("private"), ModelError("private")], transient=True)
    result = ResilientClinicalAgent(model).run(QUESTION)
    assert result["status"] == "operationally_unanswered" and len(model.requests) == 2
    assert result["response"] and result["output_origin"] == "application"


def test_permanent_provider_failure_is_not_retried():
    model = Model([ModelError("private")])
    result = ResilientClinicalAgent(model).run(QUESTION)
    assert result["status"] == "operationally_unanswered" and len(model.requests) == 1


def test_truncated_output_is_recovered_or_explicitly_unanswered_never_medically_scored_wrong():
    model = Model([UnusableModelResponse("Synthetic truncation"), FINAL])
    result = ResilientClinicalAgent(model).run(QUESTION)
    assert result["status"] == "completed" and len(model.requests) == 2


def test_budget_block_returns_terminal_message_without_a_model_generation():
    class Denied:
        def reply(self, request):
            raise BudgetBlocked("Synthetic budget denial.")
    result = ResilientClinicalAgent(Denied()).run(QUESTION)
    assert result["response"] and result["model_turns"] == 0 and result["output_origin"] == "application"


def test_duplicate_tool_text_is_compacted_without_losing_registered_evidence():
    executions = []
    model = Model([TOOL, FINAL])
    result = ResilientClinicalAgent(model, [fixture_tool(executions)]).run(QUESTION)
    output = model.requests[1]["history"][1]["result"]["output"]
    assert "sources" not in output and output["source_ids"] == ["SYNTHETIC_SOURCE"]
    assert model.requests[1]["evidence_registry"]["SYNTHETIC_SOURCE"]["text"] == "BLUE"
    assert result["history"][1]["result"]["output"]["sources"][0]["text"] == "BLUE"


def test_next_question_gets_no_previous_evidence():
    executions = []
    model = Model([TOOL, FINAL, FINAL])
    agent = ResilientClinicalAgent(model, [fixture_tool(executions)])
    agent.run(QUESTION)
    agent.run({**QUESTION, "question_id": "OTHER"})
    assert model.requests[-1]["history"] == [] and model.requests[-1]["evidence_registry"] == {}


def test_invalid_input_gets_message_without_gold_leak_or_model_call():
    model = Model([])
    result = ResilientClinicalAgent(model).run({**QUESTION, "answer_idx": "PRIVATE_GOLD_MARKER"})
    assert result["status"] == "invalid_input" and result["response"] and not model.requests
    assert "PRIVATE_GOLD_MARKER" not in json.dumps(result)


def test_configuration_must_reserve_final_calls():
    with pytest.raises(ValidationError):
        ResilientLimits(max_model_turns=3, max_research_turns=3)


def test_native_final_schema_excludes_tools_and_unknown_citation_ids():
    schema = final_schema(QUESTION, {"KNOWN_VIEW": {}})
    assert schema["properties"]["type"]["enum"] == ["final"]
    assert schema["properties"]["citations"]["items"]["enum"] == ["KNOWN_VIEW"]
    assert schema["additionalProperties"] is False and len(schema["required"]) == 6
    assert final_schema({"question": "Free text"})["properties"]["selected_option"] == {"type": "null"}


def test_backend_switches_only_final_phase_to_native_schema():
    class Backend:
        def __init__(self):
            self.binds, self.calls = [], []

        def bind(self, **kwargs):
            self.binds.append(kwargs)
            return self

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content=json.dumps(FINAL))

    inner = Backend()
    backend = PhaseBackend(inner)
    for phase in ["research", "finalize"]:
        backend.invoke([HumanMessage(content=json.dumps({"phase": phase, "question": QUESTION, "evidence_registry": {}}))])
    assert len(inner.binds) == 1
    assert inner.binds[0]["response_format"]["json_schema"]["strict"] is True
    assert [row["response_format"] for row in backend.formats] == ["json_object", "json_schema"]


def test_logical_json_fallback_backend_does_not_assume_bind_support():
    class Backend:
        def invoke(self, messages):
            return AIMessage(content=json.dumps(FINAL))
    backend = PhaseBackend(Backend(), native_final=False)
    backend.invoke([HumanMessage(content=json.dumps({"phase": "finalize", "question": QUESTION, "evidence_registry": {}}))])
    assert backend.formats[0]["response_format"] == "json_object"


def test_adapter_saves_original_visible_wrapper_while_graph_uses_canonical_action():
    class Backend:
        def invoke(self, messages):
            return AIMessage(content=json.dumps({"action": FINAL}))
    config = ModelConfig(provider="openai_compatible", model="synthetic", base_url="http://localhost.invalid/v1", max_retries=0, rate_limit_retries=0, json_repair_attempts=0)
    client = ModelClient(Backend(), config)
    model = ResilientJSONModel(client, lambda request: None)
    result = ResilientClinicalAgent(model).run(QUESTION)
    assert result["status"] == "completed"
    assert json.loads(client.calls[0]["response"]) == {"action": FINAL}
    assert result["trace"][1]["normalizations"] == ["unwrapped_single_action_object"]
    assert client.calls[0]["system_prompt"] == RESILIENT_SYSTEM
