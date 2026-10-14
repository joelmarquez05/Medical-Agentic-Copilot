"""Offline execution checks. Every model response and source is synthetic."""

import copy
import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict

from medical_copilot.clinical_graph import (
    BudgetBlocked, ClinicalAgent, EvidenceRecord, JSONActionModel, Limits,
    ToolDefinition, ToolOutput, save_trace,
)
from medical_copilot.config import ModelConfig
from medical_copilot.models import ModelClient


QUESTION = {"question_id": "SYNTHETIC-01", "question": "Which marker is in the synthetic document?",
            "options": {"A": "BLUE", "B": "GREEN"}}


def final(option="A", citations=None):
    return {"type": "final", "answer": "The fixture marker is BLUE.",
            "justification": "Synthetic execution fixture, not a medical answer.",
            "selected_option": option, "citations": citations or []}


def call(query="marker"):
    return {"type": "tool", "name": "fixture_search", "arguments": {"query": query}}


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def reply(self, request):
        self.requests.append(copy.deepcopy(request))
        response = self.responses.pop(0)
        return response(request) if callable(response) else response


def source(text="SIMULATED DOCUMENT: marker BLUE"):
    return EvidenceRecord(id="SYNTHETIC_DOC_01", title="Synthetic source for execution tests",
                          text=text, text_kind="synthetic")


def tool(execute):
    return ToolDefinition("fixture_search", "Synthetic local test lookup; no network.", SearchArgs, execute)


def test_baseline_is_one_turn_with_no_tools_or_registered_sources():
    model = ScriptedModel([final()])
    result = ClinicalAgent(model).run(QUESTION)
    assert result["status"] == "completed"
    assert result["model_turns"] == 1 and result["tool_actions"] == 0
    assert model.requests[0]["tools"] == [] and result["evidence"] == {}
    assert model.requests[0]["remaining"]["tool_actions"] == 0
    assert result["response"]["evidence_status"] == "no_registered_source_cited"


def test_real_executor_result_reaches_the_next_model_decision():
    executions = []

    def execute(args):
        executions.append(args.query)
        return ToolOutput(content="Synthetic lookup succeeded.", sources=[source()])

    def inspect(request):
        assert request["history"][-1]["role"] == "tool"
        assert request["evidence_registry"]["SYNTHETIC_DOC_01"]["text"].endswith("BLUE")
        return final(citations=["SYNTHETIC_DOC_01"])

    result = ClinicalAgent(ScriptedModel([call(), inspect]), [tool(execute)]).run(QUESTION)
    assert result["status"] == "completed" and executions == ["marker"]
    assert result["model_turns"] == 2 and result["tool_actions"] == 1


def test_empty_result_can_trigger_one_reformulation_then_final_answer():
    queries = []

    def execute(args):
        queries.append(args.query)
        return ToolOutput(content="SIMULATED: no results.")

    def reformulate(request):
        assert request["history"][-1]["result"]["output"]["sources"] == []
        return call("revised marker")

    model = ScriptedModel([call(), reformulate, final()])
    result = ClinicalAgent(model, [tool(execute)]).run(QUESTION)
    assert queries == ["marker", "revised marker"]
    assert result["model_turns"] == 3 and result["tool_actions"] == 2
    assert model.requests[-1]["tools"] == []


def test_final_permitted_turn_cannot_execute_a_third_tool_action():
    executions = []
    model = ScriptedModel([call(), call(), call()])
    result = ClinicalAgent(model, [tool(lambda args: executions.append(args.query) or ToolOutput(content="Synthetic empty result."))]).run(QUESTION)
    assert result["status"] == "limit_reached"
    assert len(executions) == 2 and len(model.requests) == 3


def test_unknown_tool_is_never_executed():
    result = ClinicalAgent(ScriptedModel([{"type": "tool", "name": "shell", "arguments": {}}])).run(QUESTION)
    assert result["status"] == "invalid_tool_request" and result["tool_actions"] == 0


def test_invalid_tool_arguments_are_rejected_before_execution_and_returned_as_error():
    executions = []
    invalid = {"type": "tool", "name": "fixture_search", "arguments": {"query": "x", "unexpected": True}}
    model = ScriptedModel([invalid, final()])
    result = ClinicalAgent(model, [tool(lambda args: executions.append(args) or ToolOutput(content="x"))]).run(QUESTION)
    assert not executions and result["tool_actions"] == 1
    assert model.requests[1]["history"][-1]["result"]["status"] == "error"
    assert not model.requests[1]["history"][-1]["result"]["execution_attempted"]


def test_exception_body_never_enters_trace_or_model_feedback():
    def execute(args):
        raise RuntimeError("FAKE_SECRET_SHOULD_NEVER_BE_STORED")

    model = ScriptedModel([call(), final()])
    result = ClinicalAgent(model, [tool(execute)]).run(QUESTION)
    assert "FAKE_SECRET" not in json.dumps(result) + json.dumps(model.requests)
    assert result["trace"][2]["error_type"] == "RuntimeError"


def test_source_collision_keeps_previous_source_and_reports_tool_error():
    count = 0

    def execute(args):
        nonlocal count
        count += 1
        return ToolOutput(content="Synthetic result.", sources=[source("original" if count == 1 else "changed")])

    model = ScriptedModel([call(), call(), final(citations=["SYNTHETIC_DOC_01"])])
    result = ClinicalAgent(model, [tool(execute)]).run(QUESTION)
    assert result["evidence"]["SYNTHETIC_DOC_01"]["text"] == "original"
    assert model.requests[-1]["history"][-1]["result"]["error_type"] == "EvidenceConflict"


def test_fresh_question_cannot_cite_or_receive_previous_question_evidence():
    model = ScriptedModel([call(), final(citations=["SYNTHETIC_DOC_01"]), final(citations=["SYNTHETIC_DOC_01"])])
    agent = ClinicalAgent(model, [tool(lambda args: ToolOutput(content="Synthetic result.", sources=[source()]))])
    assert agent.run(QUESTION)["status"] == "completed"
    result = agent.run({**QUESTION, "question_id": "SYNTHETIC-02"})
    assert model.requests[-1]["history"] == [] and model.requests[-1]["evidence_registry"] == {}
    assert result["status"] == "completed_with_citation_issues"


def test_evaluator_key_is_rejected_before_any_model_call():
    model = ScriptedModel([])
    result = ClinicalAgent(model).run({**QUESTION, "answer_idx": "A", "reference_rationale": "PRIVATE_EVALUATOR_MARKER"})
    assert result["status"] == "invalid_input" and not model.requests
    assert "PRIVATE_EVALUATOR_MARKER" not in json.dumps(result)


def test_invalid_action_does_not_trigger_a_paid_repair_or_second_turn():
    model = ScriptedModel([{"type": "unsupported_action"}])
    result = ClinicalAgent(model).run(QUESTION)
    assert result["status"] == "invalid_model_output" and len(model.requests) == 1


def test_clarification_is_terminal_and_not_a_completed_exam_answer():
    model = ScriptedModel([{"type": "clarify", "question": "Which synthetic document should be read?"}])
    result = ClinicalAgent(model).run(QUESTION)
    assert result["status"] == "awaiting_clarification" and result["model_turns"] == 1


def test_input_size_gate_blocks_before_a_model_attempt():
    model = ScriptedModel([])
    result = ClinicalAgent(model, limits=Limits(max_input_bytes=1024)).run(QUESTION)
    assert result["status"] == "input_limit_reached" and not model.requests


class TextBackend:
    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return AIMessage(content=json.dumps(final()))


def offline_client(backend, **overrides):
    return ModelClient(backend, ModelConfig(provider="openai_compatible", model="synthetic-offline",
                       base_url="http://localhost.invalid/v1", max_retries=0,
                       rate_limit_retries=0, json_repair_attempts=0, **overrides))


def test_json_adapter_is_langchain_transport_with_a_pre_request_gate():
    backend, gates = TextBackend(), []
    adapter = JSONActionModel(offline_client(backend), before_request=lambda request: gates.append(request))
    result = ClinicalAgent(adapter).run(QUESTION)
    assert result["status"] == "completed" and len(gates) == len(backend.messages) == 1
    assert json.loads(backend.messages[0][1].content)["question"] == QUESTION


def test_budget_gate_denial_does_not_call_backend():
    def deny(request):
        raise BudgetBlocked("Synthetic budget gate denial.")

    backend = TextBackend()
    result = ClinicalAgent(JSONActionModel(offline_client(backend), before_request=deny)).run(QUESTION)
    assert result["status"] == "budget_blocked" and not backend.messages
    assert result["model_turns"] == 0


def test_non_json_text_is_a_format_failure_without_a_second_request():
    class NonJSONBackend(TextBackend):
        def invoke(self, messages):
            self.messages.append(messages)
            return AIMessage(content="Synthetic non-JSON response.")

    backend = NonJSONBackend()
    result = ClinicalAgent(JSONActionModel(offline_client(backend), before_request=lambda request: None)).run(QUESTION)
    assert result["status"] == "invalid_model_output" and len(backend.messages) == 1


def test_truncated_response_is_not_a_medical_wrong_answer():
    class TruncatedBackend(TextBackend):
        def invoke(self, messages):
            self.messages.append(messages)
            return AIMessage(content="Synthetic partial JSON", response_metadata={"finish_reason": "length"})

    backend = TruncatedBackend()
    result = ClinicalAgent(JSONActionModel(offline_client(backend), before_request=lambda request: None)).run(QUESTION)
    assert result["status"] == "unusable_model_response" and result["response"] is None
    assert len(backend.messages) == 1


def test_trace_writer_preserves_existing_records(tmp_path):
    path = tmp_path / "trace.json"
    save_trace(path, {"synthetic": True})
    with pytest.raises(FileExistsError):
        save_trace(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"synthetic": True}
