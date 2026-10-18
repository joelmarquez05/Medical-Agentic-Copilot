"""Versioned recovery and reserved final-answer phase for the clinical agent."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

from medical_copilot.clinical_graph import (
    ACTION_SCHEMA, BudgetBlocked, ClinicalAgent, ClinicalQuestion, FinalAction,
    MalformedAction, State, StrictModel, SYSTEM, UnusableModelResponse,
)
from medical_copilot.models import ModelError, parse_object


EVIDENCE_INSTRUCTION = """You are an evidence-oriented medical copilot. Real, checkable references are an
important part of a useful clinical answer. Actively consider a targeted PubMed search
to support clinical recommendations or verify precise named-trial results, even when
you recognize the topic. Prefer relevant retrieved evidence over an unverified recollection.
Search remains autonomous: there is no mandatory operation or minimum number of searches.
You may answer directly when existing knowledge is adequate; then state that no external
reference was verified and make that evidence limitation explicit. Do not invent citations.
Inspect actual returned text and distinguish study population, outcome and uncertainty.
If an excerpt omits the needed finding, consider reading a returned paper's abstract.
Use only exact evidence_registry source IDs in citations; never replace them with a raw
PMID, URL or invented identifier. Unrelated or insufficient evidence must be acknowledged.
Keep the shared action schema. The top-level discriminator key is type, with value final,
tool or clarify; never use action/tool as an alternative discriminator and never nest an
action object inside another object. A tool action has type, name and arguments at the same
top level. A final action has type, answer, justification and selected_option at the same
top level. selected_option is a single A-D letter for these cases;
citations and limitations are lists, using [] when empty, never null. Answer, justification
and limitations contain public explanation only. Return one JSON action."""

VERSION = "clinical-resilient-v3"
RESILIENT_SYSTEM = SYSTEM + "\n\n" + EVIDENCE_INSTRUCTION + """

Research and finalization have separate budgets. Search is autonomous. Start PubMed queries
with a short trial name and clinical topic/drug; avoid needless conjunctions, guessed effect
sizes, and restrictive acronyms. Reformulate substantively if the original study is absent.
Use the actual phase in the request. During finalize, research has ended: do not request any
tool. Give the best supported answer using the original case, supplied evidence and existing
knowledge, acknowledging missing verification or conflicting/inadequate evidence. An empty
search is not proof of no literature. In examination cases, choose the best provided option
when you can; do not invent certainty or missing facts. If you cannot determine an answer,
explain the limitation explicitly. Return the phase's schema. Never fabricate a source.
The evidence registry contains the actual text; compact tool history points to its IDs."""


class ResilientLimits(StrictModel):
    max_research_turns: int = Field(default=3, ge=0, le=4)
    max_tool_actions: int = Field(default=3, ge=0, le=4)
    max_final_attempts: int = Field(default=2, ge=1, le=2)
    max_model_turns: int = Field(default=5, ge=1, le=6)
    max_input_bytes: int = Field(default=65536, ge=4096, le=262144)

    @model_validator(mode="after")
    def reserve_closing_calls(self):
        if self.max_model_turns < self.max_research_turns + self.max_final_attempts:
            raise ValueError("Reserve final attempts separately from research turns.")
        return self


class ResilientState(State, total=False):
    research_turns: int
    final_attempts: int
    provider_failures: int
    closing_reason: str | None
    next_step: str
    output_origin: str | None


def normalize_action(raw):
    """Only unambiguous transport changes; never rewrite medical claims/options."""
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise ValueError("Expected one action object.")
    value, notes = copy.deepcopy(raw), []
    if isinstance(value.get("action"), dict):
        if set(value) - {"action", "explanation"} or ("explanation" in value and not isinstance(value["explanation"], str)):
            raise ValueError("Ambiguous action wrapper.")
        value = value["action"]
        notes.append("unwrapped_single_action_object")
    elif "action" in value:
        if "type" in value or not isinstance(value["action"], str) or value["action"] not in {"final", "tool", "clarify"}:
            raise ValueError("Conflicting action discriminator.")
        value["type"] = value.pop("action")
        notes.append("renamed_action_discriminator_to_type")
    if "type" not in value and isinstance(value.get("tool"), str) and set(value) <= {"tool", "arguments"}:
        value["type"], value["name"] = "tool", value.pop("tool")
        notes.append("canonicalized_tool_name_alias")
    if "type" not in value and {"answer", "justification"} <= set(value) <= set(FinalAction.model_fields):
        value["type"] = "final"
        notes.append("inferred_unique_final_discriminator")
    if value.get("type") == "final":
        for key in ["citations", "limitations"]:
            if key in value and value[key] is None:
                value[key] = []
                notes.append("normalized_null_" + key + "_to_empty_list")
        option = value.get("selected_option")
        if isinstance(option, str) and option.strip().upper() in set("ABCD") and option != option.strip().upper():
            value["selected_option"] = option.strip().upper()
            notes.append("canonicalized_same_option_letter")
    action = ACTION_SCHEMA.validate_python(value).model_dump(mode="json")
    return action, notes


def final_schema(question=None, evidence=None):
    """A flat final-only schema for provider-native structured output when supported."""
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "type": {"type": "string", "enum": ["final"]},
        "answer": {"type": "string"}, "justification": {"type": "string"},
        "selected_option": {"anyOf": [{"type": "string", "enum": list((question or {}).get("options") or "ABCD")}, {"type": "null"}]},
        "citations": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    }, "required": list(FinalAction.model_fields)}
    if evidence:
        schema["properties"]["citations"]["items"]["enum"] = list(evidence)
    else:
        schema["properties"]["citations"]["maxItems"] = 0
    if question is not None and not question.get("options"):
        schema["properties"]["selected_option"] = {"type": "null"}
    return schema


class PhaseBackend:
    """Use a final-only native JSON schema where that transport was prevalidated."""

    def __init__(self, inner, native_final=True):
        self.inner, self.native_final, self.formats = inner, native_final, []

    def invoke(self, messages):
        request = json.loads(messages[-1].content)
        terminal = request["phase"] == "finalize"
        mode = "json_schema" if terminal and self.native_final else "json_object"
        self.formats.append({"phase": request["phase"], "response_format": mode})
        if mode == "json_schema":
            response_format = {"type": "json_schema", "json_schema": {"name": "medical_final_answer", "strict": True,
                "schema": final_schema(request["question"], request["evidence_registry"])}}
            return self.inner.bind(response_format=response_format).invoke(messages)
        return self.inner.invoke(messages)


class ResilientJSONModel:
    def __init__(self, client, before_request, transient_failure=None):
        self.client, self.before_request = client, before_request
        self.transient_failure = transient_failure or (lambda: False)
        if client.config.max_retries or client.config.rate_limit_retries or client.config.json_repair_attempts:
            raise ValueError("Use explicit graph recovery; hidden SDK/application retries must be disabled.")

    def reply(self, request):
        self.before_request(copy.deepcopy(request))
        try:
            text = self.client.generate(RESILIENT_SYSTEM, json.dumps(request, ensure_ascii=False), "clinical_" + request["phase"])
        except ModelError:
            if self.client.calls and self.client.calls[-1].get("status") == "unusable_response":
                raise UnusableModelResponse("No complete visible output.") from None
            raise
        try:
            return parse_object(text)
        except ValueError:
            raise MalformedAction("Visible output is not one JSON action.") from None

    def may_retry_provider(self):
        return bool(self.transient_failure())


def compact_history(history):
    output = copy.deepcopy(history)
    for item in output:
        if item["role"] == "tool":
            result = item["result"].get("output", {})
            if "sources" in result:
                result["source_ids"] = [source["id"] for source in result.pop("sources")]
    return output


class ResilientClinicalAgent(ClinicalAgent):
    """Keep the existing tool executor/validator, add an explicit closing branch."""

    def __init__(self, model, tools=None, limits=None):
        self.model, self.limits = model, limits or ResilientLimits()
        tool_list = tools or []
        self.tools = {tool.name: tool for tool in tool_list}
        if len(self.tools) != len(tool_list):
            raise ValueError("Duplicate tools.")
        graph = StateGraph(ResilientState)
        graph.add_node("initialize", self._initialize)
        graph.add_node("research", self._research)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("finalize", self._finalize)
        graph.add_node("validate_output", self._validate_output)
        graph.add_node("fallback", self._fallback)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "research")
        routes = {name: name for name in ["execute_tool", "finalize", "validate_output", "fallback", "finish"]}
        for node in ["research", "finalize", "validate_output"]:
            graph.add_conditional_edges(node, lambda state: state["next_step"], routes)
        graph.add_edge("execute_tool", "research")
        graph.add_edge("fallback", "finish")
        graph.add_edge("finish", END)
        self.graph = graph.compile()

    def _initialize(self, state):
        return {**super()._initialize(state), "research_turns": 0, "final_attempts": 0,
                "provider_failures": 0, "closing_reason": None, "next_step": "research", "output_origin": None}

    def _request_phase(self, state, phase):
        tools = [tool.catalog() for tool in self.tools.values()] if phase == "research" else []
        return {"question": state["question"], "history": compact_history(state["history"]),
                "evidence_registry": state["evidence"], "tools": tools,
                "action_schema": ACTION_SCHEMA.json_schema() if phase == "research" else final_schema(state["question"], state["evidence"]),
                "remaining": {"model_turns_including_current": self.limits.max_model_turns - state["model_turns"],
                              "tool_actions": self.limits.max_tool_actions - state["tool_actions"] if tools else 0},
                "phase": phase, "closing_reason": state["closing_reason"],
                "execution_policy": "Research is optional. Finalize must produce a final answer, not another operation. Missing evidence must be disclosed."}

    def _attempt(self, state, phase):
        request = self._request_phase(state, phase)
        size = len((RESILIENT_SYSTEM + json.dumps(request, ensure_ascii=False)).encode())
        event = {"node": phase, "turn": state["model_turns"] + 1, "input_bytes": size}
        result = {"action": None, "status": "running", "next_step": "finalize"}
        if size > self.limits.max_input_bytes or state["model_turns"] >= self.limits.max_model_turns:
            reason = "input_limit_reached" if size > self.limits.max_input_bytes else "model_budget_exhausted"
            return {**result, "closing_reason": reason, "next_step": "fallback",
                    "trace": state["trace"] + [{**event, "status": reason, "request_sent": False}]}
        started = time.perf_counter()
        try:
            raw = self.model.reply(copy.deepcopy(request))
            action, normalization = normalize_action(raw)
            if phase == "finalize" and action["type"] != "final":
                raise ValueError("Final phase cannot request operations.")
            event.update(status="ok", action=action, normalizations=normalization)
            result.update(action=action, history=state["history"] + [{"role": "model", "action": action}])
            if action["type"] == "tool":
                if action["name"] in self.tools and request["tools"]:
                    result["next_step"] = "execute_tool"
                else:
                    result["closing_reason"] = "unavailable_tool_request"
                    event["operation_refused"] = True
            else:
                result["next_step"] = "validate_output"
        except BudgetBlocked:
            return {**result, "closing_reason": "budget_blocked", "next_step": "fallback",
                    "trace": state["trace"] + [{**event, "status": "budget_blocked", "request_sent": False}]}
        except ModelError as exc:
            failures = state["provider_failures"] + 1
            retry = failures <= 1 and bool(getattr(self.model, "may_retry_provider", lambda: False)())
            result.update(provider_failures=failures, closing_reason="provider_failure", next_step="finalize" if retry else "fallback")
            event.update(status="model_error", error_type=type(exc).__name__, recoverable=retry)
        except (MalformedAction, UnusableModelResponse, ValueError) as exc:
            result.update(closing_reason="invalid_or_unusable_output")
            event.update(status="invalid_model_output", error_type=type(exc).__name__)
        except Exception as exc:
            result.update(closing_reason="unexpected_model_failure", next_step="fallback")
            event.update(status="model_error", error_type=type(exc).__name__)
        event["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        result.update(model_turns=state["model_turns"] + 1, trace=state["trace"] + [event])
        if result["action"] is None:
            result["history"] = state["history"] + [{"role": "control", "message": "Previous model operation did not yield a valid action. Finish with the original question and registered evidence; no further tool is available in finalization.", "failure_kind": result["closing_reason"]}]
        return result

    def _research(self, state):
        if state["research_turns"] >= self.limits.max_research_turns or (self.tools and state["tool_actions"] >= self.limits.max_tool_actions):
            return {"action": None, "closing_reason": "research_budget_exhausted", "next_step": "finalize"}
        return {**self._attempt(state, "research"), "research_turns": state["research_turns"] + 1}

    def _finalize(self, state):
        if state["final_attempts"] >= self.limits.max_final_attempts:
            return {"next_step": "fallback", "closing_reason": "final_attempts_exhausted"}
        result = self._attempt(state, "finalize")
        attempts = state["final_attempts"] + 1
        if result["next_step"] == "finalize" and attempts >= self.limits.max_final_attempts:
            result["next_step"] = "fallback"
        return {**result, "final_attempts": attempts}

    def _validate_output(self, state):
        result = super()._validate_output(state)
        if result["status"] == "invalid_model_output":
            result.update(status="running", response=None, closing_reason="invalid_final_selection",
                          next_step="finalize" if state["final_attempts"] < self.limits.max_final_attempts else "fallback")
        else:
            result.update(next_step="finish", output_origin="model")
        return result

    def _fallback(self, state):
        response = {"type": "operational_message", "answer": "A reliable final model answer could not be obtained within the configured limits.",
                    "justification": "The execution stopped after recorded provider, transport, input or output failures. No medical answer or option was fabricated.",
                    "selected_option": None, "citations": [], "limitations": [state["closing_reason"] or "unknown_failure"],
                    "evidence_status": "no_valid_model_answer"}
        return {"status": "operationally_unanswered", "response": response, "output_origin": "application",
                "trace": state["trace"] + [{"node": "fallback", "reason": state["closing_reason"]}]}

    def run(self, question):
        try:
            ClinicalQuestion.model_validate(question)
        except Exception as exc:
            return {"status": "invalid_input", "error_type": type(exc).__name__, "response": {
                "type": "operational_message", "answer": "The supplied question does not meet the input contract.", "selected_option": None},
                "output_origin": "application", "trace": [], "model_turns": 0, "tool_actions": 0, "evidence": {}}
        result = super().run(question)
        result["version"] = VERSION
        result["interpretation"] = "Every episode returns a terminal message; only completed model answers count as medical completions. Normalization/recovery and original provider responses remain auditable."
        return result
