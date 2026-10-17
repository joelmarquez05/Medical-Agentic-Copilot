"""A bounded LangGraph agent with model-selected tools and self-review."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from medical_copilot import prompts
from medical_copilot.evidence import GuidanceLibrary
from medical_copilot.models import ModelClient, ModelError
from medical_copilot.schemas import Plan, Review, TargetArgs
from medical_copilot.tools import make_tools


class AgentState(TypedDict, total=False):
    question: str
    plan: dict[str, Any]
    tool_results: list[dict[str, Any]]
    drafts: list[str]
    reviews: list[dict[str, Any]]
    mechanical_issues: list[str]
    trace: list[str]
    revision_count: int
    answer: str
    status: str
    sources: list[dict[str, Any]]


def source_ids(state: AgentState) -> set[str]:
    return {source["id"] for result in state.get("tool_results", []) if result["status"] == "ok" for source in result["output"].get("sources", [])} | {source for result in state.get("tool_results", []) if result["status"] == "ok" for source in result["output"].get("source_ids", [])}


def cited_source_ids(draft: str) -> set[str]:
    """Accept individual and grouped markers; do not hide IDs in bad labels."""
    return {source for bracket in re.findall(r"\[([^\]\n]+)\]", draft) for source in re.findall(r"\bS\d{2}\b", bracket)}


def citation_issues(draft: str, allowed: set[str], clinical: bool) -> list[str]:
    cited = cited_source_ids(draft)
    issues = [f"Unknown or unavailable source marker [{source}]." for source in sorted(cited - allowed)]
    if clinical and allowed and not cited:
        issues.append("No supplied source marker accompanies the clinical answer.")
    if re.search(r"https?://", draft):
        issues.append("The draft contains a URL; use supplied source markers and the application bibliography instead.")
    return issues


class MedicalCopilot:
    def __init__(self, client: ModelClient, library: GuidanceLibrary | None = None):
        self.client = client
        self.library = library or GuidanceLibrary()
        self.tools = make_tools(self.library)
        graph = StateGraph(AgentState)
        for name in ["plan", "execute_tools", "draft", "review", "revise", "finalize"]:
            graph.add_node(name, getattr(self, "_" + name))
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execute_tools")
        graph.add_edge("execute_tools", "draft")
        graph.add_edge("draft", "review")
        graph.add_conditional_edges("review", self._after_review, {"revise": "revise", "finalize": "finalize"})
        graph.add_edge("revise", "review")
        graph.add_edge("finalize", END)
        # No checkpointer or cross-question conversation memory is attached.
        self.graph = graph.compile()

    def _plan(self, state: AgentState) -> dict[str, Any]:
        catalog = {"guidance_topics": self.library.catalog(), "tools": [{"name": tool.name, "description": tool.description, "arguments_schema": tool.args_schema.model_json_schema()} for tool in self.tools.values()]}
        plan = self.client.structured(Plan, prompts.PLAN, json.dumps({"question": state["question"], "available": catalog}), "plan")
        return {"plan": plan.model_dump(), "trace": state["trace"] + ["plan"]}

    def _invoke_tool(self, name: str, arguments: dict[str, Any], question: str) -> dict[str, Any]:
        started = time.perf_counter()
        record: dict[str, Any] = {"name": name, "arguments": arguments}
        try:
            if name == "estimate_nutrition_targets":
                weight = TargetArgs.model_validate(arguments).weight_kg
                explicit_weights = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:-\s*)?(?:kg|kilograms?)\b(?!\s*/)", question, re.IGNORECASE)]
                if not any(abs(weight - value) < 1e-6 for value in explicit_weights):
                    raise ValueError("Calculator weight must be explicitly present in the question.")
            record.update(status="ok", output=self.tools[name].invoke(arguments))
        except (ValidationError, ValueError, KeyError) as exc:
            # Do not execute unknown operations or silently correct bad inputs.
            record.update(status="error", error_type=type(exc).__name__, error="Tool arguments are invalid; do not rely on this result.")
        record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        return self.client.redactor.object(record)

    def _execute_tools(self, state: AgentState) -> dict[str, Any]:
        calls = state["plan"]["tool_calls"] if state["plan"]["scope"] == "oncology_nutrition" else []
        results = [self._invoke_tool(call["name"], call["arguments"], state["question"]) for call in calls]
        return {"tool_results": results, "trace": state["trace"] + ["execute_tools"]}

    def _context(self, state: AgentState) -> dict[str, Any]:
        return {"question": state["question"], "plan": state["plan"], "tool_results": state["tool_results"], "allowed_source_ids": sorted(source_ids(state))}

    def _draft(self, state: AgentState) -> dict[str, Any]:
        draft = self.client.generate(prompts.DRAFT, json.dumps(self._context(state)), "draft")
        return {"drafts": state["drafts"] + [draft], "trace": state["trace"] + ["draft"]}

    def _review(self, state: AgentState) -> dict[str, Any]:
        context = self._context(state)
        context["draft"] = state["drafts"][-1]
        issues = citation_issues(context["draft"], source_ids(state), state["plan"]["scope"] == "oncology_nutrition")
        context["mechanical_citation_issues"] = issues
        review = self.client.structured(Review, prompts.REVIEW, json.dumps(context), "review")
        return {"reviews": state["reviews"] + [review.model_dump()], "mechanical_issues": issues, "trace": state["trace"] + ["review"]}

    def _after_review(self, state: AgentState) -> str:
        review = state["reviews"][-1]
        needs_work = review["needs_revision"] or bool(review["issues"]) or bool(review["additional_topics"]) or bool(state["mechanical_issues"])
        return "revise" if needs_work and state["revision_count"] < self.client.config.max_revisions else "finalize"

    def _revise(self, state: AgentState) -> dict[str, Any]:
        review = state["reviews"][-1]
        results = list(state["tool_results"])
        if review["additional_topics"] and state["plan"]["scope"] == "oncology_nutrition":
            results.append(self._invoke_tool("lookup_guidance", {"topics": review["additional_topics"]}, state["question"]))
        updated = dict(state, tool_results=results)
        context = self._context(updated)
        context.update(previous_draft=state["drafts"][-1], reviewer=review, mechanical_citation_issues=state["mechanical_issues"])
        draft = self.client.generate(prompts.REVISE, json.dumps(context), "revise")
        return {"tool_results": results, "drafts": state["drafts"] + [draft], "revision_count": state["revision_count"] + 1, "trace": state["trace"] + ["revise"]}

    def _finalize(self, state: AgentState) -> dict[str, Any]:
        draft = state["drafts"][-1]
        allowed = source_ids(state)
        cited = sorted(cited_source_ids(draft) & allowed)
        sources = [self.library.sources[source] for source in cited if source in self.library.sources]
        review = state["reviews"][-1]
        unresolved = bool(review["needs_revision"] or review["issues"] or review["additional_topics"] or state["mechanical_issues"])
        if state["plan"]["scope"] == "out_of_scope" or not allowed:
            status = "insufficient_evidence"
        else:
            status = "needs_human_review" if unresolved else "self_review_passed"
        if sources:
            draft += "\n\nSources consulted from the local guidance collection:\n" + "\n".join(f"- [{source['id']}] {source['authors']} ({source['year']}). {source['title']}. {source['url']}" for source in sources)
        return {"answer": draft, "status": status, "sources": sources, "trace": state["trace"] + ["finalize"]}

    def run(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question or len(question) > 12000:
            raise ValueError("Question must contain 1-12000 characters.")
        call_start = len(self.client.calls)
        started = time.perf_counter()
        initial: AgentState = {"question": question, "tool_results": [], "drafts": [], "reviews": [], "trace": [], "revision_count": 0, "mechanical_issues": []}
        completed_state: AgentState = dict(initial)
        result: dict[str, Any] = {"started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prompt_version": prompts.PROMPT_VERSION, "question": question}
        try:
            for update in self.graph.stream(initial, {"recursion_limit": 24}, stream_mode="updates"):
                for node_result in update.values():
                    if isinstance(node_result, dict):
                        completed_state.update(node_result)
            result.update(completed_state)
        except ModelError as exc:
            # Preserve completed work without presenting a partial draft as final.
            result.update(completed_state)
            if completed_state["drafts"]:
                result["partial_answer"] = completed_state["drafts"][-1]
            result.update(status="error", error=str(exc), answer="")
        result.update(elapsed_seconds=round(time.perf_counter() - started, 3), model_calls=self.client.calls[call_start:])
        result["usage"] = {name: sum(call.get("usage", {}).get(name, 0) for call in result["model_calls"]) for name in ["input_tokens", "output_tokens", "total_tokens"]}
        return self.client.redactor.object(result)
