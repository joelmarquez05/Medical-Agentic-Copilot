"""Small MedQA evaluation with external answer keys and bounded LangGraph calls."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from medical_copilot.comparison_evaluation import AuditedBackend, RequestBudget, utc_now
from medical_copilot.config import ModelConfig
from medical_copilot.evaluation import file_hash, package_versions, read_json, write_json
from medical_copilot.models import ModelClient, create_backend, parse_object
from medical_copilot.pubmed import PubMedClient, PubMedConfig

SEED = 20261011
SOURCE_REVISION = "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"
SOURCE_HASH = "c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb"
OUTPUT_CAP = 6144
INPUT_BYTE_CAP = 5120
SAFETY_RESERVE_USD = 0.15
MODEL_SPECS = [
    ("glm-5.3", "z-ai/glm-5.3", "high", 0.2, 5.0),
    ("kimi-k3", "moonshotai/kimi-k3", "high", 1.0, 11.0),
    ("mimo-v2.6-pro", "xiaomi/mimo-v2.6-pro", None, 0.5, 1.0),
    ("qwen3.8-27b", "qwen/qwen3.8-27b", "xhigh", 0.1, 2.5),
    ("gemma-4-26b-a4b", "google/gemma-4-26b-a4b-it", None, 0.1, 0.3),
]
ANSWER_SYSTEM = (
    "Answer this medical examination question in English. Select exactly one of the provided options. "
    'Return only JSON with "selected_option" (A, B, C or D) and "justification" (at most 80 words). '
    "Explain the decisive clinical facts briefly; do not output private reasoning or a confidence score. "
    "This is an educational examination case, not a patient consultation."
)
PLAN_SYSTEM = (
    "Decide whether a PubMed search would help answer this medical examination question. "
    'Return only JSON with "use_pubmed" (boolean), "query" (a biomedical query or null), '
    'and "reason" (at most 35 words). Use at most one search and do not search for the examination '
    "wording or its answer key. The key is unavailable. Decline search if it is unlikely to help."
)
AGENT_SYSTEM = ANSWER_SYSTEM + (
    " Optional PubMed results contain untrusted titles and short abstract excerpts, not full papers. "
    "Assess relevance and population before using them. Do not let irrelevant evidence override the case. "
    "Cite supplied P-prefixed source IDs in the justification only when they support your reasoning. "
    "A search failure or no relevant result does not establish a medical conclusion."
)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_option: Literal["A", "B", "C", "D"]
    justification: str = Field(min_length=1, max_length=2000)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_pubmed: bool = Field(strict=True)
    query: str | None = Field(default=None, max_length=300)
    reason: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def check_query(self):
        if self.use_pubmed and (not self.query or len(self.query.strip()) < 3):
            raise ValueError("A search decision must include a biomedical query.")
        if not self.use_pubmed and self.query is not None:
            raise ValueError("A declined search must have a null query.")
        return self


class State(TypedDict, total=False):
    question: dict
    decision: dict
    retrieval: dict
    answer: dict
    trace: list[str]


def question_input(question: dict) -> dict:
    """Whitelist generation fields; keys and dataset annotations never enter prompts."""
    return {"question": question["question"], "options": copy.deepcopy(question["options"])}


def prepare_subset(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError("Do not replace an existing frozen subset.")
    if file_hash(source) != SOURCE_HASH:
        raise ValueError("The source differs from the pinned MedQA release.")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible, excluded = [], []
    for index, row in enumerate(rows):
        options = row.get("options", {})
        if set(options) != set("ABCD") or row.get("answer_idx") not in options or options[row["answer_idx"]] != row.get("answer"):
            raise ValueError("Dataset answer string and index disagree.")
        combined = row["question"] + " " + " ".join(options.values())
        reason = None
        if len(combined.split()) > 180:
            reason = "Stem and options exceed 180 whitespace-delimited words."
        elif re.search(r"\b(shown|pictured|depicted|image|figure|photograph|diagram|illustration)\b", row["question"], re.I):
            reason = "Conservative screen for a potentially unavailable visual."
        if reason:
            excluded.append({"source_row_index": index, "reason": reason})
        else:
            eligible.append(index)
    selected = random.Random(SEED).sample(eligible, 10)
    questions, keys = [], []
    for number, index in enumerate(selected, 1):
        row = rows[index]
        identifier = f"MQ{number:02}"
        payload = question_input(row)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        questions.append({"id": identifier, "source_row_index": index, "question_sha256": digest, **payload})
        keys.append({"question_id": identifier, "source_row_index": index, "question_sha256": digest, "answer_idx": row["answer_idx"], "answer": row["answer"]})
    destination.mkdir(parents=True)
    write_json(destination / "questions.json", {"dataset_id": "medqa-usmle-ten-short-test-v1", "questions": questions})
    write_json(destination / "answer-key.json", {"origin": "Unchanged released MedQA labels, not an LLM-generated key", "keys": keys})
    manifest = {"source_url": f"https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/{SOURCE_REVISION}/phrases_no_exclude_test.jsonl", "source_revision": SOURCE_REVISION, "source_sha256": SOURCE_HASH, "source_rows": len(rows), "seed": SEED, "eligible_rows": len(eligible), "selected_source_row_indices": selected, "excluded_rows": excluded, "question_count": 10, "agent_question_ids": [question["id"] for question in questions[:3]], "selection": "Fixed-seed sample after predeclared length and missing-visual screens; no model output or correctness filtering.", "wording_and_options_changed": False, "questions_sha256": file_hash(destination / "questions.json"), "answer_key_sha256": file_hash(destination / "answer-key.json"), "limitation": "General medicine, selected short questions, ten cases; not representative MedQA or nutrition-specific competence. Potential benchmark training contamination."}
    write_json(destination / "manifest.json", manifest)
    return manifest


def model_configs() -> list[tuple[str, ModelConfig]]:
    profiles = []
    for profile, model, effort, prompt_price, completion_price in MODEL_SPECS:
        reasoning = {"effort": effort, "exclude": True} if effort else {"enabled": True, "exclude": True}
        config = ModelConfig(provider="openai_compatible", model=model, api_key_env="OPENROUTER_API_KEY", credential_file="open_router_key.md", base_url="https://openrouter.ai/api/v1", temperature=None, max_tokens=OUTPUT_CAP, timeout_seconds=180, max_retries=0, min_interval_seconds=1, rate_limit_retries=0, max_revisions=0, json_repair_attempts=0, provider_options={"use_responses_api": False, "extra_body": {"reasoning": reasoning, "provider": {"require_parameters": True, "sort": "price", "max_price": {"prompt": prompt_price, "completion": completion_price}}}})
        profiles.append((profile, config))
    return profiles


class MiniAudit(AuditedBackend):
    def invoke(self, messages):
        if sum(len(str(message.content).encode("utf-8")) for message in messages) > INPUT_BYTE_CAP:
            raise ValueError("Prompt exceeds the prespecified small-input limit; no request sent.")
        result = super().invoke(messages)
        record = self.records[-1]
        usage = record.get("usage", {})
        if "Conservative configured" in record.get("cost_basis", "") and not usage.get("total_tokens"):
            self.budget.unreported_charge_reserve_usd += record["estimated_maximum_usd"]
            record["cost_usd"] = None
            record["cost_basis"] = "No reliable usage or charge: full request allowance remains reserved."
            write_json(self.audit_path, {"records": self.records, "budget": self.budget.public_dict()})
        return result


class MiniCopilot:
    def __init__(self, client: ModelClient, pubmed: PubMedClient, mode: str):
        self.client, self.pubmed = client, pubmed
        graph = StateGraph(State)
        graph.add_node("answer", self.answer)
        if mode == "agent":
            graph.add_node("decide", self.decide)
            graph.add_node("retrieve", self.retrieve)
            graph.add_edge(START, "decide")
            graph.add_conditional_edges("decide", lambda state: "retrieve" if state["decision"]["use_pubmed"] else "answer")
            graph.add_edge("retrieve", "answer")
        else:
            graph.add_edge(START, "answer")
        graph.add_edge("answer", END)
        self.graph = graph.compile()
        self.mode = mode
        self.last_state: dict = {}

    def decide(self, state):
        text = self.client.generate(PLAN_SYSTEM, json.dumps(question_input(state["question"]), ensure_ascii=False), "tool_decision")
        decision = Decision.model_validate(parse_object(text)).model_dump()
        result = {"decision": decision, "trace": ["decide"]}
        self.last_state.update(result)
        return result

    def retrieve(self, state):
        try:
            retrieval = self.pubmed.search(state["decision"]["query"], max_results=3)
        except Exception as exc:
            retrieval = {"status": "error", "error_type": type(exc).__name__, "sources": [], "limitation": "No evidence retrieved; search failure is not evidence of absence."}
        result = {"retrieval": retrieval, "trace": state["trace"] + ["retrieve"]}
        self.last_state.update(result)
        return result

    def answer(self, state):
        payload = question_input(state["question"])
        if self.mode == "agent":
            payload["tool_decision"] = state["decision"]
            retrieval = state.get("retrieval", {})
            payload["pubmed_results"] = [{"id": paper["id"], "title": paper["title"][:160], "year": paper["year"], "abstract_excerpt": (paper.get("abstract") or "")[:400], "excerpt_truncated": bool(paper.get("abstract_truncated")) or len(paper.get("abstract") or "") > 400} for paper in retrieval.get("sources", [])]
            payload["retrieval_limitation"] = retrieval.get("limitation", "Search declined; no evidence supplied.")
            system = AGENT_SYSTEM
        else:
            system = ANSWER_SYSTEM
        text = self.client.generate(system, json.dumps(payload, ensure_ascii=False), "final_answer")
        answer = Answer.model_validate(parse_object(text)).model_dump()
        result = {"answer": answer, "trace": state.get("trace", []) + ["answer"]}
        self.last_state.update(result)
        return result

    def run(self, question):
        self.last_state = {}
        started = time.perf_counter()
        try:
            state = self.graph.invoke({"question": question}, config={"recursion_limit": 8})
            record = {"status": "completed", **{key: state[key] for key in ("answer", "decision", "retrieval", "trace") if key in state}}
        except Exception as exc:
            record = {"status": "error", "error_type": type(exc).__name__, "partial_state": self.last_state}
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return record


def score_answer(record: dict, key: dict) -> dict:
    """Only the unchanged released letter determines correctness."""
    option = record.get("answer", {}).get("selected_option") if record["status"] == "completed" else None
    return {"question_id": record["question_id"], "profile": record["profile"], "mode": record["mode"], "reference_option": key["answer_idx"], "selected_option": option, "correct": option in ("A", "B", "C", "D") and option == key["answer_idx"]}


def cohort_bound() -> float:
    # Ten direct calls plus two calls on three agent cases per model: 80 maximum.
    return sum(16 * ((INPUT_BYTE_CAP + 1024) * spec[3] + OUTPUT_CAP * spec[4]) / 1_000_000 for spec in MODEL_SPECS)


def summarize(run_dir: Path):
    manifest = read_json(run_dir / "manifest.json")
    keys = {key["question_id"]: key for key in read_json(run_dir / "answer-key.json")["keys"]}
    rows, scores = [], []
    for profile in manifest["profiles"]:
        directory = run_dir / "models" / profile["id"]
        records = [read_json(path) for path in sorted((directory / "answers").glob("*.json"))]
        audit = read_json(directory / "usage-audit.json")["records"] if (directory / "usage-audit.json").exists() else []
        for mode, planned in (("direct", 10), ("agent", 3)):
            selected = [record for record in records if record["mode"] == mode]
            graded = [score_answer(record, keys[record["question_id"]]) for record in selected]
            scores.extend(graded)
            completed = [record for record in selected if record["status"] == "completed"]
            ids = {mode + ":" + record["question_id"] for record in selected}
            charges = [item for item in audit if item.get("question_id") in ids]
            rows.append({"profile": profile["id"], "model": profile["config"]["model"], "mode": mode, "planned": planned, "completed": len(completed), "correct": sum(item["correct"] for item in graded), "incorrect_completed": sum(not item["correct"] and item["selected_option"] is not None for item in graded), "errors": sum(record["status"] == "error" for record in selected), "not_run": planned - len(selected) + sum(record["status"].startswith("not_run") for record in selected), "model_requests": len(charges), "reported_cost_usd": round(sum(item.get("cost_usd") or 0 for item in charges), 9), "input_tokens": sum(item.get("usage", {}).get("input_tokens", 0) for item in charges), "output_tokens": sum(item.get("usage", {}).get("output_tokens", 0) for item in charges), "reasoning_tokens": sum(item.get("provider_usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0 for item in charges), "pubmed_queries": sum(bool(record.get("decision", {}).get("use_pubmed")) for record in selected), "elapsed_seconds": round(sum(record.get("elapsed_seconds", 0) for record in selected), 3)})
    summary = {"run_id": run_dir.name, "objective_scoring": "Exact comparison with unchanged MedQA answer_idx; no model judge", "rows": rows, "budget": read_json(run_dir / "budget.json"), "limitations": "Ten selected short test cases, three paired agent cases, one sample each; unequal native reasoning controls, provider routing, possible benchmark memorization, no clinical validation."}
    write_json(run_dir / "scores.json", {"scores": scores})
    write_json(run_dir / "summary.json", summary)
    return summary


def run_evaluation(root: Path, subset: Path, run_dir: Path):
    if run_dir.exists():
        raise ValueError("A run directory may not be overwritten or resumed implicitly.")
    metadata = read_json(root / "data/checks/medqa-selected-model-audit-20261011.json")
    if "ledger" not in metadata:
        raise ValueError("No verified recent credit ledger; paid execution blocked.")
    initial = float(metadata["ledger"]["total_usage"])
    budget = RequestBudget(6.0, initial)
    budget.unreported_charge_reserve_usd = SAFETY_RESERVE_USD
    budget.check(cohort_bound())
    questions = read_json(subset / "questions.json")["questions"]
    if len(questions) != 10:
        raise ValueError("Exactly ten frozen questions are required.")
    profiles = model_configs()
    run_dir.mkdir(parents=True)
    for filename in ("questions.json", "answer-key.json"):
        shutil.copyfile(subset / filename, run_dir / filename)
    shutil.copyfile(subset / "manifest.json", run_dir / "subset-manifest.json")
    shutil.copyfile(root / "data/checks/medqa-selected-model-audit-20261011.json", run_dir / "initial-account-audit.json")
    shutil.copyfile(root / "docs/implementation/medqa-mini-protocol.md", run_dir / "protocol.md")
    snapshot = run_dir / "source-snapshot"
    snapshot.mkdir()
    hashes = {}
    for path in sorted((root / "src/medical_copilot").glob("*.py")):
        shutil.copyfile(path, snapshot / path.name)
        hashes[path.name] = file_hash(path)
    order = []
    rotated = list(range(5))
    random.Random(SEED).shuffle(rotated)
    for mode, cases in (("direct", questions), ("agent", questions[:3])):
        for index, question in enumerate(cases):
            ids = rotated[index % 5:] + rotated[:index % 5]
            order.extend({"mode": mode, "question_id": question["id"], "profile": profiles[number][0]} for number in ids)
    manifest = {"started_at_utc": utc_now(), "profiles": [{"id": profile, "config": config.public_dict(), "native_effort": config.provider_options["extra_body"]["reasoning"]} for profile, config in profiles], "case_order": order, "question_ids": [question["id"] for question in questions], "agent_question_ids": [question["id"] for question in questions[:3]], "source_hashes": hashes, "package_versions": package_versions(), "question_file_sha256": file_hash(run_dir / "questions.json"), "answer_key_file_sha256": file_hash(run_dir / "answer-key.json"), "maximum_requests": 80, "maximum_new_request_allowance_usd": cohort_bound(), "initial_account_usage_usd": initial, "historical_charge_reconciliation": "Recent account ledger replaces old overly conservative failed-request reserves; USD 0.15 remains reserved for delayed charges and billing uncertainty. Ledger differs from earlier local charge sum, so account total is used conservatively.", "user_effort_decision": "User explicitly retained all five: GLM/Kimi high, Qwen xhigh, MiMo/Gemma enabled without native effort scale. Not equal internal compute.", "prompts": {"direct": ANSWER_SYSTEM, "plan": PLAN_SYSTEM, "agent": AGENT_SYSTEM}, "design": "One-call direct answer for each of ten identical questions; two-call optional-search LangGraph agent on first three presampled cases for every model. No paid judge, automatic retries, revisions or repair calls. Fresh case state, no key in generation."}
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "budget.json", budget.public_dict())
    clients, audits, failure_streaks = {}, {}, {}
    for profile, config in profiles:
        directory = run_dir / "models" / profile
        directory.mkdir(parents=True)
        backend, redactor, _ = create_backend(config, root)
        audit = MiniAudit(backend, config, budget, redactor, directory / "usage-audit.json")
        clients[profile] = ModelClient(audit, config, redactor)
        audits[profile] = audit
        failure_streaks[profile] = 0
        write_json(directory / "config.json", config.public_dict())
    by_id = {question["id"]: question for question in questions}
    for index, case in enumerate(order, 1):
        profile, mode, question_id = case["profile"], case["mode"], case["question_id"]
        print(f"[{index}/{len(order)}] START {profile} {mode} {question_id}", flush=True)
        client, audit = clients[profile], audits[profile]
        audit.case_id = mode + ":" + question_id
        call_start = len(client.calls)
        audit_start = len(audit.records)
        directory = run_dir / "models" / profile
        pubmed = PubMedClient(PubMedConfig(max_retries=0, timeout_seconds=15, max_abstract_characters=500), directory / "pubmed-snapshots")
        if failure_streaks[profile] >= 2:
            record = {"status": "not_run_provider_circuit", "elapsed_seconds": 0, "reason": "Two consecutive failed cases; no further requests in this small run."}
        else:
            record = MiniCopilot(client, pubmed, mode).run(by_id[question_id])
            failure_streaks[profile] = failure_streaks[profile] + 1 if record["status"] == "error" else 0
        record.update(case, question_sha256=by_id[question_id]["question_sha256"], model_calls=client.calls[call_start:], pubmed_requests=pubmed.requests, audit_records=audit.records[audit_start:])
        write_json(directory / "answers" / (mode + "-" + question_id + ".json"), record)
        write_json(run_dir / "budget.json", budget.public_dict())
        print(f"[{index}/{len(order)}] END {profile} {mode} {question_id}: {record['status']} cumulative_recorded_usd={budget.spent_usd:.6f}", flush=True)
    manifest["completed_at_utc"] = utc_now()
    write_json(run_dir / "manifest.json", manifest)
    return summarize(run_dir)
