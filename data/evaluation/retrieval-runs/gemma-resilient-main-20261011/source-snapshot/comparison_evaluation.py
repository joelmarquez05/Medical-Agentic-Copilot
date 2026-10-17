"""Preserve a model comparison, actual usage and bounded paid requests."""

from __future__ import annotations

import copy
import json
import math
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_copilot.comparison import ComparisonCopilot, PROMPT_VERSION
from medical_copilot.config import load_config
from medical_copilot.evaluation import DIMENSIONS, file_hash, load_questions, package_versions, read_json, write_json
from medical_copilot.evidence import GuidanceLibrary
from medical_copilot.models import ModelClient, create_backend

DEFAULT_PROFILES = ["openrouter-claude-opus-5.5", "openrouter-glm-5.3", "openrouter-qwen3.8-27b", "gemini-3.8-flash", "gemma-4-31b"]
DEFAULT_COHORTS = ["gemini-pubmed-internal-20261010", "gemini-pubmed-final-20261010"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def freeze_evidence(root: Path, destination: Path, questions: list[dict], cohorts=DEFAULT_COHORTS) -> dict[str, Any]:
    """Deduplicate the union of actual old records, without relevance filtering."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Fixed evidence must be written to an empty directory.")
    destination.mkdir(parents=True, exist_ok=True)
    library = GuidanceLibrary()
    guidance = library.lookup([card["topic"] for card in library.data["cards"]])
    inventory = []
    for question in questions:
        papers, provenance = {}, []
        for cohort in cohorts:
            path = root / "data/evaluation/retrieval-runs" / cohort / "retrieval" / (question["id"] + ".json")
            record = read_json(path)
            if record["question"] != question["question"]:
                raise ValueError("Archived retrieval question differs from the internal question.")
            provenance.append({"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)})
            for paper in record["unique_papers"]:
                previous = papers.get(paper["id"])
                if previous and (previous["title"], previous["abstract"]) != (paper["title"], paper["abstract"]):
                    raise ValueError("Repeated PubMed records disagree; resolve explicitly before freezing.")
                papers.setdefault(paper["id"], copy.deepcopy(paper))
        # Remove query-specific rank; provenance stays in the separate inventory.
        ordered = [{key: value for key, value in papers[source].items() if key != "retrieval_rank"} for source in sorted(papers, key=lambda source: int(source[1:]))]
        evidence = {"mode": "fixed_previously_retrieved_pubmed_plus_guidance", "guidance_version": guidance["version"], "cards": guidance["cards"], "sources": guidance["sources"] + ordered, "limitation": "Manual guidance paraphrases and question-specific PubMed metadata/abstracts from two earlier real retrieval cohorts, not full texts or a new search. The unfiltered union may contain irrelevant or conflicting material. No evaluator labels are supplied."}
        path = destination / (question["id"] + ".json")
        write_json(path, evidence)
        inventory.append({"question_id": question["id"], "sha256": file_hash(path), "pubmed_ids": [paper["id"] for paper in ordered], "guidance_ids": [source["id"] for source in guidance["sources"]], "archived_retrieval_records": provenance})
    manifest = {"created_at_utc": utc_now(), "selection": "Unfiltered PMID-deduplicated union of the initial and final requirement-3 cohorts for each unchanged question; all ten guidance cards supplied to every model.", "guidance_sha256": file_hash(root / "src/medical_copilot/resources/oncology_guidance.json"), "questions": inventory}
    write_json(destination / "manifest.json", manifest)
    return manifest


class BudgetLimitExceeded(RuntimeError):
    """No provider request was sent because the local allowance was too small."""


class RequestBudget:
    def __init__(self, limit_usd: float, spent_usd: float = 0.0):
        if not math.isfinite(limit_usd) or limit_usd <= 0 or not math.isfinite(spent_usd) or spent_usd < 0:
            raise ValueError("Budget values must be finite and nonnegative, with a positive limit.")
        self.limit_usd, self.spent_usd = limit_usd, spent_usd
        self.unreported_charge_reserve_usd = 0.0

    def check(self, estimated_maximum: float):
        if self.spent_usd + self.unreported_charge_reserve_usd + estimated_maximum > self.limit_usd:
            raise BudgetLimitExceeded("The local paid-request allowance is exhausted.")

    def public_dict(self):
        return {"limit_usd": self.limit_usd, "reported_or_estimated_spend_usd": round(self.spent_usd, 8), "unreported_charge_reserve_usd": round(self.unreported_charge_reserve_usd, 8), "policy": "Before each paid request, reserve a conservative prompt-byte estimate plus the full output cap at configured provider price ceilings. Failed requests without usage retain that reserve. This is a local guard, not a provider billing guarantee."}


class AuditedBackend:
    """Capture numeric usage and public response metadata, never private thoughts."""

    def __init__(self, inner, config, budget: RequestBudget, redactor, audit_path: Path):
        self.inner, self.config, self.budget, self.redactor, self.audit_path = inner, config, budget, redactor, audit_path
        self.records: list[dict] = []
        self.case_id: str | None = None

    def invoke(self, messages):
        paid = self.config.provider == "openai_compatible" and "openrouter.ai" in (self.config.base_url or "")
        prices = self.config.provider_options.get("extra_body", {}).get("provider", {}).get("max_price", {})
        if paid and not {"prompt", "completion"}.issubset(prices):
            raise ValueError("Paid comparison profiles must declare provider price ceilings.")
        prompt_bound = 1024 + sum(len(str(message.content).encode("utf-8")) for message in messages)
        maximum = (prompt_bound * prices.get("prompt", 0) + self.config.max_tokens * prices.get("completion", 0)) / 1_000_000 if paid else 0.0
        record = {"question_id": self.case_id, "started_at_utc": utc_now(), "paid_request": paid, "estimated_maximum_usd": round(maximum, 8)}
        started = time.perf_counter()
        try:
            if paid:
                self.budget.check(maximum)
            record["request_sent"] = True
            message = self.inner.invoke(messages)
            metadata = message.response_metadata or {}
            provider_usage = metadata.get("token_usage", {})
            usage = message.usage_metadata or {}
            cost = provider_usage.get("cost")
            if paid and (not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0):
                cost = (usage.get("input_tokens", 0) * prices["prompt"] + usage.get("output_tokens", 0) * prices["completion"]) / 1_000_000
                record["cost_basis"] = "Conservative configured price ceilings times reported tokens; actual billing absent."
            elif paid:
                record["cost_basis"] = "OpenRouter response usage.cost in USD."
            else:
                record["cost_basis"] = "No monetary cost reported by Google; zero USD is not asserted."
            if paid:
                self.budget.spent_usd += float(cost)
            record.update(status="ok", usage=usage, provider_usage=provider_usage, cost_usd=float(cost) if paid else None, model_name=metadata.get("model_name"), model_version=metadata.get("model_version"), finish_reason=metadata.get("finish_reason"), provider_route="Not exposed by this LangChain response.")
            return message
        except Exception as exc:
            sent = bool(record.get("request_sent"))
            if paid and sent:
                self.budget.unreported_charge_reserve_usd += maximum
            status = getattr(exc, "status_code", None)
            if not isinstance(status, int):
                status = getattr(exc, "code", None)
            record.update(status="error", error_type=type(exc).__name__, request_sent=sent, error_http_status=status if isinstance(status, int) else None, cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None)
            raise
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.records.append(self.redactor.object(record))
            write_json(self.audit_path, {"records": self.records, "budget": self.budget.public_dict()})


def comparison_summary(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    models = []
    for profile in manifest["profiles"]:
        directory = run_dir / "models" / profile["id"]
        answers = [read_json(path) for path in sorted((directory / "answers").glob("*.json"))]
        audit = read_json(directory / "usage-audit.json")["records"] if (directory / "usage-audit.json").exists() else []
        attempted = [answer for answer in answers if answer["status"] != "not_run_provider_unavailable"]
        complete = [answer for answer in attempted if answer["status"] != "error" and answer.get("answer")]
        elapsed = [answer["elapsed_seconds"] for answer in attempted]
        import statistics
        models.append({"profile": profile["id"], "model": profile["config"]["model"], "recorded_cases": len(answers), "attempted_cases": len(attempted), "unattempted_provider_unavailable_cases": len(answers) - len(attempted), "completed_answers": len(complete), "statuses": dict(Counter(answer["status"] for answer in answers)), "model_calls": sum(len(answer["model_calls"]) for answer in answers), "provider_requests": sum(bool(record["request_sent"]) for record in audit), "revisions": sum(answer.get("revision_count", 0) for answer in answers), "json_repair_calls": sum("json_repair" in call["stage"] for answer in answers for call in answer["model_calls"]), "reported_tokens": {key: sum(answer["usage"][key] for answer in answers) for key in ["input_tokens", "output_tokens", "total_tokens"]}, "reported_or_estimated_cost_usd": round(sum(record.get("cost_usd") or 0 for record in audit), 8) if any(record["paid_request"] for record in audit) else None, "median_case_elapsed_seconds": round(statistics.median(elapsed), 3) if elapsed else None, "total_case_elapsed_seconds": round(sum(elapsed), 3), "live_pubmed_requests": sum(answer["live_pubmed_requests"] for answer in answers)})
    summary = {"run_id": manifest["run_id"], "evaluation": manifest["evaluation"], "question_count": len(manifest["question_ids"]), "models": models, "interpretation": "Engineering results only. Medical quality is assessed separately from model self-review and completion. Latency includes configured pacing, all graph stages and bounded retries; provider defaults differ."}
    write_json(run_dir / "engineering-summary.json", summary)
    return summary


def run_comparison(root: Path, run_dir: Path, profiles=DEFAULT_PROFILES, ids: list[str] | None = None, budget_usd=8.0, initial_spend_usd=0.0, provider_failure_limit: int = 3) -> dict[str, Any]:
    dataset_path = root / "data/evaluation/oncology_nutrition_questions.json"
    dataset = load_questions(dataset_path)
    questions = [question for question in dataset["questions"] if ids is None or question["id"] in ids]
    if not questions or (ids and set(ids) != {question["id"] for question in questions}):
        raise ValueError("Select existing question IDs.")
    if len(set(profiles)) != len(profiles) or not profiles:
        raise ValueError("Select distinct model profiles.")
    if type(provider_failure_limit) is not int or not 1 <= provider_failure_limit <= 10:
        raise ValueError("The consecutive provider-failure limit must be 1-10.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Choose an empty comparison directory; prior observations cannot be overwritten.")
    configs = [(profile, load_config(root / "config/models" / (profile + ".json"))) for profile in profiles]
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_manifest = freeze_evidence(root, run_dir / "evidence", questions)
    source_dir = run_dir / "source-snapshot"
    source_dir.mkdir()
    source_hashes = {}
    for path in sorted((root / "src/medical_copilot").glob("*.py")):
        shutil.copyfile(path, source_dir / path.name)
        source_hashes[path.name] = file_hash(path)
    shutil.copyfile(dataset_path, run_dir / "question-set.json")
    shutil.copyfile(root / "data/evaluation/comparison_rubric.json", run_dir / "qualitative-rubric.json")
    shutil.copyfile(root / "docs/implementation/requirement-5-protocol.md", run_dir / "protocol.md")
    order = []
    # A balanced rotation avoids always testing a provider first or last.
    rng = random.Random(20261010)
    base_order = list(profiles)
    rng.shuffle(base_order)
    for index, question in enumerate(questions):
        rotated = base_order[index % len(profiles):] + base_order[:index % len(profiles)]
        order.extend({"question_id": question["id"], "profile": profile} for profile in rotated)
    manifest = {"run_id": run_dir.name, "evaluation": "requirement_5_controlled_internal_model_comparison", "started_at_utc": utc_now(), "prompt_version": PROMPT_VERSION, "dataset_id": dataset["dataset_id"], "question_ids": [question["id"] for question in questions], "questions_sha256": file_hash(dataset_path), "rubric_sha256": file_hash(run_dir / "qualitative-rubric.json"), "protocol_sha256": file_hash(run_dir / "protocol.md"), "source_code_sha256": source_hashes, "package_versions": package_versions(), "evidence_manifest_sha256": file_hash(run_dir / "evidence/manifest.json"), "profiles": [{"id": profile, "config": config.public_dict(), "config_sha256": file_hash(root / "config/models" / (profile + ".json"))} for profile, config in configs], "case_order": order, "design": "One fresh graph state per model-question pair; identical fixed evidence for each question. Same system prompts, tools, shared output cap from profiles and bounded one revision/one JSON repair. Provider sampling and reasoning are not equalized: Qwen medium effort, other provider defaults. No live search or cross-question clinical memory.", "selection_rationale": "User-selected frontier proprietary and open-weight models plus compact open-weight candidates for potential future local research; selection roles are hypotheses, not proven medical rankings.", "budget": RequestBudget(budget_usd, initial_spend_usd).public_dict()}
    write_json(run_dir / "manifest.json", manifest)
    manifest["consecutive_provider_failure_limit"] = provider_failure_limit
    write_json(run_dir / "manifest.json", manifest)
    budget = RequestBudget(budget_usd, initial_spend_usd)
    clients, audits = {}, {}
    for profile, config in configs:
        backend, redactor, source = create_backend(config, root)
        directory = run_dir / "models" / profile
        directory.mkdir(parents=True)
        write_json(directory / "config.json", config.public_dict())
        audited = AuditedBackend(backend, config, budget, redactor, directory / "usage-audit.json")
        audits[profile] = audited
        clients[profile] = ModelClient(audited, config, redactor)
        write_json(directory / "credential-resolution.json", {"source": source, "value_saved": False})
    by_id = {question["id"]: question for question in questions}
    evidence_hashes = {entry["question_id"]: entry["sha256"] for entry in evidence_manifest["questions"]}
    failure_streaks = {profile: 0 for profile in profiles}
    for index, case in enumerate(order):
        profile, question_id = case["profile"], case["question_id"]
        print(f"[{index + 1}/{len(order)}] START {profile} {question_id}", flush=True)
        audits[profile].case_id = question_id
        audit_start = len(audits[profile].records)
        evidence = read_json(run_dir / "evidence" / (question_id + ".json"))
        if failure_streaks[profile] >= provider_failure_limit:
            result = {"question": by_id[question_id]["question"], "status": "not_run_provider_unavailable", "answer": "", "model_calls": [], "tool_results": [], "trace": [], "elapsed_seconds": 0, "usage": {key: 0 for key in ["input_tokens", "output_tokens", "total_tokens"]}, "live_pubmed_requests": 0, "prompt_version": PROMPT_VERSION, "evidence_mode": "fixed_previously_retrieved_pubmed_plus_guidance", "reason": f"No request sent after {provider_failure_limit} consecutive model-question failures ending in a provider request error. Not a measured medical response."}
        else:
            result = ComparisonCopilot(clients[profile], evidence).run(by_id[question_id]["question"])
            provider_error = result["status"] == "error" and result["model_calls"] and result["model_calls"][-1].get("status") == "error"
            failure_streaks[profile] = failure_streaks[profile] + 1 if provider_error else 0
        result["consecutive_provider_failures"] = failure_streaks[profile]
        result.update(question_id=question_id, profile=profile, evidence_sha256=evidence_hashes[question_id], response_audit=audits[profile].records[audit_start:])
        write_json(run_dir / "models" / profile / "answers" / (question_id + ".json"), result)
        write_json(run_dir / "budget.json", budget.public_dict())
        comparison_summary(run_dir)
        print(f"[{index + 1}/{len(order)}] {profile} {question_id}: {result['status']}; {len(result['model_calls'])} calls; {result['elapsed_seconds']} s; paid spend ${budget.spent_usd:.4f}", flush=True)
    manifest.update(completed_at_utc=utc_now(), budget=budget.public_dict())
    write_json(run_dir / "manifest.json", manifest)
    return comparison_summary(run_dir)


def summarize_comparison_assessments(run_dir: Path, assessment_path: Path) -> dict[str, Any]:
    """Validate separately authored medical ratings; never derive them from text."""
    import statistics
    manifest = read_json(run_dir / "manifest.json")
    document = read_json(assessment_path)
    assessments = document["assessments"]
    expected = {(case["profile"], case["question_id"]) for case in manifest["case_order"]}
    pairs = [(item["profile"], item["question_id"]) for item in assessments]
    if len(set(pairs)) != len(pairs) or set(pairs) != expected:
        raise ValueError("Assessments must cover each selected model-question pair exactly once.")
    for item in assessments:
        path = run_dir / "models" / item["profile"] / "answers" / (item["question_id"] + ".json")
        answer = read_json(path)
        if item["answer_file_sha256"] != file_hash(path):
            raise ValueError("An assessment does not match the saved answer fingerprint.")
        if answer["status"] == "error" or not answer.get("answer"):
            category = "not_scorable_provider_unavailable" if answer["status"] == "not_run_provider_unavailable" else "not_scorable_generation_error"
            if item["ratings"] is not None or item["assessment_status"] != category:
                raise ValueError("Generation failures must have N/A medical ratings.")
            continue
        if item["assessment_status"] != "assessed_final_answer" or set(item["ratings"]) != set(DIMENSIONS):
            raise ValueError("Completed final answers need the five prespecified dimensions.")
        if any(type(value) is not int or value not in {0, 1, 2} for value in item["ratings"].values()):
            raise ValueError("Qualitative ratings must be integers 0, 1 or 2.")
        if set(item["dimension_rationales"]) != set(DIMENSIONS) or not all(item["dimension_rationales"].values()):
            raise ValueError("Each qualitative dimension needs a substantive rationale.")
        if not item["observed_excerpts"] or not all(excerpt in answer["answer"] for excerpt in item["observed_excerpts"]):
            raise ValueError("Assessment excerpts must be present in the saved final answer.")
        if not isinstance(item["critical_failures_observed"], list):
            raise ValueError("Critical failures must be recorded separately.")
    models = []
    for profile in manifest["profiles"]:
        selected = [item for item in assessments if item["profile"] == profile["id"]]
        scored = [item for item in selected if item["ratings"] is not None]
        totals = [sum(item["ratings"].values()) for item in scored]
        models.append({"profile": profile["id"], "model": profile["config"]["model"], "scored_final_answers": len(scored), "unscored_failures": len(selected) - len(scored), "ordinal_totals_by_question": {item["question_id"]: sum(item["ratings"].values()) if item["ratings"] is not None else None for item in selected}, "mean_ordinal_total": round(statistics.mean(totals), 3) if totals else None, "median_ordinal_total": statistics.median(totals) if totals else None, "dimension_means": {dimension: round(statistics.mean(item["ratings"][dimension] for item in scored), 3) if scored else None for dimension in DIMENSIONS}, "critical_failure_cases": [item["question_id"] for item in scored if item["critical_failures_observed"]]})
    summary = {"run_id": manifest["run_id"], "assessment_method": document["assessment_method"], "assessments_sha256": file_hash(assessment_path), "rubric_sha256": manifest["rubric_sha256"], "models": models, "interpretation": "Descriptive ordinal assistant-authored findings on completed internal answers only. N/A failures are not zero medical scores; means across unequal completion counts are not directly comparable. Not benchmark accuracy or clinical validation."}
    write_json(run_dir / "qualitative-summary.json", summary)
    return summary
