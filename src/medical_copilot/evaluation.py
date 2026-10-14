"""Run the internal set and preserve reproducible, unscored model outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medical_copilot.config import ModelConfig
from medical_copilot.graph import MedicalCopilot
from medical_copilot.models import ModelClient, create_backend

DIMENSIONS = ["relevance", "correctness", "evidence_support", "actionability", "safety_uncertainty"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so an interrupted write does not leave partial JSON.
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions() -> dict[str, str]:
    names = ["medical-agentic-copilot", "langchain", "langchain-core", "langgraph", "langchain-google-genai", "langchain-openai", "google-genai", "pydantic"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def load_questions(path: Path) -> dict[str, Any]:
    dataset = read_json(path)
    questions = dataset.get("questions", [])
    ids = [item.get("id") for item in questions]
    if not questions or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Evaluation question IDs must be present and unique.")
    if any(not isinstance(item.get("question"), str) or not item["question"].strip() for item in questions):
        raise ValueError("Every evaluation question must have nonempty text.")
    return dataset


def run_evaluation(config: ModelConfig, root: Path, questions_path: Path, rubric_path: Path, run_dir: Path, ids: list[str] | None = None, delay_seconds: float = 1.0) -> dict[str, Any]:
    dataset = load_questions(questions_path)
    selected = [item for item in dataset["questions"] if ids is None or item["id"] in ids]
    if not selected or (ids and set(ids) != {item["id"] for item in selected}):
        raise ValueError("Requested question IDs must exist in the internal set.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Choose an empty run directory; previous results are never overwritten.")
    backend, redactor, credential_source = create_backend(config, root)
    agent = MedicalCopilot(ModelClient(backend, config, redactor))
    code_files = sorted((root / "src/medical_copilot").glob("*.py"))
    evidence_path = root / "src/medical_copilot/resources/oncology_guidance.json"
    manifest: dict[str, Any] = {
        "run_id": run_dir.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation": "requirement_2_internal_questions",
        "dataset_id": dataset["dataset_id"],
        "question_ids": [item["id"] for item in selected],
        "model_config": config.public_dict(),
        "credential_source": credential_source,
        "credentials_saved": False,
        "external_tracing": False,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "package_versions": package_versions(),
        "dataset_sha256": file_hash(questions_path),
        "rubric_sha256": file_hash(rubric_path),
        "guidance_sha256": file_hash(evidence_path),
        "code_sha256": {path.name: file_hash(path) for path in code_files},
        "guidance_version": agent.library.data["version"],
        "assessment_policy": "No automatic correctness score. Saved answers are reviewed separately against the frozen rubric.",
        "status": "running",
        "completed_questions": 0,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "manifest.json", redactor.object(manifest))
    (run_dir / "graph.mmd").write_text(agent.graph.get_graph().draw_mermaid(), encoding="utf-8")
    results = []
    for index, item in enumerate(selected):
        # Only question text enters the agent. Gold criteria are not passed.
        result = agent.run(item["question"])
        result.update(question_id=item["id"], topic=item["topic"])
        write_json(run_dir / "answers" / (item["id"] + ".json"), redactor.object(result))
        results.append(result)
        manifest["completed_questions"] = len(results)
        write_json(run_dir / "manifest.json", redactor.object(manifest))
        print(f"{item['id']}: {result['status']}; {len(result['model_calls'])} model calls; {result['usage']['total_tokens']} reported tokens; {result['elapsed_seconds']:.2f}s", flush=True)
        if index + 1 < len(selected) and delay_seconds:
            time.sleep(delay_seconds)
    manifest.update(status="completed" if all(item["status"] != "error" for item in results) else "completed_with_errors", finished_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    write_json(run_dir / "manifest.json", redactor.object(manifest))
    metrics = engineering_metrics(results)
    write_json(run_dir / "engineering-metrics.json", metrics)
    write_answers_markdown(run_dir, results)
    return metrics


def engineering_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [item["elapsed_seconds"] for item in results]
    return {
        "question_count": len(results),
        "statuses": dict(Counter(item["status"] for item in results)),
        "model_call_count": sum(len(item["model_calls"]) for item in results),
        "model_call_stages": dict(Counter(call["stage"] for item in results for call in item["model_calls"])),
        "reported_usage": {name: sum(item["usage"][name] for item in results) for name in ["input_tokens", "output_tokens", "total_tokens"]},
        "median_question_seconds": round(statistics.median(elapsed), 3),
        "min_question_seconds": round(min(elapsed), 3),
        "max_question_seconds": round(max(elapsed), 3),
        "sum_question_seconds": round(sum(elapsed), 3),
        "revised_questions": [item["question_id"] for item in results if item.get("revision_count", 0) > 0],
        "questions_with_mechanical_citation_issues": [item["question_id"] for item in results if item.get("mechanical_issues")],
        "successful_tool_call_count": sum(call["status"] == "ok" for item in results for call in item.get("tool_results", [])),
        "failed_tool_call_count": sum(call["status"] != "ok" for item in results for call in item.get("tool_results", [])),
        "interpretation": "Operational measurements only. Self-review status and citation-ID checks do not establish medical correctness or safety. Token counts are provider-reported; no billing cost or free-tier status is inferred.",
    }


def write_answers_markdown(run_dir: Path, results: list[dict[str, Any]]) -> None:
    sections = ["# Internal evaluation: saved agent answers\n\nThese are observed model outputs. Qualitative findings are recorded separately. The JSON files preserve drafts, prompts, tool outputs, reviews, timing, and token metadata.\n"]
    for result in results:
        sections += [f"## {result['question_id']} - {result['topic']}\n", result["question"] + "\n", f"Status: `{result['status']}`. Trace: `{' -> '.join(result.get('trace', []))}`.\n", result.get("answer") or result.get("error", "No answer."), "\n"]
    (run_dir / "answers.md").write_text("\n".join(sections), encoding="utf-8")


def summarize_assessments(run_dir: Path, assessments_path: Path, rubric_path: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    review = read_json(assessments_path)
    rubric = read_json(rubric_path)
    if review["run_id"] != manifest["run_id"] or review["rubric_id"] != rubric["rubric_id"]:
        raise ValueError("Assessment run/rubric IDs do not match.")
    items = review["assessments"]
    ids = [item["question_id"] for item in items]
    if set(ids) != set(manifest["question_ids"]) or len(ids) != len(set(ids)):
        raise ValueError("Exactly one assessment is required for each evaluated question.")
    for item in items:
        answer_path = run_dir / "answers" / (item["question_id"] + ".json")
        if item["answer_file_sha256"] != file_hash(answer_path):
            raise ValueError("An assessed answer has changed since review.")
        if set(item["ratings"]) != set(DIMENSIONS) or any(type(value) is not int or value not in {0, 1, 2} for value in item["ratings"].values()):
            raise ValueError("Ratings must contain all five rubric dimensions with integer values 0-2.")
        if set(item["rationales"]) != set(DIMENSIONS):
            raise ValueError("Each rating must have a qualitative rationale.")
    totals = {item["question_id"]: sum(item["ratings"].values()) for item in items}
    summary = {"run_id": manifest["run_id"], "rubric_id": rubric["rubric_id"], "question_count": len(items), "assessor": review["assessor"], "blinded": review["blinded"], "clinician_review": review["clinician_review"], "rating_counts_by_dimension": {dimension: {str(score): sum(item["ratings"][dimension] == score for item in items) for score in [0, 1, 2]} for dimension in DIMENSIONS}, "descriptive_totals": totals, "critical_failure_question_ids": [item["question_id"] for item in items if item["critical_failures_observed"]], "limitations": review["limitations"], "interpretation": rubric["total_policy"]}
    write_json(run_dir / "qualitative-summary.json", summary)
    lines = ["# Qualitative answer assessment\n", f"Run: `{manifest['run_id']}`. Assessor: {review['assessor']}. Blinded: {review['blinded']}. Independent clinician review: {review['clinician_review']}.\n", "Scores are ordinal aids to the written findings, not accuracy or clinical validation.\n", "| Question | Relevance | Correctness | Evidence | Actionability | Safety/uncertainty | Descriptive total |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in items:
        lines.append("| " + " | ".join([item["question_id"]] + [str(item["ratings"][dimension]) for dimension in DIMENSIONS] + [str(totals[item["question_id"]]) + "/10"]) + " |")
    lines += ["\n## Method and limits\n", *["- " + item for item in review["limitations"]]]
    for item in items:
        lines += [f"\n## {item['question_id']}\n", item["overall_finding"], "\nSupporting observed excerpts:\n", *["> " + excerpt + "\n" for excerpt in item["observed_excerpts"]], "\nDimension rationales:\n", *[f"- **{dimension} ({item['ratings'][dimension]}/2):** {item['rationales'][dimension]}" for dimension in DIMENSIONS], "\nMissing or weak points: " + ("; ".join(item["missing_or_weak_points"]) or "None identified against this limited rubric."), "\nCritical failures observed: " + ("; ".join(item["critical_failures_observed"]) or "None identified against the prespecified critical-failure list."), "\nImprovement: " + item["suggested_improvement"]]
    (run_dir / "qualitative-assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
