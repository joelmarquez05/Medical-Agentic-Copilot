"""A separate operational recovery cohort; original comparison stays intact."""
from __future__ import annotations

import copy
import shutil
import time
from pathlib import Path

from medical_copilot.comparison import ComparisonCopilot, PROMPT_VERSION
from medical_copilot.comparison_evaluation import AuditedBackend, RequestBudget, comparison_summary, utc_now
from medical_copilot.config import ModelConfig, load_config
from medical_copilot.evaluation import file_hash, package_versions, read_json, write_json
from medical_copilot.models import ModelClient, create_backend


def http_code(error):
    """Inspect typed exception metadata, not error bodies or credentials."""
    for _ in range(4):
        if error is None:
            break
        for name in ["status_code", "code"]:
            code = getattr(error, name, None)
            if isinstance(code, int):
                return code
        error = error.__cause__
    return None


class TransientRetryBackend:
    """Bound identical retries for transient server or connection failures."""

    def __init__(self, audited: AuditedBackend, retries=2, wait_seconds=30.0, sleep=time.sleep):
        self.audited, self.retries, self.wait_seconds, self.sleep = audited, retries, wait_seconds, sleep
        self._last_provider_started = 0.0
        if retries not in {0, 1, 2} or not 0 <= wait_seconds <= 60:
            raise ValueError("Recovery retries and wait must remain bounded.")

    def invoke(self, messages):
        for attempt in range(self.retries + 1):
            pace = max(0.0, self.audited.config.min_interval_seconds - (time.perf_counter() - self._last_provider_started))
            if pace:
                self.sleep(pace)
            self._last_provider_started = time.perf_counter()
            audit_start = len(self.audited.records)
            try:
                return self.audited.invoke(messages)
            except Exception as exc:
                code = http_code(exc)
                record = self.audited.records[-1]
                record.update(error_http_status=code, recovery_attempt=attempt + 1)
                connection_failure = "connectionerror" in type(exc).__name__.lower()
                retry = (code in {500, 502, 503} or connection_failure) and attempt < self.retries
                if retry:
                    record["retry_wait_seconds"] = self.wait_seconds
                write_json(self.audited.audit_path, {"records": self.audited.records, "budget": self.audited.budget.public_dict()})
                if not retry:
                    raise
                self.sleep(self.wait_seconds)
            finally:
                if len(self.audited.records) > audit_start:
                    self.audited.records[-1].update(recovery_attempt=attempt + 1, recovery_pacing_wait_seconds=round(pace, 3))
                    write_json(self.audited.audit_path, {"records": self.audited.records, "budget": self.audited.budget.public_dict()})
        raise AssertionError("Unreachable bounded recovery loop.")


def recover_failed_cases(root: Path, original: Path, destination: Path, alternate_platform=False, extra_paid_spend_usd=0.0, extra_unknown_reserve_usd=0.0):
    """Retry noncompleted cases; preserve the actual model/platform used."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty recovery directory.")
    primary = read_json(original / "manifest.json")
    if "completed_at_utc" not in primary:
        raise ValueError("Finish the primary cohort before operational recovery.")
    google = {profile["id"]: copy.deepcopy(profile) for profile in primary["profiles"]}
    if alternate_platform:
        alternate_profiles = {"gemini-3.8-flash": "openrouter-gemini-3.8-flash", "gemma-4-31b": "openrouter-gemma-4-31b"}
        for identifier, profile in google.items():
            if profile["config"]["provider"] != "google_genai":
                continue
            actual_path = root / "config/models" / (alternate_profiles[identifier] + ".json")
            config = load_config(actual_path)
            if config.model.removeprefix("google/") != profile["config"]["model"]:
                raise ValueError("Alternate platform must preserve the publicly named model family.")
            profile["primary_config"] = profile["config"]
            profile["config"] = config.public_dict()
            profile["config_sha256"] = file_hash(actual_path)
    selected = []
    for case in primary["case_order"]:
        if case["profile"] not in google:
            continue
        answer = read_json(original / "models" / case["profile"] / "answers" / (case["question_id"] + ".json"))
        if not answer.get("answer") or answer["status"] in {"error", "not_run_provider_unavailable"}:
            selected.append(case)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["question-set.json", "qualitative-rubric.json", "protocol.md"]:
        shutil.copyfile(original / name, destination / name)
    shutil.copytree(original / "evidence", destination / "evidence")
    source = destination / "source-snapshot"
    source.mkdir()
    hashes = {}
    for path in sorted((root / "src/medical_copilot").glob("*.py")):
        shutil.copyfile(path, source / path.name)
        hashes[path.name] = file_hash(path)
    manifest = {"run_id": destination.name, "evaluation": "requirement_5_separate_operational_recovery", "started_at_utc": utc_now(), "prompt_version": PROMPT_VERSION, "question_ids": sorted({case["question_id"] for case in selected}), "profiles": list(google.values()), "case_order": selected, "original_run": original.relative_to(root).as_posix(), "original_manifest_sha256": file_hash(original / "manifest.json"), "source_code_sha256": hashes, "package_versions": package_versions(), "evidence_manifest_sha256": file_hash(destination / "evidence/manifest.json"), "rubric_sha256": file_hash(destination / "qualitative-rubric.json"), "alternate_platform": alternate_platform, "recovery_policy": "Only failed/unattempted cases are eligible, never completed answers selected by quality. Preserve questions, evidence, prompts, output caps and graph bounds. Google-model recovery uses explicit OpenRouter google/ IDs and price ceilings; Gemma requests reasoning.enabled=true with reasoning.exclude=true because the alternate endpoint initially defaulted to no thinking. Other models retain primary reasoning settings. Hidden checkpoint, quantization and backend equality are not asserted. Add at most two identical retries for HTTP 500/502/503 or typed connection errors with 30-second waits; timeout/auth/model/format/clinical concerns are not retried here. Stop after three consecutive failed provider cases. Platform, reasoning-control and retry-budget differences confound latency/reliability and limit causal attribution of quality differences.", "questions_sha256": primary["questions_sha256"]}
    write_json(destination / "manifest.json", manifest)
    budget = RequestBudget(primary["budget"]["limit_usd"], primary["budget"]["reported_or_estimated_spend_usd"] + extra_paid_spend_usd)
    budget.unreported_charge_reserve_usd = primary["budget"]["unreported_charge_reserve_usd"] + extra_unknown_reserve_usd
    clients, audits = {}, {}
    for identifier, profile in google.items():
        config = ModelConfig.model_validate(profile["config"])
        backend, redactor, credential_source = create_backend(config, root)
        directory = destination / "models" / identifier
        directory.mkdir(parents=True)
        write_json(directory / "config.json", config.public_dict())
        write_json(directory / "credential-resolution.json", {"source": credential_source, "value_saved": False})
        audited = AuditedBackend(backend, config, budget, redactor, directory / "usage-audit.json")
        audits[identifier] = audited
        clients[identifier] = ModelClient(TransientRetryBackend(audited), config, redactor)
    questions = {question["id"]: question for question in read_json(original / "question-set.json")["questions"]}
    streaks = {identifier: 0 for identifier in google}
    for position, case in enumerate(selected):
        identifier, question_id = case["profile"], case["question_id"]
        print(f"Recovery [{position + 1}/{len(selected)}] START {identifier} {question_id}", flush=True)
        audits[identifier].case_id = question_id
        first = len(audits[identifier].records)
        evidence_path = destination / "evidence" / (question_id + ".json")
        if streaks[identifier] >= 3:
            result = {"question": questions[question_id]["question"], "status": "not_run_provider_unavailable", "answer": "", "model_calls": [], "tool_results": [], "trace": [], "elapsed_seconds": 0, "usage": {key: 0 for key in ["input_tokens", "output_tokens", "total_tokens"]}, "live_pubmed_requests": 0, "prompt_version": PROMPT_VERSION, "evidence_mode": "fixed_previously_retrieved_pubmed_plus_guidance", "reason": "No recovery request sent after three consecutive failed provider cases; not a measured medical answer."}
        else:
            result = ComparisonCopilot(clients[identifier], read_json(evidence_path)).run(questions[question_id]["question"])
            failed = result["status"] == "error" and result["model_calls"] and result["model_calls"][-1].get("status") == "error"
            streaks[identifier] = streaks[identifier] + 1 if failed else 0
        result.update(question_id=question_id, profile=identifier, evidence_sha256=file_hash(evidence_path), response_audit=audits[identifier].records[first:], consecutive_provider_failures=streaks[identifier], cohort="operational_recovery")
        write_json(destination / "models" / identifier / "answers" / (question_id + ".json"), result)
        write_json(destination / "budget.json", budget.public_dict())
        comparison_summary(destination)
        print(f"Recovery [{position + 1}/{len(selected)}] {identifier} {question_id}: {result['status']}; {len(result['model_calls'])} logical calls; {len(result['response_audit'])} HTTP attempts; {result['elapsed_seconds']} s", flush=True)
    manifest["completed_at_utc"] = utc_now()
    manifest["budget"] = budget.public_dict()
    write_json(destination / "manifest.json", manifest)
    return comparison_summary(destination)


def consolidate_comparison(root: Path, primary: Path, recovery: Path, destination: Path):
    """Collect the first completed output, with original origins, not new inference."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty consolidated directory.")
    first, second = read_json(primary / "manifest.json"), read_json(recovery / "manifest.json")
    if "completed_at_utc" not in first or "completed_at_utc" not in second:
        raise ValueError("Both source cohorts must be completed.")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["question-set.json", "qualitative-rubric.json", "protocol.md"]:
        shutil.copyfile(primary / name, destination / name)
    shutil.copytree(primary / "evidence", destination / "evidence")
    shutil.copytree(primary / "source-snapshot", destination / "source-snapshot")
    manifest = copy.deepcopy(first)
    manifest.update(run_id=destination.name, evaluation="requirement_5_collected_first_completed_answers_with_separate_recovery", collected_at_utc=utc_now(), collection_policy="Not a new inference run. Keep each completed primary answer; otherwise use the first completed recovery answer. If neither completed, preserve the latest operational record with N/A quality. Selection is based on finalization, never medical score. Actual platform/configuration and all attempts are recorded in origins. Latency/usage of selected graphs and all attempts are separate.", primary_budget=first["budget"], budget=second["budget"], origin_runs=[{"path": path.relative_to(root).as_posix(), "manifest_sha256": file_hash(path / "manifest.json")} for path in [primary, recovery]])
    write_json(destination / "manifest.json", manifest)
    origins = []
    for case in first["case_order"]:
        identifier, question_id = case["profile"], case["question_id"]
        original_path = primary / "models" / identifier / "answers" / (question_id + ".json")
        retry_path = recovery / "models" / identifier / "answers" / (question_id + ".json")
        original = read_json(original_path)
        chosen_path = original_path
        candidates = [original_path]
        if retry_path.exists():
            candidates.append(retry_path)
            retry = read_json(retry_path)
            if not original.get("answer") or original["status"] == "error":
                chosen_path = retry_path
        chosen = copy.deepcopy(read_json(chosen_path))
        origin_manifest = first if chosen_path == original_path else second
        actual_profile = next(profile for profile in origin_manifest["profiles"] if profile["id"] == identifier)
        origin = {"profile": identifier, "question_id": question_id, "selected_answer_file": chosen_path.relative_to(root).as_posix(), "selected_answer_sha256": file_hash(chosen_path), "cohort": "primary" if chosen_path == original_path else "operational_recovery", "actual_model_config": actual_profile["config"], "all_case_attempts": [{"file": path.relative_to(root).as_posix(), "sha256": file_hash(path), "status": read_json(path)["status"], "elapsed_seconds": read_json(path)["elapsed_seconds"]} for path in candidates]}
        chosen["collection_origin"] = origin
        write_json(destination / "models" / identifier / "answers" / (question_id + ".json"), chosen)
        origins.append(origin)
    for profile in first["profiles"]:
        identifier = profile["id"]
        records = []
        for run in [primary, recovery]:
            audit_path = run / "models" / identifier / "usage-audit.json"
            if audit_path.exists():
                records.extend(dict(record, origin_run=run.name) for record in read_json(audit_path)["records"])
        write_json(destination / "models" / identifier / "usage-audit.json", {"records": records, "interpretation": "All primary and recovery provider attempts; selected answer model_calls are one final graph, not this aggregate."})
        write_json(destination / "models" / identifier / "config.json", profile["config"])
    write_json(destination / "answer-origins.json", {"policy": manifest["collection_policy"], "cases": origins})
    summary = comparison_summary(destination)
    for model in summary["models"]:
        selected = [origin for origin in origins if origin["profile"] == model["profile"]]
        attempts = [read_json(root / item["file"]) for origin in selected for item in origin["all_case_attempts"]]
        model["all_cohort_attempted_cases"] = sum(answer["status"] != "not_run_provider_unavailable" for answer in attempts)
        model["all_cohort_model_calls"] = sum(len(answer["model_calls"]) for answer in attempts)
        model["all_cohort_reported_tokens"] = {key: sum(answer["usage"][key] for answer in attempts) for key in ["input_tokens", "output_tokens", "total_tokens"]}
        model["recovery_selected_answers"] = sum(origin["cohort"] == "operational_recovery" and read_json(destination / "models" / origin["profile"] / "answers" / (origin["question_id"] + ".json")).get("answer") != "" for origin in selected)
        model["all_cohort_elapsed_seconds"] = round(sum(answer["elapsed_seconds"] for answer in attempts), 3)
    summary["interpretation"] = "Collected first completed answers, not a new inference cohort. Selected graph latency excludes earlier failed graphs; all-cohort counts/time/tokens and separately preserved original engineering summaries describe those costs. Recovery uses additional transient retries, so raw availability/latency comparisons across providers are confounded. Medical ratings remain separately authored."
    write_json(destination / "engineering-summary.json", summary)
    return summary
