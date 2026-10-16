"""Collect a fixed synthetic patient-memory demonstration in fresh processes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from medical_copilot.clinical_evaluation import public_error
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import read_json, write_json
from medical_copilot.patient_agent import VERSION
from medical_copilot.patient_memory import PatientMemory, PatientScopeError, file_hash, utc_now
from medical_copilot.patient_runtime import NEW_SOURCES, RUNS
from medical_copilot.patient_tools import PatientTools, ReadInteractionArgs, SearchHistoryArgs

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "data/evaluation/patient-demo-20261014"


def fact(value, observed, **extras):
    return {"fact_key": "weight_kg", "value": value, "unit": "kg", "observed_at": observed,
            "source_ref": "Explicit synthetic demo input; not a clinical claim.", **extras}


def prepare_memory(database):
    fixture = read_json(SET / "fixtures.json")
    with PatientMemory(database) as memory:
        documents = {}
        for patient in fixture["patients"]:
            pid = patient["patient_id"]
            memory.create_patient(pid, patient["display_name"])
            memory.conversation(pid, pid + "-seed", create=True)
            memory.add_fact(pid, fact(patient["weight_kg"], "2026-10-10T09:00:00+00:00"), fact_id="F-" + pid + "-initial")
            documents[pid] = memory.register_pdf(pid, SET / (pid + ".pdf"), title="Synthetic " + pid + " case")
        memory.register_pdf("alpha", ROOT / "data/documents/reference/espen-cancer-2021.pdf",
                            title="ESPEN practical guideline: Clinical Nutrition in cancer", year="2021",
                            url="https://www.espen.org/files/ESPEN-Guidelines/ESPEN-practical-guideline-clinical-nutrition-in-cancer.pdf")
        memory.save_interaction("alpha", "alpha-seed", {"question_id": "SEED-SNACK", "question": "My preferred snack is " + fixture["alpha_snack"] + "."},
                                {"type": "fixture_note", "answer": "Synthetic application acknowledgment; no model generated this turn."}, {},
                                status="fixture", origin="synthetic_fixture", interaction_id="I-alpha-snack", created_at="2026-10-10T10:00:00+00:00")
        for i in range(fixture["filler_turns"]):
            memory.save_interaction("alpha", "alpha-seed", {"question_id": "SEED-FILLER-" + str(i), "question": "Unrelated synthetic scheduling note " + str(i)},
                                    {"type": "fixture_note", "answer": "Recorded by fixture script, not an LLM."}, {}, status="fixture", origin="synthetic_fixture",
                                    interaction_id="I-alpha-filler-" + str(i))
        checks = {}
        for name, operation in [
            ("foreign_document_rejected", lambda: memory.document("beta", documents["alpha"]["document_id"])),
            ("foreign_conversation_rejected", lambda: memory.conversation("beta", "alpha-seed")),
            ("foreign_interaction_rejected", lambda: PatientTools(memory, "beta").read(ReadInteractionArgs(interaction_id="I-alpha-snack"))),
        ]:
            try:
                operation()
            except PatientScopeError:
                checks[name] = True
            else:
                checks[name] = False
        checks["foreign_history_not_searched"] = not memory.search_history("beta", "snack")
        assert all(checks.values())
        return checks


def validate_prerequisite(directory, stage, hashes):
    manifest = read_json(directory / "manifest.json")
    summary = read_json(directory / "summary.json")
    if manifest.get("stage") != stage or not manifest.get("finished_at_utc") or manifest.get("source_hashes") != hashes:
        raise ValueError("Prerequisite stage/source provenance does not match.")
    if summary["valid_model_answers"] != summary["planned"]:
        raise ValueError("Prerequisite transport did not complete all episodes.")
    # Semantic correctness remains a result, never a transport selection criterion.
    return file_hash(directory / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["probe", "pilot", "main"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.parent != ROOT / RUNS or run.exists():
        raise ValueError("Use a fresh direct child of data/evaluation/patient-runs.")
    fixture_manifest = read_json(SET / "manifest.json")
    for name, expected in fixture_manifest["source_hashes"].items():
        if file_hash(SET / name) != expected:
            raise ValueError("Frozen synthetic input changed.")
    hashes = {name: file_hash(ROOT / "src/medical_copilot" / name) for name in NEW_SOURCES}
    prerequisites = {}
    if args.stage in {"pilot", "main"}:
        if not args.probe_dir:
            raise ValueError("A matching excluded native probe is required.")
        prerequisites["probe_manifest_sha256"] = validate_prerequisite(args.probe_dir.resolve(), "probe", hashes)
    if args.stage == "main":
        if not args.pilot_dir:
            raise ValueError("Matching excluded tool pilots are required.")
        prerequisites["pilot_manifest_sha256"] = validate_prerequisite(args.pilot_dir.resolve(), "pilot", hashes)
    run.mkdir(parents=True)
    for name in ("questions.json", "answer-key.json", "fixtures.json", "manifest.json"):
        shutil.copyfile(SET / name, run / ("selection-manifest.json" if name == "manifest.json" else name))
    manifest = {"requirement": 7, "version": VERSION, "stage": args.stage, "started_at_utc": utc_now(), "source_hashes": hashes,
                "selection_manifest_sha256": file_hash(SET / "manifest.json"), "aggregate_only": True, "subruns": [], **prerequisites}
    write_json(run / "manifest.json", manifest)
    database = run / "memory.sqlite3"
    storage_checks = prepare_memory(database)
    save_trace(run / "storage-checks.json", storage_checks)
    questions = read_json(SET / "questions.json")["questions"]
    if args.stage == "probe":
        questions = [{"patient_id": "alpha", "conversation_id": "alpha-probe", "memory_enabled": True, "require_reference": False,
                      "question": {"question_id": "PROBE01", "question": "Give my registered display name as a JSON object with display_name. Use the supplied patient context.", "options": None}}]
    elif args.stage == "pilot":
        questions = [{"patient_id": pid, "conversation_id": pid + "-pilot", "memory_enabled": True, "require_reference": True,
                      "question": {"question_id": "PILOT-" + pid, "question": "Read my registered synthetic case PDF and report its exact record marker in a JSON object with marker. Cite an actual source view.", "options": None}} for pid in ("alpha", "beta")]
    save_trace(run / "episode-plan.json", {"episodes": questions, "process_restart": "Every episode runs ask_patient.py in a separate fresh Python process."})
    for case in questions:
        qid = case["question"]["question_id"]
        if qid in {"PM07", "PM08"}:
            values = read_json(SET / "fixtures.json")
            with PatientMemory(database) as memory:
                if qid == "PM07":
                    memory.add_fact("alpha", fact(values["new_alpha_weight"], "2026-10-13T09:00:00+00:00"), fact_id="F-alpha-new")
                    memory.add_fact("alpha", fact(values["late_old_alpha_weight"], "2026-10-01T09:00:00+00:00"), fact_id="F-alpha-late-old")
                else:
                    memory.add_fact("alpha", fact(values["corrected_alpha_weight"], "2026-10-13T09:00:00+00:00", supersedes_fact_id="F-alpha-new"), fact_id="F-alpha-corrected")
                memory.backup(run / "update-snapshots" / (qid + ".sqlite3"))
        input_path = run / "episode-inputs" / (qid + ".json")
        save_trace(input_path, case["question"])
        child = ROOT / RUNS / (run.name + "-" + qid.lower())
        command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/ask_patient.py"), "--database", str(database),
                   "--patient", case["patient_id"], "--conversation", case["conversation_id"], "--input-json", str(input_path), "--run-dir", str(child)]
        if not case["memory_enabled"]:
            command.append("--no-memory")
        if case["require_reference"]:
            command.append("--require-reference")
        if args.stage == "probe":
            command.append("--final-only-probe")
        outcome = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        (run / "episode-inputs" / (qid + "-stdout.txt")).write_text(outcome.stdout, encoding="utf-8")
        entry = {"question_id": qid, "run_dir": child.relative_to(ROOT).as_posix(), "returncode": outcome.returncode,
                 "stderr_withheld": bool(outcome.stderr), "fresh_python_process": True}
        if (child / "answers" / (qid + ".json")).exists():
            row = read_json(child / "answers" / (qid + ".json"))
            entry.update(status=row["status"], answer_sha256=file_hash(child / "answers" / (qid + ".json")))
        else:
            entry["status"] = "no_saved_answer"
        manifest["subruns"].append(entry)
        write_json(run / "manifest.json", manifest)
        print(qid + ": " + entry["status"] + "; separate process completed", flush=True)
    summaries = [read_json(ROOT / entry["run_dir"] / "summary.json") for entry in manifest["subruns"] if (ROOT / entry["run_dir"] / "summary.json").exists()]
    summary = {key: sum(s.get(key, 0) for s in summaries) for key in ("recorded", "valid_model_answers", "terminal_messages", "saved_interactions", "provider_requests", "reported_or_estimated_cost_usd", "unknown_charge_records", "input_tokens", "output_tokens", "memory_tool_calls", "pdf_tool_calls", "literature_tool_calls")}
    summary.update(planned=len(questions), stage=args.stage, fresh_process_per_episode=True, storage_checks_passed=all(storage_checks.values()),
                   interpretation="Separate process persistence demonstration; primary reference fields and qualitative support are appraised separately.")
    manifest["finished_at_utc"] = utc_now()
    write_json(run / "manifest.json", manifest)
    write_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": public_error(exc)}))
        raise SystemExit(1) from None
