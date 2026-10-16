"""Answer new patient-scoped questions using Gemma, memory, PDF tools and PubMed."""
import argparse
import json
from pathlib import Path

from medical_copilot.clinical_evaluation import public_error
from medical_copilot.clinical_graph import ClinicalQuestion
from medical_copilot.evaluation import read_json
from medical_copilot.patient_memory import PatientMemory, identifier
from medical_copilot.patient_runtime import PatientRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--conversation", required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--no-memory", action="store_true", help="Explicit control: withhold all patient records/tools and do not save answers")
    parser.add_argument("--final-only-probe", action="store_true", help="Excluded native-format check: research disabled, normal final recovery retained")
    args = parser.parse_args()
    if not args.database.is_file():
        raise ValueError("Create the patient database explicitly first.")
    data = read_json(args.input_json)
    rows = data["questions"] if isinstance(data, dict) and set(data) == {"questions"} else [data]
    questions = [ClinicalQuestion.model_validate(row).model_dump(mode="json") for row in rows]
    ids = [identifier(row["question_id"]) for row in questions]
    if not questions or len(ids) != len(set(ids)):
        raise ValueError("Provide nonempty unique question IDs.")
    with PatientMemory(args.database) as memory:
        # Reject missing patients, foreign conversations or excess context before API access.
        memory.conversation(args.patient, args.conversation, create=True)
        if not args.no_memory:
            memory.context(args.patient, args.conversation)
        run = PatientRun(Path(__file__).resolve().parents[1], args.run_dir, config_path=args.config)
        for question in questions:
            row = run.ask(memory, args.patient, args.conversation, question, memory_enabled=not args.no_memory, require_reference=args.require_reference, probe=args.final_only_probe)
            print(json.dumps({"question_id": question["question_id"], "response": row["response"], "memory_write": row.get("memory_write")}, ensure_ascii=False))
        print(json.dumps(run.finish()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": public_error(exc)}))
        raise SystemExit(1) from None
