"""Manage local patient records explicitly; no LLM, network or credentials."""
import argparse
import json
from pathlib import Path

from medical_copilot.clinical_evaluation import public_error
from medical_copilot.evaluation import read_json
from medical_copilot.patient_memory import PatientMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    sub = parser.add_subparsers(dest="operation", required=True)
    create = sub.add_parser("create")
    create.add_argument("--patient", required=True)
    create.add_argument("--name", required=True)
    add = sub.add_parser("add-fact")
    add.add_argument("--patient", required=True)
    add.add_argument("--input-json", type=Path, required=True)
    pdf = sub.add_parser("register-pdf")
    pdf.add_argument("--patient", required=True)
    pdf.add_argument("--pdf", type=Path, required=True)
    pdf.add_argument("--title")
    pdf.add_argument("--url")
    pdf.add_argument("--year")
    show = sub.add_parser("show")
    show.add_argument("--patient", required=True)
    args = parser.parse_args()
    with PatientMemory(args.database) as memory:
        if args.operation == "create":
            result = memory.create_patient(args.patient, args.name)
        elif args.operation == "add-fact":
            result = memory.add_fact(args.patient, read_json(args.input_json))
        elif args.operation == "register-pdf":
            result = memory.register_pdf(args.patient, args.pdf, title=args.title, url=args.url, year=args.year)
        else:
            result = {"patient": memory.patient(args.patient), "current_facts": memory.current_facts(args.patient),
                      "files": memory.documents(args.patient), "recent_public_interactions": memory.recent_interactions(args.patient)}
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": public_error(exc)}))
        raise SystemExit(1) from None
