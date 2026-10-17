"""English command-line interface for demonstrations and evaluation runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from medical_copilot.config import ConfigurationError, load_config
from medical_copilot.evaluation import run_evaluation, summarize_assessments, write_json
from medical_copilot.graph import MedicalCopilot
from medical_copilot.models import ModelClient, ModelError, create_backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Oncology-nutrition research copilot (requirement 2).")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root; relative data and credential paths are resolved here.")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Answer one educational question with the LangGraph agent.")
    ask.add_argument("question")
    ask.add_argument("--config", type=Path, default=Path("config/models/gemini.json"))
    ask.add_argument("--output", type=Path, help="Optional JSON trace output; nothing is written by default.")
    evaluate = commands.add_parser("evaluate", help="Run and preserve the internal evaluation set; no automatic medical grading.")
    evaluate.add_argument("--config", type=Path, default=Path("config/models/gemini.json"))
    evaluate.add_argument("--questions", type=Path, default=Path("data/evaluation/oncology_nutrition_questions.json"))
    evaluate.add_argument("--rubric", type=Path, default=Path("data/evaluation/qualitative_rubric.json"))
    evaluate.add_argument("--run-dir", type=Path)
    evaluate.add_argument("--ids", nargs="+", help="Optional internal question IDs for a pilot run.")
    evaluate.add_argument("--delay", type=float, default=1.0, help="Seconds between questions (0-30).")
    show = commands.add_parser("show-config", help="Validate and show a profile without reading credentials or making API calls.")
    show.add_argument("--config", type=Path, default=Path("config/models/gemini.json"))
    summarize = commands.add_parser("summarize", help="Validate separately authored qualitative reviews and build the summary.")
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--assessments", type=Path, required=True)
    summarize.add_argument("--rubric", type=Path, default=Path("data/evaluation/qualitative_rubric.json"))
    args = parser.parse_args(argv)
    root = args.root.resolve()

    def local(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    try:
        if args.command == "summarize":
            result = summarize_assessments(local(args.run_dir), local(args.assessments), local(args.rubric))
            print(json.dumps(result, indent=2))
            return 0
        config = load_config(local(args.config))
        if args.command == "show-config":
            print(json.dumps(config.public_dict(), indent=2))
            return 0
        if args.command == "evaluate":
            if not 0 <= args.delay <= 30:
                raise ValueError("Evaluation delay must be between 0 and 30 seconds.")
            run_dir = local(args.run_dir) if args.run_dir else root / "data/evaluation/runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            result = run_evaluation(config, root, local(args.questions), local(args.rubric), run_dir, args.ids, args.delay)
            print(json.dumps(result, indent=2))
            return 1 if result["statuses"].get("error") else 0
        backend, redactor, _ = create_backend(config, root)
        result = MedicalCopilot(ModelClient(backend, config, redactor)).run(args.question)
        if args.output:
            write_json(local(args.output), redactor.object(result))
        print(f"Agent status: {result['status']}\n")
        print(result.get("answer") or result.get("error", "No answer."))
        return 1 if result["status"] == "error" else 0
    except (ConfigurationError, ModelError, ValueError, OSError) as exc:
        # Application exceptions contain safe summaries; no provider traceback.
        if isinstance(exc, (ConfigurationError, ModelError)):
            print(str(exc), file=sys.stderr)
        else:
            print(f"Operation failed ({type(exc).__name__}); check input paths and arguments.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
