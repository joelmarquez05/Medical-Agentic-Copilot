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
from medical_copilot.literature import LiteratureCopilot
from medical_copilot.pubmed import PubMedClient, PubMedError, load_pubmed_config
from medical_copilot.retrieval_evaluation import run_retrieval_evaluation, summarize_retrieval_assessments
from medical_copilot.comparison_evaluation import DEFAULT_PROFILES, run_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Oncology-nutrition research copilot with PubMed retrieval.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root; relative data and credential paths are resolved here.")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Answer one educational question with the LangGraph agent.")
    ask.add_argument("question")
    ask.add_argument("--config", type=Path, default=Path("config/models/gemini.json"))
    ask.add_argument("--output", type=Path, help="Optional JSON trace output; nothing is written by default.")
    ask.add_argument("--evidence", choices=["pubmed", "local"], default="pubmed", help="Evidence mode; pubmed adds live paper search, local reproduces the requirement-2 workflow.")
    ask.add_argument("--pubmed-config", type=Path, default=Path("config/pubmed.json"))
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
    search = commands.add_parser("search-pubmed", help="Search PubMed without an LLM or model API key.")
    search.add_argument("query")
    search.add_argument("--max-results", type=int, default=5)
    search.add_argument("--sort", choices=["relevance", "pub_date"], default="relevance")
    search.add_argument("--pubmed-config", type=Path, default=Path("config/pubmed.json"))
    search.add_argument("--output", type=Path)
    retrieve = commands.add_parser("evaluate-retrieval", help="Run the unchanged internal questions with PubMed and save separate retrieval observations.")
    retrieve.add_argument("--config", type=Path, default=Path("config/models/gemini.json"))
    retrieve.add_argument("--pubmed-config", type=Path, default=Path("config/pubmed.json"))
    retrieve.add_argument("--questions", type=Path, default=Path("data/evaluation/oncology_nutrition_questions.json"))
    retrieve.add_argument("--rubric", type=Path, default=Path("data/evaluation/retrieval_rubric.json"))
    retrieve.add_argument("--run-dir", type=Path)
    retrieve.add_argument("--ids", nargs="+")
    retrieve.add_argument("--delay", type=float, default=4, help="Seconds between questions (0-30).")
    retrieval_summary = commands.add_parser("summarize-retrieval", help="Validate separately authored paper-relevance assessments.")
    retrieval_summary.add_argument("--run-dir", type=Path, required=True)
    retrieval_summary.add_argument("--assessments", type=Path, required=True)
    retrieval_summary.add_argument("--rubric", type=Path, default=Path("data/evaluation/retrieval_rubric.json"))
    compare = commands.add_parser("compare-models", help="Compare complete agents on identical fixed evidence; preserve usage and failures separately from medical ratings.")
    compare.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    compare.add_argument("--run-dir", type=Path, required=True)
    compare.add_argument("--ids", nargs="+", help="Optional question IDs for a separately recorded pilot.")
    compare.add_argument("--budget-usd", type=float, default=8.0, help="Local total paid-request allowance, including initial-spend (default USD 8).")
    compare.add_argument("--initial-spend-usd", type=float, default=0.0, help="Already incurred paid preflight/pilot cost counted against the allowance.")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    def local(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    try:
        if args.command == "compare-models":
            result = run_comparison(root, local(args.run_dir), args.profiles, args.ids, args.budget_usd, args.initial_spend_usd)
            print(json.dumps(result, indent=2))
            return 1 if any(model["completed_answers"] != model["attempted_cases"] for model in result["models"]) else 0
        if args.command == "search-pubmed":
            from dotenv import load_dotenv
            load_dotenv(root / ".env", override=False)
            result = PubMedClient(load_pubmed_config(local(args.pubmed_config))).search(args.query, args.max_results, args.sort)
            if args.output:
                write_json(local(args.output), result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "summarize-retrieval":
            result = summarize_retrieval_assessments(local(args.run_dir), local(args.assessments), local(args.rubric))
            print(json.dumps(result, indent=2))
            return 0
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
        if args.command == "evaluate-retrieval":
            if not 0 <= args.delay <= 30:
                raise ValueError("Evaluation delay must be between 0 and 30 seconds.")
            run_dir = local(args.run_dir) if args.run_dir else root / "data/evaluation/retrieval-runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            result = run_retrieval_evaluation(config, load_pubmed_config(local(args.pubmed_config)), root, local(args.questions), local(args.rubric), run_dir, args.ids, args.delay)
            print(json.dumps(result, indent=2))
            return 1 if result["statuses"].get("error") else 0
        backend, redactor, _ = create_backend(config, root)
        client = ModelClient(backend, config, redactor)
        agent = LiteratureCopilot(client, PubMedClient(load_pubmed_config(local(args.pubmed_config)))) if args.evidence == "pubmed" else MedicalCopilot(client)
        result = agent.run(args.question)
        if args.output:
            write_json(local(args.output), redactor.object(result))
        print(f"Agent status: {result['status']}\n")
        print(result.get("answer") or result.get("error", "No answer."))
        return 1 if result["status"] == "error" else 0
    except (ConfigurationError, ModelError, PubMedError, ValueError, OSError) as exc:
        # Application exceptions contain safe summaries; no provider traceback.
        if isinstance(exc, (ConfigurationError, ModelError, PubMedError)):
            print(str(exc), file=sys.stderr)
        else:
            print(f"Operation failed ({type(exc).__name__}); check input paths and arguments.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
