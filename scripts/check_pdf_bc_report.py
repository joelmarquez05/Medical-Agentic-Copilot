"""Check the selected historical PDF comparison without network or credentials."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

from medical_copilot.config import load_config


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/pdf-runs/gemma-pdf-main-20261012"
CONDITIONS = ("B-full-context", "C-pdf-tools")
DEFAULT_RECEIPT = ROOT / "data/checks/requirement-6-selected-bc-summary-20261012.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = read(RUN / "manifest.json")
    original_summary = read(RUN / "summary.json")["conditions"]
    reviews = read(RUN / "manual-reviews.json")["reviews"]
    qualitative = read(RUN / "qualitative-summary.json")["conditions"]
    replay_path = ROOT / "data/checks/requirement-6-main-replay-20261012.json"
    replay = read(replay_path)
    assert replay["passed"] and replay["new_provider_requests"] == replay["credentials_read"] == 0
    replayed = {(e["condition"], e["question_id"]): e for e in replay["episodes"]}
    review_map = {(r["condition"], r["question_id"]): r for r in reviews}
    keys = {k["question_id"]: k["answer_idx"] for k in read(RUN / "answer-key.json")["keys"]}
    files = [("document.json", "document_sha256"), ("protocol.md", "protocol_sha256"),
             ("policy.json", "policy_sha256"), ("qualitative-rubric.json", "rubric_sha256"),
             ("questions.json", "questions_sha256"), ("answer-key.json", "answer_key_sha256"),
             ("selection-manifest.json", "selection_manifest_sha256")]
    for filename, field in files:
        assert digest(RUN / filename) == manifest[field], filename
    profile = ROOT / "config/models/openrouter-gemma-4-reference.json"
    assert digest(profile) == manifest["config_sha256"]
    # The run copy is a public projection, not a byte-identical configuration file.
    assert read(RUN / "config.json") == load_config(profile).public_dict()
    for filename, expected in manifest["source_hashes"].items():
        assert digest(ROOT / "src/medical_copilot" / filename) == expected
        assert digest(RUN / "source-snapshot" / filename) == expected
    assert digest(ROOT / "data/documents/reference/espen-cancer-2021.pdf") == manifest["pdf_sha256"]
    previous_artifacts = read(ROOT / "data/checks/requirement-6-final-artifacts-20261012.json")
    assert digest(RUN / "manual-reviews.json") == previous_artifacts["manual_reviews_sha256"]
    assert digest(RUN / "results.md") == previous_artifacts["report_sha256"]

    selected, results = [], {}
    for condition in CONDITIONS:
        rows, audit, events, grades, grounding = [], [], [], [], []
        for qid in manifest["question_ids"]:
            path = RUN / condition / "answers" / (qid + ".json")
            row, review = read(path), review_map[(condition, qid)]
            assert digest(path) == review["answer_sha256"] == replayed[(condition, qid)]["answer_sha256"]
            assert row["condition"] == condition and row["question"]["question_id"] == qid
            assert row["status"] in ("completed", "completed_with_citation_issues")
            assert row["response"]["type"] == "final"
            assert len(row["model_calls"]) == replayed[(condition, qid)]["model_responses"]
            rows.append(row)
            audit.extend(row["response_audit"])
            events.extend(row["document_tools"]["events"])
            grades.append(str(review["explanation_grade"]))
            grounding.append(str(review["document_grounding_grade"]))
            selected.append({"condition": condition, "question_id": qid, "answer_sha256": digest(path),
                             "selected_option": row["response"]["selected_option"], "key": keys[qid]})
        assert all(a["request_sent"] and a["status"] == "ok" for a in audit)
        assert audit == read(RUN / condition / "usage-audit.json")["records"]
        calculated = {
            "recorded": len(rows), "valid_completed": len(rows),
            "correct": sum(r["response"]["selected_option"] == keys[r["question"]["question_id"]] for r in rows),
            "provider_requests": len(audit),
            "input_tokens": sum(a["usage"]["input_tokens"] for a in audit),
            "output_tokens": sum(a["usage"]["output_tokens"] for a in audit),
            "reasoning_tokens": sum(a["provider_usage"].get("reasoning_tokens", 0) for a in audit),
            "cost_usd": float(sum(Decimal(str(a["cost_usd"])) for a in audit)),
            "elapsed_provider_seconds": sum(a["elapsed_seconds"] for a in audit),
            "successful_pdf_tool_calls": sum(e["status"] == "ok" for e in events),
        }
        for metric, value in calculated.items():
            assert math.isclose(value, original_summary[condition][metric], rel_tol=1e-10, abs_tol=1e-9), (condition, metric)
        assert dict(Counter(grades)) == qualitative[condition]["explanation_grades"]
        assert dict(Counter(grounding)) == qualitative[condition]["grounding_grades"]
        results[condition] = {**calculated,
                              "successful_operations_by_name": dict(Counter(e["name"] for e in events if e["status"] == "ok")),
                              "qualitative_appraisal": qualitative[condition]}

    b, c = (results[k] for k in CONDITIONS)
    reductions = {k: (1 - c[k] / b[k]) * 100 for k in ("input_tokens", "cost_usd", "elapsed_provider_seconds")}
    source_files = [RUN / n for n in ("manifest.json", "summary.json", "qualitative-summary.json", "manual-reviews.json",
                                     "qualitative-rubric.json", "answer-key.json", "questions.json", "document.json", "config.json",
                                     "protocol.md", "policy.json", "selection-manifest.json", "results.md")]
    source_files += [RUN / r["condition"] / "answers" / (r["question_id"] + ".json") for r in selected]
    source_files += [replay_path, profile]
    docs = [ROOT / n for n in ("README.md", "handoff.md", "docs/implementation/requirement-6-pdf.md",
                               "docs/implementation/requirement-6-pdf-efficiency.md",
                               "data/evaluation/pdf-runs/gemma-article-efficiency-main-20261012/ARCHIVE.md")]
    checked = 0
    for doc in docs:
        for link in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", doc.read_text(encoding="utf-8-sig")):
            if link.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = (doc.parent / link.split("#")[0].strip("<>")).resolve()
            assert target == DEFAULT_RECEIPT or target.exists(), (doc, link)
            checked += 1
    report = docs[3].read_text(encoding="utf-8")
    for exact in ("220,686", "57,410", "0.02203620", "0.00718602", "73.99%", "67.39%", "0.02922222", "5 / 1 / 0", "6 / 0 / 0"):
        assert exact in report, exact
    return {
        "passed": True, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_run": manifest["run_id"],
        "selection": "All six cases per B/C condition; retrospective user-selected presentation, not new generation.",
        "conditions": results, "relative_reduction_percent": reductions,
        "selected_provider_requests": b["provider_requests"] + c["provider_requests"],
        "selected_cost_usd": round(b["cost_usd"] + c["cost_usd"], 8), "selected_records": selected,
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files},
        "original_frozen_inputs_and_source_code_match_manifest": True,
        "selected_answers_match_prior_replay_and_unchanged_reviews": True,
        "all_selected_metric_totals_recomputed_from_saved_response_audits": True,
        "checked_local_document_links": checked,
        "latency_scope": "Sum of provider waits, not end-to-end latency or a controlled speedup.",
        "quality_scope": "Frozen authored option keys plus unblinded assistant review; no clinical equivalence conclusion.",
        "new_provider_requests": 0, "credentials_read": 0, "network_requests": 0,
        "later_article_detail_cohort": "Archived separately with all outcomes; not pooled into these savings.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional fresh verification receipt; otherwise print only")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise SystemExit("Preserve earlier receipts; choose a fresh output filename.")
    result = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "selected_provider_requests", "selected_cost_usd",
                                            "relative_reduction_percent", "checked_local_document_links",
                                            "new_provider_requests", "credentials_read")}, indent=2))
