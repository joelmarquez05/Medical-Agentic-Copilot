"""Build reviewable LaTeX assets from preserved code and evaluation artifacts.

No credentials, inference, network calls, or changes to evaluation records.
Run from any directory, then compile docs/technical-report/report.tex twice.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/technical-report"
BASE = ROOT / "data/evaluation/internal-runs/gemma-baseline-restart-20261011"
INTERNAL = ROOT / "data/evaluation/retrieval-runs/gemma-resilient-main-20261011"
GEMMA = ROOT / "data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"
PANEL = ROOT / "data/evaluation/comparison-runs/medqa30-four-model-main-20261011"
PATIENT = ROOT / "data/evaluation/patient-runs/gemma-patient-main-20261014"
MODELS = [("gemma", "Gemma", GEMMA), ("glm-5.3", "GLM", PANEL / "glm-5.3"),
          ("kimi-k3", "Kimi", PANEL / "kimi-k3"), ("mimo-v2.6-pro", "MiMo", PANEL / "mimo-v2.6-pro"),
          ("qwen3.8-27b", "Qwen", PANEL / "qwen3.8-27b")]
INPUTS: dict[str, str] = {}


def track(path: Path) -> bytes:
    data = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
    return data


def read(path: Path):
    return json.loads(track(path).decode("utf-8-sig"))


def tex(value) -> str:
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}", "\u2264": r"$\leq$", "\u2265": r"$\geq$",
               "\u2192": r"$\rightarrow$", "\u03b2": r"$\beta$", "\u03b1": r"$\alpha$",
               "\u00b1": r"$\pm$", "\u03bc": r"$\mu$", "\u00d7": r"$\times$",
               "\u2103": r"$^\circ$C", "\u2109": r"$^\circ$F", "\u00b3": r"$^3$"}
    return "".join(mapping.get(char, char) for char in str(value)).replace("\n", " ")


def write(name, content):
    target = OUT / "generated" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content + "\n", encoding="utf-8")


def snippet(filename, symbol, name, *, max_lines=None):
    path = ROOT / "src/medical_copilot" / filename
    source = track(path).decode("utf-8-sig")
    if symbol is None:
        first, last = 1, len(source.splitlines())
    else:
        parts = symbol.split(".")
        candidates = ast.parse(source).body
        for part in parts:
            node = next(n for n in candidates if getattr(n, "name", None) == part)
            candidates = getattr(node, "body", [])
        first, last = node.lineno, node.end_lineno
    if max_lines:
        last = min(last, first + max_lines - 1)
    lines = source.splitlines()[first - 1:last]
    (OUT / "snippets").mkdir(parents=True, exist_ok=True)
    (OUT / "snippets" / (name + ".py")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"file": path.relative_to(ROOT).as_posix(), "symbol": symbol, "first_line": first,
            "last_line": last, "snippet": "snippets/" + name + ".py"}


def main():
    specs = [
        ("clinical_graph.py", "State", "state"),
        ("clinical_graph.py", "FinalAction", "final-action"),
        ("clinical_graph.py", "ToolDefinition", "tool-definition"),
        ("clinical_graph.py", "ClinicalAgent._execute_tool", "execute-tool"),
        ("clinical_graph.py", "ClinicalAgent._validate_output", "validate-output"),
        ("clinical_resilient.py", "ResilientClinicalAgent.__init__", "graph"),
        ("clinical_resilient.py", "ResilientClinicalAgent._request_phase", "request-phase"),
        ("clinical_resilient.py", "ResilientClinicalAgent._research", "research"),
        ("clinical_resilient.py", "final_schema", "final-schema"),
        ("clinical_resilient.py", "normalize_action", "normalize-action"),
        ("models.py", "create_backend", "model-adapter"),
        ("clinical_literature.py", "evidence_view", "pubmed-view"),
        ("clinical_literature.py", "ClinicalLiteratureTools.read", "pubmed-read"),
        ("clinical_literature.py", "ClinicalLiteratureTools.search", "pubmed-search"),
        ("pubmed.py", "PubMedClient.search", "pubmed-http-search"),
        ("clinical_evaluation.py", "score_record", "scoring"),
        ("clinical_model_comparison.py", "episode_order", "model-order"),
        ("pdf_documents.py", "normalize_text", "pdf-normalize"),
        ("pdf_documents.py", "positioned_blocks", "pdf-positioned"),
        ("pdf_documents.py", "ordered_blocks", "pdf-order"),
        ("pdf_documents.py", "split_page", "pdf-split"),
        ("pdf_documents.py", "PDFLibrary.search", "pdf-rank"),
        ("pdf_documents.py", "PDFTools.read", "pdf-read"),
        ("patient_memory.py", "FactInput", "fact-input"),
        ("patient_memory.py", "PatientMemory.current_facts", "current-facts"),
        ("patient_memory.py", "PatientMemory.fact_history", "fact-history"),
        ("patient_memory.py", "PatientMemory.context", "patient-context"),
        ("patient_memory.py", "PatientMemory.search_history", "history-search"),
        ("patient_memory.py", "PatientMemory.document", "owned-document"),
        ("patient_memory.py", "PatientMemory.save_interaction", "save-interaction"),
        ("patient_tools.py", "PatientTools.read", "restore-evidence"),
        ("patient_agent.py", "PatientAgent._validate_output", "patient-source-gate"),
        ("patient_agent.py", "PatientAgent._save_memory", "save-node"),
    ]
    locations = {name: snippet(filename, symbol, name) for filename, symbol, name in specs}
    rows = []
    for name, loc in locations.items():
        rows.append(tex(name) + " & " + r"\path{" + loc["file"] + "}" + " & " + f'{loc["first_line"]}--{loc["last_line"]}' + r" \\")
    write("snippet-map.tex", "\n".join(rows))

    summaries = {slug: read(run / "summary.json") for slug, _, run in MODELS}
    assessments = read(PANEL / "assessment-summary.json")["models"]
    assessments["gemma"] = read(GEMMA / "assessment-summary-v2.json")
    rows, dimensions = [], []
    for slug, label, _ in MODELS:
        summary, assessment = summaries[slug], assessments[slug]
        rows.append(" & ".join([label, f'{summary["correct"]}/30', str(summary["episodes_with_search"]),
            str(summary["tool_names"].get("search_pubmed", 0)), str(summary["tool_names"].get("read_pubmed_abstract", 0)),
            str(summary["episodes_with_citations"]), str(summary["provider_requests"]), f'{summary["reported_or_estimated_cost_usd"]:.8f}']) + r" \\")
        for dim, title in [("case_reasoning", "Case reasoning"), ("reference_concordance", "Reference concordance"),
                           ("calibration_and_provenance", "Calibration / provenance")]:
            counts = assessment["answer_dimension_counts"][dim]
            dimensions.append(" & ".join([label, title] + [str(counts.get(v, 0)) for v in ["consistent", "minor_limitation", "material_issue", "not_evaluable"]]) + r" \\")
    write("panel-table.tex", "\n".join(rows))
    write("dimension-table.tex", "\n".join(dimensions))
    for name, key, categories in [("citations", "citation_support", ["supported", "partially_supported", "unsupported", "not_assessable"]),
                                   ("relevance", "paper_relevance", ["directly_relevant", "partially_relevant", "irrelevant", "not_assessable"])]:
        rows = []
        for slug, label, _ in MODELS:
            counts = assessments[slug][key]
            values = [counts.get(c, 0) for c in categories]
            rows.append(" & ".join([label] + [str(v) for v in values] + [str(sum(values))]) + r" \\")
        write(name + "-table.tex", "\n".join(rows))

    base_reviews = None
    for candidate in BASE.glob("*.json"):
        if "qualitative" in candidate.name or "assessment" in candidate.name:
            value = read(candidate)
            if isinstance(value, dict) and ("reviews" in value or "assessments" in value):
                base_reviews = value.get("reviews", value.get("assessments"))
    current_reviews = {v["question_id"]: v for v in read(INTERNAL / "answer-assessments.json")["reviews"]}
    keys = {k["question_id"]: k for k in read(BASE / "answer-key.json")["keys"]}
    appendix = []
    for path in sorted((BASE / "answers").glob("*.json")):
        baseline = read(path)
        qid = baseline["question"]["question_id"]
        current = read(INTERNAL / "answers" / (qid + ".json"))
        appendix.append(r"\subsection{" + qid + r": internal paired case}")
        appendix.append(r"\textbf{Question context:} " + tex(baseline["question"]["question"]) + "\n")
        appendix.append(r"\textbf{Frozen key:} " + tex(keys[qid]["answer_idx"]) + ". " + tex(keys[qid]["key_type"]) + ".\n")
        for title, row in [("Tools-disabled baseline", baseline), ("PubMed-enabled v3", current)]:
            response = row["response"]
            appendix.append(r"\textbf{" + title + "}: choice " + tex(response.get("selected_option")) + "; " + tex(response["answer"]) + "\n\n" + tex(response["justification"]) + "\n")
        review = current_reviews[qid]
        appendix.append(r"\textbf{Separate v3 review:} " + tex(review["brief_observed_notes"]) + "\n")
        labels = {"case_reasoning": "Reasoning", "reference_concordance": "concordance", "calibration_and_provenance": "calibration/provenance"}
        appendix.append(r"{\small\textbf{Ratings:} " + "; ".join(labels[k] + ": " + tex(v) for k, v in review["dimension_ratings"].items()) + r".}\par" + "\n")
    write("internal-cases.tex", "\n".join(appendix))

    key_map = {k["question_id"]: k for k in read(GEMMA / "answer-key.json")["keys"]}
    all_reviews = {}
    for slug, _, run in MODELS:
        name = "answer-assessments-v2.json" if slug == "gemma" else "answer-assessments.json"
        all_reviews[slug] = {r["question_id"]: r for r in read(run / name)["reviews"]}
    appendix = []
    for qid in key_map:
        reference = read(GEMMA / "answers" / (qid + ".json"))
        appendix.append(r"\subsection{" + qid + ": key " + tex(key_map[qid]["answer_idx"]) + "}")
        appendix.append(r"\textbf{Case context:} " + tex(reference["question"]["question"]) + "\n")
        appendix.append(r"\textbf{Released answer text:} " + tex(key_map[qid]["answer"]) + ".\n")
        for slug, label, run in MODELS:
            row = read(run / "answers" / (qid + ".json"))
            review = all_reviews[slug][qid]
            choice = row["response"]["selected_option"]
            appendix.append(r"\paragraph{" + label + ": " + tex(choice) + (" (key match)" if choice == key_map[qid]["answer_idx"] else " (key mismatch)") + "}")
            appendix.append(tex(review["brief_observed_notes"]) + "\n")
            ratings = review["dimension_ratings"]
            appendix.append(r"{\small Reasoning: " + tex(ratings["case_reasoning"]) + "; concordance: " + tex(ratings["reference_concordance"]) + "; calibration/provenance: " + tex(ratings["calibration_and_provenance"]) + r".}\par")
    write("benchmark-cases.tex", "\n".join(appendix))
    scores = read(PATIENT / "scores.json")
    rows = []
    for case in scores["cases"]:
        actual = "; ".join(f'{f["field"]}=' + ("(field absent)" if not f["present"] else "null" if f["actual"] is None else str(f["actual"])) for f in case["fields"])
        rows.append(" & ".join([case["question_id"], tex(case["patient_id"]), tex(actual),
            f'{case["matched_fields"]}/{case["total_fields"]}']) + r" \\")
    write("patient-table.tex", "\n".join(rows))
    receipt = {"date": "2026-10-15", "purpose": "Technical report asset generation only", "new_inference_requests": 0,
               "new_pubmed_requests": 0, "credential_files_read": 0, "snippets": locations, "input_sha256": INPUTS,
               "checks": {"models": len(MODELS), "benchmark_reviews": sum(len(v) for v in all_reviews.values()),
                          "internal_paired_cases": len(current_reviews), "patient_protocol_cases": len(scores["cases"])}}
    (OUT / "report-input-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generated": str(OUT), "tracked_inputs": len(INPUTS), "snippets": len(locations), **receipt["checks"]}))


if __name__ == "__main__":
    main()
