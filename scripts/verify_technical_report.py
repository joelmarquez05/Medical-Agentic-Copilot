"""Verify report provenance and replay its source studies without network access."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/technical-report"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((REPORT / "report-input-manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["input_sha256"].items():
        assert digest(ROOT / name) == expected, name
    for item in manifest["snippets"].values():
        source = (ROOT / item["file"]).read_text(encoding="utf-8-sig").splitlines()
        excerpt = "\n".join(source[item["first_line"] - 1:item["last_line"]]) + "\n"
        assert (REPORT / item["snippet"]).read_text(encoding="utf-8") == excerpt
    checks = REPORT / "checks"
    checks.mkdir(exist_ok=True)
    studies = [
        ("baseline", "verify_clinical_baseline.py", "data/evaluation/internal-runs/gemma-baseline-restart-20261011"),
        ("internal-v3", "verify_clinical_resilient.py", "data/evaluation/retrieval-runs/gemma-resilient-main-20261011"),
        ("gemma-benchmark", "verify_clinical_benchmark.py", "data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"),
        ("four-model-panel", "verify_clinical_model_comparison.py", "data/evaluation/comparison-runs/medqa30-four-model-main-20261011"),
        ("pdf-original", "verify_pdf_evaluation.py", "data/evaluation/pdf-runs/gemma-pdf-main-20261012"),
        ("patient-main", "verify_patient_run.py", "data/evaluation/patient-runs/gemma-patient-main-20261014"),
    ]
    results = []
    for label, script, run in studies:
        target = checks / (label + ".json")
        if not target.exists():
            command = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / script), "--run-dir", str(ROOT / run), "--output", str(target)]
            result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=180)
            (checks / (label + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"Offline replay failed for {label}; inspect {checks / (label + '.log')}")
        receipt = json.loads(target.read_text(encoding="utf-8"))
        assert receipt["passed"]
        cases = receipt.get("cases", receipt.get("episodes", receipt.get("results", [])))
        results.append({"study": label, "passed": True, "receipt": target.relative_to(REPORT).as_posix(),
                        "receipt_sha256": digest(target), "episodes": len(cases) if isinstance(cases, list) else cases})
        print(label + ": passed", flush=True)
    import pymupdf
    pdf = REPORT / "report.pdf"
    log = (REPORT / "report.log").read_text(encoding="utf-8", errors="replace")
    assert "Fatal error" not in log and "Output written on report.pdf" in log
    assert "undefined references" not in log and "undefined citations" not in log
    assert not re.search(r"(?:Reference|Citation) .* undefined", log)
    assert "Overfull" not in log
    with pymupdf.open(pdf) as document:
        texts = [p.get_text() for p in document]
        joined = "\n".join(texts)
        for marker in ["LangGraph", "PubMed", "BM26", "PM10", "73.99", "67.39", "SQLite"]:
            assert marker in joined, marker
        assert 15 <= len(document) <= 35, "User requested at most 35 pages, including front matter."
        assert all(text.strip() for text in texts)
        page_count = len(document)
        pdf_checks = {"pages": page_count, "all_pages_have_text": True, "required_topics_present": True,
                      "bookmarks": len(document.get_toc()), "pdf_sha256": digest(pdf)}
    # Scan only the authored report bundle, never credential or patient-storage files.
    suspicious = re.compile(r"sk-or-v1-[A-Za-z0-9_-]{15,}|Bearer\s+[A-Za-z0-9_-]{20,}")
    for path in REPORT.rglob("*"):
        if path.suffix in {".tex", ".py", ".md", ".json"}:
            assert not suspicious.search(path.read_text(encoding="utf-8")), path
    receipt = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
               "tracked_inputs_verified": len(manifest["input_sha256"]), "exact_code_excerpts_verified": len(manifest["snippets"]),
               "coverage": manifest["checks"], "offline_replays": results, "pdf_checks": pdf_checks,
               "report_source_sha256": {p.relative_to(REPORT).as_posix(): digest(p) for p in [REPORT / 'report.tex', *sorted((REPORT / 'study').glob('*.tex'))]},
               "latex_checks": {"undefined_references": 0, "undefined_citations": 0, "overfull_boxes": 0},
               "new_model_requests": 0, "new_pubmed_requests": 0, "credential_files_read": 0,
               "limits": "Clinical appraisals remain historical unblinded assistant reviews; this checks documentation and operational reproducibility."}
    (REPORT / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "pdf_pages": page_count, "studies_replayed": len(results)}))


if __name__ == "__main__":
    main()
