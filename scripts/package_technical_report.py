"""Package only the study guide's active LaTeX inputs and check standalone compilation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/technical-report"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verification = json.loads((REPORT / "verification.json").read_text(encoding="utf-8"))
    assert verification["passed"] and verification["pdf_checks"]["pages"] <= 35
    assert digest(REPORT / "report.pdf") == verification["pdf_checks"]["pdf_sha256"]
    chapters = sorted((REPORT / "study").glob("*.tex"))
    active_text = "\n".join(p.read_text(encoding="utf-8") for p in chapters)
    files = {REPORT / "report.tex", REPORT / "README.md", REPORT / "report-input-manifest.json", REPORT / "verification.json", *chapters}
    files.update(REPORT / item for item in re.findall(r"\\tableinput\{([^}]+)\}", active_text))
    files.update(REPORT / "snippets" / (item + ".py") for item in re.findall(r"\\snippet\{([^}]+)\}", active_text))
    for path in files:
        assert path.is_file() and path.resolve().is_relative_to(REPORT.resolve())
    archive_path = REPORT / "latex-source.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(REPORT).as_posix())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    check_dir = (ROOT / "tmp" / ("study-guide-package-check-" + stamp)).resolve()
    assert check_dir.is_relative_to((ROOT / "tmp").resolve()) and not check_dir.exists()
    check_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(check_dir)
    for number in (1, 2):
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                                cwd=check_dir, capture_output=True, timeout=90)
        (check_dir / ("compile-" + str(number) + ".txt")).write_bytes(result.stdout + result.stderr)
        assert result.returncode == 0, "Standalone LaTeX compilation failed."
    import pymupdf
    with pymupdf.open(check_dir / "report.pdf") as document:
        assert len(document) == verification["pdf_checks"]["pages"] <= 35
    receipt = {"passed": True, "archive": "latex-source.zip", "archive_sha256": digest(archive_path),
               "file_count": len(files), "active_code_excerpts": len(re.findall(r"\\snippet\{([^}]+)\}", active_text)),
               "standalone_compilation_passes": 2, "pdf_pages": verification["pdf_checks"]["pages"],
               "package_file_sha256": {p.relative_to(REPORT).as_posix(): digest(p) for p in sorted(files)},
               "new_model_requests": 0, "new_pubmed_requests": 0, "credential_files_read": 0}
    (REPORT / "package-check.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "source_files": len(files), "code_excerpts": receipt["active_code_excerpts"], "pages": receipt["pdf_pages"]}))


if __name__ == "__main__":
    main()
