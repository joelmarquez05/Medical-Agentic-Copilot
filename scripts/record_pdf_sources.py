"""Archive implementation-source metadata and the exact pre-generation read scope."""
import json
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import read_json

ROOT = Path(__file__).resolve().parents[1]
prior = read_json(ROOT / "data/research/requirement-6-pdf-selection-sources-20261012.json")["sources"]
clinical = dict(prior[0])
clinical["read_level"] = "Original source inspected before model generation: rendered PDF pages 2, 4, 5, 6, 7, 12 and 13 for the five positive fact keys; all sixteen text-bearing pages checked for absence of pembrolizumab. Page 3 layout inspected previously. Exact chunks/geometry retained. No complete guideline evidence appraisal, no OCR/image-exhaustive absence proof and no latest-guidance claim."
clinical["supports"] = "Frozen six authored document-fact option keys, page locators and retrospective explanation/citation appraisal, independently of model answers. Not released exam labels."
sources = [clinical, prior[1]]
for identifier, title, url, scope, support in [
    ("R6-TECH02", "Text - PyMuPDF documentation", "https://pymupdf.readthedocs.io/en/latest/recipes-text.html", "Text extraction, natural reading order and table-content sections.", "Creator-order text may differ from reading order; position-aware reordering is needed and table interpretation is a separate problem."),
    ("R6-TECH03", "Page - PyMuPDF documentation", "https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_text", "get_text API formats, extraction flags, sort and clipping.", "Dictionary extraction and controlling image inclusion; coordinate sort alone is not a universal semantic reading-order solution."),
    ("R6-TECH04", "TextPage - PyMuPDF documentation", "https://pymupdf.readthedocs.io/en/latest/textpage.html#structure-of-dictionary-outputs", "extractDICT and dictionary-output block/line/span structure, including bbox and span text.", "Actual positioned-line/span geometry used to split narrow column lines merged into one block and retain raw provenance."),
    ("R6-TECH05", "Appendix 1: Details on Text Extraction - PyMuPDF documentation", "https://pymupdf.readthedocs.io/en/latest/app1.html", "Referenced API page; substantive retrieval failed with HTTP 503. Page/TextPage primary API consulted instead.", "Unverified access; no implementation-specific conclusion depends on unseen appendix content.")]:
    sources.append({"id": identifier, "title": title, "authors_or_organization": "Artifex Software and PyMuPDF contributors",
        "year": "Living documentation, accessed 2026", "url": url, "preprint": False, "accessed": "2026-10-12",
        "read_level": scope, "supports": support, "verified_access": identifier != "R6-TECH05"})
save_trace(ROOT / "data/research/requirement-6-implementation-sources-20261012.json", {"sources": sources,
    "separation": "Researcher PDF/reference verification is not supplied as evaluator labels or gold excerpts to any answering model. All B views are the complete extraction; C views require actual tool delivery."})
print(json.dumps({"saved_sources": len(sources), "hosted_model_requests": 0}))
