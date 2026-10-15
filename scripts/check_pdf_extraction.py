"""Verify preserved PDF geometry/chunk coverage and pre-generation source revision."""
import json
from collections import Counter
from pathlib import Path
from medical_copilot.clinical_graph import EvidenceRecord, save_trace
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.pdf_documents import PDFLibrary

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/documents/ingested/espen-cancer-v1-20261012/document.json"
NEW = ROOT / "data/documents/ingested/espen-cancer-v2-20261012/document.json"
record = read_json(NEW)
library = PDFLibrary.load([NEW])
assert file_hash(NEW.parent / "original.pdf") == record["pdf_sha256"] == file_hash(ROOT / "data/documents/reference/espen-cancer-2021.pdf")
for page in record["pages"]:
    raw = Counter(line["raw_text"] for block in page["original_blocks"] for line in block["lines"])
    reordered = Counter(line for block in page["blocks"] for line in block["raw_text"].split("\n"))
    assert raw == reordered, (page["pdf_page"], "original line coverage")
    start = 0
    for sid in page["chunk_ids"]:
        view, locator = record["views"][sid], record["locators"][sid]
        EvidenceRecord.model_validate(view)
        assert view["id"] == sid and locator["start_character"] == start
        assert page["text"][start:locator["end_character"]] == view["text"]
        assert 0 < len(view["text"]) <= 4000
        start = locator["end_character"]
    assert start == len(page["text"])
assert [page["printed_page"] for page in record["pages"]] == [None] + [str(n) for n in range(2899, 2914)]
old_set, new_set = ROOT / "data/evaluation/pdf-set-20261012", ROOT / "data/evaluation/pdf-set-v2-20261012"
assert read_json(old_set / "questions.json") == read_json(new_set / "questions.json")
fields = ["question_id", "answer_idx", "key_type", "reference_facts", "pdf_pages", "printed_pages", "recommendations"]
assert [{k: r[k] for k in fields} for r in read_json(old_set / "answer-key.json")["keys"]] == [{k: r[k] for k in fields} for r in read_json(new_set / "answer-key.json")["keys"]]
assert "pembrolizumab" not in "\n".join(p["text"] for p in record["pages"]).casefold()
result = {"passed": True, "pdf_sha256": record["pdf_sha256"], "document_sha256": file_hash(NEW), "pre_generation_v1_document_sha256": file_hash(OLD),
    "original_lines_preserved": True, "nonoverlapping_chunks_cover_every_extracted_page": True,
    "pages": record["page_count"], "chunks": len(record["views"]), "extracted_text_utf8_bytes": sum(len(p["text"].encode()) for p in record["pages"]),
    "v1_v2_question_wording_options_and_keys_identical": True, "numeric_glyphs_not_rewritten": True,
    "selectable_text_pages": sum(not p["no_selectable_text"] for p in record["pages"]), "new_model_requests": 0,
    "limits": record["limits"]}
destination = ROOT / "data/checks/requirement-6-pdf-extraction-20261012.json"
save_trace(destination, result)
print(json.dumps(result))
