import copy
import json
from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from medical_copilot.pdf_documents import (
    PDFLibrary, PDFTools, ReadPDFPageArgs, SearchPDFArgs, ingest_pdf, normalize_text, ordered_blocks, split_page,
)
from medical_copilot.pdf_agent import CONDITIONS, PDFClinicalAgent, PDF_SYSTEM
from medical_copilot.clinical_resilient import ResilientLimits


@pytest.fixture
def document(tmp_path):
    path = tmp_path / "source.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        # Deliberately insert the right column before the left column.
        page.insert_text((320, 100), "Right continuation: report nausea.")
        page.insert_text((40, 100), "Left introduction: assess intake.")
        page.insert_text((40, 150), "Malnutrition requires assessment.")
        second = pdf.new_page()
        second.insert_text((40, 100), "Second page: chronic constipation.")
        pdf.save(path)
    return ingest_pdf(path, tmp_path / "ingested", title="Fixture clinical document")


def final(citations=()):
    return {"type": "final", "answer": "Assessment is needed.", "justification": "The provided question concerns assessment.",
            "selected_option": None, "citations": list(citations), "limitations": []}


class Scripted:
    def __init__(self, actions):
        self.actions, self.requests = iter(actions), []

    def reply(self, request):
        self.requests.append(copy.deepcopy(request))
        return next(self.actions)


def test_actual_pdf_columns_and_all_text_preserved(document):
    text = document["pages"][0]["text"]
    assert text.index("Left introduction") < text.index("Right continuation")
    assert "Malnutrition" in text
    for page in document["pages"]:
        assert "".join(document["views"][sid]["text"] for sid in page["chunk_ids"]) == page["text"]


def test_full_width_band_separates_column_order():
    values = [(0, [30, 60, 250, 80]), (1, [320, 70, 570, 90]), (2, [30, 200, 570, 215]),
              (3, [320, 240, 570, 270]), (4, [30, 260, 250, 290])]
    blocks = [{"block_number": i, "bbox": b} for i, b in values]
    assert [b["block_number"] for b in ordered_blocks(blocks, 600)] == [0, 1, 2, 4, 3]


@pytest.mark.parametrize("text", ["short", "word\n" * 2400, "x" * 12001])
def test_chunk_offsets_cover_entire_page_without_overlap(text):
    pieces = list(split_page(text))
    assert "".join(p[2] for p in pieces) == text
    assert all(len(p[2]) <= 4000 for p in pieces)
    assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:]))


def test_normalization_does_not_reinterpret_numeric_glyphs():
    assert normalize_text("deﬁciency\nrecom-\nmend 50e75% and 1e9") == "deficiency\nrecommend 50e75% and 1e9"


def test_empty_text_pdf_is_explicitly_unsupported(tmp_path):
    source = tmp_path / "blank.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(source)
    with pytest.raises(ValueError, match="OCR"):
        ingest_pdf(source, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_encrypted_pdf_is_not_silently_parsed(tmp_path):
    source = tmp_path / "encrypted.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(source, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="fixture-owner", user_pw="fixture-user")
    with pytest.raises(ValueError, match="Encrypted"):
        ingest_pdf(source, tmp_path / "new")


def test_existing_ingestion_refused_before_pdf_open(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        ingest_pdf(tmp_path / "missing.pdf", output)


def test_document_search_and_page_read_use_identical_immutable_views(document):
    library = PDFLibrary([document])
    tool = PDFTools(library)
    identifier = document["document_id"]
    result = tool.search(SearchPDFArgs(document_id=identifier, query="chronic constipation"))
    page = tool.read(ReadPDFPageArgs(document_id=identifier, page=2))
    assert result.sources[0].model_dump() == page.sources[0].model_dump()
    assert result.sources[0].id in library.all_views()
    assert json.loads(result.content)["results"][0]["pdf_page"] == 2


def test_unknown_document_id_cannot_open_arbitrary_file(document):
    with pytest.raises(ValueError, match="catalog"):
        PDFTools(PDFLibrary([document])).read(ReadPDFPageArgs(document_id="../../another-patient.pdf", page=1))


@pytest.mark.parametrize("page", [0, 251])
def test_page_argument_bounds(page):
    with pytest.raises(ValidationError):
        ReadPDFPageArgs(document_id="Dfixture", page=page)


def test_page_outside_actual_document_rejected(document):
    with pytest.raises(ValueError, match="outside"):
        PDFTools(PDFLibrary([document])).read(ReadPDFPageArgs(document_id=document["document_id"], page=3))


def test_empty_lexical_search_has_explicit_coverage_without_fake_source(document):
    result = PDFTools(PDFLibrary([document])).search(SearchPDFArgs(document_id=document["document_id"], query="pembrolizumab"))
    assert result.sources == []
    assert json.loads(result.content)["no_matching_passages"] and json.loads(result.content)["searchable_pages"] == 2


def test_corrupted_chunk_coverage_rejected(document):
    altered = copy.deepcopy(document)
    altered["views"][next(iter(altered["views"]))]["text"] += "invented"
    with pytest.raises(AssertionError):
        PDFLibrary([altered])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_three_conditions_initialize_with_correct_content_and_tools(document, condition):
    library = PDFLibrary([document])
    model = Scripted([final()])
    tool = PDFTools(library) if condition == "C-pdf-tools" else None
    result = PDFClinicalAgent(model, library, condition, pdf_tools=tool).run({"question_id": "fresh", "question": "What should be assessed?"})
    assert result["status"] == "completed" and result["output_origin"] == "model"
    request = model.requests[0]
    assert request["history"] == []
    assert bool(request["tools"]) == (condition == "C-pdf-tools")
    assert request["evidence_registry"] == (library.all_views() if condition == "B-full-context" else {})
    assert len((PDF_SYSTEM + json.dumps(request, ensure_ascii=False)).encode()) == result["trace"][1]["input_bytes"]


def test_pdf_search_results_reach_next_model_turn(document):
    library = PDFLibrary([document])
    sid = library.search(document["document_id"], "constipation")[0][0]
    model = Scripted([{"type": "tool", "name": "search_pdf", "arguments": {"document_id": document["document_id"], "query": "constipation"}}, final([sid])])
    result = PDFClinicalAgent(model, library, pdf_tools=PDFTools(library)).run({"question_id": "fresh", "question": "What does the document say about constipation?"})
    assert result["response"]["citations"] == [sid]
    assert sid in model.requests[1]["evidence_registry"]
    assert result["tool_actions"] == 1


def test_new_episode_cannot_see_prior_tool_registry(document):
    library = PDFLibrary([document])
    first = PDFClinicalAgent(Scripted([{"type": "tool", "name": "read_pdf_page", "arguments": {"document_id": document["document_id"], "page": 1}}, final()]), library, pdf_tools=PDFTools(library))
    first.run({"question_id": "one", "question": "Read the document."})
    model = Scripted([final()])
    second = PDFClinicalAgent(model, library, pdf_tools=PDFTools(library))
    second.run({"question_id": "two", "question": "A different question."})
    assert model.requests[0]["evidence_registry"] == {} and model.requests[0]["history"] == []


def test_evaluator_fields_rejected_without_model_call(document):
    model = Scripted([])
    result = PDFClinicalAgent(model, PDFLibrary([document]), "A-no-document").run({"question_id": "one", "question": "Question", "answer_idx": "A"})
    assert result["status"] == "invalid_input" and model.requests == []


def test_pdf_specific_input_guard_stops_before_call(document):
    altered = copy.deepcopy(document)
    page = altered["pages"][0]
    # Build a valid larger source with multiple chunks to cross the small test guard.
    sid = page["chunk_ids"][0]
    altered["views"][sid]["text"] = "large content " * 550
    page["text"] = altered["views"][sid]["text"]
    import hashlib
    page["text_sha256"] = hashlib.sha256(page["text"].encode()).hexdigest()
    model = Scripted([])
    result = PDFClinicalAgent(model, PDFLibrary([altered]), "B-full-context", limits=ResilientLimits(max_input_bytes=4096)).run({"question_id": "one", "question": "Question"})
    assert result["status"] == "operationally_unanswered" and model.requests == []
