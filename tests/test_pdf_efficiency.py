"""Regressions for document-only answering and exact article-field evidence checks."""
import copy
import json
from pathlib import Path
import pytest
from medical_copilot.evaluation import read_json
from medical_copilot.pdf_documents import PDFLibrary, PDFTools
from medical_copilot.pdf_efficiency_agent import ArticlePDFAgent, CONDITIONS
from medical_copilot.pdf_efficiency_evaluation import score_fields, order, validate_set

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def library():
    return PDFLibrary.load([ROOT / "data/documents/ingested/espen-cancer-v2-20261012/document.json"])

class Scripted:
    def __init__(self, actions):
        self.actions, self.requests = iter(actions), []
    def reply(self, request):
        self.requests.append(copy.deepcopy(request))
        return next(self.actions)

def final(ids=(), answer='{"patients_count": 472}'):
    return {"type": "final", "answer": answer, "justification": "Fixture source check, not a clinical assessment.", "selected_option": None, "citations": list(ids), "limitations": []}

def test_no_document_condition_removed_from_active_agent(library):
    assert CONDITIONS == ("B-full-context", "C-pdf-tools")
    with pytest.raises(ValueError, match="Only"):
        ArticlePDFAgent(Scripted([]), library, "A-no-document")

def test_direct_answer_without_passages_returns_to_actual_research(library):
    record = library.document(library.catalog()[0]["document_id"])
    sid = record["pages"][6]["chunk_ids"][0]
    tool = {"type": "tool", "name": "search_pdf", "arguments": {"document_id": record["document_id"], "query": "HMB", "max_results": 1}}
    model = Scripted([final([sid]), tool, final([sid])])
    row = ArticlePDFAgent(model, library, pdf_tools=PDFTools(library)).run({"question_id": "FIX", "question": "Extract the reported HMB trial size."})
    assert row["status"] == "completed" and row["model_turns"] == 3 and row["tool_actions"] == 1
    assert model.requests[0]["evidence_registry"] == {} and sid in model.requests[-1]["evidence_registry"]
    assert any(e["node"] == "source_gate" and e["reason"] == "no_supplied_passages" for e in row["trace"])

def test_missing_sources_remain_bounded_and_never_get_invented(library):
    model = Scripted([final()] * 5)
    row = ArticlePDFAgent(model, library, pdf_tools=PDFTools(library)).run({"question_id": "FIX", "question": "Extract a fixture fact."})
    assert row["status"] == "operationally_unanswered" and row["model_turns"] == 5
    assert row["tool_actions"] == 0 and row["evidence"] == {} and row["response"]["citations"] == []

def test_unknown_source_hash_is_rejected_without_substitution(library):
    sid = next(iter(library.all_views()))
    invalid = "D000000000000_p1_c1_badbadbadbad"
    row = ArticlePDFAgent(Scripted([final([invalid]), final([sid])]), library, "B-full-context").run({"question_id": "FIX", "question": "Fixture fact."})
    assert row["status"] == "completed" and row["model_turns"] == 2 and row["response"]["citations"] == [sid]
    rejected = [e for e in row["trace"] if e["node"] == "source_gate" and e["status"] == "rejected"]
    assert rejected[0]["invalid_ids"] == [invalid]

def test_exact_registered_inline_ids_get_structured_without_medical_edit(library):
    sid = next(iter(library.all_views()))
    action = final()
    action["justification"] = "Fixture wording [" + sid + "]."
    row = ArticlePDFAgent(Scripted([action]), library, "B-full-context").run({"question_id": "FIX", "question": "Fixture fact."})
    assert row["response"]["citations"] == [sid] and row["response"]["justification"] == action["justification"]
    assert row["history"][-1]["action"]["citations"] == []  # Original generated action remains auditable.
    assert any(e["node"] == "source_gate" and e["status"] == "normalized" for e in row["trace"])

@pytest.mark.parametrize("value,status", [(False, "match"), (0, "wrong_type"), (None, "null"), (True, "mismatch")])
def test_boolean_reference_does_not_accept_numeric_or_missing_value(value, status):
    row = {"status": "completed", "output_origin": "model", "response": final(["S"], json.dumps({"significant": value})), "evidence": {"S": {}}}
    key, support = {"question_id": "FIX", "expected_fields": {"significant": False}}, {"field_support": {"significant": ["S"]}}
    scored = score_fields(row, key, support)
    assert scored["checks"][0]["status"] == status
    assert scored["correct_fields"] == int(status == "match")

def test_right_numeric_value_with_wrong_known_passage_is_not_cited_support():
    row = {"status": "completed", "output_origin": "model", "response": final(["OTHER"]), "evidence": {"SUPPORT": {}, "OTHER": {}}}
    key, support = {"question_id": "FIX", "expected_fields": {"patients_count": 472}}, {"field_support": {"patients_count": ["SUPPORT"]}}
    scored = score_fields(row, key, support)
    assert scored["correct_fields"] == 1 and scored["correct_fields_with_cited_reference_passage"] == 0
    assert scored["checks"][0]["reference_passage_supplied"]

def test_cross_page_reference_is_not_credited_if_unread_and_uncited():
    row = {"status": "completed", "output_origin": "model", "response": final(["PAGE7"], '{"dexamethasone": 3}'), "evidence": {"PAGE7": {}}}
    key, support = {"question_id": "FIX", "expected_fields": {"dexamethasone": 3}}, {"field_support": {"dexamethasone": ["PAGE8"]}}
    scored = score_fields(row, key, support)
    assert scored["correct_fields"] == 1 and not scored["checks"][0]["reference_passage_supplied"]
    assert scored["correct_fields_with_cited_reference_passage"] == 0

def test_new_frozen_set_has_only_positive_open_article_tasks():
    questions, keys = validate_set(ROOT)
    assert len(questions) == 6 and sum(len(k["expected_fields"]) for k in keys) == 24
    assert all(q["options"] is None for q in questions)
    schedule = order(questions)
    assert len(schedule) == len(set(schedule)) == 12
    assert [schedule[n][1] for n in range(0, 12, 2)] == ["B-full-context", "C-pdf-tools"] * 3
