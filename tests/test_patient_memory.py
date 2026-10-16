"""Patient continuity, provenance, scope and bounded-agent regressions."""
import copy
import json

import pymupdf
import pytest
from pydantic import ValidationError

from medical_copilot.patient_agent import PatientAgent
from medical_copilot.patient_memory import FactInput, PatientMemory, PatientScopeError
from medical_copilot.patient_tools import PatientTools, ReadFactsArgs, ReadInteractionArgs, SearchHistoryArgs


def final(answer="Fixture answer.", citations=()):
    return {"type": "final", "answer": answer, "justification": "Synthetic fixture, not a medical recommendation.",
            "selected_option": None, "citations": list(citations), "limitations": []}


class Scripted:
    def __init__(self, actions):
        self.actions, self.requests = iter(actions), []

    def reply(self, request):
        self.requests.append(copy.deepcopy(request))
        return next(self.actions)


def fact(value=65, **updates):
    return {"fact_key": "weight_kg", "value": value, "unit": "kg", "observed_at": "2026-10-10T09:00:00+00:00", "source_ref": "Synthetic explicit input", **updates}


@pytest.fixture
def memory(tmp_path):
    with PatientMemory(tmp_path / "memory.sqlite3") as store:
        store.create_patient("alpha", "Synthetic Alpha")
        store.create_patient("beta", "Synthetic Beta")
        store.conversation("alpha", "alpha-old", create=True)
        store.conversation("beta", "beta-old", create=True)
        yield store


def add_pdf(memory, tmp_path, patient="alpha", marker="ALPHA-ONLY"):
    source = tmp_path / (patient + ".pdf")
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 80), "SYNTHETIC CASE\nPrivate marker: " + marker + "\nRecorded supplement cartons: 2.")
        pdf.save(source)
    return memory.register_pdf(patient, source, title="Synthetic " + patient + " case")


def seed(store, patient="alpha", conversation="alpha-old", iid="I-old", question="My preferred snack is sesame crackers.", evidence=None):
    return store.save_interaction(patient, conversation, {"question_id": "SEED", "question": question},
                                  {"type": "fixture_note", "answer": "Recorded synthetic user statement."}, evidence or {},
                                  status="fixture", origin="synthetic_fixture", interaction_id=iid)


def test_disk_reopen_preserves_patient_facts_files_and_public_history(memory, tmp_path):
    memory.add_fact("alpha", fact())
    doc = add_pdf(memory, tmp_path)
    seed(memory)
    with PatientMemory(memory.database) as reopened:
        assert reopened.current_facts("alpha")["weight_kg"]["value"] == 65
        assert reopened.documents("alpha")[0]["document_id"] == doc["document_id"]
        assert reopened.interaction("alpha", "I-old")["question"]["question"].endswith("sesame crackers.")


def test_new_conversation_reuses_patient_scope(memory):
    memory.add_fact("alpha", fact())
    seed(memory)
    memory.conversation("alpha", "alpha-new", create=True)
    context = memory.context("alpha", "alpha-new")
    assert context["current_facts"]["weight_kg"]["value"] == 65
    assert context["recent_interactions"][0]["conversation_id"] == "alpha-old"


def test_late_old_observation_does_not_replace_current_measurement(memory):
    memory.add_fact("alpha", fact(64, observed_at="2026-10-13T09:00:00+00:00"))
    memory.add_fact("alpha", fact(70, observed_at="2026-10-01T09:00:00+00:00"))
    assert memory.current_facts("alpha")["weight_kg"]["value"] == 64
    assert len(memory.fact_history("alpha", "weight_kg")) == 2


def test_correction_keeps_trace_and_supersedes_same_observation(memory):
    old = memory.add_fact("alpha", fact(65), fact_id="F-old")
    memory.add_fact("alpha", fact(64, supersedes_fact_id=old["fact_id"]), fact_id="F-corrected")
    assert memory.current_facts("alpha")["weight_kg"]["value"] == 64
    assert next(f for f in memory.fact_history("alpha") if f["fact_id"] == "F-old")["superseded"]
    with pytest.raises(ValueError):
        memory.add_fact("alpha", fact(63, supersedes_fact_id="F-old"))


def test_fact_correction_cannot_cross_patient_or_key(memory):
    memory.add_fact("alpha", fact(), fact_id="F-alpha")
    with pytest.raises(PatientScopeError):
        memory.add_fact("beta", fact(82, supersedes_fact_id="F-alpha"))
    with pytest.raises(PatientScopeError):
        memory.add_fact("alpha", fact("other", fact_key="preference", supersedes_fact_id="F-alpha"))


def test_foreign_conversation_rejected_before_model_use(memory):
    model = Scripted([])
    with pytest.raises(PatientScopeError):
        PatientAgent(model, memory, "beta", "alpha-old")
    assert model.requests == []


def test_query_and_context_do_not_search_other_patients(memory):
    memory.add_fact("alpha", fact(65))
    memory.add_fact("beta", fact(82))
    seed(memory)
    assert memory.search_history("beta", "sesame") == []
    context = memory.context("beta", "beta-old")
    assert context["current_facts"]["weight_kg"]["value"] == 82
    assert "sesame" not in json.dumps(context)


def test_owned_pdf_only_and_model_metadata_has_no_paths(memory, tmp_path):
    doc = add_pdf(memory, tmp_path)
    with pytest.raises(PatientScopeError):
        memory.document("beta", doc["document_id"])
    assert "relative_path" not in json.dumps(memory.documents("alpha"))
    assert str(tmp_path) not in json.dumps(memory.documents("alpha"))
    assert not PatientTools(memory, "beta").library


def test_owned_ingestion_tampering_fails_before_content_delivery(memory, tmp_path):
    doc = add_pdf(memory, tmp_path)
    path = memory.document_root / "alpha" / doc["document_id"] / "document.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(PatientScopeError):
        PatientTools(memory, "alpha")


def test_document_linked_fact_requires_actual_owned_source_view(memory, tmp_path):
    doc = add_pdf(memory, tmp_path)
    sid = next(iter(memory.document("alpha", doc["document_id"])["views"]))
    value = fact(2, fact_key="cartons", unit=None, origin="document_entry", document_id=doc["document_id"], source_id=sid)
    memory.add_fact("alpha", value)
    with pytest.raises(PatientScopeError):
        memory.add_fact("beta", value)


def test_patient_identity_not_accepted_in_memory_tool_arguments():
    for schema, args in [(SearchHistoryArgs, {"query": "snack"}), (ReadInteractionArgs, {"interaction_id": "I-old"}), (ReadFactsArgs, {"fact_key": "weight_kg"})]:
        with pytest.raises(ValidationError):
            schema.model_validate({**args, "patient_id": "beta"})


def test_history_older_than_recent_window_remains_retrievable(memory):
    seed(memory)
    for i in range(5):
        seed(memory, iid="I-filler" + str(i), question="Unrelated fixture interaction " + str(i))
    assert all(row["interaction_id"] != "I-old" for row in memory.context("alpha", "alpha-old")["recent_interactions"])
    found = PatientTools(memory, "alpha").search(SearchHistoryArgs(query="sesame"))
    assert "I-old" in found.content and "sesame crackers" in found.content


def test_foreign_interaction_rejected_without_content(memory):
    seed(memory)
    tools = PatientTools(memory, "beta")
    with pytest.raises(PatientScopeError):
        tools.read(ReadInteractionArgs(interaction_id="I-old"))
    assert tools.events[-1]["status"] == "rejected" and "generation_view" not in tools.events[-1]


def test_saved_source_restoration_delivers_exact_view_and_keeps_origin(memory, tmp_path):
    doc = add_pdf(memory, tmp_path)
    views = memory.document("alpha", doc["document_id"])["views"]
    sid = next(iter(views))
    seed(memory, evidence=views)
    output = PatientTools(memory, "alpha").read(ReadInteractionArgs(interaction_id="I-old", restore_source_ids=[sid]))
    assert output.sources[0].model_dump(mode="json") == views[sid]
    assert "no new literature search" in output.content
    with pytest.raises(PatientScopeError):
        PatientTools(memory, "alpha").read(ReadInteractionArgs(interaction_id="I-old", restore_source_ids=["UNKNOWN"]))


def test_context_budget_reports_omissions_instead_of_silent_cutting(memory):
    seed(memory, question="x" * 10000)
    context = memory.context("alpha", "alpha-old", max_bytes=1500)
    assert context["recent_interactions"] == [] and context["omitted_recent_interactions"] == 1
    memory.add_fact("alpha", fact("x" * 1000))
    with pytest.raises(ValueError, match="Core"):
        memory.context("alpha", "alpha-old", max_bytes=500)


@pytest.mark.parametrize("updates", [{"observed_at": "2026-10-10T09:00:00"}, {"value": float("nan")}, {"value": "x" * 2000}, {"origin": "assistant_inference"}])
def test_fact_contract_rejects_ambiguous_or_unsupported_entries(updates):
    with pytest.raises((ValueError, ValidationError)):
        FactInput.model_validate(fact(**updates))


def test_public_response_cannot_include_private_reasoning(memory):
    with pytest.raises(ValueError, match="private"):
        memory.save_interaction("alpha", "alpha-old", {"question_id": "X", "question": "Fixture"},
                                {"type": "final", "answer": "Fixture", "reasoning_content": "must not persist"}, {}, status="completed", origin="model")


def test_agent_loads_context_uses_fresh_actions_and_persists_public_output(memory):
    memory.add_fact("alpha", fact())
    model = Scripted([final('65')])
    row = PatientAgent(model, memory, "alpha", "alpha-new").run({"question_id": "PM", "question": "What is my recorded weight?"})
    assert row["status"] == "completed" and row["memory_write"]["saved"]
    assert model.requests[0]["history"] == [] and model.requests[0]["evidence_registry"] == {}
    assert model.requests[0]["patient_context"]["current_facts"]["weight_kg"]["value"] == 65
    assert [e["node"] for e in row["trace"]][-2:] == ["finish", "save_patient_interaction"]
    assert set(memory.current_facts("alpha")) == {"weight_kg"}
    next_model = Scripted([final()])
    PatientAgent(next_model, memory, "alpha", "alpha-newer").run({"question_id": "PM2", "question": "Follow-up"})
    assert next_model.requests[0]["history"] == [] and next_model.requests[0]["patient_context"]["recent_interactions"]


def test_memory_disabled_control_withholds_facts_and_does_not_pollute_history(memory):
    memory.add_fact("alpha", fact())
    model = Scripted([final('{"weight_kg": null}')])
    row = PatientAgent(model, memory, "alpha", "alpha-new", memory_enabled=False).run({"question_id": "CONTROL", "question": "My recorded weight?"})
    assert model.requests[0]["patient_context"]["current_facts"] == {} and model.requests[0]["tools"] == []
    assert not row["memory_write"]["saved"] and not memory.recent_interactions("alpha")


def test_agent_reads_actual_scoped_pdf_and_validates_source(memory, tmp_path):
    doc = add_pdf(memory, tmp_path)
    sid = next(iter(memory.document("alpha", doc["document_id"])["views"]))
    model = Scripted([{"type": "tool", "name": "read_pdf_page", "arguments": {"document_id": doc["document_id"], "page": 1}}, final("2", [sid])])
    row = PatientAgent(model, memory, "alpha", "alpha-new", require_reference=True).run({"question_id": "PDF", "question": "Read my case file."})
    assert row["status"] == "completed" and row["tool_actions"] == 1
    assert "ALPHA-ONLY" in row["evidence"][sid]["text"]
    assert sid in memory.interaction("alpha", row["interaction_id"])["evidence"]


def test_invalid_citations_recover_within_five_calls_and_remain_honest(memory):
    model = Scripted([final(citations=["FAKE"])] * 5)
    row = PatientAgent(model, memory, "alpha", "alpha-new").run({"question_id": "BAD", "question": "Fixture"})
    assert row["status"] == "operationally_unanswered" and row["model_turns"] == 5
    assert row["evidence"] == {} and row["response"]["citations"] == []
    assert row["memory_write"]["saved"]


def test_persistence_failure_is_explicit_without_rewriting_model_answer(memory, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Synthetic storage failure")
    monkeypatch.setattr(memory, "save_interaction", fail)
    row = PatientAgent(Scripted([final("Original")]), memory, "alpha", "alpha-new").run({"question_id": "WRITE", "question": "Fixture"})
    assert row["response"]["answer"] == "Original" and not row["memory_write"]["saved"]
    assert row["memory_write"]["reason"] == "persistence_failed"
