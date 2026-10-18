"""Guided setup must preserve real scoped state without inventing model history."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from medical_copilot.patient_memory import PatientScopeError
from medical_copilot.web.agent import WorkspaceAgent
from medical_copilot.web.app import create_app

ROOT = Path(__file__).resolve().parents[1]
FINAL = {"type": "final", "answer": "Synthetic offline test response about PROTEOS.",
         "justification": "Test transport only.", "citations": [], "limitations": []}


def prepare(client, example, model="gemma"):
    response = client.post(f"/api/examples/{example}/prepare", json={"model_id": model})
    assert response.status_code == 200, response.text
    return response.json()


def test_examples_keep_original_choices_and_preparation_never_runs_inference(tmp_path):
    calls = []
    with TestClient(create_app(tmp_path, lambda *args: calls.append(True), testing=True)) as client:
        examples = client.get("/api/bootstrap").json()["examples"]
        assert [e["id"] for e in examples if e["featured"]] == ["literature", "pdf", "memory"]
        assert len([e for e in examples if e["flow"] == "continuity"]) == 3
        for example in examples:
            selected = prepare(client, example["id"], "kimi-k3")
            context = client.get("/api/sessions/" + selected["session_id"]).json()
            assert selected["setup_only"] and selected["model_inference_requests"] == 0
            assert context["messages"] == [] and context["session"]["model_id"] == "kimi-k3"
        assert calls == []
        assert client.post("/api/examples/unknown/prepare", json={"model_id": "gemma"}).status_code == 400
        assert client.post("/api/examples/pdf/prepare", json={"model_id": "arbitrary"}).status_code == 422


def test_followup_opens_new_conversation_with_real_prior_history_and_owned_sources(tmp_path):
    app = create_app(tmp_path, testing=True)
    with TestClient(app) as client:
        initial = prepare(client, "continuity-case")
        first = app.state.store.session(initial["session_id"])
        context = app.state.store.context(first["id"])
        assert context["facts"]["weight_kg"]["value"] == 64.2
        assert len(context["files"]) == 2
        case_id = next(f["document_id"] for f in context["files"] if f["page_count"] == 1)
        class Model:
            def __init__(self, actions):
                self.actions, self.requests = list(actions), []
            def reply(self, request):
                self.requests.append(request)
                if self.actions:
                    return self.actions.pop(0)
                return {**FINAL, "citations": list(request["evidence_registry"])[:1]}
        before_model = Model([{"type": "tool", "name": "search_pdf", "arguments": {"document_id": case_id, "query": "preferred snack"}}])
        with app.state.store.memory() as memory:
            before = WorkspaceAgent(before_model, memory, first["owner_id"], first["id"], patient_selected=True).run(
                {"question_id": "CASE", "question": initial["question"]})
        assert before["status"] == "completed" and before["memory_write"]["saved"]
        assert "lactose-free plain yogurt" in " ".join(view["text"] for view in before["evidence"].values())
        followup = prepare(client, "continuity-followup")
        second = app.state.store.session(followup["session_id"])
        assert second["id"] != first["id"] and second["patient_id"] == first["patient_id"]
        context = app.state.store.context(second["id"])
        assert context["facts"]["weight_kg"]["value"] == 62.4 and len(context["files"]) == 3
        assert context["facts"]["weight_kg"]["observed_at"].startswith("2026-10-13")
        source_id = next(iter(before["evidence"]))
        after_model = Model([
            {"type": "tool", "name": "search_patient_history", "arguments": {"query": "PROTEOS"}},
            {"type": "tool", "name": "read_patient_interaction", "arguments": {
                "interaction_id": before["interaction_id"], "restore_source_ids": [source_id]}},
            {"type": "tool", "name": "read_patient_facts", "arguments": {"fact_key": "weight_kg"}},
        ])
        with app.state.store.memory() as memory:
            after = WorkspaceAgent(after_model, memory, second["owner_id"], second["id"], patient_selected=True).run(
                {"question_id": "FOLLOWUP", "question": followup["question"]})
        assert after["status"] == "completed" and after["tool_actions"] == 3
        assert after["evidence"][source_id]["text"] == before["evidence"][source_id]["text"]
        assert "PROTEOS" in json.dumps(after_model.requests[0]["workspace_context"])
        events = after["patient_tools"]["memory_events"]
        assert before["interaction_id"] in json.dumps(events[0])
        assert all(str(value) in json.dumps(events[-1]) for value in [64.2, 62.4])


def test_repeated_followup_preserves_explicit_corrections_and_avoids_duplicate_facts(tmp_path):
    app = create_app(tmp_path, testing=True)
    with TestClient(app) as client:
        followup = prepare(client, "continuity-followup")
        context = app.state.store.context(followup["session_id"])
        patient_id = context["session"]["patient_id"]
        fact = context["facts"]["weight_kg"]
        app.state.store.add_fact(patient_id, {"fact_key": "weight_kg", "value": 62.1, "unit": "kg",
            "observed_at": fact["observed_at"], "source_ref": "Synthetic explicit correction for this test.",
            "origin": "user_provided", "supersedes_fact_id": fact["fact_id"]})
        before = app.state.store.context(followup["session_id"])
        again = prepare(client, "continuity-followup")
        after = app.state.store.context(again["session_id"])
        assert after["facts"]["weight_kg"]["value"] == 62.1
        assert len(before["fact_history"]) == len(after["fact_history"]) == 4
        assert len(after["files"]) == 3 and len(app.state.store.patients()) == 2


def test_new_clinical_cases_stay_separate_and_general_mode_has_no_patient_data(tmp_path):
    app = create_app(tmp_path, testing=True)
    with TestClient(app) as client:
        alpha = prepare(client, "continuity-case")
        alpha_context = app.state.store.context(alpha["session_id"])
        beta = next(p for p in app.state.store.patients() if p["name"] == "Patient Beta")
        beta_session = app.state.store.create_session(beta["patient_id"])
        beta_context = app.state.store.context(beta_session["id"])
        assert beta_context["facts"]["weight_kg"]["value"] == 81.7
        assert beta_context["facts"]["snack"]["value"] == "banana and oat porridge"
        assert len(beta_context["files"]) == 1
        with pytest.raises(PatientScopeError):
            app.state.store.pdf_path(beta_session["id"], alpha_context["files"][0]["document_id"])
        general = prepare(client, "continuity-evidence")
        context = app.state.store.context(general["session_id"])
        assert context["facts"] == {} and context["files"] == []
        assert context["session"]["patient_id"] is None


def test_patient_rename_preserves_clinical_memory_and_removes_original_demos(tmp_path):
    app = create_app(tmp_path, testing=True)
    with TestClient(app) as client:
        initial = prepare(client, "continuity-followup")
        store = app.state.store
        before = store.context(initial["session_id"])
        clinical_ids = {p["patient_id"] for p in store.patients()}
        patient_id = before["session"]["patient_id"]
        job, _ = store.begin(initial["session_id"], "Keep this clinical history.", "clinical-history")
        store.finish(job["id"], {"response": FINAL})
        class Model:
            def reply(self, request):
                return FINAL
        with store.memory() as memory:
            record = WorkspaceAgent(Model(), memory, patient_id, initial["session_id"], patient_selected=True).run(
                {"question_id": "RENAME", "question": "Keep this clinical history."})
            history = memory.interaction(patient_id, record["interaction_id"])

        # Reproduce the four patients and schema saved by the previous release.
        for code in ("Alpha", "Beta"):
            legacy = store.create_patient("Patient " + code, demo=True)
            session = store.create_session(legacy["patient_id"], demo=True)
            store.add_fact(legacy["patient_id"], {"fact_key": "weight_kg", "value": 999,
                "observed_at": "2026-10-10T09:00:00+00:00", "source_ref": "Obsolete synthetic demo."})
            store.register_pdf(session["id"], ROOT / "demo/cases" / (code.lower() + "_initial.pdf"), "Obsolete demo")
            with store.db() as db:
                db.execute("UPDATE people SET name=? WHERE demo_key=?", (code + " - Clinical demo", "clinical-" + code.lower()))
        custom = store.create_patient("Patient Alpha")
        with store.db() as db:
            db.execute("DROP INDEX people_demo_key")
            db.execute("ALTER TABLE people DROP COLUMN demo_key")

    for _ in range(2):
        with TestClient(create_app(tmp_path, testing=True)) as client:
            store = client.app.state.store
            people = client.get("/api/bootstrap").json()["patients"]
            demos = [p for p in people if p["demo"]]
            assert {p["name"] for p in demos} == {"Patient Alpha", "Patient Beta"}
            assert {p["patient_id"] for p in demos} == clinical_ids
            assert store.patient(custom["patient_id"])["demo"] == 0
            after = store.context(initial["session_id"])
            assert after["files"] == before["files"] and after["fact_history"] == before["fact_history"]
            assert store.messages(initial["session_id"])[0]["id"] == job["id"]
            with store.memory() as memory:
                assert memory.patient(patient_id)["display_name"] == "Patient Alpha"
                assert memory.interaction(patient_id, record["interaction_id"]) == history
                assert list(memory.connection.execute("PRAGMA foreign_key_check")) == []
            assert client.post("/api/examples/prepare").status_code == 200
            followup = prepare(client, "continuity-followup")
            assert store.session(followup["session_id"])["patient_id"] == patient_id
            assert store.context(followup["session_id"])["fact_history"] == before["fact_history"]
            assert len(store.context(followup["session_id"])["files"]) == len(before["files"])
            assert len(store.patients()) == 3


def test_demo_case_files_are_one_page_with_explicit_fictional_labels():
    import pymupdf
    from medical_copilot.patient_memory import file_hash
    for filename in ["alpha_initial.pdf", "beta_initial.pdf", "alpha_followup.pdf"]:
        with pymupdf.open(ROOT / "demo/cases" / filename) as document:
            assert len(document) == 1
            text = document[0].get_text()
            assert "FICTIONAL TEACHING CASE" in text and len(text) > 900
