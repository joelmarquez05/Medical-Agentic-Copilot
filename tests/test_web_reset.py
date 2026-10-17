"""Reset both histories atomically while retaining clinical records, files and audit."""
import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from medical_copilot.web.agent import WorkspaceAgent
from medical_copilot.web.app import create_app
from medical_copilot.web.runtime import public_result


class Model:
    def reply(self, request):
        return {"type": "final", "answer": "Synthetic reset-test history marker.",
                "justification": "Offline test fixture.", "citations": [], "limitations": []}


def seed(app):
    store = app.state.store
    person = store.create_patient("Reset test patient")
    session = store.create_session(person["patient_id"], model_id="kimi-k3")
    store.add_fact(person["patient_id"], {"fact_key": "weight_kg", "value": 70, "unit": "kg",
        "observed_at": "2026-10-10T12:00:00+00:00", "origin": "user_provided", "source_ref": "Synthetic reset-test fixture."})
    root = Path(__file__).resolve().parents[1]
    document = store.register_pdf(session["id"], root / "demo/cases/alpha_initial.pdf", "Synthetic reset-test document")
    job, _ = store.begin(session["id"], "Remember the reset-test conversation marker.", "reset-test-request")
    with store.memory() as memory:
        record = WorkspaceAgent(Model(), memory, person["patient_id"], session["id"], patient_selected=True).run(
            {"question_id": "RESET", "question": job["question"]})
    store.finish(job["id"], public_result(record, session["id"]))
    return person, session, document


def test_reset_removes_both_histories_and_keeps_records_files_settings_and_audit(tmp_path):
    app = create_app(tmp_path, testing=True)
    with TestClient(app) as client:
        person, session, document = seed(app)
        store = app.state.store
        appearance = {**store.settings(), "name": "Personalized workspace"}
        store.save_settings(appearance)
        audit = tmp_path / "runs/preserved-audit/usage-audit.json"
        audit.parent.mkdir(parents=True)
        audit.write_text(json.dumps({"records": [{"request_sent": True, "cost_usd": .001}]}))
        before_audit = audit.read_bytes()
        assert client.post("/api/reset", json={"confirm": False}).status_code == 422
        assert len(store.messages(session["id"])) == 1
        result = client.post("/api/reset", json={"confirm": True})
        assert result.status_code == 200 and result.json()["cleared_memory_interactions"] == 1
        assert client.get("/api/bootstrap").json()["sessions"] == []
        assert store.patients() == [person] and store.settings() == appearance
        assert audit.read_bytes() == before_audit
        with store.memory() as memory:
            assert memory.recent_interactions(person["patient_id"]) == []
            assert memory.current_facts(person["patient_id"])["weight_kg"]["value"] == 70
            assert memory.document(person["patient_id"], document["document_id"])
        again = store.create_session(person["patient_id"])
        assert len(store.context(again["id"])["files"]) == 1
        assert client.get(f"/api/sessions/{again['id']}/documents/{document['document_id']}/file").status_code == 200


def test_reset_waits_for_queued_or_running_requests(tmp_path):
    app = create_app(tmp_path, testing=True)
    with TestClient(app) as client:
        person, session, _ = seed(app)
        job, _ = app.state.store.begin(session["id"], "Queued test", "still-pending")
        for state in ["queued", "running"]:
            if state == "running":
                app.state.store.running(job["id"])
            assert client.post("/api/reset", json={"confirm": True}).status_code == 409
            assert len(app.state.store.messages(session["id"])) == 2
            with app.state.store.memory() as memory:
                assert len(memory.recent_interactions(person["patient_id"])) == 1


def test_cross_database_reset_rolls_back_both_histories_on_failure(tmp_path):
    app = create_app(tmp_path, testing=True)
    with TestClient(app):
        person, session, _ = seed(app)
        with app.state.store.memory() as memory:
            with memory.connection:
                memory.connection.execute("CREATE TRIGGER stop_reset BEFORE DELETE ON interactions BEGIN SELECT RAISE(ABORT, 'Synthetic reset failure'); END")
        with pytest.raises(sqlite3.IntegrityError, match="Synthetic reset failure"):
            app.state.store.reset_conversations()
        assert len(app.state.store.messages(session["id"])) == 1
        with app.state.store.memory() as memory:
            assert len(memory.recent_interactions(person["patient_id"])) == 1
