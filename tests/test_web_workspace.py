"""Meaningful app boundaries: scope, persistence, billing idempotency and real tools."""
import json
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import pytest

from medical_copilot.patient_memory import PatientScopeError
from medical_copilot.clinical_resilient import ResilientLimits
from medical_copilot.web.agent import WorkspaceAgent, WorkspaceLimits
from medical_copilot.web.app import create_app
from medical_copilot.web.runtime import prior_app_usage, public_result
from medical_copilot.web.models import profile_path
from medical_copilot.web.store import WorkspaceStore

ROOT = Path(__file__).resolve().parents[1]
FINAL = {"type": "final", "answer": "Public answer", "justification": "Brief support.", "citations": [], "limitations": []}


class Scripted:
    def __init__(self, actions):
        self.actions, self.requests = list(actions), []
    def reply(self, request):
        self.requests.append(request)
        return self.actions.pop(0)


@pytest.mark.parametrize("explicit_limits,expected_tools", [
    (None, 20),
    (ResilientLimits(max_research_turns=1, max_tool_actions=1, max_model_turns=3), 1),
])
def test_research_budget_preserves_two_closing_attempts_and_saves_long_run(tmp_path, explicit_limits, expected_tools):
    store = WorkspaceStore(tmp_path)
    session = store.create_session()
    tool = {"type": "tool", "name": "search_conversation_history", "arguments": {"query": "synthetic marker"}}
    # The first finalization incorrectly requests another tool; it must be refused.
    model = Scripted([tool] * expected_tools + [tool, FINAL, tool])
    with store.memory() as memory:
        agent = WorkspaceAgent(model, memory, store.session(session["id"])["owner_id"], session["id"], limits=explicit_limits)
        result = agent.run({"question_id": "LONG", "question": "Check the conversation history and give a supported answer."})
        saved = memory.interaction(agent.patient_id, agent.interaction_id)
    assert result["status"] == "completed" and result["output_origin"] == "model"
    assert result["tool_actions"] == expected_tools and result["research_turns"] == expected_tools
    assert result["final_attempts"] == 2 and result["model_turns"] == expected_tools + 2
    assert [r["phase"] for r in model.requests] == ["research"] * expected_tools + ["finalize"] * 2
    assert all(r["tools"] == [] and r["remaining"]["tool_actions"] == 0 for r in model.requests[-2:])
    assert len(result["patient_tools"]["memory_events"]) == expected_tools
    assert all(e["status"] == "ok" for e in result["patient_tools"]["memory_events"])
    assert result["memory_write"]["saved"] and saved["response"]["answer"] == FINAL["answer"]
    assert len(model.actions) == 1


def test_larger_allowance_still_finishes_early_and_prevents_duplicate_save(tmp_path):
    store = WorkspaceStore(tmp_path)
    session = store.create_session()
    model = Scripted([FINAL])
    with store.memory() as memory:
        agent = WorkspaceAgent(model, memory, store.session(session["id"])["owner_id"], session["id"])
        result = agent.run({"question_id": "EARLY", "question": "A question requiring no tool."})
        with pytest.raises(ValueError, match="fresh WorkspaceAgent"):
            agent.run({"question_id": "EARLY", "question": "A question requiring no tool."})
    assert result["model_turns"] == 1 and result["tool_actions"] == 0
    assert len(model.requests) == 1 and model.requests[0]["remaining"]["tool_actions"] == 20


def test_web_limits_cannot_spend_reserved_finalization_turns():
    with pytest.raises(ValueError, match="Reserve final attempts"):
        WorkspaceLimits(max_model_turns=20)


def test_model_migration_keeps_old_answers_and_conversations_as_gemma(tmp_path):
    store = WorkspaceStore(tmp_path)
    session = store.create_session()
    job, _ = store.begin(session["id"], "Historical question", "old-request")
    store.finish(job["id"], {"response": FINAL})
    with store.db() as db:
        db.execute("ALTER TABLE sessions DROP COLUMN model_id")
        db.execute("ALTER TABLE messages DROP COLUMN model_id")
    reopened = WorkspaceStore(tmp_path)
    assert reopened.session(session["id"])["model_id"] == "gemma"
    assert reopened.message(job["id"])["model_id"] == "gemma"
    assert reopened.message(job["id"])["result"]["response"] == FINAL
    assert reopened.create_session(model_id="kimi-k3")["model_id"] == "kimi-k3"


def test_model_selection_is_persistent_and_jobs_keep_their_original_model(tmp_path):
    started, release = threading.Event(), threading.Event()
    calls = []
    def runner(store, job, emit, root):
        calls.append(job["model_id"])
        started.set()
        assert release.wait(5)
        return {"response": FINAL, "status": "completed"}
    with TestClient(create_app(tmp_path, runner, testing=True)) as client:
        assert [m["id"] for m in client.get("/api/bootstrap").json()["models"]] == ["gemma", "kimi-k3"]
        assert client.post("/api/sessions", json={"model_id": "arbitrary-provider"}).status_code == 422
        session = client.post("/api/sessions", json={}).json()
        path = "/api/sessions/" + session["id"]
        assert client.put(path + "/model", json={"model_id": "kimi-k3"}).json()["model_id"] == "kimi-k3"
        assert WorkspaceStore(tmp_path).session(session["id"])["model_id"] == "kimi-k3"
        body = {"question": "Same question", "request_id": "selected-kimi"}
        job = client.post(path + "/messages", json=body).json()
        try:
            assert started.wait(5)
            assert job["model_id"] == "kimi-k3"
            assert client.put(path + "/model", json={"model_id": "gemma"}).status_code == 409
        finally:
            release.set()
        client.get("/api/runs/" + job["id"] + "/events")
        assert client.put(path + "/model", json={"model_id": "gemma"}).status_code == 200
        assert client.put(path + "/model", json={"model_id": "other"}).status_code == 422
        assert client.post(path + "/messages", json=body).json()["model_id"] == "kimi-k3"
        other = client.post(path + "/messages", json={**body, "request_id": "selected-gemma"}).json()
        client.get("/api/runs/" + other["id"] + "/events")
        assert calls == ["kimi-k3", "gemma"]
        assert [m["model_id"] for m in client.get(path).json()["messages"]] == ["kimi-k3", "gemma"]


@pytest.mark.parametrize("model_id,provider_model", [("gemma", "google/gemma-4-26b-a4b-it"), ("kimi-k3", "moonshotai/kimi-k3")])
def test_live_runtime_routes_selected_profile_and_records_actual_model_offline(tmp_path, monkeypatch, model_id, provider_model):
    from langchain_core.messages import AIMessage
    from medical_copilot.models import Redactor
    import medical_copilot.web.runtime as runtime

    store = WorkspaceStore(tmp_path)
    session = store.create_session(model_id=model_id)
    job, _ = store.begin(session["id"], "Read this conversation's earlier notes.", "routing-test")
    configured, formats = [], []
    class Transport:
        def __init__(self):
            self.actions = iter([
                {"type": "tool", "name": "search_conversation_history", "arguments": {"query": "synthetic marker"}},
                {"type": "invalid"}, FINAL,
            ])
        def bind(self, **kwargs):
            formats.append(kwargs)
            return self
        def invoke(self, messages):
            return AIMessage(content=json.dumps(next(self.actions)),
                response_metadata={"finish_reason": "stop", "token_usage": {"prompt_tokens": 100, "completion_tokens": 40, "cost": .00002}},
                usage_metadata={"input_tokens": 100, "output_tokens": 40, "total_tokens": 140})
    def backend(config, root):
        configured.append(config.model)
        return Transport(), Redactor(), "synthetic_test_transport"
    monkeypatch.setattr(runtime, "create_backend", backend)
    monkeypatch.setattr(runtime, "preflight", lambda config, root: {
        "eligible_endpoints": [{"supported_parameters": ["structured_outputs"]}], "ledger": {"total_usage": 0}})
    monkeypatch.setattr(runtime, "prior_app_usage", lambda root, directory: {
        "account_anchor_usd": 0, "known_cost_usd": 0, "unknown_reserve_usd": 0})
    monkeypatch.setattr(runtime, "observe_ledger", lambda config, root: {"total_usage": 0})
    result = runtime.run_live(store, job, lambda *args, **kwargs: None, ROOT)
    directory = tmp_path / "runs" / job["id"]
    manifest = json.loads((directory / "manifest.json").read_text())
    record = json.loads((directory / "answer.json").read_text())
    assert configured == [provider_model]
    assert result["status"] == "completed" and result["model"]["id"] == model_id
    assert manifest["workspace_model"]["id"] == model_id and manifest["profile_path"] == profile_path(model_id)
    assert record["workspace_model"]["id"] == model_id and record["tool_actions"] == 1
    assert record["configuration"]["limits"]["max_tool_actions"] == 20
    assert result["usage"]["model_requests"] == 3 and formats[0]["response_format"]["type"] == "json_schema"


def test_general_conversations_have_independent_scopes_and_survive_reopen(tmp_path):
    store = WorkspaceStore(tmp_path)
    a, b = store.create_session(), store.create_session()
    owner = store.session(a["id"])["owner_id"]
    with store.memory() as memory:
        result = WorkspaceAgent(Scripted([FINAL]), memory, owner, a["id"]).run({"question_id": "Q1", "question": "My distinctive project note is red maple."})
        assert result["memory_write"]["saved"]
    again = WorkspaceStore(tmp_path)
    with again.memory() as memory:
        assert "red maple" in json.dumps(memory.context(owner, a["id"]))
        other_owner = again.session(b["id"])["owner_id"]
        assert "red maple" not in json.dumps(memory.context(other_owner, b["id"]))
    assert again.patients() == []


def test_general_mode_has_history_tools_but_no_patient_facts_or_foreign_context(tmp_path):
    store = WorkspaceStore(tmp_path)
    person = store.create_patient("Fictional person")
    store.add_fact(person["patient_id"], {"fact_key": "private_marker", "value": "NEVER-IN-GENERAL", "observed_at": "2026-10-15T10:00:00+00:00", "source_ref": "Synthetic input"})
    session = store.create_session()
    model = Scripted([FINAL])
    with store.memory() as memory:
        WorkspaceAgent(model, memory, store.session(session["id"])["owner_id"], session["id"]).run({"question_id": "Q", "question": "General evidence question"})
    request = model.requests[0]
    assert request["workspace_mode"] == "general"
    assert "patient_context" not in request and "patient_id" not in request["workspace_context"]
    assert request["workspace_context"]["current_facts"] == {}
    assert "NEVER-IN-GENERAL" not in json.dumps(request)
    assert "read_patient_facts" not in [t["name"] for t in request["tools"]]
    assert "search_conversation_history" in [t["name"] for t in request["tools"]]


def test_patient_new_conversation_reuses_records_but_general_stays_empty(tmp_path):
    store = WorkspaceStore(tmp_path)
    patient = store.create_patient("Fictional patient")
    store.add_fact(patient["patient_id"], {"fact_key": "weight_kg", "value": 64.2, "observed_at": "2026-10-15T10:00:00+00:00", "source_ref": "Synthetic input"})
    a = store.create_session(patient["patient_id"])
    b = store.create_session(patient["patient_id"])
    c = store.create_session()
    assert store.context(a["id"])["facts"]["weight_kg"]["value"] == store.context(b["id"])["facts"]["weight_kg"]["value"] == 64.2
    assert store.context(c["id"])["facts"] == {}
    with pytest.raises(PatientScopeError):
        store.create_session("W-not-a-public-patient")


def test_pdf_tool_uses_owned_passage_and_public_result_routes_to_same_session(tmp_path):
    store = WorkspaceStore(tmp_path)
    session = store.create_session()
    document = store.register_pdf(session["id"], ROOT / "data/evaluation/patient-demo-20261014/alpha.pdf", "Fictional case")
    model = Scripted([{"type": "tool", "name": "search_pdf", "arguments": {"document_id": document["document_id"], "query": "supplement cartons"}}])
    original_reply = model.reply
    def reply(request):
        if model.actions:
            return original_reply(request)
        model.requests.append(request)
        sid = next(iter(request["evidence_registry"]))
        return {**FINAL, "answer": "Two cartons in the fictional record.", "citations": [sid]}
    model.reply = reply
    events = []
    with store.memory() as memory:
        result = WorkspaceAgent(model, memory, store.session(session["id"])["owner_id"], session["id"], emit=lambda label, **kw: events.append((label, kw))).run({"question_id": "DOC", "question": "Read the supplement cartons in this PDF."})
    public = public_result(result, session["id"])
    assert public["status"] == "completed" and public["sources"][0]["pdf_page"] == 1
    assert "Recorded supplement cartons per day: 2" in public["sources"][0]["text"]
    assert public["sources"][0]["file_url"].startswith("/api/sessions/" + session["id"])
    assert any(event[1].get("kind") == "tool_start" for event in events)
    other = store.create_session()
    with pytest.raises(PatientScopeError):
        store.pdf_path(other["id"], document["document_id"])


def test_request_idempotency_does_not_repeat_inference_and_survives_reload(tmp_path):
    store = WorkspaceStore(tmp_path)
    session = store.create_session()
    first, fresh = store.begin(session["id"], "Question", "request-1")
    second, fresh_again = store.begin(session["id"], "Question", "request-1")
    assert fresh and not fresh_again and first["id"] == second["id"]
    with pytest.raises(ValueError):
        store.begin(session["id"], "Changed question", "request-1")
    with pytest.raises(ValueError):
        store.begin(session["id"], "Another request", "request-2")
    store.finish(first["id"], {"response": {"answer": "Saved answer"}})
    reopened = WorkspaceStore(tmp_path)
    assert reopened.messages(session["id"])[0]["result"]["response"]["answer"] == "Saved answer"


def test_restarted_job_is_not_automatically_charged_again(tmp_path):
    store = WorkspaceStore(tmp_path)
    session = store.create_session()
    job, _ = store.begin(session["id"], "Pending question", "request-1")
    store.running(job["id"])
    store.recover_interrupted()
    restored = store.message(job["id"])
    assert restored["state"] == "error" and "may have incurred a charge" in restored["result"]["error"]


def test_crash_allowance_is_counted_before_known_response_accounting(tmp_path, monkeypatch):
    import medical_copilot.web.runtime as runtime
    monkeypatch.setattr(runtime, "prior_usage", lambda root: {"saved_requests": 1, "known_cost_usd": 1, "unknown_reserve_usd": .1, "account_anchor_usd": 2})
    run = tmp_path / "runs/R1"
    run.mkdir(parents=True)
    (run / "usage-audit.json").write_text(json.dumps({"records": [{"request_sent": True, "cost_usd": .02}]}))
    (run / "in-flight.json").write_text(json.dumps({"reserved_usd": .03}))
    totals = prior_app_usage(tmp_path, tmp_path)
    assert totals["saved_requests"] == 2 and totals["known_cost_usd"] == 1.02 and totals["unknown_reserve_usd"] == .13


def test_http_post_event_stream_and_idempotency_share_real_job(tmp_path):
    calls = []
    def runner(store, job, emit, root):
        calls.append(job["id"])
        emit("Synthetic test runner invoked", kind="model")
        return {"response": FINAL, "sources": [], "status": "completed", "output_origin": "test_fixture", "memory_saved": False}
    with TestClient(create_app(tmp_path, runner, testing=True)) as client:
        session = client.post('/api/sessions', json={}).json()
        body = {"question": "Test question", "request_id": "request-1"}
        job = client.post('/api/sessions/' + session['id'] + '/messages', json=body).json()
        stream = client.get('/api/runs/' + job['id'] + '/events')
        assert '"type": "complete"' in stream.text and 'Synthetic test runner invoked' in stream.text
        again = client.post('/api/sessions/' + session['id'] + '/messages', json=body).json()
        assert again['id'] == job['id'] and calls == [job['id']]
        assert client.get('/api/sessions/' + session['id']).json()['messages'][0]['result']['response']['answer'] == 'Public answer'


def test_http_foreign_document_and_cross_origin_post_are_rejected(tmp_path):
    with TestClient(create_app(tmp_path, testing=True)) as client:
        a = client.post('/api/sessions', json={}).json()
        b = client.post('/api/sessions', json={}).json()
        payload = (ROOT / 'data/evaluation/patient-demo-20261014/alpha.pdf').read_bytes()
        doc = client.post('/api/sessions/' + a['id'] + '/documents', files={'file': ('case.pdf', payload, 'application/pdf')}).json()
        assert client.get(f"/api/sessions/{a['id']}/documents/{doc['document_id']}/file").status_code == 200
        assert client.get(f"/api/sessions/{b['id']}/documents/{doc['document_id']}/file").status_code == 404
        assert client.post('/api/sessions', json={}, headers={'Origin':'https://unrelated.example'}).status_code == 403


def test_invalid_pdf_is_actionable_without_model_access(tmp_path):
    with TestClient(create_app(tmp_path, testing=True)) as client:
        session = client.post('/api/sessions', json={}).json()
        response = client.post('/api/sessions/' + session['id'] + '/documents', files={'file': ('fake.pdf', b'not a PDF', 'application/pdf')})
        assert response.status_code == 400 and 'valid PDF' in response.json()['detail']
        assert not list((tmp_path / 'uploads').iterdir())
        corrupt = client.post('/api/sessions/' + session['id'] + '/documents', files={'file': ('corrupt.pdf', b'%PDF-1.7\nnot a document', 'application/pdf')})
        assert corrupt.status_code == 400 and 'could not be ingested' in corrupt.json()['detail']


def test_appearance_persists_but_script_logo_is_rejected(tmp_path):
    with TestClient(create_app(tmp_path, testing=True)) as client:
        appearance = client.get('/api/bootstrap').json()['appearance']
        changed = {**appearance, 'name':'Custom research desk', 'accent':'#920000', 'font_scale':1.1}
        assert client.put('/api/appearance', json=changed).json() == changed
        assert client.get('/api/bootstrap').json()['appearance'] == changed
        assert client.put('/api/appearance', json={**changed, 'logo':'data:image/svg+xml;base64,PHN2Zz4='}).status_code == 422


def test_demo_setup_is_repeatable_and_saved_example_is_labelled(tmp_path):
    with TestClient(create_app(tmp_path, testing=True)) as client:
        first = client.post('/api/examples/prepare').json()
        second = client.post('/api/examples/prepare').json()
        assert first == second
        assert len(client.get('/api/bootstrap').json()['patients']) == 2
        saved = client.post('/api/examples/saved').json()
        context = client.get('/api/sessions/' + saved['id']).json()
        assert context['session']['patient_id'] is None
        assert len(context['files']) == 1
        clinical = next(p for p in client.get('/api/bootstrap').json()['patients'] if p['name'] == 'Patient Alpha')
        clinical_session = client.post('/api/sessions', json={'patient_id': clinical['patient_id']}).json()
        clinical_files = client.get('/api/sessions/' + clinical_session['id']).json()['files']
        assert len(clinical_files) == 1 and clinical_files[0]['document_id'] != context['files'][0]['document_id']
        for document in context['files']:
            assert client.get(f"/api/sessions/{saved['id']}/documents/{document['document_id']}/file").status_code == 200
        row = context['messages'][0]
        assert row['result']['archive']['new_inference'] is False
        assert 'Saved model response' in row['result']['archive']['label']
