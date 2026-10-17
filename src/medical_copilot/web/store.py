"""Durable UI sessions, explicit patients and idempotent public answer jobs."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import uuid

from medical_copilot.patient_memory import PatientMemory, PatientScopeError, identifier, utc_now
from medical_copilot.web.models import DEFAULT_MODEL, public_model

DEFAULT_APPEARANCE = {"name": "Medical Copilot", "accent": "#B90000", "font": "editorial", "font_scale": 1,
                      "density": "comfortable", "logo": None}


class WorkspaceStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "workspace.sqlite3"
        self.memory_database = self.directory / "memory.sqlite3"
        with self.memory():
            pass
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS people (patient_id TEXT PRIMARY KEY, name TEXT NOT NULL, demo INTEGER NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL, patient_id TEXT REFERENCES people(patient_id),
                    owner_id TEXT NOT NULL, demo INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), request_id TEXT UNIQUE NOT NULL,
                    question TEXT NOT NULL, state TEXT NOT NULL, result TEXT, events TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id,created_at);
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
            """)
            # Existing conversations and answers were all generated with Gemma.
            for table in ("sessions", "messages"):
                if "model_id" not in {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN model_id TEXT NOT NULL DEFAULT 'gemma'")
            db.execute("INSERT OR IGNORE INTO settings VALUES (1,?)", (json.dumps(DEFAULT_APPEARANCE),))
            if "demo_key" not in {row["name"] for row in db.execute("PRAGMA table_info(people)")}:
                db.execute("ALTER TABLE people ADD COLUMN demo_key TEXT")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS people_demo_key ON people(demo_key)")
        self._migrate_demo_patients()

    def _migrate_demo_patients(self):
        """Retire the original demos and keep the newer clinical patients' IDs."""
        with self.db() as db:
            db.execute("ATTACH DATABASE ? AS patient_memory", (str(self.memory_database),))
            db.execute("BEGIN IMMEDIATE")
            for code in ("Alpha", "Beta"):
                name, key = "Patient " + code, "clinical-" + code.lower()
                obsolete = db.execute("SELECT patient_id FROM people WHERE demo=1 AND demo_key IS NULL AND name=?", (name,)).fetchall()
                for row in obsolete:
                    patient_id = row["patient_id"]
                    db.execute("DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE patient_id=?)", (patient_id,))
                    db.execute("DELETE FROM sessions WHERE patient_id=?", (patient_id,))
                    for table in ("interactions", "facts", "documents", "conversations", "patients"):
                        db.execute(f"DELETE FROM patient_memory.{table} WHERE patient_id=?", (patient_id,))
                    db.execute("DELETE FROM people WHERE patient_id=?", (patient_id,))
                db.execute("UPDATE people SET name=?,demo_key=? WHERE demo=1 AND name=? AND demo_key IS NULL",
                           (name, key, code + " - Clinical demo"))
                db.execute("UPDATE patient_memory.patients SET display_name=? WHERE patient_id IN (SELECT patient_id FROM people WHERE demo_key=?)",
                           (name, key))
                db.execute("UPDATE sessions SET title=? WHERE title=? AND patient_id IN (SELECT patient_id FROM people WHERE demo_key=?)",
                           (name + " consultation", code + " - Clinical demo", key))

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.database, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def memory(self):
        return PatientMemory(self.memory_database)

    def patients(self):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM people ORDER BY created_at")]

    def patient(self, patient_id):
        identifier(patient_id)
        with self.db() as db:
            row = db.execute("SELECT * FROM people WHERE patient_id=?", (patient_id,)).fetchone()
        if row is None:
            raise PatientScopeError("Patient is unavailable.")
        return dict(row)

    def create_patient(self, name, demo=False, *, demo_key=None):
        patient_id, stamp = "P" + uuid.uuid4().hex, utc_now()
        with self.memory() as memory:
            memory.create_patient(patient_id, name.strip())
        with self.db() as db:
            db.execute("INSERT INTO people (patient_id,name,demo,created_at,demo_key) VALUES (?,?,?,?,?)",
                       (patient_id, name.strip(), int(demo), stamp, demo_key))
        return self.patient(patient_id)

    def sessions(self):
        with self.db() as db:
            return [self.public_session(dict(r)) for r in db.execute("SELECT s.*,p.name patient_name,(SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) message_count FROM sessions s LEFT JOIN people p ON p.patient_id=s.patient_id ORDER BY updated_at DESC")]

    @staticmethod
    def public_session(row):
        return {k: row[k] for k in ("id", "title", "patient_id", "demo", "created_at", "updated_at", "model_id")} | {"patient_name": row.get("patient_name"), "message_count": row.get("message_count", 0)}

    def session(self, session_id):
        identifier(session_id)
        with self.db() as db:
            row = db.execute("SELECT s.*,p.name patient_name FROM sessions s LEFT JOIN people p ON p.patient_id=s.patient_id WHERE s.id=?", (session_id,)).fetchone()
        if row is None:
            raise PatientScopeError("Conversation is unavailable.")
        return dict(row)

    def create_session(self, patient_id=None, title="New consultation", demo=False, model_id=DEFAULT_MODEL):
        public_model(model_id)
        if patient_id:
            self.patient(patient_id)
        session_id, stamp = "C" + uuid.uuid4().hex, utc_now()
        owner = patient_id or "W" + uuid.uuid4().hex
        with self.memory() as memory:
            if not patient_id:
                memory.create_patient(owner, "General research workspace")
            memory.conversation(owner, session_id, create=True)
        with self.db() as db:
            db.execute("INSERT INTO sessions (id,title,patient_id,owner_id,demo,created_at,updated_at,model_id) VALUES (?,?,?,?,?,?,?,?)",
                       (session_id, title, patient_id, owner, int(demo), stamp, stamp, model_id))
        return self.public_session(self.session(session_id))

    def set_model(self, session_id, model_id):
        public_model(model_id)
        self.session(session_id)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM messages WHERE session_id=? AND state IN ('queued','running')", (session_id,)).fetchone():
                raise ValueError("Wait for the current answer before changing models.")
            db.execute("UPDATE sessions SET model_id=?,updated_at=? WHERE id=?", (model_id, utc_now(), session_id))
        return self.public_session(self.session(session_id))

    def context(self, session_id):
        session = self.session(session_id)
        with self.memory() as memory:
            return {"session": self.public_session(session), "files": memory.documents(session["owner_id"]),
                    "facts": memory.current_facts(session["owner_id"]) if session["patient_id"] else {},
                    "fact_history": memory.fact_history(session["owner_id"]) if session["patient_id"] else []}

    def add_fact(self, patient_id, fact):
        self.patient(patient_id)
        with self.memory() as memory:
            return memory.add_fact(patient_id, fact)

    def register_pdf(self, session_id, path, title):
        session = self.session(session_id)
        with self.memory() as memory:
            return memory.register_pdf(session["owner_id"], path, title=title)

    def pdf_path(self, session_id, document_id):
        identifier(document_id)
        session = self.session(session_id)
        with self.memory() as memory:
            memory.document(session["owner_id"], document_id)
            return memory.document_root / session["owner_id"] / document_id / "original.pdf"

    def settings(self):
        with self.db() as db:
            return json.loads(db.execute("SELECT payload FROM settings WHERE id=1").fetchone()[0])

    def save_settings(self, settings):
        with self.db() as db:
            db.execute("UPDATE settings SET payload=? WHERE id=1", (json.dumps(settings),))
        return settings

    @staticmethod
    def public_message(row):
        return {k: row[k] for k in ("id", "session_id", "request_id", "question", "state", "created_at", "updated_at", "model_id")} | {
            "result": json.loads(row["result"]) if row["result"] else None, "events": json.loads(row["events"])}

    def messages(self, session_id):
        self.session(session_id)
        with self.db() as db:
            return [self.public_message(dict(r)) for r in db.execute("SELECT * FROM messages WHERE session_id=? ORDER BY created_at", (session_id,))]

    def message(self, message_id):
        identifier(message_id)
        with self.db() as db:
            row = db.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        if row is None:
            raise PatientScopeError("Request is unavailable.")
        return self.public_message(dict(row))

    def begin(self, session_id, question, request_id):
        self.session(session_id)
        identifier(request_id)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM messages WHERE request_id=?", (request_id,)).fetchone()
            if old:
                if old["session_id"] != session_id or old["question"] != question:
                    raise ValueError("Request ID was already used for different content.")
                return self.public_message(dict(old)), False
            if db.execute("SELECT 1 FROM messages WHERE session_id=? AND state IN ('queued','running')", (session_id,)).fetchone():
                raise ValueError("Wait for the current answer before sending another question.")
            job, stamp = "R" + uuid.uuid4().hex, utc_now()
            model_id = db.execute("SELECT model_id FROM sessions WHERE id=?", (session_id,)).fetchone()[0]
            db.execute("INSERT INTO messages (id,session_id,request_id,question,state,result,events,created_at,updated_at,model_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (job, session_id, request_id, question, "queued", None, "[]", stamp, stamp, model_id))
            db.execute("UPDATE sessions SET updated_at=?,title=CASE WHEN title='New consultation' THEN ? ELSE title END WHERE id=?", (stamp, question[:62], session_id))
        return self.message(job), True

    def event(self, job_id, label, *, kind="status", **details):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT events FROM messages WHERE id=?", (job_id,)).fetchone()
            events = json.loads(row[0])
            event = {"sequence": len(events) + 1, "label": label, "kind": kind, "at": utc_now(), **details}
            events.append(event)
            db.execute("UPDATE messages SET events=?,updated_at=? WHERE id=?", (json.dumps(events), utc_now(), job_id))
        return event

    def running(self, job_id):
        with self.db() as db:
            db.execute("UPDATE messages SET state='running',updated_at=? WHERE id=?", (utc_now(), job_id))

    def finish(self, job_id, result, *, error=False):
        with self.db() as db:
            db.execute("UPDATE messages SET state=?,result=?,updated_at=? WHERE id=?", ("error" if error else "complete", json.dumps(result), utc_now(), job_id))

    def recover_interrupted(self):
        with self.db() as db:
            rows = [r[0] for r in db.execute("SELECT id FROM messages WHERE state IN ('queued','running')")]
        for job in rows:
            self.event(job, "Request interrupted by an application restart", kind="error")
            self.finish(job, {"error": "The application restarted before this answer was saved. You can send a new request; the earlier request may have incurred a charge."}, error=True)

    def reset_conversations(self):
        """Clear UI chats and public memory history together; retain records/files/audits."""
        with self.db() as db:
            db.execute("ATTACH DATABASE ? AS patient_memory", (str(self.memory_database),))
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM messages WHERE state IN ('queued','running') LIMIT 1").fetchone():
                raise ValueError("Wait for all current answers to finish before resetting conversations.")
            sessions = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            messages = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            interactions = db.execute("SELECT COUNT(*) FROM patient_memory.interactions").fetchone()[0]
            db.execute("DELETE FROM messages")
            db.execute("DELETE FROM sessions")
            db.execute("DELETE FROM patient_memory.interactions")
            db.execute("DELETE FROM patient_memory.conversations")
        return {"cleared_conversations": sessions, "cleared_messages": messages,
                "cleared_memory_interactions": interactions, "patient_records_and_files_preserved": True}
