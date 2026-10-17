"""Local patient-scoped persistence; identities are supplied by application code.

The namespace/read-use-save pattern is adapted from LangChain's MIT memory-agent
reference. This application store is not a LangGraph BaseStore implementation.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from medical_copilot.clinical_graph import ClinicalQuestion, EvidenceRecord, StrictModel
from medical_copilot.pdf_documents import PDFLibrary, ingest_pdf, tokens

VERSION = "patient-memory-v1"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


class PatientScopeError(ValueError):
    """The requested item is unavailable in the selected patient context."""


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Use a bounded identifier containing letters, digits, hyphens or underscores.")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def date_string(value):
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Dates must include a timezone.")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FactInput(StrictModel):
    fact_key: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    value: JsonValue
    unit: str | None = Field(default=None, max_length=40)
    observed_at: AwareDatetime
    source_ref: str = Field(min_length=1, max_length=300)
    origin: str = Field(default="user_provided", pattern=r"^(user_provided|document_entry)$")
    document_id: str | None = Field(default=None, max_length=80)
    source_id: str | None = Field(default=None, max_length=80)
    supersedes_fact_id: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def bounded_payload(self):
        if len(json_text(self.value).encode("utf-8")) > 1500:
            raise ValueError("Keep individual fact values below 1500 UTF-8 bytes.")
        if self.origin == "document_entry" and not (self.document_id and self.source_id):
            raise ValueError("Document entries require an owned document and exact source-view ID.")
        if self.origin == "user_provided" and (self.document_id or self.source_id):
            raise ValueError("User entries cannot imply document verification.")
        return self


class PatientMemory:
    """File-backed SQLite records and owned, immutable PDF ingestion directories."""

    def __init__(self, database, *, document_root=None):
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.document_root = Path(document_root).resolve() if document_root else self.database.parent / (self.database.stem + "-files")
        self.connection = sqlite3.connect(self.database, timeout=15)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.close()
            raise ValueError("Unsupported patient database schema version.")
        if version == 0:
            if self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                self.close()
                raise ValueError("Refuse to modify an unrelated database.")
            self.connection.executescript("""
                CREATE TABLE patients (patient_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    created_at TEXT NOT NULL, UNIQUE(patient_id,conversation_id));
                CREATE TABLE facts (sequence INTEGER PRIMARY KEY AUTOINCREMENT, fact_id TEXT UNIQUE NOT NULL,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id), fact_key TEXT NOT NULL, payload TEXT NOT NULL,
                    observed_at TEXT NOT NULL, recorded_at TEXT NOT NULL, supersedes_fact_id TEXT REFERENCES facts(fact_id));
                CREATE INDEX facts_patient ON facts(patient_id,fact_key,observed_at,sequence);
                CREATE TABLE documents (patient_id TEXT NOT NULL REFERENCES patients(patient_id), document_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL, record_sha256 TEXT NOT NULL, pdf_sha256 TEXT NOT NULL, registered_at TEXT NOT NULL,
                    PRIMARY KEY(patient_id,document_id));
                CREATE TABLE interactions (sequence INTEGER PRIMARY KEY AUTOINCREMENT, interaction_id TEXT UNIQUE NOT NULL,
                    patient_id TEXT NOT NULL, conversation_id TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL,
                    FOREIGN KEY(patient_id,conversation_id) REFERENCES conversations(patient_id,conversation_id));
                CREATE INDEX interactions_patient ON interactions(patient_id,sequence);
                PRAGMA user_version = 1;
            """)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def patient(self, patient_id):
        identifier(patient_id)
        row = self.connection.execute("SELECT patient_id,display_name,created_at FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
        if row is None:
            raise PatientScopeError("Selected patient is not registered.")
        return dict(row)

    def create_patient(self, patient_id, display_name):
        identifier(patient_id)
        if not isinstance(display_name, str) or not display_name.strip() or len(display_name) > 120:
            raise ValueError("Use a nonempty display name below 120 characters.")
        with self.connection:
            self.connection.execute("INSERT INTO patients VALUES (?,?,?)", (patient_id, display_name, utc_now()))
        return self.patient(patient_id)

    def conversation(self, patient_id, conversation_id, *, create=False):
        self.patient(patient_id)
        identifier(conversation_id)
        row = self.connection.execute("SELECT patient_id FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
        if row is not None and row["patient_id"] != patient_id:
            raise PatientScopeError("Conversation is unavailable for the selected patient.")
        if row is None:
            if not create:
                raise PatientScopeError("Conversation is unavailable for the selected patient.")
            with self.connection:
                self.connection.execute("INSERT INTO conversations VALUES (?,?,?)", (conversation_id, patient_id, utc_now()))
        return {"patient_id": patient_id, "conversation_id": conversation_id}

    def add_fact(self, patient_id, value, *, fact_id=None, recorded_at=None):
        self.patient(patient_id)
        fact = FactInput.model_validate(value)
        fact_id = identifier(fact_id or "F" + uuid.uuid4().hex)
        if fact.supersedes_fact_id:
            old = self.connection.execute("SELECT fact_key FROM facts WHERE patient_id=? AND fact_id=?", (patient_id, fact.supersedes_fact_id)).fetchone()
            if old is None or old["fact_key"] != fact.fact_key:
                raise PatientScopeError("Correction must refer to the selected patient's same fact key.")
            if self.connection.execute("SELECT 1 FROM facts WHERE patient_id=? AND supersedes_fact_id=?", (patient_id, fact.supersedes_fact_id)).fetchone():
                raise ValueError("This observation was already superseded; correct its replacement.")
        if fact.document_id:
            record = self.document(patient_id, fact.document_id)
            if fact.source_id not in record["views"]:
                raise PatientScopeError("Source view is unavailable for the selected document.")
        payload = fact.model_dump(mode="json")
        payload["observed_at"] = date_string(fact.observed_at)
        stamp = date_string(recorded_at or utc_now())
        with self.connection:
            self.connection.execute("INSERT INTO facts(fact_id,patient_id,fact_key,payload,observed_at,recorded_at,supersedes_fact_id) VALUES (?,?,?,?,?,?,?)",
                                    (fact_id, patient_id, fact.fact_key, json_text(payload), payload["observed_at"], stamp, fact.supersedes_fact_id))
        return {**payload, "fact_id": fact_id, "recorded_at": stamp}

    def fact_history(self, patient_id, fact_key=None):
        self.patient(patient_id)
        rows = self.connection.execute("SELECT * FROM facts WHERE patient_id=?" + (" AND fact_key=?" if fact_key else "") + " ORDER BY observed_at DESC,recorded_at DESC,sequence DESC",
                                       (patient_id, fact_key) if fact_key else (patient_id,)).fetchall()
        superseded = {r["supersedes_fact_id"] for r in self.connection.execute("SELECT supersedes_fact_id FROM facts WHERE patient_id=?", (patient_id,)) if r["supersedes_fact_id"]}
        return [{**json.loads(r["payload"]), "fact_id": r["fact_id"], "recorded_at": r["recorded_at"], "superseded": r["fact_id"] in superseded,
                 "verification": "Explicit input with recorded origin; not independently clinically verified."} for r in rows]

    def current_facts(self, patient_id):
        current = {}
        for fact in self.fact_history(patient_id):
            if not fact["superseded"] and fact["fact_key"] not in current:
                current[fact["fact_key"]] = fact
        return current

    def register_pdf(self, patient_id, pdf, *, title=None, url=None, year=None):
        self.patient(patient_id)
        pdf = Path(pdf).resolve()
        # Match the parser's bound before hashing an explicitly selected input.
        if not pdf.is_file() or pdf.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("Select an existing PDF below 50 MiB.")
        digest = file_hash(pdf)
        document_id = "D" + digest[:12]
        existing = self.connection.execute("SELECT 1 FROM documents WHERE patient_id=? AND document_id=?", (patient_id, document_id)).fetchone()
        if existing:
            return self.document_metadata(patient_id, document_id)
        destination = (self.document_root / patient_id / document_id).resolve()
        if not destination.is_relative_to(self.document_root.resolve() / patient_id) or destination.exists():
            raise ValueError("Choose a fresh owned ingestion destination.")
        record = ingest_pdf(pdf, destination, title=title, url=url, year=year)
        if record["pdf_sha256"] != digest:
            raise ValueError("Input PDF changed during ingestion.")
        relative = (destination / "document.json").relative_to(self.document_root.resolve()).as_posix()
        with self.connection:
            self.connection.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)", (patient_id, document_id, relative, file_hash(destination / "document.json"), digest, utc_now()))
        return self.document_metadata(patient_id, document_id)

    def document(self, patient_id, document_id):
        self.patient(patient_id)
        row = self.connection.execute("SELECT * FROM documents WHERE patient_id=? AND document_id=?", (patient_id, document_id)).fetchone()
        if row is None:
            raise PatientScopeError("Document is unavailable for the selected patient.")
        path = (self.document_root / row["relative_path"]).resolve()
        expected = (self.document_root / patient_id / document_id / "document.json").resolve()
        if path != expected or not path.is_relative_to(self.document_root.resolve() / patient_id) or file_hash(path) != row["record_sha256"]:
            raise PatientScopeError("Owned ingestion record failed integrity validation.")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["document_id"] != document_id or record["pdf_sha256"] != row["pdf_sha256"] or file_hash(path.parent / "original.pdf") != row["pdf_sha256"]:
            raise PatientScopeError("Owned PDF failed integrity validation.")
        PDFLibrary([record])
        return record

    def document_metadata(self, patient_id, document_id):
        record = self.document(patient_id, document_id)
        return {k: record[k] for k in ("document_id", "title", "pdf_sha256", "page_count", "limits")}

    def documents(self, patient_id):
        self.patient(patient_id)
        ids = [r[0] for r in self.connection.execute("SELECT document_id FROM documents WHERE patient_id=? ORDER BY document_id", (patient_id,))]
        return [self.document_metadata(patient_id, doc_id) for doc_id in ids]

    def save_interaction(self, patient_id, conversation_id, question, response, evidence, *, status, origin, interaction_id=None, created_at=None):
        self.conversation(patient_id, conversation_id)
        question = ClinicalQuestion.model_validate(question).model_dump(mode="json")
        if not isinstance(response, dict) or not isinstance(response.get("type"), str):
            raise ValueError("Persist an explicit public response or application status.")
        allowed = {"type", "answer", "justification", "selected_option", "citations", "limitations", "evidence_status", "question"}
        if set(response) - allowed:
            raise ValueError("Do not persist private/provider fields as a public response.")
        views = {sid: EvidenceRecord.model_validate(view).model_dump(mode="json") for sid, view in evidence.items()}
        if any(sid != view["id"] for sid, view in views.items()):
            raise ValueError("Saved evidence identities must match their keys.")
        interaction_id = identifier(interaction_id or "I" + uuid.uuid4().hex)
        payload = {"question": question, "response": copy.deepcopy(response), "status": status, "output_origin": origin,
                   "evidence": views, "interpretation": "Historical public interaction; assistant statements are not verified patient facts."}
        if len(json_text(payload).encode()) > 262144:
            raise ValueError("Public interaction exceeds the storage bound.")
        stamp = date_string(created_at or utc_now())
        with self.connection:
            self.connection.execute("INSERT INTO interactions(interaction_id,patient_id,conversation_id,created_at,payload) VALUES (?,?,?,?,?)",
                                    (interaction_id, patient_id, conversation_id, stamp, json_text(payload)))
        return self.interaction(patient_id, interaction_id)

    def interaction(self, patient_id, interaction_id):
        self.patient(patient_id)
        row = self.connection.execute("SELECT * FROM interactions WHERE patient_id=? AND interaction_id=?", (patient_id, interaction_id)).fetchone()
        if row is None:
            raise PatientScopeError("Interaction is unavailable for the selected patient.")
        return {**json.loads(row["payload"]), "interaction_id": row["interaction_id"], "conversation_id": row["conversation_id"], "created_at": row["created_at"]}

    def recent_interactions(self, patient_id, limit=3):
        self.patient(patient_id)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= 50:
            raise ValueError("Use a bounded interaction limit.")
        ids = [r[0] for r in self.connection.execute("SELECT interaction_id FROM interactions WHERE patient_id=? ORDER BY sequence DESC LIMIT ?", (patient_id, limit))]
        return [self.interaction(patient_id, item) for item in ids]

    def search_history(self, patient_id, query, limit=3):
        self.patient(patient_id)
        wanted = set(tokens(query))
        # The SQL patient constraint precedes ranking; other patients are not searched.
        rows = self.connection.execute("SELECT interaction_id,payload,sequence FROM interactions WHERE patient_id=? ORDER BY sequence DESC LIMIT 1000", (patient_id,)).fetchall()
        ranked = []
        for row in rows:
            payload = json.loads(row["payload"])
            text = payload["question"]["question"] + " " + str(payload["response"].get("answer", ""))
            score = len(wanted.intersection(tokens(text)))
            if score or not wanted:
                ranked.append((score, row["sequence"], row["interaction_id"]))
        return [self.interaction(patient_id, item) for _, _, item in sorted(ranked, reverse=True)[:limit]]

    def context(self, patient_id, conversation_id, *, max_bytes=8192, recent_turns=3):
        self.conversation(patient_id, conversation_id)
        patient = self.patient(patient_id)
        context = {"patient_id": patient_id, "conversation_id": conversation_id, "display_name": patient["display_name"],
                   "current_facts": self.current_facts(patient_id), "files": self.documents(patient_id), "recent_interactions": [],
                   "history_is_unverified": True, "omitted_recent_interactions": 0, "limits": "Only this patient's registered data. Omitted older history remains tool-accessible."}
        if len(json_text(context).encode()) > max_bytes:
            raise ValueError("Core patient context exceeds its bound; reduce explicit profile/file metadata.")
        for row in self.recent_interactions(patient_id, recent_turns):
            item = {k: row[k] for k in ("interaction_id", "conversation_id", "created_at", "question", "response", "status", "output_origin")}
            candidate = copy.deepcopy(context)
            candidate["recent_interactions"].append(item)
            if len(json_text(candidate).encode()) > max_bytes:
                context["omitted_recent_interactions"] += 1
            else:
                context = candidate
        if len(json_text(context).encode()) > max_bytes:
            raise ValueError("Bounded patient context could not be assembled.")
        return context

    def backup(self, destination):
        destination = Path(destination)
        if destination.exists():
            raise ValueError("Preserve snapshots; choose a fresh destination.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(destination) as target:
            self.connection.backup(target)
