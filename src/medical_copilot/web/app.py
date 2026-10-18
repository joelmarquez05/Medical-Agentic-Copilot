"""Loopback FastAPI service and built React application, with live job events."""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from medical_copilot.clinical_graph import StrictModel
from medical_copilot.patient_memory import FactInput, PatientScopeError
from medical_copilot.web.agent import WorkspaceLimits
from medical_copilot.web.models import DEFAULT_MODEL, PROFILES, ModelId, public_model
from medical_copilot.web.examples import EXAMPLES, prepare_clinical_patients, prepare_example
from medical_copilot.web.runtime import run_live
from medical_copilot.web.store import WorkspaceStore

ROOT = Path(__file__).resolve().parents[3]


class NewSession(StrictModel):
    patient_id: str | None = None
    model_id: ModelId = DEFAULT_MODEL


class ModelChoice(StrictModel):
    model_id: ModelId


class ResetConversations(StrictModel):
    confirm: Literal[True]


class NewPatient(StrictModel):
    name: str = Field(min_length=1, max_length=120)

class NewQuestion(StrictModel):
    question: str = Field(min_length=1, max_length=12000)
    request_id: str = Field(min_length=1, max_length=80)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Write a question first.")
        return value.strip()


class Appearance(StrictModel):
    name: str = Field(min_length=1, max_length=50)
    accent: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    font: str = Field(pattern=r"^(editorial|modern)$")
    font_scale: float = Field(ge=.9, le=1.25)
    density: str = Field(pattern=r"^(comfortable|compact)$")
    logo: str | None = Field(default=None, max_length=300000)

    @field_validator("logo")
    @classmethod
    def image_only(cls, value):
        if value:
            prefix, data = value.split(",", 1)
            if prefix not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
                raise ValueError("Choose a PNG, JPEG or WebP logo.")
            decoded = base64.b64decode(data, validate=True)
            if not decoded.startswith((b"\x89PNG", b"\xff\xd8\xff", b"RIFF")):
                raise ValueError("Invalid image data.")
        return value


def prepare_examples(store, root):
    _, people = prepare_clinical_patients(store, root, DEFAULT_MODEL)
    research = next((s for s in store.sessions() if s["demo"] and not s["patient_id"] and s["title"] == "Nutrition evidence"), None)
    if not research:
        research = store.create_session(title="Nutrition evidence", demo=True)
    store.register_pdf(research["id"], root / "demo/01_espen_cancer_2021.pdf", "ESPEN practical guideline: Clinical Nutrition in cancer (2021)")
    return {"research_session": research["id"], "patients": people}


def create_app(directory=None, runner=None, *, testing=False):
    store = WorkspaceStore(directory or os.environ.get("MEDICAL_COPILOT_DATA_DIR", ROOT / "runtime"))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="medical-copilot")
    execute = runner or run_live

    @asynccontextmanager
    async def lifespan(app):
        store.recover_interrupted()
        yield
        executor.shutdown(wait=False)

    app = FastAPI(title="Medical Copilot", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", *(["testserver"] if testing else [])])

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        allowed = {str(request.base_url).rstrip("/"), "http://localhost:5173", "http://127.0.0.1:5173"}
        if request.method not in {"GET", "HEAD", "OPTIONS"} and (request.headers.get("sec-fetch-site") == "cross-site" or (origin and origin not in allowed)):
            return JSONResponse({"detail": "Open the application locally to perform this action."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(PatientScopeError)
    async def unavailable(request, exc):
        return JSONResponse({"detail": "This record is unavailable in the selected context."}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)[:300]}, status_code=400)

    @app.get("/api/bootstrap")
    def bootstrap():
        return {"appearance": store.settings(), "patients": store.patients(), "sessions": store.sessions(),
                "agent_limits": {**WorkspaceLimits().model_dump(mode="json"), "graph_recursion_limit": WorkspaceLimits().graph_recursion_limit},
                "model": public_model(DEFAULT_MODEL), "models": [public_model(model_id) for model_id in PROFILES],
                "examples": EXAMPLES}

    @app.post("/api/sessions")
    def new_session(body: NewSession):
        return store.create_session(body.patient_id, model_id=body.model_id)

    @app.put("/api/sessions/{session_id}/model")
    def change_model(session_id: str, body: ModelChoice):
        try:
            return store.set_model(session_id, body.model_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str):
        return {**store.context(session_id), "messages": store.messages(session_id)}

    @app.post("/api/patients")
    def new_patient(body: NewPatient):
        return store.create_patient(body.name)

    @app.post("/api/patients/{patient_id}/facts")
    def add_fact(patient_id: str, body: FactInput):
        return store.add_fact(patient_id, body.model_dump(mode="json"))

    @app.put("/api/appearance")
    def appearance(body: Appearance):
        return store.save_settings(body.model_dump(mode="json"))

    @app.post("/api/reset")
    def reset_conversations(body: ResetConversations):
        try:
            return store.reset_conversations()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/sessions/{session_id}/documents")
    async def upload(session_id: str, file: UploadFile = File()):
        store.session(session_id)
        uploads = store.directory / "uploads"
        uploads.mkdir(exist_ok=True)
        handle, name = tempfile.mkstemp(suffix=".pdf", dir=uploads)
        path = Path(name)
        try:
            size = 0
            with os.fdopen(handle, "wb") as output:
                while data := await file.read(1024 * 1024):
                    size += len(data)
                    if size > 50 * 1024 * 1024:
                        raise HTTPException(413, "Choose a PDF smaller than 50 MiB.")
                    output.write(data)
            if path.read_bytes()[:5] != b"%PDF-":
                raise HTTPException(400, "Choose a valid PDF file.")
            title = (file.filename or "Attached document").replace("\\", "/").split("/")[-1].removesuffix(".pdf")[:180]
            def validate_and_ingest():
                import pymupdf
                # A failed file-based constructor can retain a Windows file handle.
                # Validate the container in memory before the page-aware file parser.
                with pymupdf.open(stream=path.read_bytes(), filetype="pdf") as document:
                    if document.needs_pass or not 0 < len(document) <= 250:
                        raise ValueError("Unsupported PDF container.")
                return store.register_pdf(session_id, path, title)
            return await asyncio.to_thread(validate_and_ingest)
        except (ValueError, RuntimeError):
            raise HTTPException(400, "This PDF could not be ingested. Use an unencrypted PDF with selectable text, up to 250 pages.") from None
        finally:
            await file.close()
            path.unlink(missing_ok=True)

    @app.get("/api/sessions/{session_id}/documents/{document_id}/file")
    def pdf_file(session_id: str, document_id: str):
        return FileResponse(store.pdf_path(session_id, document_id), media_type="application/pdf", content_disposition_type="inline")

    def work(job):
        store.running(job["id"])
        emit = lambda label, **details: store.event(job["id"], label, **details)
        try:
            result = execute(store, job, emit, ROOT)
            store.finish(job["id"], result)
        except Exception as exc:
            name = type(exc).__name__
            message = "The model request could not complete. Your question is saved; you can try again."
            if name in {"BudgetLimitExceeded", "BudgetBlocked"}:
                message = "The configured cumulative spending allowance has been reached. No further model calls were sent after the guard declined."
            elif name in {"ValueError", "ConfigurationError"}:
                message = "The model configuration or workspace context could not be used. Check the configured endpoint and reduce oversized context before retrying."
            emit(message, kind="error")
            store.finish(job["id"], {"error": message, "error_type": name}, error=True)

    @app.post("/api/sessions/{session_id}/messages", status_code=202)
    def ask(session_id: str, body: NewQuestion):
        try:
            job, fresh = store.begin(session_id, body.question, body.request_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        if fresh:
            store.event(job["id"], "Request queued", kind="queued")
            executor.submit(work, job)
        return store.message(job["id"])

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, last_event_id: str | None = Header(default=None)):
        store.message(run_id)
        try:
            after = int(last_event_id or 0)
        except ValueError:
            after = 0

        async def stream():
            position = after
            while True:
                row = await asyncio.to_thread(store.message, run_id)
                for event in row["events"]:
                    if event["sequence"] > position:
                        position = event["sequence"]
                        yield "id: " + str(position) + "\ndata: " + json.dumps({"type": "activity", "event": event}) + "\n\n"
                if row["state"] in {"complete", "error"}:
                    yield "data: " + json.dumps({"type": "complete", "message": row}) + "\n\n"
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(.6)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    @app.post("/api/examples/prepare")
    def examples():
        return prepare_examples(store, ROOT)

    @app.post("/api/examples/{example_id}/prepare")
    def guided_example(example_id: str, body: ModelChoice):
        return prepare_example(store, ROOT, example_id, body.model_id)

    distribution = ROOT / "web/dist"
    if (distribution / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=distribution / "assets"), name="assets")

    @app.get("/")
    def index():
        if not (distribution / "index.html").exists():
            raise HTTPException(503, "Build the frontend with npm run build in web/ first.")
        return FileResponse(distribution / "index.html", headers={"Cache-Control": "no-cache"})

    return app
