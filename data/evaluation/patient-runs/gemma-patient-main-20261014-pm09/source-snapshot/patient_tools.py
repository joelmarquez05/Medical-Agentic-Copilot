"""Bounded memory/PDF tools whose patient identity is never model-controlled."""
import copy
import json

from pydantic import Field

from medical_copilot.clinical_graph import StrictModel, ToolDefinition, ToolOutput
from medical_copilot.patient_memory import PatientScopeError, json_text
from medical_copilot.pdf_documents import PDFLibrary, PDFTools


class SearchHistoryArgs(StrictModel):
    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=3, ge=1, le=3, strict=True)


class ReadInteractionArgs(StrictModel):
    interaction_id: str = Field(min_length=1, max_length=80)
    restore_source_ids: list[str] = Field(default_factory=list, max_length=3)


class ReadFactsArgs(StrictModel):
    fact_key: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    max_results: int = Field(default=5, ge=1, le=5, strict=True)


class PatientTools:
    def __init__(self, memory, patient_id):
        memory.patient(patient_id)
        self.memory, self.patient_id, self.events = memory, patient_id, []
        metadata = memory.documents(patient_id)
        self.library = PDFLibrary([memory.document(patient_id, row["document_id"]) for row in metadata]) if metadata else None
        self.pdf = PDFTools(self.library) if self.library else None

    def definitions(self):
        tools = [
            ToolDefinition("search_patient_history", "Search only the selected patient's older public questions/answers by keywords. Up to three excerpts with interaction IDs. Historical assistant claims are unverified; exact phrasing and lexical coverage can matter. Use read_patient_interaction for full content or explicit restoration of actually saved source views.", SearchHistoryArgs, self.search),
            ToolDefinition("read_patient_interaction", "Read a known interaction belonging to this patient. Optionally restore up to three source IDs saved in that interaction; their original view text is actually delivered and its historical provenance is recorded. This is not a new PubMed search. Never guess or request another patient's interaction.", ReadInteractionArgs, self.read),
            ToolDefinition("read_patient_facts", "Read dated observations/corrections for one key from the selected patient's explicit facts. Up to five entries, newest observation first, marked if superseded. Distinguish observation date from recording date. These are case data, not medical publications.", ReadFactsArgs, self.facts),
        ]
        if self.pdf:
            for definition in self.pdf.definitions():
                def execute(args, original=definition.execute):
                    # Recheck ownership and integrity at execution, not just graph setup.
                    self.memory.document(self.patient_id, args.document_id)
                    return original(args)
                tools.append(ToolDefinition(definition.name, definition.description, definition.arguments_schema, execute))
        return tools

    def _operation(self, name, args, operation):
        event = {"name": name, "arguments": args.model_dump(mode="json")}
        try:
            output = operation()
            event.update(status="ok", generation_view=output.model_dump(mode="json"))
            self.events.append(event)
            return output
        except Exception as exc:
            event.update(status="rejected", error_type=type(exc).__name__)
            self.events.append(event)
            raise

    def search(self, args):
        def perform():
            rows = self.memory.search_history(self.patient_id, args.query, args.max_results)
            items = [{"interaction_id": r["interaction_id"], "conversation_id": r["conversation_id"], "created_at": r["created_at"],
                      "question_excerpt": r["question"]["question"][:600], "answer_excerpt": r["response"].get("answer", r["response"].get("question", ""))[:700],
                      "excerpt_truncated": len(r["question"]["question"]) > 600 or len(r["response"].get("answer", "")) > 700,
                      "saved_source_ids": list(r["evidence"]), "status": r["status"], "output_origin": r["output_origin"]} for r in rows]
            return ToolOutput(content=json_text({"results": items, "search_scope": "Only selected patient; newest 1000 interactions; lexical ranking.",
                                                 "limitation": "Historical assistant outputs are not independently verified facts."}))
        return self._operation("search_patient_history", args, perform)

    def read(self, args):
        def perform():
            row = self.memory.interaction(self.patient_id, args.interaction_id)
            if len(set(args.restore_source_ids)) != len(args.restore_source_ids) or any(sid not in row["evidence"] for sid in args.restore_source_ids):
                raise PatientScopeError("Requested source view is unavailable in this patient's selected interaction.")
            metadata = {k: row[k] for k in ("interaction_id", "conversation_id", "created_at", "question", "response", "status", "output_origin", "interpretation")}
            metadata["restored_source_ids"] = args.restore_source_ids
            metadata["source_provenance"] = "Original saved views explicitly delivered now; no new literature search implied."
            # Reject over-large reads instead of silently discarding decisive facts.
            return ToolOutput(content=json_text(metadata), sources=[row["evidence"][sid] for sid in args.restore_source_ids])
        return self._operation("read_patient_interaction", args, perform)

    def facts(self, args):
        def perform():
            facts = self.memory.fact_history(self.patient_id, args.fact_key)
            return ToolOutput(content=json_text({"fact_key": args.fact_key, "observations": facts[:args.max_results], "omitted_observations": max(0, len(facts) - args.max_results),
                                                 "order": "Observation time descending, then recording time. Superseded entries must not be used as current values."}))
        return self._operation("read_patient_facts", args, perform)

    def public_record(self):
        return {"memory_events": copy.deepcopy(self.events), "pdf_events": copy.deepcopy(self.pdf.events) if self.pdf else [],
                "scope": "Application-bound patient; no model-facing patient ID argument."}
