"""Replay saved PDF responses, schemas and tools offline, without credentials."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from medical_copilot.clinical_evaluation import score_record
from medical_copilot.clinical_graph import ClinicalQuestion, save_trace
from medical_copilot.clinical_resilient import PhaseBackend, ResilientLimits, final_schema
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.models import ModelClient, Redactor
from medical_copilot.pdf_agent import PDF_SYSTEM, PDFClinicalAgent, PDFJSONModel
from medical_copilot.pdf_documents import PDFLibrary, PDFTools
from verify_clinical_resilient import BoundReplay
from verify_clinical_literature import without_transient


def verify(run):
    root = Path(__file__).resolve().parents[1]
    manifest = read_json(run / "manifest.json")
    assert manifest["finished_at_utc"] and manifest["requirement"] == 6
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(run / "source-snapshot" / name) == file_hash(root / "src/medical_copilot" / name) == digest
    for filename, field in [("document.json", "document_sha256"), ("config.json", "config_sha256"), ("protocol.md", "protocol_sha256"),
                            ("policy.json", "policy_sha256"), ("qualitative-rubric.json", "rubric_sha256"), ("questions.json", "questions_sha256"),
                            ("selection-manifest.json", "selection_manifest_sha256")]:
        # The public config is serialized differently; its origin hash is separately checked.
        if filename != "config.json":
            assert file_hash(run / filename) == manifest[field], filename
    assert read_json(run / "config.json") == load_config(root / "config/models/openrouter-gemma-4-reference.json").public_dict()
    assert manifest["effective_system_sha256"] == hashlib.sha256(PDF_SYSTEM.encode()).hexdigest()
    assert read_json(run / "system-instructions.json")["effective_system"] == PDF_SYSTEM
    config = load_config(root / "config/models/openrouter-gemma-4-reference.json")
    library = PDFLibrary.load([run / "document.json"])
    questions = {q["question_id"]: q for q in read_json(run / "questions.json")["questions"]}
    keys = {k["question_id"]: k for k in read_json(run / "answer-key.json")["keys"]} if manifest["stage"] == "main" else {}
    if keys:
        assert file_hash(run / "answer-key.json") == manifest["answer_key_sha256"]
    results = []
    for arm in manifest["conditions"]:
        directory = run / arm
        records = [read_json(p) for p in sorted((directory / "answers").glob("*.json"))]
        assert {r["question"]["question_id"] for r in records} == set(manifest["question_ids"])
        audit = read_json(directory / "usage-audit.json")["records"]
        assert audit == [a for row in records for a in row["response_audit"]]
        for row in records:
            identifier = row["question"]["question_id"]
            assert row["question"] == ClinicalQuestion.model_validate(questions[identifier]).model_dump(mode="json")
            limits = ResilientLimits(**row["configuration"]["limits"])
            expected_limits = dict(manifest["limits"])
            if manifest["stage"] == "probe":
                expected_limits["max_research_turns"] = 0
            assert limits.model_dump() == expected_limits
            first = json.loads(row["model_calls"][0]["user_prompt"])
            assert first["history"] == []
            assert first["evidence_registry"] == (library.all_views() if arm == "B-full-context" else {})
            for call in row["model_calls"]:
                request = json.loads(call["user_prompt"])
                assert call["system_prompt"] == PDF_SYSTEM and request["question"] == row["question"]
                assert call["stage"] == "pdf_" + request["phase"] and len(call["attempts"]) == 1
                assert len((call["system_prompt"] + call["user_prompt"]).encode()) <= limits.max_input_bytes
                assert not {"answer_idx", "reference_facts", "supporting_view_ids", "reference_option"}.intersection(request["question"])
                assert all(name in {"search_pdf", "read_pdf_page"} for name in [t["name"] for t in request["tools"]])
                if request["phase"] == "finalize":
                    assert request["tools"] == [] and request["remaining"]["tool_actions"] == 0
                    assert request["action_schema"] == final_schema(request["question"], request["evidence_registry"])
                else:
                    assert [t["name"] for t in request["tools"]] == (["search_pdf", "read_pdf_page"] if arm == "C-pdf-tools" else [])
                if arm == "A-no-document":
                    assert request["evidence_registry"] == {}
                if arm == "B-full-context":
                    assert request["evidence_registry"] == library.all_views()
            assert all(c["status"] != "error" for c in row["model_calls"]), "Operational error replay needs an explicit failure fixture; do not invent a successful response."
            backend = BoundReplay(row["model_calls"])
            phased = PhaseBackend(backend, True)
            client = ModelClient(phased, config, Redactor())
            model = PDFJSONModel(client, lambda request: None)
            tools = PDFTools(library)
            replayed = PDFClinicalAgent(model, library, arm, limits=limits, pdf_tools=tools if arm == "C-pdf-tools" else None).run(row["question"])
            for field in ["status", "response", "history", "evidence", "tool_actions", "model_turns", "issues", "trace",
                          "research_turns", "final_attempts", "provider_failures", "closing_reason", "output_origin", "document_tools", "initial_evidence_source_count"]:
                assert without_transient(replayed[field]) == without_transient(row[field]), (arm, identifier, field)
            assert phased.formats == row["transport_formats"] and backend.index == len(row["model_calls"])
            assert row["model_turns"] <= limits.max_model_turns and row["tool_actions"] <= limits.max_tool_actions
            assert len(row["response_audit"]) == len(row["model_calls"])
            for sid, view in row["evidence"].items():
                assert library.all_views()[sid] == view
            results.append({"condition": arm, "question_id": identifier, "model_responses": backend.index,
                            "native_final_calls": len(backend.schemas), "pdf_operations": len(tools.events), "answer_sha256": file_hash(directory / "answers" / (identifier + ".json"))})
        summary = read_json(directory / "summary.json")
        assert summary["recorded"] == len(records) == summary["terminal_message_coverage"]
        assert summary["provider_requests"] == len(audit)
        assert math.isclose(sum(r.get("cost_usd") or 0 for r in audit), summary["cost_usd"], abs_tol=1e-12)
        if keys:
            assert read_json(directory / "scores.json")["scores"] == [score_record(r, keys[r["question"]["question_id"]]) for r in records]
    suspicious = re.compile(r"sk-or-v1-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9_-]{15,}")
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".py"}:
            raw = path.read_text(encoding="utf-8")
            # Source code contains redaction field names but never provider thought values.
            assert not suspicious.search(raw), "Credential-shaped content withheld."
            if path.suffix == ".json":
                assert '"reasoning_content"' not in raw and '"reasoning_text"' not in raw
    return {"passed": True, "run_id": run.name, "episodes": results, "new_provider_requests": 0, "credentials_read": 0,
            "all_initial_and_delivered_contexts_replayed": True, "no_pubmed_tools_or_gold_fields_in_requests": True,
            "interpretation": "Offline operational provenance, not clinical quality or an independent factual judge."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve())
    save_trace(args.output, result)
    print(json.dumps({"passed": result["passed"], "episodes": len(result["episodes"]), "replayed_model_responses": sum(r["model_responses"] for r in result["episodes"]), "new_provider_requests": 0}))

if __name__ == "__main__":
    main()
