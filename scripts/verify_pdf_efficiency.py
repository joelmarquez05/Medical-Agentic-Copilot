"""Offline replay of document-required article answers, tools, fields and schemas."""
import argparse
import hashlib
import json
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.clinical_resilient import PhaseBackend, ResilientLimits
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.models import ModelClient, Redactor
from medical_copilot.pdf_documents import PDFLibrary, PDFTools
from medical_copilot.pdf_efficiency_agent import ARTICLE_SYSTEM, ArticleJSONModel, ArticlePDFAgent, VERSION
from medical_copilot.pdf_efficiency_evaluation import score_fields
from verify_clinical_resilient import BoundReplay
from verify_clinical_literature import without_transient

def verify(run):
    root = Path(__file__).resolve().parents[1]
    manifest = read_json(run / "manifest.json")
    assert manifest["version"] == VERSION and manifest["finished_at_utc"]
    assert set(manifest["conditions"]) <= {"B-full-context", "C-pdf-tools"}
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(run / "source-snapshot" / name) == file_hash(root / "src/medical_copilot" / name) == digest
    for name, field in [("document.json", "document_sha256"), ("questions.json", "questions_sha256"), ("policy.json", "policy_sha256"),
        ("protocol.md", "protocol_sha256"), ("selection-manifest.json", "selection_manifest_sha256"), ("answer-key.json", "answer_key_sha256"),
        ("field-evidence-key.json", "field_evidence_key_sha256"), ("qualitative-rubric.json", "rubric_sha256")]:
        assert file_hash(run / name) == manifest[field]
    assert hashlib.sha256(ARTICLE_SYSTEM.encode()).hexdigest() == manifest["effective_system_sha256"]
    config = load_config(root / "config/models/openrouter-gemma-4-reference.json")
    assert read_json(run / "config.json") == config.public_dict()
    library = PDFLibrary.load([run / "document.json"])
    questions = {q["question_id"]: q for q in read_json(run / "questions.json")["questions"]}
    keys = {q["question_id"]: q for q in read_json(run / "answer-key.json")["keys"]}
    support = {q["question_id"]: q for q in read_json(run / "field-evidence-key.json")["keys"]}
    results = []
    for arm in manifest["conditions"]:
        directory = run / arm
        rows = [read_json(p) for p in sorted((directory / "answers").glob("*.json"))]
        assert {r["question"]["question_id"] for r in rows} == set(manifest["question_ids"])
        assert read_json(directory / "usage-audit.json")["records"] == [a for r in rows for a in r["response_audit"]]
        for row in rows:
            assert row["question"] == questions[row["question"]["question_id"]]
            limits = ResilientLimits(**row["configuration"]["limits"])
            expected = dict(manifest["limits"])
            if manifest["stage"] == "probe":
                expected["max_research_turns"] = 0
            assert limits.model_dump() == expected
            calls = row["model_calls"]
            assert all(c["status"] != "error" for c in calls), "Preserved provider failures require an explicit failure replay fixture."
            for n, call in enumerate(calls):
                request = json.loads(call["user_prompt"])
                assert call["system_prompt"] == ARTICLE_SYSTEM and call["stage"] == "pdf_article_" + request["phase"]
                assert len((call["system_prompt"] + call["user_prompt"]).encode()) <= limits.max_input_bytes
                assert request["question"] == row["question"] and len(call["attempts"]) == 1
                assert not set(request["question"]) - {"question_id", "question", "options"}
                if n == 0:
                    assert request["history"] == []
                if arm == "B-full-context":
                    assert request["evidence_registry"] == library.all_views() and request["tools"] == []
                elif n == 0:
                    assert request["evidence_registry"] == {}
                if request["phase"] == "finalize":
                    assert request["tools"] == []
            backend = BoundReplay(calls)
            phased = PhaseBackend(backend, True)
            client = ModelClient(phased, config, Redactor())
            model = ArticleJSONModel(client, lambda request: None)
            tools = PDFTools(library)
            replayed = ArticlePDFAgent(model, library, arm, limits=limits, pdf_tools=tools if arm == "C-pdf-tools" else None).run(row["question"])
            for field in ["status", "response", "history", "evidence", "tool_actions", "model_turns", "issues", "trace",
                "research_turns", "final_attempts", "provider_failures", "closing_reason", "output_origin", "document_tools", "initial_evidence_source_count"]:
                assert without_transient(replayed[field]) == without_transient(row[field]), (arm, row["question"]["question_id"], field)
            assert phased.formats == row["transport_formats"] and model.local_normalizations == row["answer_object_normalizations"]
            assert backend.index == len(calls)
            for sid, view in row["evidence"].items():
                assert view == library.all_views()[sid]
            if row["status"] == "completed":
                assert row["response"]["citations"] and set(row["response"]["citations"]) <= set(row["evidence"])
                if arm == "C-pdf-tools":
                    assert tools.events
            results.append({"condition": arm, "question_id": row["question"]["question_id"], "model_responses": backend.index,
                "pdf_operations": len(tools.events), "answer_sha256": file_hash(directory / "answers" / (row["question"]["question_id"] + ".json"))})
        if manifest["stage"] == "main":
            assert read_json(directory / "scores.json")["scores"] == [score_fields(r, keys[r["question"]["question_id"]], support[r["question"]["question_id"]]) for r in rows]
    return {"passed": True, "run_id": run.name, "episodes": results, "new_model_requests": 0, "credentials_read": 0,
        "no_knowledge_only_condition": True, "identical_complete_B_and_delivered_C_views": True, "no_gold_answer_fields_in_requests": True,
        "interpretation": "Exact saved operational replay and code-based field checks, not independent clinical validation."}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve())
    save_trace(args.output, result)
    print(json.dumps({"passed": True, "episodes": len(result["episodes"]), "model_responses": sum(r["model_responses"] for r in result["episodes"]), "new_model_requests": 0}))

if __name__ == "__main__":
    main()
