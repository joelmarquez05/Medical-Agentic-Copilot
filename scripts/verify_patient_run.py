"""Replay patient memory, model responses and PubMed locally without API access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

from medical_copilot.clinical_evaluation import DEFAULT_PROFILE
from medical_copilot.clinical_graph import save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.clinical_resilient import PhaseBackend, ResilientLimits
from medical_copilot.config import load_config
from medical_copilot.evaluation import read_json
from medical_copilot.models import ModelClient, Redactor
from medical_copilot.patient_agent import PATIENT_SYSTEM, PatientAgent, PatientJSONModel
from medical_copilot.patient_memory import PatientMemory, file_hash
from medical_copilot.pubmed import PubMedClient, PubMedConfig
from verify_clinical_literature import HTTPReplay, without_transient
from verify_clinical_resilient import BoundReplay

ROOT = Path(__file__).resolve().parents[1]


class PatientReplay(BoundReplay):
    def invoke(self, messages):
        call = self.calls[self.index]
        if call["status"] == "error":
            assert messages[0].content == call["system_prompt"] and messages[1].content == call["user_prompt"]
            self.index += 1
            name = call.get("error_type") or "RecordedProviderFailure"
            assert name.isidentifier()
            error = type(name, (Exception,), {})
            raise error("Offline replay of the saved provider failure; no private body.")
        return super().invoke(messages)


def verify_episode(run):
    manifest = read_json(run / "manifest.json")
    assert manifest["finished_at_utc"] and manifest["requirement"] == 7
    for name, expected in manifest["source_hashes"].items():
        assert file_hash(run / "source-snapshot" / name) == expected
        # The v1.1 generation-only classifier adds the observed connection alias.
        # Historical routing is replayed from the original saved decision below.
        if name != "patient_runtime.py":
            assert file_hash(ROOT / "src/medical_copilot" / name) == expected
    assert manifest["effective_system_sha256"] == hashlib.sha256(PATIENT_SYSTEM.encode()).hexdigest()
    assert file_hash(ROOT / DEFAULT_PROFILE) == manifest["config_sha256"]
    assert read_json(run / "system-instructions.json")["effective_system"] == PATIENT_SYSTEM
    results = []
    for episode in manifest["episodes"]:
        row = read_json(run / "answers" / (episode["question_id"] + ".json"))
        snapshot = run / episode["memory_snapshot"]
        assert file_hash(snapshot) == episode["memory_snapshot_sha256"] == row["memory_snapshot_sha256"]
        backend = PatientReplay(row["model_calls"])
        phased = PhaseBackend(backend, native_final=True)
        client = ModelClient(phased, load_config(ROOT / DEFAULT_PROFILE), Redactor())
        recovery = {e["turn"]: e.get("recoverable", False) for e in row["trace"] if e.get("status") == "model_error"}
        model = PatientJSONModel(client, lambda request: None, lambda: recovery.get(backend.index, False))
        http = HTTPReplay(row["literature"]["http_requests"], run / "http-snapshots")
        literature = ClinicalLiteratureTools(PubMedClient(PubMedConfig.model_validate(read_json(run / "pubmed-config.json")), transport=http))
        with tempfile.TemporaryDirectory(prefix="medical-patient-replay-") as directory:
            temporary = Path(directory).resolve()
            assert temporary.is_relative_to(Path(tempfile.gettempdir()).resolve())
            database = temporary / "memory.sqlite3"
            shutil.copyfile(snapshot, database)
            with PatientMemory(database, document_root=episode["document_root"]) as memory:
                agent = PatientAgent(model, memory, episode["patient_id"], episode["conversation_id"], literature=literature,
                                     memory_enabled=episode["memory_enabled"], require_reference=episode["require_reference"],
                                     limits=ResilientLimits.model_validate(episode["limits"]), interaction_id=episode["interaction_id"], created_at=episode["created_at"])
                with patch("medical_copilot.pubmed.time.sleep", lambda seconds: None):
                    actual = agent.run(row["question"])
                for field in ("status", "response", "history", "evidence", "trace", "model_turns", "tool_actions", "research_turns",
                              "final_attempts", "closing_reason", "provider_failures", "output_origin", "patient_context", "memory_write", "patient_tools", "literature"):
                    assert without_transient(actual[field]) == without_transient(row[field]), (episode["question_id"], field)
                assert phased.formats == row["transport_formats"] and backend.index == len(row["model_calls"])
                assert http.index == len(row["literature"]["http_requests"])
                assert getattr(model, "normalizations", []) == row["answer_object_normalizations"]
                if row["memory_write"]["saved"]:
                    saved = memory.interaction(episode["patient_id"], episode["interaction_id"])
                    assert saved["response"] == row["response"] and saved["evidence"] == row["evidence"]
        results.append({"question_id": episode["question_id"], "answer_sha256": file_hash(run / "answers" / (episode["question_id"] + ".json")),
                        "replayed_model_responses": backend.index, "replayed_http_requests": http.index, "memory_snapshot_replayed": True,
                        "new_source_views_restored_only_after_tool_delivery": True,
                        "historical_generation_runtime_sha256": manifest["source_hashes"]["patient_runtime.py"],
                        "current_generation_runtime_sha256": file_hash(ROOT / "src/medical_copilot/patient_runtime.py"),
                        "provider_failure_routing_replayed_as_recorded": bool(recovery)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = read_json(run / "manifest.json")
    results = []
    if manifest.get("aggregate_only"):
        assert manifest["finished_at_utc"]
        for child in manifest["subruns"]:
            results.extend(verify_episode(ROOT / child["run_dir"]))
    else:
        results = verify_episode(run)
    receipt = {"passed": True, "run_id": run.name, "episodes": results, "replayed_model_responses": sum(r["replayed_model_responses"] for r in results),
               "replayed_http_requests": sum(r["replayed_http_requests"] for r in results), "new_inference_requests": 0,
               "credential_files_read": 0, "new_pubmed_requests": 0, "interpretation": "Exact prompt/schema/action/source/memory replay; not an independent clinical judgment."}
    if args.output:
        save_trace(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
