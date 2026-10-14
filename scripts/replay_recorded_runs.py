"""Offline engineering replay of stored visible responses, never new inference."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from medical_copilot.config import ModelConfig
from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.graph import MedicalCopilot
from medical_copilot.models import ModelClient

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    if isinstance(value, dict):
        return {key: canonical(child) for key, child in value.items() if key not in {"elapsed_seconds", "started_at_utc"}}
    if isinstance(value, list):
        return [canonical(child) for child in value]
    return value


def prompt_value(text):
    try:
        return canonical(json.loads(text))
    except ValueError:
        return text


class RecordedBackend:
    def __init__(self, calls):
        self.calls = list(calls)
        self.position = 0

    def invoke(self, messages):
        call = self.calls[self.position]
        self.position += 1
        assert messages[0].content == call["system_prompt"], "System prompt changed during replay."
        assert prompt_value(messages[1].content) == prompt_value(call["user_prompt"]), "Substantive model input changed during replay."
        if call["status"] == "error":
            error = type(call["error_type"], (RuntimeError,), {})
            raise error("Recorded provider failure; this replay makes no network request.")
        return AIMessage(content=call["response"], usage_metadata=call.get("usage") or None, response_metadata=call.get("response_metadata") or {})


def main():
    results = []
    run_ids = ["gemini-internal-final-20261010", "nvidia-gpt-oss-20b-demo-20261010", "nvidia-kimi-k3-demo-20261010", "nvidia-deepseek-v4.1-flash-demo-20261010"]
    for run_id in run_ids:
        run = ROOT / "data/evaluation/runs" / run_id
        manifest = read_json(run / "manifest.json")
        settings = ModelConfig.model_validate(manifest["model_config"]).model_copy(update={"min_interval_seconds": 0, "rate_limit_retries": 0})
        for path in sorted((run / "answers").glob("*.json")):
            observed = read_json(path)
            backend = RecordedBackend(observed["model_calls"])
            replay = MedicalCopilot(ModelClient(backend, settings)).run(observed["question"])
            assert backend.position == len(observed["model_calls"])
            assert replay["status"] == observed["status"]
            if observed["status"] != "error":
                for key in ["answer", "sources", "trace", "revision_count", "mechanical_issues"]:
                    assert replay[key] == observed[key], f"Replay changed {run_id}/{path.stem}/{key}."
            else:
                # Original failure outputs stay unchanged; new partial state is separate.
                partial = {key: replay[key] for key in ["status", "error", "trace", "plan", "tool_results", "drafts", "reviews", "partial_answer", "revision_count"] if key in replay}
                partial["provenance"] = "Offline replay of the saved visible responses and saved failure type. No new medical answer or API response was generated."
                partial["original_answer_file_sha256"] = file_hash(path)
                write_json(run / "replayed-partial-state.json", partial)
            results.append({"run_id": run_id, "question_id": path.stem, "status": replay["status"], "model_inputs_matched": True, "final_answer_unchanged": observed["status"] != "error", "completed_tool_calls_preserved": len(replay.get("tool_results", []))})
    report = {"purpose": "Verify successful behavior is unchanged after preserving interrupted graph state; recover reviewable partial state for the two interrupted NVIDIA demos.", "live_api_calls": 0, "cases": results, "source_changes_since_main_run": {"cli.py": {"reason": "Requirement 3 adds PubMed commands and makes ask default to literature retrieval; --evidence local retains the original graph. The recorded requirement-2 runs and this replay use MedicalCopilot directly.", "current_sha256": file_hash(ROOT / "src/medical_copilot/cli.py")}, "graph.py": {"reason": "Stream completed node updates so provider failures retain plan, evidence, draft, and review history.", "current_sha256": file_hash(ROOT / "src/medical_copilot/graph.py")}}, "interpretation": "This is offline engineering verification using observed responses. It is not a second medical evaluation or additional live provider success."}
    write_json(ROOT / "data/checks/requirement-2-replay.json", report)
    print(f"Verified {len(results)} recorded cases; no network calls; successful outputs unchanged.")


if __name__ == "__main__":
    main()
