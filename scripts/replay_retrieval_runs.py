"""Replay archived model/tool outputs without new retrieval or inference."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage
from medical_copilot.config import ModelConfig
from medical_copilot.evaluation import read_json, write_json, file_hash
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
        self.calls, self.position = calls, 0
    def invoke(self, messages):
        call = self.calls[self.position]
        self.position += 1
        assert messages[0].content == call["system_prompt"], "Recorded system prompt changed."
        assert prompt_value(messages[1].content) == prompt_value(call["user_prompt"]), "Recorded substantive input changed."
        assert call["status"] == "ok"
        return AIMessage(content=call["response"], usage_metadata=call.get("usage") or None, response_metadata=call.get("response_metadata") or {})


class RecordedPubMed:
    def __init__(self, tools):
        self.tools = [tool for tool in tools if tool["name"] == "search_pubmed"]
        self.position, self.requests = 0, []
    def search(self, query: str, max_results: int = 5, sort: str = "relevance"):
        tool = self.tools[self.position]
        self.position += 1
        expected = dict(max_results=5, sort="relevance", **{key: value for key, value in tool["arguments"].items() if key not in {"max_results", "sort"}})
        expected.update(tool["arguments"])
        assert {"query": query, "max_results": max_results, "sort": sort} == expected
        assert tool["status"] == "ok"
        result = copy.deepcopy(tool["output"])
        self.requests.extend(result["http_requests"])
        return result


def main():
    cases = []
    for index, run_id in enumerate(["gemini-pubmed-internal-20261010", "gemini-pubmed-final-20261010"]):
        run = ROOT / "data/evaluation/retrieval-runs" / run_id
        manifest = read_json(run / "manifest.json")
        name = f"_archived_literature_replay_{index}"
        source = run / "source-snapshot/literature.py"
        assert file_hash(source) == manifest["code_sha256"]["literature.py"]
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        settings = ModelConfig.model_validate(manifest["model_config"]).model_copy(update={"min_interval_seconds": 0, "rate_limit_retries": 0})
        for path in sorted((run / "answers").glob("*.json")):
            observed = read_json(path)
            backend, pubmed = RecordedBackend(observed["model_calls"]), RecordedPubMed(observed["tool_results"])
            result = module.LiteratureCopilot(ModelClient(backend, settings), pubmed).run(observed["question"])
            assert backend.position == len(observed["model_calls"])
            assert pubmed.position == len(pubmed.tools)
            for field in ["status", "answer", "sources", "trace", "revision_count", "mechanical_issues"]:
                assert result[field] == observed[field], f"Replay mismatch: {run_id}/{path.stem}/{field}."
            cases.append({"run_id": run_id, "question_id": path.stem, "archived_prompt_version": module.PROMPT_VERSION, "model_inputs_matched": True, "answer_and_flags_unchanged": True})
    report = {"live_model_calls": 0, "live_pubmed_requests": 0, "cases": cases, "interpretation": "Engineering replay of recorded outputs with each archived prompt version. It is not a new clinical evaluation, new literature retrieval, or validation of source attribution."}
    write_json(ROOT / "data/checks/requirement-3-replay.json", report)
    print(f"Replayed {len(cases)} cases with archived versions; zero network calls; outputs unchanged.")


if __name__ == "__main__":
    main()
