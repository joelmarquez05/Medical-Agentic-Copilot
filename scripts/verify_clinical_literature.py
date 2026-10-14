"""Replay saved model and PubMed responses through the graph with no live calls."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError, HTTPError
from urllib.parse import parse_qs

from langchain_core.messages import AIMessage

from medical_copilot.clinical_evaluation import DEFAULT_PROFILE, score_record
from medical_copilot.clinical_graph import ClinicalAgent, JSONActionModel, Limits, SYSTEM, save_trace
from medical_copilot.clinical_literature import ClinicalLiteratureTools
from medical_copilot.config import load_config
from medical_copilot.evaluation import file_hash, read_json
from medical_copilot.models import ModelClient, Redactor
from medical_copilot.pubmed import PubMedClient, PubMedConfig


def without_transient(value):
    if isinstance(value, dict):
        return {key: without_transient(item) for key, item in value.items()
                if key not in {"elapsed_seconds", "started_at_utc", "retrieved_at_utc", "snapshot_file"}}
    if isinstance(value, list):
        return [without_transient(item) for item in value]
    return value


class ModelReplay:
    def __init__(self, calls):
        self.calls, self.index = calls, 0

    def invoke(self, messages):
        call = self.calls[self.index]
        self.index += 1
        assert messages[0].content == call["system_prompt"]
        assert messages[1].content == call["user_prompt"]
        if call["status"] == "error":
            raise RuntimeError("Offline replay of saved provider failure.")
        return AIMessage(content=call.get("response") or "", response_metadata=call.get("response_metadata", {}),
                         usage_metadata=call.get("usage") or None)


class PromptReplay:
    def __init__(self, client, saved_system):
        self.client, self.config, self.calls, self.saved_system = client, client.config, client.calls, saved_system

    def generate(self, system, user, stage):
        assert system == SYSTEM
        return self.client.generate(self.saved_system, user, stage)


class HTTPReplay:
    def __init__(self, requests, snapshot_dir):
        self.requests, self.snapshot_dir = requests, snapshot_dir
        self.index, self.attempt = 0, 0

    def __call__(self, url, data, timeout, limit):
        request = self.requests[self.index]
        assert url == request["endpoint"]
        actual = parse_qs(data.decode())
        for key, value in request["parameters"].items():
            assert actual[key] == [str(value)]
        attempt = request["attempts"][self.attempt]
        self.attempt += 1
        if self.attempt == len(request["attempts"]):
            self.index += 1
            self.attempt = 0
        if attempt["status"] != "ok":
            if attempt.get("http_status"):
                raise HTTPError(url, attempt["http_status"], "Offline saved HTTP failure", None, None)
            raise URLError("Offline saved transport failure")
        path = self.snapshot_dir / request["snapshot_file"]
        assert file_hash(path) == request["response_sha256"]
        body = path.read_bytes()
        assert len(body) == request["response_bytes"] <= limit
        return attempt["http_status"], body


def verify(root, run):
    manifest = read_json(run / "manifest.json")
    assert manifest.get("finished_at_utc"), "Collection is still active."
    assert manifest["config_sha256"] == file_hash(root / DEFAULT_PROFILE)
    assert manifest["system_sha256"] == hashlib.sha256(SYSTEM.encode()).hexdigest()
    for name, digest in manifest["source_hashes"].items():
        assert file_hash(run / "source-snapshot" / name) == digest
    baseline = root / manifest["baseline_origin"]
    bm = read_json(baseline / "manifest.json")
    assert file_hash(baseline / "manifest.json") == manifest["baseline_manifest_sha256"]
    for name, digest in bm["source_hashes"].items():
        assert file_hash(root / "src/medical_copilot" / name) == digest == manifest["source_hashes"][name]
    for name, key in [("frozen-ten-questions.json", "questions_sha256"), ("answer-key.json", "answer_key_sha256"),
                       ("protocol.md", "protocol_sha256"), ("retrieval-rubric.json", "retrieval_rubric_sha256"),
                       ("answer-rubric.json", "answer_rubric_sha256"), ("pubmed-config.json", "pubmed_config_sha256")]:
        assert file_hash(run / name) == manifest[key]
    assert manifest["questions_sha256"] == bm["questions_sha256"]
    assert manifest["answer_key_sha256"] == bm["answer_key_sha256"]
    config = load_config(root / DEFAULT_PROFILE)
    pm_config = PubMedConfig.model_validate(read_json(run / "pubmed-config.json"))
    inputs = {item["question_id"]: item for item in read_json(run / "frozen-ten-questions.json")["questions"]}
    keys = {item["question_id"]: item for item in read_json(run / "answer-key.json")["keys"]}
    records = [read_json(path) for path in sorted((run / "answers").glob("*.json"))]
    assert len(records) == len(manifest["question_ids"])
    audit = read_json(run / "usage-audit.json")["records"]
    assert len(audit) == sum(len(row["model_calls"]) for row in records)
    results = []
    for row in records:
        identifier = row["question"]["question_id"]
        if manifest["stage"] == "autonomous":
            assert row["question"] == inputs[identifier]
        else:
            assert row["question"]["question"] not in {item["question"] for item in inputs.values()}
        assert 1 <= len(row["model_calls"]) <= 3 and row["tool_actions"] <= 2
        first = json.loads(row["model_calls"][0]["user_prompt"])
        assert first["history"] == [] and first["evidence_registry"] == {}
        assert [tool["name"] for tool in first["tools"]] == ["search_pubmed", "read_pubmed_abstract"]
        for call in row["model_calls"]:
            prompt = json.loads(call["user_prompt"])
            assert set(prompt) == {"question", "history", "evidence_registry", "tools", "action_schema", "remaining"}
            assert prompt["question"] == row["question"]
            assert len(call["attempts"]) == 1
            assert len((call["system_prompt"] + call["user_prompt"]).encode()) <= 65536
            expected = manifest.get("effective_system_sha256", manifest["system_sha256"])
            assert hashlib.sha256(call["system_prompt"].encode()).hexdigest() == expected
        http = HTTPReplay(row["literature"]["http_requests"], run / "http-snapshots")
        tools = ClinicalLiteratureTools(PubMedClient(pm_config, transport=http))
        backend = ModelReplay(row["model_calls"])
        client = ModelClient(backend, config, Redactor())
        adapter = JSONActionModel(PromptReplay(client, row["model_calls"][0]["system_prompt"]), lambda request: None)
        with patch("medical_copilot.pubmed.time.sleep", lambda seconds: None):
            replayed = ClinicalAgent(adapter, tools.definitions(), Limits(**manifest["limits"])).run(row["question"])
        for field in ["status", "response", "history", "evidence", "tool_actions", "model_turns", "issues", "trace"]:
            assert without_transient(replayed[field]) == without_transient(row[field]), (identifier, field)
        assert backend.index == len(row["model_calls"])
        assert http.index == len(row["literature"]["http_requests"])
        assert without_transient(tools.events) == without_transient(row["literature"]["events"])
        results.append({"question_id": identifier, "answer_sha256": file_hash(run / "answers" / (identifier + ".json")),
                        "replayed_model_responses": backend.index, "replayed_logical_http_requests": http.index,
                        "status": row["status"], "tool_actions": row["tool_actions"]})
    summary = read_json(run / "summary.json")
    assert len(records) == summary["recorded"]
    assert len(audit) == summary["provider_requests"]
    assert math.isclose(sum(item.get("cost_usd") or 0 for item in audit), summary["reported_or_estimated_cost_usd"], abs_tol=1e-12)
    if manifest["stage"] == "autonomous":
        calculated = [score_record(row, keys[row["question"]["question_id"]]) for row in records]
        assert calculated == read_json(run / "scores.json")["scores"]
        assert sum(score["correct"] is True for score in calculated) == summary["correct"]
    suspicious = re.compile(r"sk-or-v1-[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9_-]{15,}")
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".xml"}:
            raw = path.read_text(encoding="utf-8")
            assert not suspicious.search(raw), "Credential-shaped value found; path/content withheld."
            assert '"reasoning_content"' not in raw and '"reasoning_text"' not in raw
    return {"passed": True, "run_id": manifest["run_id"], "stage": manifest["stage"], "cases": results,
            "baseline_core_source_unchanged": True, "question_key_profile_unchanged": True,
            "current_extension_source_matches_snapshot": {name: file_hash(root / "src/medical_copilot" / name) == digest for name, digest in manifest["source_hashes"].items()},
            "new_provider_requests": 0, "new_pubmed_requests": 0, "credential_files_read": 0,
            "interpretation": "Exact saved-prompt/action/source/HTTP replay plus hashes and input isolation; not medical correctness or substantive citation support validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parents[1], args.run_dir.resolve())
    if args.output:
        save_trace(args.output.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
