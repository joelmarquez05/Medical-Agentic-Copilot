"""Offline synthetic HTTP/model fixtures; never medical evaluation results."""

import copy
import json
from urllib.parse import parse_qs

import pytest
from pydantic import ValidationError

from medical_copilot.clinical_graph import ClinicalAgent
from medical_copilot.clinical_literature import (
    ClinicalLiteratureTools, ClinicalSearchArgs, ReadAbstractArgs, UnknownPubMedSelection,
)
from medical_copilot.clinical_literature_evaluation import EvidencePromptClient, EVIDENCE_SYSTEM, check_baseline, paired_transition
from medical_copilot.pubmed import PubMedClient, PubMedConfig


def xml(abstract="SIMULATED " + "x" * 800, title="Synthetic test paper", retracted=False):
    text = f"<Abstract><AbstractText>{abstract}</AbstractText></Abstract>" if abstract else ""
    types = "<PublicationTypeList><PublicationType>Retracted Publication</PublicationType></PublicationTypeList>" if retracted else ""
    return f'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID><Article><ArticleTitle>{title}</ArticleTitle><Journal><Title>Fixture journal</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>{text}{types}<AuthorList><Author><LastName>Fixture</LastName><ForeName>Synthetic</ForeName></Author></AuthorList></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'.encode()


def tools_for(monkeypatch, *, abstract="SIMULATED " + "x" * 800, empty=False, title="Synthetic test paper", read_retracted=False):
    monkeypatch.setattr("medical_copilot.pubmed.time.sleep", lambda seconds: None)
    requests = []

    def transport(url, data, timeout, limit):
        args = parse_qs(data.decode())
        requests.append((url, args))
        if "esearch" in url:
            return 200, json.dumps({"esearchresult": {"count": "0" if empty else "1", "idlist": [] if empty else ["12345"]}}).encode()
        return 200, xml(abstract, title, read_retracted and len(requests) > 2)

    client = PubMedClient(PubMedConfig(max_retries=0, max_abstract_characters=6000), transport=transport)
    return ClinicalLiteratureTools(client), requests


def test_search_hides_later_abstract_text_and_keeps_researcher_record(monkeypatch):
    tools, requests = tools_for(monkeypatch, abstract="x" * 500 + "HIDDEN_LATER_FINDING")
    output = tools.search(ClinicalSearchArgs(query="synthetic marker"))
    assert len(requests) == 2 and len(output.sources) == 1
    assert len(output.sources[0].text) == 400 and output.sources[0].truncated
    assert "HIDDEN_LATER_FINDING" not in output.model_dump_json()
    assert "HIDDEN_LATER_FINDING" in tools.public_record()["events"][0]["papers"][0]["abstract"]


def test_read_uses_fresh_fetch_and_distinct_immutable_view(monkeypatch):
    tools, requests = tools_for(monkeypatch)
    search = tools.search(ClinicalSearchArgs(query="synthetic marker"))
    read = tools.read(ReadAbstractArgs(pmids=["12345"]))
    assert len(requests) == 3 and "efetch" in requests[-1][0]
    assert requests[-1][1]["id"] == ["12345"]
    assert search.sources[0].id != read.sources[0].id
    assert search.sources[0].text_kind == "abstract_excerpt" and read.sources[0].text_kind == "abstract"
    assert tools.public_record()["events"][-1]["generation_view"] == read.model_dump(mode="json")


def test_read_cannot_guess_ids_or_use_previous_episode_selection(monkeypatch):
    first, _ = tools_for(monkeypatch)
    first.search(ClinicalSearchArgs(query="synthetic marker"))
    second, requests = tools_for(monkeypatch)
    with pytest.raises(UnknownPubMedSelection):
        second.read(ReadAbstractArgs(pmids=["12345"]))
    assert requests == [] and second.events[0]["status"] == "error"


@pytest.mark.parametrize("pmids", [["12345", "12345"], ["P12345"], ["1", "2", "3", "4"], ["0"], [12345]])
def test_invalid_read_selections_fail_schema_before_execution(pmids):
    with pytest.raises(ValidationError):
        ReadAbstractArgs(pmids=pmids)


@pytest.mark.parametrize("values", [{"max_results": 6}, {"max_results": True}, {"sort": "pub_date"}])
def test_search_bounds_are_host_enforced(values):
    with pytest.raises(ValidationError):
        ClinicalSearchArgs(query="synthetic marker", **values)


def test_missing_abstract_is_explicit_title_only(monkeypatch):
    tools, _ = tools_for(monkeypatch, abstract="")
    search = tools.search(ClinicalSearchArgs(query="synthetic marker"))
    read = tools.read(ReadAbstractArgs(pmids=["12345"]))
    assert search.sources[0].text == read.sources[0].text == ""
    assert read.sources[0].text_kind == "title_only"
    assert json.loads(read.content)["records"][0]["abstract_present"] is False


def test_read_truncation_is_preserved_not_claimed_full_text(monkeypatch):
    tools, _ = tools_for(monkeypatch, abstract="x" * 5000)
    tools.search(ClinicalSearchArgs(query="synthetic marker"))
    read = tools.read(ReadAbstractArgs(pmids=["12345"]))
    assert len(read.sources[0].text) == 3500 and read.sources[0].truncated
    assert read.sources[0].text_kind == "abstract_excerpt"


def test_read_filters_new_retraction_metadata(monkeypatch):
    tools, _ = tools_for(monkeypatch, read_retracted=True)
    tools.search(ClinicalSearchArgs(query="synthetic marker"))
    read = tools.read(ReadAbstractArgs(pmids=["12345"]))
    assert read.sources == []
    assert json.loads(read.content)["excluded_records"][0]["pmid"] == "12345"


def test_long_title_is_bounded_with_explicit_marker(monkeypatch):
    tools, _ = tools_for(monkeypatch, title="x" * 800)
    search = tools.search(ClinicalSearchArgs(query="synthetic marker"))
    assert len(search.sources[0].title) == 500
    assert json.loads(search.content)["records"][0]["title_truncated"]


class Model:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def reply(self, request):
        self.requests.append(copy.deepcopy(request))
        response = self.replies.pop(0)
        return response(request) if callable(response) else response


QUESTION = {"question_id": "SYNTHETIC", "question": "Which fixture marker is present?", "options": {"A": "BLUE", "B": "RED"}}
FINAL = {"type": "final", "answer": "BLUE", "justification": "Synthetic fixture only.", "selected_option": "A"}
SEARCH = {"type": "tool", "name": "search_pubmed", "arguments": {"query": "synthetic marker"}}
READ = {"type": "tool", "name": "read_pubmed_abstract", "arguments": {"pmids": ["12345"]}}


def test_agent_receives_real_wrapper_feedback_and_can_cite_read_view(monkeypatch):
    tools, requests = tools_for(monkeypatch, abstract="x" * 450 + " BLUE")

    def decide_read(request):
        assert "BLUE" not in request["history"][-1]["result"]["output"]["sources"][0]["text"]
        return READ

    def answer(request):
        sources = request["history"][-1]["result"]["output"]["sources"]
        assert sources[0]["text"].endswith("BLUE")
        return {**FINAL, "citations": [sources[0]["id"]]}

    model = Model([SEARCH, decide_read, answer])
    result = ClinicalAgent(model, tools.definitions()).run(QUESTION)
    assert result["status"] == "completed" and result["model_turns"] == 3 and result["tool_actions"] == 2
    assert len(requests) == 3 and len(result["evidence"]) == 2
    assert model.requests[-1]["tools"] == []
    assert result["response"]["evidence_status"] == "retrieved_sources_cited"


def test_autonomous_direct_answer_is_not_followed_by_forced_search(monkeypatch):
    tools, requests = tools_for(monkeypatch)
    result = ClinicalAgent(Model([FINAL]), tools.definitions()).run(QUESTION)
    assert result["status"] == "completed" and result["model_turns"] == 1
    assert requests == [] and result["tool_actions"] == 0 and tools.events == []


def test_empty_search_can_reformulate_without_fabricated_papers(monkeypatch):
    tools, requests = tools_for(monkeypatch, empty=True)
    model = Model([SEARCH, {**SEARCH, "arguments": {"query": "another synthetic marker"}}, FINAL])
    result = ClinicalAgent(model, tools.definitions()).run(QUESTION)
    assert result["status"] == "completed" and len(requests) == 2
    assert result["evidence"] == {} and result["tool_actions"] == 2


def test_archived_baseline_integrity_and_changed_core_guard():
    import hashlib
    from pathlib import Path
    from medical_copilot.clinical_evaluation import DEFAULT_PROFILE, DEFAULT_SET
    from medical_copilot.clinical_literature_evaluation import BASELINE
    from medical_copilot.evaluation import file_hash, read_json
    from medical_copilot.clinical_graph import SYSTEM

    root = Path(__file__).resolve().parents[1]
    archive = root / BASELINE
    manifest = read_json(archive / "manifest.json")
    assert manifest["run_id"] == "gemma-baseline-restart-20261011"
    for name, expected in manifest["source_hashes"].items():
        assert file_hash(archive / "source-snapshot" / name) == expected
    assert file_hash(root / DEFAULT_PROFILE) == manifest["config_sha256"]
    assert hashlib.sha256(SYSTEM.encode()).hexdigest() == manifest["system_sha256"]
    for name, key in [("questions.json", "questions_sha256"), ("answer-key.json", "answer_key_sha256")]:
        assert file_hash(root / DEFAULT_SET / name) == manifest[key]
    # Relocated credentials change the live adapter. The historical guard must
    # still refuse to claim that a new run uses the unmodified baseline code.
    with pytest.raises(ValueError, match="Core baseline source changed"):
        check_baseline(root)


def test_evidence_envelope_encourages_without_rewriting_core_prompt_or_action_adapter():
    from langchain_core.messages import AIMessage
    from medical_copilot.clinical_graph import JSONActionModel, SYSTEM
    from medical_copilot.config import ModelConfig
    from medical_copilot.models import ModelClient

    class Backend:
        def __init__(self):
            self.prompts = []

        def invoke(self, messages):
            self.prompts.append(messages)
            return AIMessage(content=json.dumps(FINAL))

    backend = Backend()
    client = ModelClient(backend, ModelConfig(provider="openai_compatible", model="synthetic", base_url="http://localhost.invalid/v1", max_retries=0,
                                            rate_limit_retries=0, json_repair_attempts=0))
    result = ClinicalAgent(JSONActionModel(EvidencePromptClient(client), lambda request: None)).run(QUESTION)
    assert result["status"] == "completed" and len(backend.prompts) == 1
    assert backend.prompts[0][0].content == EVIDENCE_SYSTEM
    assert EVIDENCE_SYSTEM.startswith(SYSTEM + "\n\n")
    assert "no mandatory operation" in EVIDENCE_SYSTEM and "exact evidence_registry source IDs" in EVIDENCE_SYSTEM
    assert client.calls[0]["system_prompt"] == EVIDENCE_SYSTEM


@pytest.mark.parametrize("a,b,expected", [(True, True, "both_correct"), (False, True, "wrong_to_correct"),
                                          (True, False, "correct_to_wrong"), (False, False, "both_wrong"),
                                          (True, None, "operationally_incomplete")])
def test_pairing_keeps_operational_failure_separate(a, b, expected):
    assert paired_transition(a, b) == expected
