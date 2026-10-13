"""Offline transport/graph checks; fictional fixtures are never study outputs."""

import json
import urllib.error
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from medical_copilot.config import ModelConfig
from medical_copilot.literature import LiteratureCopilot, LiteraturePlan, literature_citation_issues
from medical_copilot.models import ModelClient
from medical_copilot.pubmed import PubMedClient, PubMedConfig, PubMedError, SearchPubMedArgs, parse_pubmed_xml
from medical_copilot.retrieval_evaluation import retrieval_record

XML = b'''<?xml version="1.0"?><PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
<Journal><Title>Fictional test journal</Title><JournalIssue><PubDate><MedlineDate>2020 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
<ArticleTitle>Nutrition <i>in</i> fictional cancer</ArticleTitle>
<Abstract><AbstractText Label="RESULTS">A fictional unit-test result.</AbstractText><AbstractText Label="CONCLUSION">Not clinical evaluation data.</AbstractText></Abstract>
<AuthorList><Author><ForeName>Test</ForeName><LastName>Author</LastName></Author><Author><CollectiveName>Test Group</CollectiveName></Author></AuthorList>
<Language>eng</Language><PublicationTypeList><PublicationType>Review</PublicationType></PublicationTypeList>
</Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.0000/test-only</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>456</PMID><Article><ArticleTitle>No abstract fixture</ArticleTitle><PublicationTypeList><PublicationType>Preprint</PublicationType></PublicationTypeList></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>'''


def transport_for(xml=XML, ids=None):
    calls = []
    def transport(url, data, timeout, limit):
        calls.append((url, data))
        if url.endswith("esearch.fcgi"):
            return 200, json.dumps({"esearchresult": {"idlist": ids or ["456", "123"], "count": "2", "querytranslation": "test translation"}}).encode()
        return 200, xml
    return transport, calls


def test_parser_preserves_mixed_text_authors_structured_abstract_and_missing_fields():
    records = parse_pubmed_xml(XML)
    assert records[0]["title"] == "Nutrition in fictional cancer"
    assert records[0]["authors"] == "Test Author, Test Group"
    assert records[0]["year"] == "2020"
    assert records[0]["abstract"].startswith("RESULTS: A fictional")
    assert records[0]["doi"] == "10.0000/test-only"
    assert records[1]["abstract"] is None and records[1]["preprint"]
    assert records[1]["read_level"] == "PubMed bibliographic metadata only"


def test_parser_records_truncation_and_retraction_metadata():
    modified = XML.replace(b"<PublicationType>Review</PublicationType>", b"<PublicationType>Retracted Publication</PublicationType>")
    record = parse_pubmed_xml(modified, 12)[0]
    assert record["retracted"] and record["abstract_truncated"]
    assert len(record["abstract"]) == 12


@pytest.mark.parametrize("body", [b"<broken>", b"<ERROR>failure</ERROR>", b'<!DOCTYPE x [<!ENTITY y "expanded">]><PubmedArticleSet/>'])
def test_bad_xml_or_entity_declaration_fails_safely(body):
    with pytest.raises(PubMedError):
        parse_pubmed_xml(body)


@pytest.mark.parametrize("args", [{"query": "  "}, {"query": "cancer\nsecret"}, {"query": "https://evil.test"}, {"query": "cancer", "max_results": 9}, {"query": "cancer", "max_results": True}, {"query": "cancer", "sort": "invented"}])
def test_query_limits_reject_invalid_requests(args):
    with pytest.raises(ValidationError):
        SearchPubMedArgs(**args)


def test_search_preserves_esearch_order_batches_fetch_and_saves_hashed_responses(tmp_path, monkeypatch):
    monkeypatch.setenv("NCBI_EMAIL", "test-contact@example.test")
    transport, calls = transport_for()
    client = PubMedClient(PubMedConfig(), tmp_path, transport)
    result = client.search("cancer nutrition")
    assert [record["pmid"] for record in result["sources"]] == ["456", "123"]
    assert [record["retrieval_rank"] for record in result["sources"]] == [1, 2]
    assert len(calls) == 2 and b"id=456%2C123" in calls[1][1]
    assert "test-contact@example.test" not in json.dumps(result)
    assert "hasretractionin" in result["effective_query"]
    assert len(list(tmp_path.glob("*"))) == 2
    assert all(request["status"] == "ok" for request in client.requests)


def test_empty_search_does_not_fetch_or_fabricate_papers():
    calls = []
    def transport(url, data, timeout, limit):
        calls.append(url)
        return 200, b'{"esearchresult":{"idlist":[],"count":"0"}}'
    result = PubMedClient(transport=transport).search("fictional no match")
    assert result["sources"] == [] and len(calls) == 1


def test_retraction_record_is_excluded_even_when_esearch_returns_it():
    xml = XML.replace(b"<PublicationType>Review</PublicationType>", b"<PublicationType>Retracted Publication</PublicationType>")
    transport, _ = transport_for(xml, ["123"])
    result = PubMedClient(transport=transport).search("cancer nutrition")
    assert result["sources"] == []
    assert result["excluded_records"][0]["pmid"] == "123"


def test_transient_http_retry_is_bounded_and_does_not_log_error_body(monkeypatch):
    monkeypatch.setattr("medical_copilot.pubmed.time.sleep", lambda seconds: None)
    calls = []
    def transport(url, data, timeout, limit):
        calls.append(url)
        raise urllib.error.HTTPError(url, 429, "secret error body must not leak", None, None)
    client = PubMedClient(transport=transport)
    with pytest.raises(PubMedError) as caught:
        client.search("cancer nutrition")
    assert len(calls) == 3
    assert "secret" not in str(caught.value) + json.dumps(client.requests)
    assert client.requests[0]["status"] == "error"


def test_nonretryable_http_error_is_not_repeated():
    calls = []
    def transport(url, data, timeout, limit):
        calls.append(url)
        raise urllib.error.HTTPError(url, 400, "bad", None, None)
    with pytest.raises(PubMedError):
        PubMedClient(transport=transport).search("cancer nutrition")
    assert len(calls) == 1


class Backend:
    def __init__(self, outputs):
        self.outputs, self.messages = list(outputs), []
    def invoke(self, messages):
        self.messages.append(messages)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return AIMessage(content=json.dumps(value) if isinstance(value, dict) else value)


def plan():
    return {"scope": "oncology_nutrition", "answer_focus": ["Answer"], "missing_information": [], "tool_calls": [{"name": "search_pubmed", "arguments": {"query": "cancer nutrition", "max_results": 5}}, {"name": "lookup_guidance", "arguments": {"topics": ["requirements"]}}]}


def review(queries=None):
    return {"needs_revision": bool(queries), "issues": [], "additional_topics": [], "additional_queries": queries or [], "uncertainty_note": "Internal check only.", "retrieval_fit": "direct", "paper_checks": [{"source_id": "P123", "reported_result": "Fictional test-only observation.", "draft_claim_supported": True, "issue": ""}]}


def copilot(outputs):
    transport, calls = transport_for()
    backend = Backend(outputs)
    config = ModelConfig(provider="google_genai", model="test-only", max_revisions=1)
    return LiteratureCopilot(ModelClient(backend, config), PubMedClient(transport=transport)), backend, calls


def test_real_tool_context_reaches_draft_and_bibliography_is_registry_controlled():
    agent, backend, calls = copilot([plan(), "Fictional fixture observation [P123, S01].", review()])
    result = agent.run("What are general nutrition targets in adult cancer?")
    assert result["status"] == "self_review_passed" and len(calls) == 2
    assert {source["id"] for source in result["sources"]} == {"P123", "S01"}
    assert "pubmed.ncbi.nlm.nih.gov/123/" in result["answer"]
    draft_context = json.loads(backend.messages[1][1].content)
    assert "test translation" in json.dumps(draft_context)
    assert "http_requests" not in json.dumps(draft_context)
    assert "untrusted evidence" in backend.messages[1][0].content
    result["question_id"] = "TEST"
    assert len(retrieval_record(result)["unique_papers"]) == 2


def test_reviewer_refinement_retrieves_once_and_question_state_resets():
    outputs = [plan(), "Initial [P123].", review(["cancer nutrition guideline"]), "Revised [P123, S01].", review(), plan(), "Second independent question [P123].", review()]
    agent, _, calls = copilot(outputs)
    first = agent.run("First adult cancer nutrition question.")
    second = agent.run("Second adult cancer nutrition question.")
    assert first["revision_count"] == 1 and len(first["pubmed_http_requests"]) == 4
    assert second["revision_count"] == 0 and len(second["pubmed_http_requests"]) == 2
    assert len(calls) == 6 and agent.search_count == 1


def test_unknown_pubmed_marker_is_flagged_and_not_attached():
    agent, _, _ = copilot([plan(), "Initial [P999].", review(), "Still wrong [P999].", review()])
    result = agent.run("Adult cancer nutrition.")
    assert result["status"] == "needs_human_review"
    assert result["sources"] == []
    assert any("[P999]" in issue for issue in result["mechanical_issues"])
    assert literature_citation_issues("Grouped [S01, P123, P999].", {"S01", "P123"}, True) == ["Unknown or unavailable source marker [P999]."]


def test_in_scope_plan_requires_search_and_out_of_scope_calls_are_rejected():
    value = plan()
    value["tool_calls"] = []
    with pytest.raises(ValidationError):
        LiteraturePlan.model_validate(value)
    value = plan()
    value["scope"] = "out_of_scope"
    with pytest.raises(ValidationError):
        LiteraturePlan.model_validate(value)


def test_pubmed_failure_is_preserved_and_answer_uses_only_actual_guidance():
    backend = Backend([plan(), "Retrieval failed; approximate guidance only [S01].", review(), "PubMed was unavailable; guidance only [S01].", review()])
    def transport(url, data, timeout, limit):
        raise urllib.error.HTTPError(url, 400, "do not echo failure body", None, None)
    client = ModelClient(backend, ModelConfig(provider="google_genai", model="test-only"))
    result = LiteratureCopilot(client, PubMedClient(transport=transport)).run("Adult cancer nutrition.")
    assert result["status"] == "needs_human_review"
    assert result["tool_results"][0]["status"] == "error"
    assert not any(source["id"].startswith("P") for source in result["sources"])
    assert result["pubmed_http_requests"][0]["status"] == "error"


def test_new_evidence_is_retained_when_revision_model_call_fails():
    agent, _, _ = copilot([plan(), "Initial [P123].", review(["cancer nutrition guideline"]), TimeoutError("unit-test failure")])
    result = agent.run("Adult cancer nutrition.")
    assert result["status"] == "error" and result["answer"] == ""
    assert result["partial_answer"] == "Initial [P123]."
    assert result["tool_results_recovered_after_interrupted_node"]
    assert sum(tool["name"] == "search_pubmed" for tool in result["tool_results"]) == 2
    assert len(result["pubmed_http_requests"]) == 4


def test_source_attribution_failure_forces_revision_even_with_no_general_issue():
    failed = review()
    failed["paper_checks"][0].update(draft_claim_supported=False, issue="The draft reverses the fictional fixture's result.")
    agent, _, _ = copilot([plan(), "Incorrect attribution [P123].", failed, "Corrected fixture attribution [P123].", review()])
    result = agent.run("Adult cancer nutrition.")
    assert result["revision_count"] == 1 and result["status"] == "self_review_passed"
    assert not result["reviews"][0]["paper_checks"][0]["draft_claim_supported"]
    assert result["answer"].startswith("Corrected")


def test_missing_source_specific_check_retains_review_flag():
    absent = review()
    absent["paper_checks"] = []
    agent, _, _ = copilot([plan(), "Fixture [P123].", absent, "Fixture again [P123].", absent])
    result = agent.run("Adult cancer nutrition.")
    assert result["status"] == "needs_human_review"
    assert any("attribution check" in issue for issue in result["mechanical_issues"])


def test_two_search_limit_prevents_third_network_call():
    agent, _, calls = copilot([])
    for _ in range(2):
        assert agent._invoke_tool("search_pubmed", {"query": "cancer nutrition"}, "Adult cancer.")["status"] == "ok"
    blocked = agent._invoke_tool("search_pubmed", {"query": "cancer nutrition"}, "Adult cancer.")
    assert blocked["status"] == "error" and "budget" in blocked["error"]
    assert len(calls) == 4


def test_search_cli_does_not_initialize_a_model_or_require_its_credential(monkeypatch, capsys):
    from medical_copilot import cli
    monkeypatch.setattr(cli.PubMedClient, "search", lambda self, *args: {"sources": [], "test_only": True})
    def prohibited(*args):
        raise AssertionError("Direct PubMed search must not initialize an LLM.")
    monkeypatch.setattr(cli, "create_backend", prohibited)
    root = Path(__file__).resolve().parents[1]
    assert cli.main(["--root", str(root), "search-pubmed", "cancer nutrition"]) == 0
    assert '"test_only": true' in capsys.readouterr().out
