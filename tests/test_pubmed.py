"""Offline transport/graph checks; fictional fixtures are never study outputs."""

import json
import urllib.error
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from medical_copilot.config import ModelConfig
from medical_copilot.models import ModelClient
from medical_copilot.pubmed import PubMedClient, PubMedConfig, PubMedError, SearchPubMedArgs, parse_pubmed_xml

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
