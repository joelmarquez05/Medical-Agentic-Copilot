"""Bounded, read-only PubMed search through the official NCBI E-utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLIENT_VERSION = "requirement-3-v1"
RETRACTION_FILTER = "NOT (retracted publication[pt] OR retraction notice[pt] OR hasretractionin OR hasretractionof)"
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PubMedError(RuntimeError):
    """Only safe summaries, never response bodies or request headers."""


class SearchPubMedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=600)
    max_results: int = Field(default=5, ge=1, le=8, strict=True)
    sort: Literal["relevance", "pub_date"] = "relevance"

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3 or re.search(r"[\x00-\x1f\x7f]|https?://", value):
            raise ValueError("Use a short biomedical search query without URLs or control characters.")
        return value


class PubMedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    timeout_seconds: float = Field(default=20, ge=1, le=60)
    min_interval_seconds: float = Field(default=0.4, ge=0.4, le=5)
    max_retries: int = Field(default=2, ge=0, le=2, strict=True)
    retry_wait_seconds: float = Field(default=1, ge=1, le=5)
    max_response_bytes: int = Field(default=4_000_000, ge=10000, le=8_000_000, strict=True)
    max_abstract_characters: int = Field(default=4500, ge=500, le=10000, strict=True)
    email_env: str = "NCBI_EMAIL"


def load_pubmed_config(path: Path) -> PubMedConfig:
    try:
        return PubMedConfig.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))
    except (OSError, ValueError) as exc:
        raise PubMedError(f"Could not load PubMed configuration ({type(exc).__name__}).") from None


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def parse_pubmed_xml(body: bytes, max_abstract_characters: int = 4500) -> list[dict[str, Any]]:
    # EFetch includes a public external DTD. Do not fetch it or expand entities.
    if b"<!ENTITY" in body.upper():
        raise PubMedError("PubMed XML contains an unsupported entity declaration.")
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        raise PubMedError("PubMed returned invalid XML.") from None
    if root.tag != "PubmedArticleSet" or root.find(".//ERROR") is not None:
        raise PubMedError("PubMed did not return a valid article collection.")
    records = []
    for item in root.findall("PubmedArticle"):
        citation = item.find("MedlineCitation")
        if citation is None:
            continue
        article = citation.find("Article")
        pmid = _text(citation.find("PMID"))
        if article is None or not re.fullmatch(r"[1-9]\d{0,9}", pmid):
            continue
        title = _text(article.find("ArticleTitle"))
        if not title:
            continue
        author_names = []
        for author in article.findall("AuthorList/Author"):
            collective = _text(author.find("CollectiveName"))
            personal = " ".join(filter(None, [_text(author.find("ForeName")) or _text(author.find("Initials")), _text(author.find("LastName"))]))
            if collective or personal:
                author_names.append(collective or personal)
        date = article.find("Journal/JournalIssue/PubDate")
        year = _text(date.find("Year")) if date is not None else ""
        date_text = " ".join(_text(child) for child in date) if date is not None else ""
        if not year:
            match = re.search(r"\b(?:19|20)\d{2}\b", date_text)
            year = match.group() if match else _text(article.find("ArticleDate/Year"))
        identifiers = {entry.get("IdType"): _text(entry) for entry in item.findall("PubmedData/ArticleIdList/ArticleId")}
        doi = identifiers.get("doi") or next((_text(entry) for entry in article.findall("ELocationID") if entry.get("EIdType") == "doi"), None)
        sections = []
        for section in article.findall("Abstract/AbstractText"):
            content = _text(section)
            if content:
                label = section.get("Label") or section.get("NlmCategory")
                sections.append((label + ": " if label and label != "UNASSIGNED" else "") + content)
        full_abstract = "\n".join(sections)
        abstract = full_abstract[:max_abstract_characters] or None
        types = [_text(entry) for entry in article.findall("PublicationTypeList/PublicationType")]
        corrections = [{"type": entry.get("RefType"), "pmid": _text(entry.find("PMID")), "reference": _text(entry.find("RefSource"))} for entry in citation.findall("CommentsCorrectionsList/CommentsCorrections")]
        retracted = "Retracted Publication" in types or any(entry["type"] in {"RetractionIn", "RetractedAndRepublishedIn"} for entry in corrections)
        notice = any(value in types for value in ["Retraction of Publication", "Retraction Notice"]) or any(entry["type"] == "RetractionOf" for entry in corrections)
        concern = any(entry["type"] in {"ExpressionOfConcernIn", "ExpressionOfConcernFor"} for entry in corrections)
        records.append({
            "id": "P" + pmid, "pmid": pmid, "title": title,
            "authors": ", ".join(author_names) or "Authors not supplied in the PubMed record",
            "author_list": author_names, "year": year or "Year not supplied",
            "publication_date": date_text or None,
            "journal": _text(article.find("Journal/Title")), "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "pmcid": identifiers.get("pmc"), "publication_types": types,
            "languages": [_text(entry) for entry in article.findall("Language")],
            "abstract": abstract, "abstract_present": bool(full_abstract),
            "abstract_truncated": len(full_abstract) > max_abstract_characters,
            "abstract_original_characters": len(full_abstract),
            "copyright_information": _text(article.find("Abstract/CopyrightInformation")) or None,
            "read_level": "PubMed title and abstract" if full_abstract else "PubMed bibliographic metadata only",
            "preprint": "Preprint" in types,
            "retracted": retracted, "retraction_notice": notice,
            "expression_of_concern": concern, "corrections": corrections,
            "limitation": "PubMed indexing is not quality appraisal. Full text was not fetched; recommendations, subgroups, and adverse effects may be absent from the abstract.",
        })
    return records


class PubMedClient:
    def __init__(self, config: PubMedConfig | None = None, snapshot_dir: Path | None = None,
                 transport: Callable[[str, bytes, float, int], tuple[int, bytes]] | None = None):
        self.config = config or PubMedConfig()
        self.snapshot_dir = snapshot_dir
        self.transport = transport or self._transport
        self.requests: list[dict[str, Any]] = []

    @staticmethod
    def _transport(url: str, data: bytes, timeout: float, limit: int) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=data, headers={"User-Agent": "MedicalAgenticCopilot/0.1 (research proof of concept)", "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(limit + 1)
            if len(body) > limit:
                raise PubMedError("PubMed response exceeded the configured size limit.")
            return response.status, body

    def _request(self, endpoint: str, parameters: dict[str, Any], extension: str) -> bytes:
        global _LAST_REQUEST
        url = BASE_URL + "/" + endpoint
        data = dict(parameters, tool="medical_agentic_copilot")
        email = os.environ.get(self.config.email_env, "").strip()
        if email and re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            data["email"] = email
        encoded = urllib.parse.urlencode(data).encode("utf-8")
        record: dict[str, Any] = {"endpoint": url, "parameters": parameters, "started_at_utc": utc_now(), "attempts": [], "cache_hit": False}
        self.requests.append(record)
        for attempt in range(self.config.max_retries + 1):
            with _REQUEST_LOCK:
                pause = self.config.min_interval_seconds - (time.monotonic() - _LAST_REQUEST)
                if pause > 0:
                    time.sleep(pause)
                _LAST_REQUEST = time.monotonic()
            started = time.perf_counter()
            status = None
            try:
                status, body = self.transport(url, encoded, self.config.timeout_seconds, self.config.max_response_bytes)
                if status != 200:
                    raise urllib.error.HTTPError(url, status, "PubMed HTTP failure", None, None)
                record["attempts"].append({"attempt": attempt + 1, "status": "ok", "http_status": status, "elapsed_seconds": round(time.perf_counter() - started, 3)})
                digest = hashlib.sha256(body).hexdigest()
                filename = digest + "." + extension
                if self.snapshot_dir is not None:
                    self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                    target = self.snapshot_dir / filename
                    if not target.exists():
                        target.write_bytes(body)
                record.update(status="ok", response_sha256=digest, response_bytes=len(body), snapshot_file=filename if self.snapshot_dir is not None else None, retrieved_at_utc=utc_now())
                return body
            except (urllib.error.URLError, TimeoutError, OSError, PubMedError) as exc:
                code = getattr(exc, "code", status)
                record["attempts"].append({"attempt": attempt + 1, "status": "error", "http_status": code, "error_type": type(exc).__name__, "elapsed_seconds": round(time.perf_counter() - started, 3)})
                retryable = code in {429, 500, 502, 503, 504} or (code is None and not isinstance(exc, PubMedError))
                if attempt < self.config.max_retries and retryable:
                    time.sleep(min(5, self.config.retry_wait_seconds * (attempt + 1)))
                    continue
                record["status"] = "error"
                reason = f"HTTP {code}" if code is not None else type(exc).__name__
                raise PubMedError(f"PubMed request failed ({reason}); no papers were fabricated or substituted.") from None
        raise PubMedError("PubMed retry limit reached.")

    def search(self, query: str, max_results: int = 5, sort: str = "relevance") -> dict[str, Any]:
        args = SearchPubMedArgs(query=query, max_results=max_results, sort=sort)
        request_start = len(self.requests)
        effective_query = f"({args.query}) {RETRACTION_FILTER}"
        raw = self._request("esearch.fcgi", {"db": "pubmed", "term": effective_query, "retmode": "json", "retmax": args.max_results, "sort": args.sort}, "json")
        try:
            response = json.loads(raw)
            result = response["esearchresult"]
            if not isinstance(result, dict) or not isinstance(result["idlist"], list):
                raise ValueError
            pmids = list(dict.fromkeys(result["idlist"]))
            count = int(result["count"])
            if count < 0 or len(pmids) > args.max_results or any(not isinstance(value, str) or not re.fullmatch(r"[1-9]\d{0,9}", value) for value in pmids) or response.get("error") or result.get("ERROR"):
                raise ValueError
        except (ValueError, KeyError, TypeError, AttributeError):
            raise PubMedError("PubMed returned an invalid search response.") from None
        sources = []
        excluded = []
        if pmids:
            xml = self._request("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}, "xml")
            parsed = {article["pmid"]: article for article in parse_pubmed_xml(xml, self.config.max_abstract_characters)}
            for rank, pmid in enumerate(pmids, 1):
                article = parsed.get(pmid)
                if article is None:
                    excluded.append({"pmid": pmid, "reason": "No parseable journal-article record was returned."})
                elif article["retracted"] or article["retraction_notice"]:
                    excluded.append({"pmid": pmid, "reason": "Retraction metadata detected; excluded from answer context."})
                else:
                    article.update(retrieval_rank=rank, retrieved_at_utc=utc_now())
                    sources.append(article)
        return {"mode": "live_pubmed_title_abstract", "client_version": CLIENT_VERSION,
                "query": args.query, "effective_query": effective_query, "sort": args.sort,
                "max_results": args.max_results, "total_matches": count,
                "query_translation": result.get("querytranslation"),
                "warnings": {key: result[key] for key in ["warninglist", "errorlist"] if result.get(key)},
                "esearch_pmids": pmids, "sources": sources, "excluded_records": excluded,
                "http_requests": self.requests[request_start:], "retrieved_at_utc": utc_now(),
                "limitation": "A small ranked search, not a systematic review. No full text, citation-network search, clinical appraisal, or medRxiv-specific integration. No persistent cache is used."}
