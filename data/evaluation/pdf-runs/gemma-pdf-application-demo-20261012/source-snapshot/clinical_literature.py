"""Optional PubMed tools for the unchanged restarted clinical graph."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from medical_copilot.clinical_graph import EvidenceRecord, ToolDefinition, ToolOutput
from medical_copilot.pubmed import PubMedClient, PubMedError, SearchPubMedArgs, parse_pubmed_xml, utc_now

EXCERPT_CHARACTERS = 400
ABSTRACT_CHARACTERS = 3500


class ClinicalSearchArgs(SearchPubMedArgs):
    max_results: int = Field(default=5, ge=1, le=5, strict=True)
    sort: Literal["relevance"] = "relevance"


class ReadAbstractArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pmids: list[str] = Field(min_length=1, max_length=3)

    @field_validator("pmids")
    @classmethod
    def validate_pmids(cls, values):
        if len(values) != len(set(values)) or any(not re.fullmatch(r"[1-9]\d{0,9}", value) for value in values):
            raise ValueError("Use one to three distinct numeric PubMed IDs from this episode's search results.")
        return values


class UnknownPubMedSelection(PubMedError):
    """No reading of guessed IDs or papers obtained in a previous episode."""


def evidence_view(paper, *, read=False):
    abstract = paper.get("abstract") or ""
    text = abstract[:ABSTRACT_CHARACTERS if read else EXCERPT_CHARACTERS]
    truncated = bool(paper.get("abstract_truncated")) or len(abstract) > len(text)
    kind = "title_only" if not text else "abstract_excerpt" if truncated else "abstract"
    view = {"id": "pending", "title": paper["title"][:500], "text": text,
            "text_kind": kind, "truncated": truncated, "url": paper["url"],
            "pmid": paper["pmid"], "year": paper.get("year")}
    digest = hashlib.sha256(json.dumps({key: value for key, value in view.items() if key != "id"},
                                     sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    # Views are immutable: a longer read does not overwrite an earlier excerpt.
    view["id"] = "P" + paper["pmid"] + "_" + ("read_" if read else "search_") + digest
    return EvidenceRecord.model_validate(view)


class ClinicalLiteratureTools:
    """One instance per episode; raw HTTP snapshots remain outside model input."""

    def __init__(self, client: PubMedClient):
        self.client = client
        self.found = {}
        self.events = []

    def definitions(self):
        return [ToolDefinition(
            "search_pubmed",
            "Search PubMed for papers relevant to a clinical question or a named trial. Optional: answer directly when existing knowledge is adequate. Returns at most five real titles, publication metadata and up to 400 abstract characters, with source IDs. An excerpt may omit the needed finding. Use read_pubmed_abstract with returned numeric PMIDs to inspect more text. No full text; PubMed indexing is not quality appraisal.",
            ClinicalSearchArgs, self.search), ToolDefinition(
            "read_pubmed_abstract",
            "Read up to three abstracts for numeric PMIDs actually returned by this episode's PubMed searches. Returns up to 3500 characters per abstract, explicitly flagged if missing/truncated. Cite the returned source IDs; search/read IDs are distinct immutable views. Reads abstracts, not full-text articles. Do not guess IDs or assume a paper supports a claim from its title.",
            ReadAbstractArgs, self.read)]

    @staticmethod
    def metadata(paper, view):
        return {"pmid": paper["pmid"], "source_id": view.id, "title": view.title,
                "title_truncated": len(paper["title"]) > 500, "year": paper.get("year"),
                "journal": (paper.get("journal") or "")[:150], "doi": (paper.get("doi") or "")[:200],
                "url": paper["url"], "abstract_present": paper.get("abstract_present"),
                "preprint": paper.get("preprint"), "expression_of_concern": paper.get("expression_of_concern"),
                "text_kind": view.text_kind, "text_truncated": view.truncated,
                "model_visible_characters": len(view.text)}

    def _operation(self, name, arguments, operation):
        start = len(self.client.requests)
        event = {"name": name, "arguments": arguments.model_dump(mode="json"), "started_at_utc": utc_now()}
        try:
            papers, metadata = operation()
            views = [evidence_view(paper, read=name == "read_pubmed_abstract") for paper in papers]
            content = {**metadata, "records": [self.metadata(paper, view) for paper, view in zip(papers, views)],
                       "limitation": "No full text. Source identity does not establish relevance or claim support. Search excerpts can omit findings; absent abstracts/empty searches are not evidence that literature does not exist.",
                       "ncbi_disclaimer_url": "https://www.ncbi.nlm.nih.gov/About/disclaimer.html"}
            output = ToolOutput(content=json.dumps(content, ensure_ascii=False), sources=views)
            event.update(status="ok", papers=copy.deepcopy(papers), generation_view=output.model_dump(mode="json"))
            return output
        except Exception as exc:
            event.update(status="error", error_type=type(exc).__name__, papers=[])
            raise
        finally:
            event["http_requests"] = copy.deepcopy(self.client.requests[start:])
            self.events.append(event)

    def search(self, arguments: ClinicalSearchArgs):
        def perform():
            result = self.client.search(arguments.query, arguments.max_results, arguments.sort)
            for paper in result["sources"]:
                self.found[paper["pmid"]] = copy.deepcopy(paper)
            return result["sources"], {key: result[key] for key in ["query", "effective_query", "total_matches", "query_translation", "warnings", "excluded_records"]}
        return self._operation("search_pubmed", arguments, perform)

    def read(self, arguments: ReadAbstractArgs):
        def perform():
            if any(pmid not in self.found for pmid in arguments.pmids):
                raise UnknownPubMedSelection("Selection was not returned by a search in this episode.")
            xml = self.client._request("efetch.fcgi", {"db": "pubmed", "id": ",".join(arguments.pmids), "retmode": "xml"}, "xml")
            parsed = {paper["pmid"]: paper for paper in parse_pubmed_xml(xml, self.client.config.max_abstract_characters)}
            papers, excluded = [], []
            for pmid in arguments.pmids:
                paper = parsed.get(pmid)
                if paper is None:
                    excluded.append({"pmid": pmid, "reason": "No parseable record returned."})
                elif paper["retracted"] or paper["retraction_notice"]:
                    excluded.append({"pmid": pmid, "reason": "Retraction metadata detected on reading."})
                else:
                    paper["retrieved_at_utc"] = utc_now()
                    papers.append(paper)
            return papers, {"requested_pmids": arguments.pmids, "excluded_records": excluded,
                            "operation": "Fresh PubMed EFetch of previously found IDs; not a preloaded evaluator abstract."}
        return self._operation("read_pubmed_abstract", arguments, perform)

    def public_record(self):
        return {"events": copy.deepcopy(self.events), "http_requests": copy.deepcopy(self.client.requests),
                "interpretation": "Raw paper metadata/abstracts are researcher artifacts. Only generation_view plus the graph's evidence registry were delivered to the model."}
