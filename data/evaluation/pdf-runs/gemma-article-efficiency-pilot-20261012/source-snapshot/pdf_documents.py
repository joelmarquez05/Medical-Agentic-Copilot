"""Local, page-aware PDF ingestion and lexical retrieval with immutable views."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pydantic import Field

from medical_copilot.clinical_graph import EvidenceRecord, StrictModel, ToolDefinition, ToolOutput, save_trace
from medical_copilot.evaluation import file_hash, read_json

VERSION = "page-aware-pdf-v2"
CHUNK_CHARS = 4000
STOPWORDS = set("a an and are as at be by for from how in is it of on or that the their this to was what when which with according document pdf guideline recommendations recommendation please".split())


def normalize_text(text):
    """Normalize ligatures/line hyphenation; never rewrite ambiguous numeric ranges."""
    value = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    value = re.sub(r"(?<=[A-Za-z])[-\u2010]\n(?=[a-z])", "", value)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def ordered_blocks(blocks, width):
    """Read columns within bands separated by full-width blocks; retain every block."""
    middle = width / 2
    anchors = sorted([b for b in blocks if b["bbox"][0] < middle - 5 and b["bbox"][2] > middle + 5],
                     key=lambda b: (b["bbox"][1], b["bbox"][0]))
    remaining = [b for b in blocks if b not in anchors]
    output = []
    for anchor in anchors:
        before = [b for b in remaining if (b["bbox"][1] + b["bbox"][3]) / 2 < anchor["bbox"][1]]
        output.extend(sorted(before, key=lambda b: (0 if (b["bbox"][0] + b["bbox"][2]) / 2 < middle else 1,
                                                  b["bbox"][1], b["bbox"][0])))
        remaining = [b for b in remaining if b not in before]
        output.append(anchor)
    output.extend(sorted(remaining, key=lambda b: (0 if (b["bbox"][0] + b["bbox"][2]) / 2 < middle else 1,
                                                 b["bbox"][1], b["bbox"][0])))
    assert sorted(b["block_number"] for b in output) == sorted(b["block_number"] for b in blocks)
    return output


def positioned_blocks(page):
    """Split merged column blocks using individual line geometry, retaining originals."""
    import pymupdf

    originals, positioned = [], []
    middle = page.rect.width / 2
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT, sort=False)
    for block in data["blocks"]:
        if block["type"] != 0:
            continue
        lines = [{"bbox": list(line["bbox"]),
                  "raw_text": "".join(span["text"] for span in line["spans"])} for line in block["lines"]]
        if not any(line["raw_text"].strip() for line in lines):
            continue
        originals.append({"block_number": block["number"], "bbox": list(block["bbox"]), "lines": lines})
        sides = ["wide" if line["bbox"][0] < middle - 5 and line["bbox"][2] > middle + 5
                 else "left" if sum(line["bbox"][::2]) / 2 < middle else "right" for line in lines]
        # A true wide line remains an anchor; only separate narrow column lines are split.
        groups = [[line for line, side in zip(lines, sides) if side == target] for target in ("left", "right")] \
            if "wide" not in sides and "left" in sides and "right" in sides else [lines]
        for group in groups:
            group = sorted(group, key=lambda line: (line["bbox"][1], line["bbox"][0]))
            raw_text = "\n".join(line["raw_text"] for line in group)
            bbox = [min(line["bbox"][0] for line in group), min(line["bbox"][1] for line in group),
                    max(line["bbox"][2] for line in group), max(line["bbox"][3] for line in group)]
            positioned.append({"block_number": len(positioned), "original_block_number": block["number"],
                               "bbox": bbox, "raw_text": raw_text, "text": normalize_text(raw_text)})
    return positioned, originals


def split_page(text, max_chars=CHUNK_CHARS):
    """Nonoverlapping offsets cover the exact entire extracted page text."""
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + max_chars // 2, end)
            if boundary != -1:
                end = boundary + 1
        yield start, end, text[start:end]
        start = end


def ingest_pdf(path, destination, *, title=None, url=None, year=None):
    """Extract an explicit local input; refuse replacement of an existing ingestion."""
    import pymupdf

    path, destination = Path(path).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Ingestion requires a fresh directory.")
    if path.stat().st_size > 50 * 1024 * 1024 or path.read_bytes()[:5] != b"%PDF-":
        raise ValueError("Expected a PDF under 50 MiB.")
    digest = file_hash(path)
    identifier = "D" + digest[:12]
    name = title or path.stem
    EvidenceRecord(id="validation", title=name, text_kind="pdf_excerpt", url=url, year=year)
    pages, views, locators = [], {}, {}
    with pymupdf.open(path) as document:
        if document.needs_pass or not 0 < document.page_count <= 250:
            raise ValueError("Encrypted or unsupported page-count document.")
        for number, page in enumerate(document, 1):
            raw_blocks, original_blocks = positioned_blocks(page)
            blocks = ordered_blocks(raw_blocks, page.rect.width)
            text = "\n\n".join(b["text"] for b in blocks)
            footer = [b["text"] for b in blocks if b["bbox"][1] > page.rect.height * .9 and re.fullmatch(r"\d{1,5}", b["text"])]
            chunk_ids = []
            for ordinal, (start, end, content) in enumerate(split_page(text), 1):
                locator = {"document_id": identifier, "pdf_page": number, "printed_page": footer[-1] if footer else None,
                           "chunk": ordinal, "start_character": start, "end_character": end}
                view_data = {"title": f"{name} | PDF page {number}", "text": content, "text_kind": "pdf_excerpt",
                             "truncated": False, "url": (url.split("#")[0] + "#page=" + str(number)) if url else None,
                             "pmid": None, "year": year}
                token = hashlib.sha256(json.dumps({"pdf_sha256": digest, **locator, **view_data}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
                sid = f"{identifier}_p{number}_c{ordinal}_{token}"
                views[sid] = EvidenceRecord(id=sid, **view_data).model_dump(mode="json")
                locators[sid] = locator
                chunk_ids.append(sid)
            pages.append({"pdf_page": number, "printed_page": footer[-1] if footer else None,
                          "text": text, "blocks": blocks, "original_blocks": original_blocks, "chunk_ids": chunk_ids,
                          "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "embedded_image_count": len(page.get_images()), "no_selectable_text": not bool(text.strip())})
        parser_version = pymupdf.VersionBind
    if not any(p["text"].strip() for p in pages):
        raise ValueError("No selectable text found; OCR is not implemented.")
    destination.mkdir(parents=True)
    with (destination / "original.pdf").open("xb") as output:
        output.write(path.read_bytes())
    record = {"version": VERSION, "document_id": identifier, "title": name, "url": url, "year": year,
              "pdf_sha256": digest, "page_count": len(pages), "parser": "PyMuPDF", "parser_version": parser_version,
              "pages": pages, "views": views, "locators": locators,
              "extraction": "Merged narrow column lines split using geometry, then blocks grouped into full-width bands and left/right columns; Unicode ligatures and word line-hyphenation normalized. Original lines/blocks/coordinates retained; numeric ranges not rewritten.",
              "limits": "Text layer only; no OCR, image/flowchart reasoning, table semantics or universal reading-order guarantee. Unknown glyph mappings and cross-page continuations remain visible."}
    save_trace(destination / "document.json", record)
    return record


def tokens(text):
    return [word for word in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold()) if word not in STOPWORDS]


class PDFLibrary:
    def __init__(self, records):
        self.documents = {r["document_id"]: copy.deepcopy(r) for r in records}
        if len(self.documents) != len(records) or not records:
            raise ValueError("Use one or more unique document records.")
        for record in records:
            assert record["version"] == VERSION
            for page in record["pages"]:
                assert "".join(record["views"][sid]["text"] for sid in page["chunk_ids"]) == page["text"]
                assert hashlib.sha256(page["text"].encode()).hexdigest() == page["text_sha256"]

    @classmethod
    def load(cls, paths):
        return cls([read_json(Path(path)) for path in paths])

    def catalog(self):
        return [{k: r[k] for k in ["document_id", "title", "pdf_sha256", "page_count", "limits"]} for r in self.documents.values()]

    def document(self, identifier):
        if identifier not in self.documents:
            raise ValueError("Document is not in this session's catalog.")
        return self.documents[identifier]

    def all_views(self):
        return copy.deepcopy({sid: view for r in self.documents.values() for sid, view in r["views"].items()})

    def search(self, identifier, query, max_results=4):
        record = self.document(identifier)
        queries = set(tokens(query))
        corpus = [(sid, Counter(tokens(v["text"]))) for sid, v in record["views"].items()]
        if not queries:
            return []
        size = len(corpus)
        average = sum(sum(c.values()) for _, c in corpus) / max(size, 1)
        frequencies = {term: sum(term in c for _, c in corpus) for term in queries}
        ranked = []
        for sid, counts in corpus:
            length, score = sum(counts.values()), 0.0
            for term in queries:
                count = counts[term]
                if count:
                    idf = math.log(1 + (size - frequencies[term] + .5) / (frequencies[term] + .5))
                    score += idf * count * 2.5 / (count + 1.5 * (.25 + .75 * length / max(average, 1)))
            if score > 0:
                ranked.append((score, record["locators"][sid]["pdf_page"], record["locators"][sid]["chunk"], sid))
        return [(sid, round(score, 6)) for score, _, _, sid in sorted(ranked, key=lambda x: (-x[0], x[1], x[2]))[:max_results]]


class SearchPDFArgs(StrictModel):
    document_id: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=4, ge=1, le=5)


class ReadPDFPageArgs(StrictModel):
    document_id: str = Field(min_length=1, max_length=80)
    page: int = Field(ge=1, le=250)


class PDFTools:
    """Per-episode operations against explicit ingested documents; no network."""
    def __init__(self, library):
        self.library, self.events = library, []

    def definitions(self):
        return [ToolDefinition("search_pdf", "Search the ingested PDF text with a short specific lexical query. Use an exact recommendation/topic/drug when useful. Returns up to five bounded passages with immutable source IDs and PDF page numbers; zero lexical hits do not prove semantic absence. No OCR or image understanding.", SearchPDFArgs, self.search),
                ToolDefinition("read_pdf_page", "Read all extracted text on one one-based PDF page of a catalog document. Pages can be selected using search results or the document catalog. Returns its nonoverlapping chunks and source IDs. Printed journal page and PDF page are different. No full-image or table interpretation.", ReadPDFPageArgs, self.read)]

    def _output(self, name, arguments, selections, record, **context):
        selected = [record["views"][sid] for sid, _ in selections]
        metadata = [{"source_id": sid, **record["locators"][sid], "lexical_score": score} for sid, score in selections]
        output = ToolOutput(content=json.dumps({"document_id": record["document_id"], "searchable_pages": record["page_count"],
            "results": metadata, "limitation": record["limits"], **context}, ensure_ascii=False), sources=selected)
        self.events.append({"name": name, "arguments": arguments.model_dump(mode="json"), "status": "ok",
                            "generation_view": output.model_dump(mode="json")})
        return output

    def search(self, arguments):
        record = self.library.document(arguments.document_id)
        selections = self.library.search(arguments.document_id, arguments.query, arguments.max_results)
        return self._output("search_pdf", arguments, selections, record, query=arguments.query,
                            no_matching_passages=not bool(selections), method="BM25-style lexical ranking of all indexed chunks")

    def read(self, arguments):
        record = self.library.document(arguments.document_id)
        if arguments.page > record["page_count"]:
            raise ValueError("Page is outside this document.")
        ids = record["pages"][arguments.page - 1]["chunk_ids"]
        if len(ids) > 5:
            raise ValueError("This page exceeds the tool view allowance; use selective search.")
        return self._output("read_pdf_page", arguments, [(sid, None) for sid in ids], record, entire_extracted_page_delivered=True)
