"""Register every cited or assessed real paper and the technical sources."""
from pathlib import Path
from medical_copilot.evaluation import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    technical = [
        {"id": "R3T01", "title": "A General Introduction to the E-utilities", "authors": "Eric Sayers / National Center for Biotechnology Information", "year": "Living documentation, accessed 2026", "url": "https://www.ncbi.nlm.nih.gov/books/NBK25497/", "supports": "Request-rate limits, client identification, and batching. The containing Entrez Programming Utilities Help book is dated 2010 onward."},
        {"id": "R3T02", "title": "The 9 E-utilities and Associated Parameters", "authors": "National Library of Medicine", "year": "Living documentation, accessed 2026", "url": "https://www.nlm.nih.gov/dataguide/eutilities/utilities.html", "supports": "ESearch sorting, query parameters and batch EFetch."},
        {"id": "R3T03", "title": "PubMed User Guide", "authors": "National Library of Medicine", "year": "Living documentation, accessed 2026", "url": "https://pubmed.ncbi.nlm.nih.gov/help/", "supports": "Boolean/field-tag searching, automatic term mapping, indexing/filter caveats, abstracts, retractions, preprints and data notices."},
    ]
    papers = {}
    for run_id in ["gemini-pubmed-internal-20261010", "gemini-pubmed-final-20261010"]:
        folder = ROOT / "data/evaluation/retrieval-runs" / run_id / "retrieval"
        for path in sorted(folder.glob("*.json")):
            record = read_json(path)
            for source in record["unique_papers"]:
                entry = papers.setdefault(source["pmid"], {key: source.get(key) for key in ["id", "pmid", "title", "authors", "author_list", "year", "journal", "doi", "url", "publication_types", "preprint", "read_level", "abstract_truncated", "copyright_information"]})
                entry.setdefault("observed_in", []).append({"run_id": run_id, "question_id": record["question_id"]})
    for source in read_json(ROOT / "data/checks/pubmed-preflight.json")["sources"]:
        entry = papers.setdefault(source["pmid"], {key: source.get(key) for key in ["id", "pmid", "title", "authors", "author_list", "year", "journal", "doi", "url", "publication_types", "preprint", "read_level", "abstract_truncated", "copyright_information"]})
        entry.setdefault("observed_in", []).append({"run_id": "pubmed-preflight", "question_id": None})
    for entry in papers.values():
        contexts = sorted({item["question_id"] for item in entry["observed_in"] if item["question_id"]})
        entry["supports"] = "Observed retrieved record and relevance assessment for " + (", ".join(contexts) if contexts else "the API preflight") + "; relevance is not clinical endorsement."
    catalogue = {"catalogue_id": "requirement-3-sources-20261010", "technical_sources": technical, "papers": [papers[key] for key in sorted(papers, key=int)], "provenance": "Bibliographic metadata from saved real NCBI EFetch responses. Titles/abstracts were the appraisal basis; no full-text appraisal is claimed. Full authors remain in the metadata. Local guidance and LangChain sources reuse the requirement-2 catalogue/journal."}
    write_json(ROOT / "data/research/requirement-3-source-catalogue.json", catalogue)
    text = ["# Requirement 3 source record\n", catalogue["provenance"] + "\n"]
    for entry in technical + catalogue["papers"]:
        authors = entry["authors"]
        if len(entry.get("author_list") or []) > 3:
            authors = ", ".join(entry["author_list"][:3]) + ", and colleagues"
        identifiers = (" PMID " + entry["pmid"] if entry.get("pmid") else "") + ("; DOI " + entry["doi"] if entry.get("doi") else "")
        text.append(f"- **{entry['id']}** - {authors} ({entry['year']}), [{entry['title']}]({entry['url']}).{identifiers} Supports: {entry['supports']}" + (f" Read level: {entry['read_level']}." if entry.get("read_level") else ""))
    (ROOT / "docs/implementation/requirement-3-sources.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Registered {len(papers)} real paper records and {len(technical)} technical sources.")


if __name__ == "__main__":
    main()
