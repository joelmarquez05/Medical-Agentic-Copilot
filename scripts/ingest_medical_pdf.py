"""Extract a local PDF into a fresh page-aware document directory."""
import argparse
import json
from pathlib import Path
from medical_copilot.pdf_documents import ingest_pdf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pdf", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--title")
parser.add_argument("--url")
parser.add_argument("--year")
args = parser.parse_args()
record = ingest_pdf(args.pdf, args.output_dir, title=args.title, url=args.url, year=args.year)
print(json.dumps({"document_id": record["document_id"], "pages": record["page_count"], "chunks": len(record["views"]),
                  "extracted_text_bytes": sum(len(p["text"].encode()) for p in record["pages"]),
                  "empty_text_pages": [p["pdf_page"] for p in record["pages"] if p["no_selectable_text"]]}))
