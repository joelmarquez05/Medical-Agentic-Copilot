"""Build three concise, selectable-text fictional case PDFs for the guided demo."""
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "data/examples/clinical-demo-20261016.json").read_text(encoding="utf-8"))
    directory = ROOT / "demo/cases"
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for item in [*data["patients"], data["followup"]]:
        document = pymupdf.open()
        page = document.new_page(width=595, height=842)
        page.draw_rect(pymupdf.Rect(42, 42, 46, 97), color=None, fill=(.73, 0, 0))
        page.insert_text((59, 56), "FICTIONAL TEACHING CASE", fontsize=9, fontname="hebo", color=(.73, 0, 0))
        page.insert_text((59, 80), item["title"], fontsize=16, fontname="hebo", color=(.1, .1, .1))
        page.insert_text((59, 98), "Project-authored vignette | No real patient information", fontsize=9, color=(.4, .4, .4))
        y = 128
        for heading, text in item["sections"]:
            page.insert_text((46, y), heading, fontsize=10.5, fontname="hebo", color=(.15, .15, .15))
            y += 8
            height = 94
            available = page.insert_textbox(pymupdf.Rect(46, y, 549, y + height), text,
                                            fontsize=10.5, fontname="helv", lineheight=1.4, color=(.22, .22, .22))
            assert available >= 0, f"Section does not fit: {heading}"
            y += height - available + 22
        assert y < 790, (item["filename"], y)
        page.draw_line((46, 800), (549, 800), color=(.85, .85, .85))
        page.insert_text((46, 817), "Clinical research workspace | Fictional demo | 1 / 1", fontsize=8, color=(.45, .45, .45))
        document.set_metadata({"title": item["title"], "author": "Medical Copilot project - fictional teaching data"})
        destination = directory / item["filename"]
        document.save(destination, garbage=4, deflate=True, no_new_id=True)
        document.close()
        with pymupdf.open(destination) as check:
            assert len(check) == 1 and len(check[0].get_text()) > 900
        results.append({"file": str(destination.relative_to(ROOT)), "pages": 1})
    print(json.dumps(results))


if __name__ == "__main__":
    main()
