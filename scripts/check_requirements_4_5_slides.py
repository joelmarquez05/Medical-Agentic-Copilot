"""Verify presentation preservation and metrics using local artifacts only."""
import hashlib
import json
import re
import struct
import zlib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "docs/presentation/requirements-4-5-image-generation.json"
SPEC = ROOT / "docs/presentation/requirements-4-5-slides.md"
RECEIPT = ROOT / "data/checks/requirements-4-5-slides-20261012.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_dimensions(path):
    """Check PNG chunk integrity and header without adding imaging dependencies."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position, dimensions, seen_data = 8, None, False
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        end = position + 8 + length
        assert end + 4 <= len(data)
        payload = data[position + 8:end]
        crc = struct.unpack(">I", data[end:end + 4])[0]
        assert zlib.crc32(kind + payload) & 0xffffffff == crc
        if kind == b"IHDR":
            assert position == 8 and length == 13
            dimensions = list(struct.unpack(">II", payload[:8]))
        elif kind == b"IDAT":
            seen_data = True
        elif kind == b"IEND":
            assert length == 0 and end + 4 == len(data) and seen_data
            return dimensions
        position = end + 4
    raise AssertionError("Incomplete PNG")


def verify():
    metadata = read(META)
    script = ROOT / "guion.md"
    data = script.read_bytes()
    preservation = metadata["preservation"]
    size = preservation["original_script_bytes"]
    assert hashlib.sha256(data[:size]).hexdigest() == preservation["original_script_sha256"]
    for name, expected in preservation["original_slides"].items():
        assert digest(ROOT / "slides" / name) == expected
    markers = list(map(int, re.findall(r"^\[Slide (\d+)\]$", data.decode("utf-8-sig"), re.M)))
    assert markers == list(range(23)), markers
    assert len(preservation["original_slides"]) == 15
    identifiers = {str(i) for i in range(15, 23)}
    assert set(metadata["prompts"]) == set(metadata["outputs"]) == identifiers
    images = []
    for number in range(15, 23):
        output = metadata["outputs"][str(number)]
        assert output["visually_reviewed"] is True
        target = ROOT / "slides" / ("diapositiva_" + str(number) + ".png")
        assert target == ROOT / output["workspace"]
        assert digest(target) == digest(Path(output["source"]))
        dimensions = png_dimensions(target)
        width, height = dimensions
        assert abs(width / height - 16 / 9) < 0.015
        assert width >= 1500 and height >= 850
        images.append({"slide_index": number, "printed_page": number + 1, "path": output["workspace"],
                       "sha256": digest(target), "dimensions": dimensions, "manual_visual_review": True})

    benchmark = ROOT / "data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"
    comparison = ROOT / "data/evaluation/comparison-runs/medqa30-four-model-main-20261011"
    selection = read(ROOT / "data/evaluation/medqa-benchmark-v3-20261011/manifest.json")
    assert (selection["source_rows"], selection["eligible_rows"], len(selection["excluded_rows"]), selection["question_count"], selection["seed"]) == (1273, 1076, 197, 30, 202610114)
    summaries = {"gemma": read(benchmark / "summary.json"), **read(comparison / "summary.json")["models"]}
    appraisals = {"gemma": read(benchmark / "assessment-summary-v2.json"), **read(comparison / "assessment-summary.json")["models"]}
    gemma_reviews = read(benchmark / "answer-assessments-v2.json")["reviews"]
    gemma_material = sum("material_issue" in r["dimension_ratings"].values() for r in gemma_reviews)
    expected = {
        "gemma": (30, 17, 22, 1, 29, 21, 3, 4, 3, 1, 2),
        "glm-5.3": (29, 20, 23, 2, 26, 23, 3, 23, 11, 2, 0),
        "kimi-k3": (29, 24, 35, 5, 28, 28, 2, 25, 16, 0, 0),
        "mimo-v2.6-pro": (27, 30, 43, 8, 24, 22, 6, 50, 17, 0, 0),
        "qwen3.8-27b": (29, 19, 23, 4, 28, 28, 1, 18, 7, 1, 0),
    }
    observed = {}
    for slug, numbers in expected.items():
        summary, review = summaries[slug], appraisals[slug]
        assert summary["planned"] == summary["valid_completed"] == review["answer_reviews"] == 30
        counts = review["answer_dimension_counts"]
        support = review["citation_support"]
        material = gemma_material if slug == "gemma" else len(review["material_issue_cases"])
        actual = (summary["correct"], summary["episodes_with_search"], summary["tool_names"]["search_pubmed"],
                  summary["tool_names"]["read_pubmed_abstract"], counts["case_reasoning"]["consistent"],
                  counts["reference_concordance"]["consistent"], material, support.get("supported", 0),
                  support.get("partially_supported", 0), support.get("unsupported", 0), support.get("not_assessable", 0))
        assert actual == numbers, (slug, actual)
        observed[slug] = {"key_matches": actual[0], "cases_with_search": actual[1], "search_calls": actual[2],
            "read_calls": actual[3], "consistent_case_reasoning": actual[4], "consistent_reference_concordance": actual[5],
            "material_issue_cases": material, "citation_support": support}
    assert sum(r["paper_pairs"] for slug, r in appraisals.items() if slug != "gemma") == 348
    assert appraisals["gemma"]["unique_question_paper_pairs"] == 42
    cost = read(ROOT / "data/checks/requirement-5-cost-audit-20261011.json")
    assert round(cost["known_reported_or_estimated_total_usd"], 3) == 0.893
    assert round(cost["latest_ledger"]["total_usage"], 3) == 4.802
    assert read(ROOT / "data/checks/requirement-5-final-artifacts-20261011.json")["full_test_suite"]["previously_observed_passed"] == 184

    links = 0
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", SPEC.read_text(encoding="utf-8")):
        if urlsplit(target).scheme or target.startswith("#"):
            continue
        local = (SPEC.parent / unquote(target.split("#")[0])).resolve()
        if local == RECEIPT:
            continue
        assert local.exists(), target
        links += 1
    result = {"date": "2026-10-12", "passed": True, "generation_mode": "built-in image_gen",
              "new_slide_indices": list(range(15, 23)), "earlier_script_unchanged": True,
              "preserved_script_bytes": size, "preserved_script_sha256": preservation["original_script_sha256"],
              "earlier_15_slide_hashes_unchanged": True, "slide_markers": markers, "new_slides": images,
              "prompt_metadata_sha256": digest(META), "script_sha256": digest(script), "specification_sha256": digest(SPEC),
              "numerical_source_crosschecks": observed, "specification_local_links_checked": links,
              "model_and_pubmed_evaluation_requests": 0, "test_suite_rerun": False,
              "limits": "Image text/layout manually inspected; checks validate files, preservation and source metrics, not automatic OCR or independent clinical judgment. Generated chart bars are illustrative; labels carry exact counts."}
    return result


if __name__ == "__main__":
    result = verify()
    if not RECEIPT.exists():
        with RECEIPT.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps({"passed": True, "new_slides": len(result["new_slides"]), "earlier_slides_preserved": 15,
                      "continuous_script_markers": len(result["slide_markers"]), "new_evaluation_requests": 0}))
