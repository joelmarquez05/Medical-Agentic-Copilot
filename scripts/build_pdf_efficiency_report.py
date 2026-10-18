"""Render the new article-detail efficiency study from actual saved records/reviews."""
import json
import statistics
from pathlib import Path
from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.pdf_efficiency_agent import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/pdf-runs/gemma-article-efficiency-main-20261012"

def render():
    manifest, summary = read_json(RUN / "manifest.json"), read_json(RUN / "summary.json")
    questions = read_json(RUN / "questions.json")["questions"]
    keys = {k["question_id"]: k for k in read_json(RUN / "answer-key.json")["keys"]}
    reviews = read_json(RUN / "manual-reviews.json")["reviews"]
    by_review = {(r["condition"], r["question_id"]): r for r in reviews}
    assert len(by_review) == 12
    b, c = [dict(summary["conditions"][arm]) for arm in CONDITIONS]
    reviewed_scores = {arm: read_json(RUN / arm / "scores-reviewed.json")["scores"] for arm in CONDITIONS}
    for arm, s in zip(CONDITIONS, [b, c]):
        s["correct_fields_with_cited_reference_passage"] = sum(row["correct_fields_with_cited_reference_passage"] for row in reviewed_scores[arm])
    efficiency = {"input_token_reduction_percent": (1 - c["input_tokens"] / b["input_tokens"]) * 100,
        "reported_cost_reduction_percent": (1 - c["cost_usd"] / b["cost_usd"]) * 100,
        "total_episode_time_reduction_percent": (1 - c["total_episode_seconds"] / b["total_episode_seconds"]) * 100,
        "median_episode_time_reduction_percent": (1 - c["median_episode_seconds"] / b["median_episode_seconds"]) * 100,
        "quality_reference": "24 correlated article fields in six cases, plus actual cited-passage and manual qualifier checks",
        "old_ABC_percentages_reused": False, "complete_quality_preserved": b["correct_fields"] == c["correct_fields"] == 24}
    write_json(RUN / "efficiency.json", efficiency)
    lines = ["# Requirement 6: article-grounded PDF access efficiency", "",
        "The active evaluation now has **two document-required conditions**: complete extracted article in context, or intelligent local PDF search/page reading over exactly the same text. The user removed knowledge-only answering and refocused the task on efficiency. Six new open-response questions extract concrete study details from the article, with 24 frozen numeric/boolean reference checks. Earlier ABC results remain historical and are not reused as this study's outcomes.", "",
        "## Efficiency with content checks", "",
        "| Condition | Correct reference fields | Correct fields with cited reference passage | Cases with every field correct | Input tokens | Requests | Cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in CONDITIONS:
        s = b if arm == "B-full-context" else c
        lines.append(f"| {arm} | {s['correct_fields']}/24 | {s['correct_fields_with_cited_reference_passage']}/24 | {s['all_fields_correct_cases']}/6 | {s['input_tokens']:,} | {s['provider_requests']} | {s['cost_usd']:.8f} |")
    lines += ["", f"Retrieval used **{efficiency['input_token_reduction_percent']:.1f}% fewer input tokens** and **{efficiency['reported_cost_reduction_percent']:.1f}% lower recorded inference charges** than full context in this cohort. **This saving did not preserve complete coverage:** C returned null for three requested fields in ART05/ART06, while B recovered all twenty-four. Do not describe this as equally complete answers delivered more efficiently. These are new observed ratios; membership in the registry alone is not passage support.", "",
        "| Condition | Output tokens | Reasoning tokens within output | Total question wall seconds | Median question wall seconds | Successful PDF operations |",
        "|---|---:|---:|---:|---:|---|"]
    for arm in CONDITIONS:
        s = summary["conditions"][arm]
        lines.append(f"| {arm} | {s['output_tokens']:,} | {s['reasoning_tokens']:,} | {s['total_episode_seconds']:.3f} | {s['median_episode_seconds']:.3f} | {s['successful_pdf_operations']} |")
    lines += ["", "Wall times cover each graph episode, including its model calls/local tools, excluding stage preflight/setup. They are single-run descriptive observations with uncontrolled provider queue/cache conditions. Fewer input tokens do not guarantee lower latency; repeated tool/model calls can dominate. Available input-cache metadata remains in raw records, without claiming cache control.", "",
        "## Paired cases", "", "| Case | B correct fields | C correct fields | B/C cited support | B/C input tokens | B/C seconds |", "|---|---:|---:|---|---|---|"]
    rows = {}
    scores = {arm: {s["question_id"]: s for s in reviewed_scores[arm]} for arm in CONDITIONS}
    for q in questions:
        identifier = q["question_id"]
        cases = []
        for arm in CONDITIONS:
            row = read_json(RUN / arm / "answers" / (identifier + ".json"))
            rows[(arm, identifier)] = row
            assert file_hash(RUN / arm / "answers" / (identifier + ".json")) == by_review[(arm, identifier)]["answer_sha256"]
            cases.append(row)
        sb, sc = [scores[arm][identifier] for arm in CONDITIONS]
        input_counts = [sum(a.get("usage", {}).get("input_tokens", 0) for a in row["response_audit"]) for row in cases]
        lines.append(f"| {identifier} | {sb['correct_fields']}/4 | {sc['correct_fields']}/4 | {sb['correct_fields_with_cited_reference_passage']}/4 / {sc['correct_fields_with_cited_reference_passage']}/4 | {input_counts[0]:,} / {input_counts[1]:,} | {cases[0]['elapsed_seconds']:.3f} / {cases[1]['elapsed_seconds']:.3f} |")
    lines += ["", "## Source, questions and evaluation method", "",
        "Muscaritoli, Arends, Bachmann and colleagues/ESPEN, *ESPEN practical guideline: Clinical Nutrition in cancer* (2021), DOI [10.1016/j.clnu.2021.02.005](https://doi.org/10.1016/j.clnu.2021.02.005), PMID 33946039, [official PDF](https://www.espen.org/files/ESPEN-Guidelines/ESPEN-practical-guideline-clinical-nutrition-in-cancer.pdf). It is a guideline article summarizing studies, not a newly selected single trial. [Full source/read-scope catalogue](../../../research/requirement-6-article-efficiency-sources-20261012.json). The questions concern what this fixed article reports, not whether the reported regimens should be prescribed or original trials independently replicated.", "",
        "Original-render-verified topics: hypermetabolic study percentages and energy range; physician vitamin-trial count/doses/follow-up; HMB-mixture RCT count/duration/completion/significance; cannabinoid trial versus pilot details; three-arm cachexia-trial count/doses across two consecutive pages; glutamine review versus controlled-subset counts. [Frozen questions](../../pdf-article-details-20261012/questions.md), [numeric/boolean keys](answer-key.json), [field-to-passage keys](field-evidence-key.json). All reference values/eligibility were frozen before new generation. No MC answer options or missing-information case. The model sees requested field names/types but not correct values, gold passages or field-to-source mapping.", "",
        "Code parses the final answer object's numeric/boolean fields and compares them to the official-document references. Missing, null, wrong-type and mismatched fields remain distinct. Booleans are not treated as numbers; extra/missing field names are a separate format issue. A correct field is separately credited for cited support only when its containing passage is actually supplied and cited. Twenty-four fields are correlated within six questions. This mechanical passage eligibility supplements, rather than replaces, [manual reviews](manual-reviews.json) of exact wording, qualifiers and source content; no LLM assigns the reference answers.", "",
        "One source-eligibility metadata correction is explicit: ART06 initially allowed either page-10 chunk for each field, although whole-review counts occur in chunk 1 and controlled-subset counts in chunk 2. [Reviewed eligibility](field-evidence-key-reviewed.json) narrows each field to its actual containing chunk after full-view review. Original numeric/boolean values, questions, source eligibility, raw scores, prompts and generations are unchanged; no source is retroactively supplied. Reviewed support counts remain B 24/C 21, identical to the original counts. Frozen value keys still predate inference; corrected passage eligibility is not misrepresented as pre-generation. Original-score replay and separate reviewed-score reproduction remain available.", "",
        "## Retrieval findings and incomplete cases", "",
        "C ART05 searched the three-arm trial, reread page 7 and then used reserved finalization after a malformed research response. It never read page 8, so dexamethasone_mg_per_day remained null even though the next page contains it. C ART06 obtained page-10 chunk 1, searched its dangling phrase and read page 11; the missing subset counts are in chunk 2 of the **same page 10**, which was never supplied. Its two null fields are honest abstentions for missing delivered evidence, not false numbers or actual absence from the article. These cases show a navigation/continuation limitation: the agent confuses chunk boundaries with document-page continuations. Future generic adjacent-chunk/full-page expansion could address this; it was not retroactively added or used to replace these outcomes.", "",
        "C's four single-passage cases are complete and faster than their paired B cases in this run. The two difficult continuation cases dominate its total time; C's median is lower while its total is higher. All six cases remain in the primary aggregate, with no favorable-case selection. Neither generic provider speedup nor a uniform efficiency gain at equal quality is established.", "",
        "## Agent and reproducibility", "",
        "Both conditions use the same Gemma 4 26B A4B IT/NextBit profile, enabled reasoning, temperature zero, 16,384 output cap, 180-second timeout, common article-only prompt and 131,072-byte bound. Three research actions/two reserved final attempts fit within five turns. Every episode is fresh. PubMed, legacy curated guidance, other PDFs and previous answers are disabled. Full context supplies all thirty identical immutable views; C supplies them only after actual autonomous search/read execution. No embeddings or external retrieval API. The parser's selectable-text/column/numeric/image limits remain unchanged.", "",
        "The [frozen protocol](protocol.md) declares source-required routing: an unsourced factual answer returns to available research/closing within the budget; unknown source IDs are rejected without substitution. Exact known inline IDs can be added to an empty structured citation array, and an unambiguous answer object can be serialized unchanged into the required answer string; raw output and explicit normalization remain visible. The gate enforces availability/registered citations, not hidden reasoning provenance or universal semantic entailment. Original ABC operational modules/cohorts remain unchanged, while the active application CLI offers only B/C.", "",
        "A separate full-context native probe and two excluded pilots passed before twelve primary episodes. The starting condition alternates B/C then C/B. Every original response/failure and all tool operations remain stored, with no quality-based rerun or question/key change after generation. [Main offline replay](../../../checks/requirement-6-article-efficiency-main-replay-20261012.json) verifies exact prompts, schemas, bounded routing, actual evidence and field scores without new inference/credentials. The full offline suite passed **230 tests** before collection.", "",
        "## Limitations and spending", "",
        "Six convenience questions, one familiar historical article, one model/provider and unblinded assistant review do not establish general clinical performance or a statistically reliable speedup. Both conditions use identical parsing; a naive-parser comparison was not run. Source availability/valid IDs do not reveal private model reasoning or guarantee every claim is based exclusively on the document. Numeric glyph ambiguities, raster figures, scanned PDFs and table semantics remain unsupported. Persistent patient memory remains requirement 7.", "",
        f"The primary cohort's recorded cost is **USD {summary['cost_usd']:.8f}**. [New-stage cost audit](../../../checks/requirement-6-article-efficiency-cost-20261012.json) includes the new probe/pilots separately, current cumulative account observation and preserved historical uncertain charges. The standing USD 6 per-request guard / USD 0.15 historical headroom and prior PDF charge accounting remain in force. No private reasoning or credential values are saved; no compiled report or paid LLM judge was created.", "",
        "## Exact answers and qualitative observations", ""]
    for q in questions:
        identifier = q["question_id"]
        lines += ["### " + identifier, "", q["question"], "", "**Frozen article reference values:**", "", "```json", json.dumps(keys[identifier]["expected_fields"], indent=2), "```", ""]
        for arm in CONDITIONS:
            row = rows[(arm, identifier)]
            response, review = row["response"], by_review[(arm, identifier)]
            lines += ["#### " + arm, "", "**Answer (exact accepted string):**", "", "```text", response["answer"], "```", "",
                "**Justification (exact):** " + response["justification"], "", "**Citations (exact accepted IDs):** " + ", ".join(response.get("citations", [])), "",
                "**Limitations (exact):** " + json.dumps(response.get("limitations", []), ensure_ascii=False), "", "**Manual observation:** " + review["note"], ""]
    (RUN / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rendered_episodes": 12, "efficiency": efficiency, "new_model_requests": 0}))

if __name__ == "__main__":
    render()
