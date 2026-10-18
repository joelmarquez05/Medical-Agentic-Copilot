"""Render actual PDF QA results and explicit manual appraisals, with no model calls."""
import json
import re
from collections import Counter
from pathlib import Path
from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.pdf_agent import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/pdf-runs/gemma-pdf-main-20261012"


def render():
    summary, manifest = read_json(RUN / "summary.json"), read_json(RUN / "manifest.json")
    questions = read_json(RUN / "questions.json")["questions"]
    keys = {k["question_id"]: k for k in read_json(RUN / "answer-key.json")["keys"]}
    reviews = read_json(RUN / "manual-reviews.json")["reviews"]
    assert {(r["condition"], r["question_id"]) for r in reviews} == {(arm, q["question_id"]) for arm in CONDITIONS for q in questions}
    review_map = {(r["condition"], r["question_id"]): r for r in reviews}
    rows = {(arm, q["question_id"]): read_json(RUN / arm / "answers" / (q["question_id"] + ".json")) for arm in CONDITIONS for q in questions}
    for pair, row in rows.items():
        review = review_map[pair]
        assert review["answer_sha256"] == file_hash(RUN / pair[0] / "answers" / (pair[1] + ".json"))
        assert [v["source_id"] for v in review["citations"]] == row["response"].get("citations", [])
        inline = set(re.findall(r"D[0-9a-f]{12}_p\d+_c\d+_[0-9a-f]{12}", row["response"]["answer"] + " " + row["response"]["justification"])) - set(row["response"].get("citations", []))
        assert {v["source_id"] for v in review["inline_only_citations"]} == inline
        returned = {v["id"] for e in row["document_tools"]["events"] for v in e["generation_view"]["sources"]}
        assert {v["source_id"] for v in review["retrieval_views"]} == returned
        for v in review["retrieval_views"]:
            assert v["view_sha256"] == file_hash_view(row["evidence"][v["source_id"]])
        for v in review["citations"] + review["inline_only_citations"]:
            if v["source_id"] in row["evidence"]:
                assert v["view_sha256"] == file_hash_view(row["evidence"][v["source_id"]])
            else:
                assert v["grade"] == "invalid-id"
    qualitative = {arm: {"explanation_grades": dict(Counter(r["explanation_grade"] for r in reviews if r["condition"] == arm)),
        "grounding_grades": dict(Counter(r["document_grounding_grade"] for r in reviews if r["condition"] == arm)),
        "citation_support": dict(Counter(v["grade"] for r in reviews if r["condition"] == arm for v in r["citations"])),
        "inline_only_citation_support": dict(Counter(v["grade"] for r in reviews if r["condition"] == arm for v in r["inline_only_citations"])),
        "retrieval_relevance": dict(Counter(v["grade"] for r in reviews if r["condition"] == arm for v in r["retrieval_views"]))} for arm in CONDITIONS}
    write_json(RUN / "qualitative-summary.json", {"conditions": qualitative, "reviewer": "Unblinded assistant; no independent clinician or paid LLM judge", "keys_changed": False})
    cost = read_json(ROOT / "data/checks/requirement-6-cost-audit-20261012.json")
    lines = ["# Requirement 6: Gemma PDF question answering", "",
        "The page-aware PDF ingestion and local document-search/page-reading workflow is implemented and demonstrated. This separate study compares no document (A), the complete extracted text (B), and autonomous PDF tools over the same text (C), with eighteen frozen primary episodes. Requirements 2-5 remain untouched; patient-specific memory is still requirement 7.", "",
        "## Primary results", "", "| Condition | Exact key matches | Valid answers | Requests | PDF operations | Input tokens | Cost USD |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in CONDITIONS:
        s = summary["conditions"][arm]
        lines += [f"| {arm} | {s['correct']}/6 | {s['valid_completed']}/6 | {s['provider_requests']} | {s['successful_pdf_tool_calls']} | {s['input_tokens']:,} | {s['cost_usd']:.8f} |"]
    lines += ["", "Exact scoring uses six project-authored official-document-fact keys frozen before generation, not MedQA or an examination board. Explanation, document grounding, citation support and retrieval relevance are independent manual observations; they do not assign or alter the correct options.", "",
        "| Case | Official-document key | A | B | C |", "|---|---|---|---|---|"]
    for q in questions:
        identifier = q["question_id"]
        options = [(rows[(arm, identifier)]["response"].get("selected_option") or "unanswered") for arm in CONDITIONS]
        lines += [f"| {identifier} | {keys[identifier]['answer_idx']} | " + " | ".join(options) + " |"]
    lines += ["", "## Explanation, grounding and citations", "", "Grades 2/1/0 follow the frozen rubric: complete/partial/material issue for explanation; verified/partial/unverified or unsupported for document grounding. Citation grades concern associated claims in the actual supplied view, not every statement in the answer. No composite medical score.", "",
              "| Condition | Explanation grades 2 / 1 / 0 | Grounding grades 2 / 1 / 0 | Structured supported / partial / unsupported / invalid | Additional inline-only supported |",
              "|---|---|---|---|---:|"]
    for arm in CONDITIONS:
        value = qualitative[arm]
        explanations = " / ".join(str(value["explanation_grades"].get(g, 0)) for g in (2, 1, 0))
        grounding = " / ".join(str(value["grounding_grades"].get(g, 0)) for g in (2, 1, 0))
        citations = " / ".join(str(value["citation_support"].get(g, 0)) for g in ("supported", "partial", "unsupported", "invalid-id"))
        lines += [f"| {arm} | {explanations} | {grounding} | {citations} | {value['inline_only_citation_support'].get('supported', 0)} |"]
    lines += ["", "C returned **21 unique question-view pairs: 7 direct, 7 partial, 7 irrelevant**. The first result was directly useful in all five positive-fact cases; both two-recommendation questions had their necessary paired-page passages. All six C episodes searched, with six successful searches and no primary page reads. The separately preserved live pilot demonstrated page reading. Common numeric recommendation/reference identifiers can produce bibliography noise; generic stopword-only searches can return no passages.", "",
              "C used **about 74% fewer input tokens and 67% lower recorded inference charges than B** in these six cases, despite more model calls. All arms had 6/6 option matches: this study shows document access/verification and context-efficiency differences, without an observed accuracy gain.", ""]
    lines += ["", "[Explicit case/view notes](manual-reviews.json) record all eighteen answers, every structured and additional inline-only cited view, and every unique C question-view pair. Repeated delivery of one source in the same case remains visible in raw tool events but is counted once for relevance. C PDF04 gives two correct source IDs inline while leaving its structured citations array empty; the automatic evidence-status field undercounts them. B PDF04 gives one misconstructed unregistered ID alongside a valid one; the genuine page exists in the full registry, but the reference token is invalid and was not silently repaired. A missing-information answer does not require a fabricated positive citation; zero lexical hits alone cannot prove semantic absence. C PDF06 correctly selects absence but overstates what its zero-hit lexical search establishes; its wording acknowledges lexical matching without the full image/semantic coverage caveat.", "",
        "## Source and extraction", "", "Muscaritoli, Arends, Bachmann and colleagues/ESPEN, *ESPEN practical guideline: Clinical Nutrition in cancer* (2021), DOI [10.1016/j.clnu.2021.02.005](https://doi.org/10.1016/j.clnu.2021.02.005), PMID 33946039, [official sixteen-page PDF](https://www.espen.org/files/ESPEN-Guidelines/ESPEN-practical-guideline-clinical-nutrition-in-cancer.pdf). [Complete source metadata/read scope](../../../research/requirement-6-implementation-sources-20261012.json). This is a fixed historical guideline, not latest-practice advice.", "",
        "The original PDF hash is `" + manifest["pdf_sha256"] + "`. PyMuPDF 1.28.2 preserves all original extracted lines, separates narrow merged-column lines, orders positioned blocks, and creates thirty immutable nonoverlapping chunks of at most 4,000 characters covering 90,660 UTF-8 text bytes. Bibliography remains included. Each view has PDF/printed page and offsets; original line geometry remains auditable. Page 1 has no detected numeric footer and is not assigned a guessed one. [Extraction receipt](../../../checks/requirement-6-pdf-extraction-20261012.json).", "",
        "Text-only extraction does not implement OCR, raster-flowchart interpretation or table semantics. The printed 50–75% range appears as `50e75%` in the text layer; this ambiguity is retained identically in B/C rather than silently rewritten. Reference keys were independently checked on original rendered pages before generation. All sixteen searchable pages contain no pembrolizumab term; that is a selectable-text check, not exhaustive image/semantic absence proof.", "",
        "## Comparable setup and engineering history", "", "The [frozen protocol](protocol.md) uses the same Gemma 4 26B A4B IT/NextBit profile, enabled reasoning, temperature zero, 16,384 output tokens, 180-second timeout and common 131,072-byte input guard. No hidden SDK retries or paid JSON repair. The new PDF prompt and LangGraph extension preserve the prior graph's three research actions and two reserved final attempts within five turns, native final schema and explicit nonanswer fallback. PubMed, prior curated guidance, evaluator keys and previous answers are disabled in all arms. Each case starts fresh. Actual context/call lengths differ by condition.", "",
        "The initial v1 no-document count pilot exhausted output length; its failed request and unknown-charge reserve are retained. V2 allowed that error to reach the existing closing phase once and discouraged prolonged speculation. The next A pilot returned three honest but optionless uncertainty answers; the MC examination contract rejected them. V3 changed only the excluded transport pilot to free-form so uncertainty could demonstrate working transport. All six primary questions/options/keys/chunks stayed unchanged; no primary case was repeated after observing quality. Original failed/successful engineering stages remain in sibling folders and the cost audit. Matching v3 full-context native probe and A/B/C pilots passed before primary collection.", "",
        "A versus B/C examines availability of document content. B versus C uses identical extraction and therefore evaluates document access and efficiency, **not parsing superiority**. A naive-parser baseline was not run. Six convenience MC questions and a familiar guideline can make knowledge-only answers competitive; a higher citation count alone does not show better support or safer clinical behavior.", "",
        "## Tokens, latency and spending", "", "| Condition | Output tokens | Reasoning tokens within output | Provider seconds summed | Maximum serialized input bytes |", "|---|---:|---:|---:|---:|"]
    for arm in CONDITIONS:
        s = summary["conditions"][arm]
        lines += [f"| {arm} | {s['output_tokens']:,} | {s['reasoning_tokens']:,} | {s['elapsed_provider_seconds']:.3f} | {s['max_input_bytes']:,} |"]
    lines += ["", f"Primary observed charges: **USD {summary['cost_usd']:.8f}**. All requirement-6 stages, including engineering failures/probes/pilots and the free-form application demo: **USD {cost['known_cost_usd']:.8f} reported**, plus **USD {cost['unknown_request_reserve_usd']:.8f} conservative reserve** for {cost['unknown_charge_records']} request(s) without a saved bill. Latest authenticated cumulative account observation: **USD {cost['latest_ledger_usage_usd']:.9f}**, below the standing USD 6 ceiling. Account changes can include delayed/provider activity, so unreported charges are not treated as zero or attributed as exact bills. The USD 0.15 historical headroom remains in the local guard. [Cost audit](../../../checks/requirement-6-cost-audit-20261012.json).", "",
        "Latency totals sum provider waits, not end-to-end UI response measurements. Reasoning tokens are reported numeric counts; private reasoning text is omitted. The local guard reserves actual prompt-byte estimates plus the full output cap at configured price ceilings before each paid request and preserves earlier costs when refreshed billing lags.", "",
        "## Verification and reproduction", "", "The observed full offline suite passed 218 tests. Saved replay verifies exact primary/pilot/probe prompts, phases, native schemas, responses, all B document content and all C delivered views without credentials or new provider/PubMed calls. [Main replay receipt](../../../checks/requirement-6-main-replay-20261012.json). The [usage guide](../../../../docs/implementation/requirement-6-pdf.md) includes ingestion, new free-form questions, fresh-run evaluation and offline replay commands. [Application demonstration](../gemma-pdf-application-demo-20261012) exercises the reusable QA CLI separately from primary MC scoring.", "",
        "## Interpretation limits", "", "These six authored source-fact questions, MC cues, familiar historical document, single model/provider and unblinded assistant review do not establish a general clinical ranking, contamination-free evaluation, causal parser/RAG benefit, or clinician-validated safety. Qualitative reviews use actual delivered views and preserve missing-information caution. Input-byte capacity and advertised token context are distinct. Provider completion and exact-option success remain separate from citation support and medical usability. Persistent patient contexts and an integrated cross-patient isolation demo are pending requirement 7.", "",
        "## Exact questions and answers", ""]
    for q in questions:
        identifier = q["question_id"]
        lines += ["### " + identifier, "", q["question"], ""]
        lines += ["- **" + k + ":** " + v for k, v in q["options"].items()]
        lines += ["", "**Frozen reference:** " + keys[identifier]["answer_idx"] + "; PDF pages " + str(keys[identifier]["pdf_pages"]) + ".", ""]
        for arm in CONDITIONS:
            row, review = rows[(arm, identifier)], review_map[(arm, identifier)]
            response = row["response"]
            lines += ["#### " + arm, "", f"Status: {row['status']}; selected: {response.get('selected_option')}; explanation grade: {review['explanation_grade']}; document-grounding grade: {review['document_grounding_grade']}.", "",
                "**Answer (exact visible final):** " + response["answer"], "", "**Justification (exact):** " + response["justification"], "",
                "**Citations (exact IDs):** " + ", ".join(response.get("citations", [])), "", "**Limitations (exact):** " + json.dumps(response.get("limitations", []), ensure_ascii=False), "",
                "**Manual observation:** " + review["note"], ""]
    (RUN / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rendered_cases": len(rows), "qualitative": qualitative, "new_model_calls": 0}))


def file_hash_view(view):
    import hashlib
    return hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


if __name__ == "__main__":
    render()
