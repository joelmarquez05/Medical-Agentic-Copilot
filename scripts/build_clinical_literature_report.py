"""Render the preserved autonomous evaluation and separate relevance/appraisal."""

import json
import re
from collections import Counter
from os.path import relpath
from pathlib import Path

from medical_copilot.evaluation import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/retrieval-runs/gemma-evidence-main-20261011"
BASE = ROOT / "data/evaluation/internal-runs/gemma-baseline-restart-20261011"


def main():
    summary, manifest = read_json(RUN / "summary.json"), read_json(RUN / "manifest.json")
    answers_review = read_json(RUN / "answer-assessments.json")
    retrieval_review = read_json(RUN / "retrieval-assessments.json")
    reviews = {row["question_id"]: row for row in answers_review["reviews"]}
    retrieval = {row["question_id"]: row for row in retrieval_review["episodes"]}
    pairs = {row["question_id"]: row for row in read_json(RUN / "paired-scores.json")["pairs"]}
    keys = {row["question_id"]: row for row in read_json(RUN / "answer-key.json")["keys"]}
    assert set(reviews) == set(retrieval) == set(keys) == set(pairs)
    answers = ["# Exact questions and preserved tool-enabled answers", "", "Current evidence-encouraged autonomous v2 cohort. Evaluator keys shown here were never generation input. No final answer is invented for operational failures.", ""]
    quality = ["# Separate answer explanation appraisal", "", answers_review["reviewer"] + ".",
               "", "The [unchanged answer rubric](answer-rubric.json) is separate from [exact option scoring](scores.json) and [retrieval appraisal](retrieval-assessment.md).", "",
               "| Case | Reasoning | Reference concordance | Calibration/provenance |", "|---|---|---|---|"]
    relevance = ["# Separate retrieval and citation assessment", "", retrieval_review["reviewer"] + ".",
                 "", "Uses the [prespecified retrieval rubric](retrieval-rubric.json). Titles plus actually delivered excerpts/abstracts support these qualitative grades; hidden host-fetched text is not credited as model reading. These judgments never replace the option keys.", ""]
    pair_table = ["| Case | Baseline key match | Tool-enabled match | Status | Tools | Transition |", "|---|---|---|---|---|---|"]
    paper_counts = Counter()
    for identifier in manifest["question_ids"]:
        path = RUN / "answers" / (identifier + ".json")
        row, rev, ret, pair, key = read_json(path), reviews[identifier], retrieval[identifier], pairs[identifier], keys[identifier]
        assert rev["answer_sha256"] == ret["answer_sha256"] == file_hash(path)
        selected = (row.get("response") or {}).get("selected_option")
        after = "Yes" if pair["tool_enabled"]["correct"] is True else "No" if pair["tool_enabled"]["correct"] is False else "N/A"
        before = "Yes" if pair["baseline"]["correct"] else "No"
        pair_table.append(f"| [{identifier}](answers.md#{identifier.lower()}) | {before} | {after} | {row['status']} | {row['tool_actions']} | {pair['transition']} |")
        ratings = rev["dimension_ratings"]
        quality.append(f"| {identifier} | {ratings['case_reasoning']} | {ratings['reference_concordance']} | {ratings['calibration_and_provenance']} |")
        question = row["question"]
        answers += [f"## {identifier}", "", question["question"], ""]
        answers += [f"- {letter}: {value}" for letter, value in question["options"].items()]
        answers += ["", f"Evaluator key: **{key['answer_idx']} — {key['answer']}** ({key['key_type']}). Selected: **{selected or 'No valid final answer'}**. Status: **{row['status']}**.", ""]
        response = row.get("response")
        if response:
            answers += ["Model answer:", "", response["answer"], "", "Model justification:", "", response["justification"], "",
                        "Model limitations: " + ("; ".join(response["limitations"]) or "None returned."), "",
                        "Citations: " + (", ".join(f"[{sid}]({row['evidence'][sid]['url']})" if sid in row["evidence"] else sid for sid in response["citations"]) or "None.")]
        else:
            answers += ["No usable final response. The last visible action remains in the original JSON record; it is not converted into a medical answer."]
        answers += ["", f"Model turns: {row['model_turns']}; tool actions: {row['tool_actions']}. [Original record](answers/{identifier}.json).", "", rev["brief_observed_notes"], ""]
        relevance += [f"## {identifier}", "", f"Decision: **{ret['search_decision']}**. Status: **{row['status']}**.", "", ret["answer_evidence_notes"], ""]
        for query in ret["query_assessments"]:
            relevance += [f"Query: `{query['query']}`. Rating: **{query['rating']}**. " + query["brief_reason"], ""]
        if ret["paper_assessments"]:
            relevance += ["| Actual paper | Delivered views | Relevance | Reason |", "|---|---|---|---|"]
            for paper in ret["paper_assessments"]:
                paper_counts[paper["paper_relevance"]] += 1
                relevance.append(f"| [{paper['pmid']}: {paper['title']}]({paper['url']}) | {', '.join(paper['model_visible_text_kinds'])} | {paper['paper_relevance']} | {paper['brief_reason']} |")
            relevance += ["", f"Original named-trial reference example found: **{ret['reference_example_found']}** (PMID {ret['reference_example_pmid']}). This is an identity observation, not exhaustive recall.", ""]
        for citation in ret["citation_assessments"]:
            relevance += [f"Citation **{citation['source_id']}**: **{citation['rating']}** from the actually supplied abstract. Claims:", ""]
            relevance += [f"- {claim['claim']}: {claim['rating']} ({claim['locator']})." for claim in citation["claim_checks"]]
            relevance += ["", citation["limits"], ""]
        relevance += [f"[Raw views, returned metadata and request traces](answers/{identifier}.json). Answer fingerprint: `{ret['answer_sha256']}`.", ""]
    for identifier, review in reviews.items():
        quality += ["", f"## {identifier}", "", review["brief_observed_notes"], "", "Evaluator context source IDs: " + ", ".join(review["reference_source_ids"]) + ".",
                    "", f"Saved-answer SHA-256: `{review['answer_sha256']}`."]
    relevance += ["## Aggregate observations", "", "Twelve unique question-paper pairs: **1 directly relevant, 9 partially relevant, 2 irrelevant**. Repeated SOUL results and the later PREVENT CLOT read are not new pairs. Only three episodes searched; the other seven have no retrieval relevance grade.", "",
                  "One valid cited source supports two explicit outcome statements in IC09. The BALANCE/SOUL primary reports were absent from current search results. Conference material, pilots, protocols, secondary analyses and projections are distinguished from the requested original results.", "",
                  "No-source/direct-answer cases are allowed autonomous choices; they demonstrate no source verification. A successful HTTP response is not a relevant-paper or supported-answer success. These small selected observations do not establish general retrieval precision/recall or a full-text literature review.", ""]
    quality += ["", "## Comparison limits", "", "Baseline's IC09 hip-fracture-only claim is absent from the new concise answer, and its stated numerical claims now have a verified delivered citation. The new answer does not expressly explain eligibility. IC04's key/prioritization discrepancy remains and now introduces an inadequately distinguished irreversible-category alternative. Most direct answers now acknowledge absent verification, but remain knowledge-based.", "",
                "Do not average subjective ratings into a validated clinical score. The two episodes without final answers are not medical option errors. The additional evidence instruction and extra turns prevent attributing changes solely to literature tools.", ""]
    cost = read_json(ROOT / "data/checks/clinical-literature-cost-audit-20261011.json")
    prior = read_json(ROOT / "data/evaluation/retrieval-runs/gemma-autonomous-main-20261011/summary.json")
    assessment_summary = {"reviewed_answer_cases": 10, "retrieval_reviewed_episodes": 10, "episodes_with_search": summary["episodes_with_search"],
        "unique_question_paper_pairs": sum(paper_counts.values()), "paper_relevance_counts": dict(paper_counts),
        "valid_supported_cited_sources": 1, "supported_explicit_claims": 2,
        "answer_dimension_counts": {dimension: dict(Counter(row["dimension_ratings"][dimension] for row in reviews.values())) for dimension in next(iter(reviews.values()))["dimension_ratings"]},
        "keys_changed": 0, "paid_judge_calls": 0, "limitation": "Unblinded subjective assistant review; only three retrieval episodes and no clinician adjudication."}
    write_json(RUN / "assessment-summary.json", assessment_summary)
    duration = sum(read_json(RUN / "answers" / (identifier + ".json"))["elapsed_seconds"] for identifier in manifest["question_ids"])
    results = ["# Requirement 3: autonomous PubMed implementation and evaluation", "",
        "Implemented two real PubMed tools in the same LangGraph agent and evaluated the unchanged ten questions with Gemma 4. The current user-requested evidence-oriented prompt **encourages search and real references while preserving autonomous choice**. Core graph/model configuration/questions/keys remain unchanged; the additional instruction is explicitly preserved.", "",
        "**Current result:** 8/10 valid final answers, **7/10 exact-key matches** (7/8 among valid answers), one completed key mismatch and two operationally unanswered episodes. The baseline had 10/10 valid answers and 9/10 matches. No observed wrong-to-correct or correct-to-wrong medical-option transition: seven both-correct, one both-wrong and two operationally incomplete pairs.", "",
        "## Actual agent/tool behavior", "",
        "Gemma chose to search in **3/10 cases**: BALANCE, PREVENT CLOT and SOUL. The other seven were direct answers. It executed **4 searches and 1 abstract read**, making **9 successful PubMed HTTP requests** without transient retries. These were genuine model-selected operations with actual results returned before subsequent model decisions; no forced minimum search or post-answer automatic operation.", "",
        "The search tool uses NCBI ESearch and batch EFetch, returning at most five titles/metadata and 400 abstract characters. Reading accepts at most three numeric PMIDs found in that episode and makes a fresh EFetch, exposing up to 3500 abstract characters with missing/truncation flags. Immutable content-bound view IDs distinguish excerpt and read. Citation existence is checked by the host; substantive support is appraised separately. No native vendor function-call claim, full-text reading or medRxiv-specific integration.", "",
        "## Paired answer results", "", *pair_table, "",
        "Seven released MedQA cases: 6/7 key matches, all completed. Three published-fact cases: 1/3 key matches with two no-final-answer failures. These are small internal development cases, not an external held-out benchmark or general clinical safety estimate.", "",
        "## Separate retrieval and support findings", "",
        "Twelve unique question-paper pairs were reviewed using the visible titles/excerpts/abstract: **1 directly relevant, 9 partially relevant, 2 irrelevant**. These counts cover the three actual retrieval episodes; no relevance percentage is assigned to episodes that declined search. [Detailed retrieval/citation assessment](retrieval-assessment.md).", "",
        "**PREVENT CLOT (IC09):** the original trial was found, its abstract read, and the final DVT/PE statements supported by the exact delivered read-view citation. The option was already correct at baseline. This demonstrates a functioning search–read–answer path and evidence support for that answer, not an accuracy gain. [Original trial](https://pubmed.ncbi.nlm.nih.gov/36652352/).", "",
        "**BALANCE (IC08):** the query returned related meta-analysis/pilot/secondary material and two irrelevant items, but not the requested main report. Gemma's attempted read used a nested action wrapper, violating the action contract; no read or final answer followed. The query's literal `vs` and several conjunctive terms were retained in NCBI's translation. They are plausible contributors to the missing original report, not a verified single cause. No post hoc broader query or source packet was injected.", "",
        "**SOUL (IC10):** two near-identical queries returned the same two related projection/forecasting papers, not the original trial. At the final permitted turn, Gemma requested another search even though the tool catalogue was empty. The host correctly stopped at the two-action limit. The reformulation did not broaden retrieval; no fourth model call was added.", "",
        "## Explanation review", "",
        "The [separate answer appraisal](answer-assessment.md) applies the original baseline rubric and binds all judgments to saved record hashes. Direct cases remain uncited knowledge-based answers, now acknowledging lack of verification. IC02 retains an unstated pleuritic descriptor; IC03's differential/length has limitations; IC04 keeps the initial-treatment versus definitive-rescue key discrepancy and introduces an inadequately distinguished possible irreversible category. IC09 avoids the previous hip-only description and has support for its specific numbers, but does not explain eligibility. [Clinical reference catalogue](../../../research/requirement-2-restart-sources.json).", "",
        "## Prompt versions and preserved earlier cohort", "",
        "The first autonomous cohort had already completed when the user requested motivating searches and real references. It remains preserved: **6/10 valid answers, 5 exact-key matches**, four action-format failures, 2/10 episodes searched, one unknown citation ID, 13 provider requests. [Initial summary](../gemma-autonomous-main-20261011/summary.json), [initial protocol](protocol-v1-link-placeholder).", "",
        "The new v2 cohort follows that explicit instruction and reruns all ten as a distinct condition. It is not a replacement/recovery of individual poor answers. The envelope reinforces the existing JSON/citation contract, adds no medical key/hint and permits direct answers. [Exact effective instructions](system-instructions.json), [readable frozen v2 protocol](protocol-readable.md).", "",
        "**Comparability limit:** tool availability, the evidence instruction, live feedback and additional model turns change together compared with the baseline. This is an added-workflow comparison, not an isolated causal effect of tool access. Differences between the two tool cohorts are single observations with different prompts; no reliable prompt-effect estimate or global model ranking is claimed.", "",
        "## Checks and cost", "",
        "**133 offline tests pass**, including 25 new literature/envelope tests. The separate [two-tool live smoke check](../../../checks/clinical-pubmed-smoke-20261011/smoke.json) made three successful public HTTP requests and no model call, outside all ten cases. Each prompt version has a separately saved one-case Gemma pilot; both chose direct answers. Smoke queries are researcher-chosen and never presented as Gemma behavior.", "",
        "All four cohorts—two pilots and two ten-case studies—replay saved provider responses and PubMed snapshots through the graph with **zero new calls or credential reads**. The current [main verification](../../../checks/clinical-evidence-main-verification-20261011.json) confirms frozen input/key/profile/core code, actual effective prompt, fresh history/selections, exact actions/views and HTTP hashes. The earlier extension-module hash differs because of v2, while its original snapshot remains intact. Raw response/citation/limit failures remain failures in replay.", "",
        f"Current main: **15 model requests, USD {summary['reported_or_estimated_cost_usd']:.10f}** provider-reported; {summary['input_tokens']} input tokens, {summary['output_tokens']} output tokens including {summary['reasoning_tokens']} numeric reasoning tokens. Summed per-case wall time: {duration:.1f} seconds. Private reasoning text is excluded.", "",
        f"All new point-3 work, including the preserved first cohort and both pilots: **30 provider requests, USD {cost['provider_reported_total_usd']:.10f}**. Later authenticated account usage: **USD {cost['latest_ledger']['total_usage']:.9f}**, below USD 6. Immediate account snapshots lagged some charges; later response totals/account delta agree within rounding. [Complete cost observations](../../../checks/clinical-literature-cost-audit-20261011.json). No paid judge, model retry/format repair or primary quality-based rerun was used.", "",
        "## Inspect or run", "", "From the project root:", "", "```powershell",
        ".\\.venv\\Scripts\\python.exe -X utf8 scripts/verify_clinical_literature.py --run-dir data/evaluation/retrieval-runs/gemma-evidence-main-20261011",
        ".\\.venv\\Scripts\\python.exe -X utf8 scripts/build_clinical_literature_report.py",
        "```", "", "These inspect saved artifacts only. For a new separately authorized run, use `scripts/run_clinical_literature.py --stage smoke|pilot|autonomous --run-dir <fresh-directory>`, supplying `--smoke-dir` to hosted stages and `--pilot-dir` to the main stage. Existing directories are refused; fresh account/endpoint checks precede generation.", "",
        "[All questions/answers](answers.md), [exact paired scores](paired-scores.json), [raw records](answers), [source snapshots](source-snapshot), [manifest/dependency versions](manifest.json), [PubMed configuration](pubmed-config.json). Complete technical source metadata/read levels: [NCBI catalogue](../../../research/requirement-3-restart-sources.json). All **18 distinct actually observed PubMed records** across current/initial/smoke work, with authors/year/PMID/DOI/URL and read provenance: [observed-paper catalogue](../../../research/requirement-3-restart-observed-papers-20261011.json). Clinical/model/question references reuse the journal's prior source entries.", "",
        "## Completion and remaining limitations", "",
        "Requirement 3 now has connected, demonstrated PubMed tools and a separate qualitative relevance/support evaluation on the same internal cases. The observed workflow does not improve option accuracy and loses final-answer coverage in two cases. Query formulation, strict action-format reliability and behavior at the final permitted turn need refinement before a larger benchmark. Any refinement must be versioned and validated without rewriting these results or selecting replacement cases.", "",
        "Requirements 4–5 remain a larger distinct benchmark and six-model comparison. PDF ingestion and patient memory remain requirements 6–7. No new compiled LaTeX/PDF report is generated; project memory is the journal.", ""]
    results = [line.replace("protocol-v1-link-placeholder", "../gemma-autonomous-main-20261011/protocol.md") for line in results]
    # Preserve the hash-bound snapshot; resolve its source-relative links in a view.
    origin = ROOT / "docs/implementation/requirement-3-evidence-encouraged-protocol.md"

    def protocol_link(match):
        label, target = match.group(1), match.group(2)
        if "://" in target or target.startswith("#"):
            return match.group(0)
        path, separator, anchor = target.partition("#")
        resolved = Path(relpath((origin.parent / path).resolve(), RUN)).as_posix()
        return f"[{label}]({resolved}{separator}{anchor})"

    portable = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", protocol_link, (RUN / "protocol.md").read_text(encoding="utf-8"))
    (RUN / "protocol-readable.md").write_text("Rendered from the unchanged [hash-bound protocol snapshot](protocol.md). Relative links resolve against its original source document.\n\n" + portable, encoding="utf-8")
    for name, lines in [("answers.md", answers), ("answer-assessment.md", quality), ("retrieval-assessment.md", relevance), ("results.md", results)]:
        (RUN / name).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rendered_cases": 10, "paper_relevance_counts": dict(paper_counts), "correct": summary["correct"], "valid_completed": summary["valid_completed"], "new_inference_requests": 0}))


if __name__ == "__main__":
    main()
