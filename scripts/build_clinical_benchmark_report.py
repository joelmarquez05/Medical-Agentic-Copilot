"""Render the fixed Gemma benchmark and separate answer/citation findings from artifacts."""
import json
from collections import Counter
from pathlib import Path

from medical_copilot.clinical_benchmark import BENCHMARK
from medical_copilot.evaluation import read_json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"


def main():
    summary=read_json(RUN/"summary.json")
    scores={s["question_id"]:s for s in read_json(RUN/"scores.json")["scores"]}
    answer_review=read_json(RUN/"answer-assessments-v2.json")
    retrieval_review=read_json(RUN/"retrieval-assessments-v2.json")
    assessment=read_json(RUN/"assessment-summary-v2.json")
    reviews={r["question_id"]:r for r in answer_review["reviews"]}
    retrieval={r["question_id"]:r for r in retrieval_review["episodes"]}
    sources={s["id"]:s for s in read_json(ROOT/"data/research/requirement-4-clinical-review-sources-20261011.json")["sources"]}
    records=[read_json(p) for p in sorted((RUN/"answers").glob("*.json"))]
    assert len(records)==30 and set(scores)==set(reviews)==set(retrieval)
    exact=["# Exact benchmark questions and model answers","",
        "Thirty unchanged released MedQA test questions. Evaluator keys shown below were never generation input. All model answer/justification/limitation text is preserved verbatim; appraisals are separate.",""]
    quality=["# Separate answer explanation appraisal","",answer_review["reviewer"]+".","",
        "All thirty responses are reviewed under [the frozen rubric](rubric.json). Every option matches the unchanged released label, but the explanation/citation findings are assessed separately. Focused primary-reference checks are [researcher context](../../../research/requirement-4-clinical-review-sources-20261011.json), never retroactive agent evidence. This is not independent clinician validation.",""]
    evidence=["# Separate retrieval and citation appraisal","",retrieval_review["reviewer"]+".","",
        "All returned question-paper pairs and cited identifiers are assessed. Grade only actual delivered titles/excerpts/abstracts; source existence, relevance and claim support are separate. [Structured notes](retrieval-assessments-v2.json), [retrieval rubric](retrieval-rubric.json), [observed-paper catalogue](../../../research/requirement-4-observed-papers-20261011.json). Query grades are manually based on wording/topic, never determined by empty results alone.",""]
    table=["| Case | Key / choice | Status | Model calls | Tool actions | Papers | Cited IDs |",
           "|---|---|---|---:|---:|---:|---:|"]
    for row in records:
        identifier=row["question"]["question_id"]
        question,response,score=row["question"],row["response"],scores[identifier]
        episode=retrieval[identifier]
        table.append(f"| {identifier} | {score['reference_option']} / {score['selected_option']} | {row['status']} | {len(row['model_calls'])} | {row['tool_actions']} | {len(episode['paper_assessments'])} | {len(response['citations'])} |")
        exact += ["## "+identifier,"",question["question"],""]
        exact += ["- **"+letter+":** "+option for letter,option in question["options"].items()]
        exact += ["",f"Released key: **{score['reference_option']}**. Model choice: **{response['selected_option']}**. Status: `{row['status']}`.","",
            "**Exact model answer:**","",response["answer"],"","**Exact model justification:**","",response["justification"],"","**Exact model limitations:**",""]
        exact += ["- "+item for item in response["limitations"]] or ["Empty list returned."]
        exact += ["","**Exact cited identifiers:**",""]
        for sid in response["citations"]:
            source=row["evidence"].get(sid)
            if source:
                exact.append(f"- `{sid}`: [PMID {source['pmid']}]({source['url']}), {source['title']} ({source['text_kind']}).")
            else:
                exact.append(f"- `{sid}`: unknown registered view ID; original response is not repaired.")
        if not response["citations"]:
            exact.append("No source cited.")
        exact += ["",f"[Full original record](answers/{identifier}.json).",""]
        review=reviews[identifier]
        quality += ["## "+identifier,"","; ".join(k+": "+v for k,v in review["dimension_ratings"].items())+".","",review["brief_observed_notes"],""]
        for sid in review["researcher_reference_ids"]:
            source=sources[sid]
            quality += [f"Researcher reference {sid}: [{source['title']}]({source['url']}) ({source['year']}). {source['read_level']}",""]
        evidence += ["## "+identifier,"",f"Policy observation: **{episode['search_decision']}**; recorded tool errors: {episode['tool_errors']}.",""]
        for query in episode["query_assessments"]:
            evidence += [f"Query: `{query['query']}` -- **{query['rating']}**. {query['notes']}",""]
        if episode["query_assessments"] and not episode["paper_assessments"]:
            evidence += ["Zero papers returned; not proof that relevant literature does not exist.",""]
        for paper in episode["paper_assessments"]:
            evidence.append(f"- [PMID {paper['pmid']}]({paper['url']}), {paper['title']}: **{paper['paper_relevance']}**. {paper['brief_reason']}")
        evidence += ["","Citation support:",""]
        for citation in episode["citation_assessments"]:
            evidence.append(f"- `{citation['source_id']}`: **{citation['rating']}**; registry ID valid: **{citation['view_exists']}**. {citation['brief_reason']}")
        if not episode["citation_assessments"]:
            evidence.append("No cited view; external answer support remains absent/unverified.")
        evidence += [""]
    material=[r["question_id"] for r in reviews.values() if "material_issue" in r["dimension_ratings"].values()]
    search_events=[e for row in records for e in row["literature"]["events"] if e["name"]=="search_pubmed"]
    empty=sum(not e["generation_view"]["sources"] for e in search_events)
    trials=read_json(ROOT/BENCHMARK/"manifest.json")
    reasons=Counter(e["reason"] for e in trials["excluded_rows"])
    cost=read_json(ROOT/"data/checks/requirement-4-cost-audit-20261011.json")
    lines=["# Requirement 4: Gemma 4 full-agent MedQA results","",
        "**Primary objective result: 30/30 exact released-key matches (100%) and 30/30 valid model answers.** Wilson 95% interval: **88.65%-100%**. No operational nonanswers, provider errors, truncations or medical-quality reruns. Twenty-nine responses are `completed`; BM26 is `completed_with_citation_issues` and still selects the correct option. This status difference is preserved.","",
        "The user requested the complete literature-enabled agent first with Gemma 4, with separate answer and citation appraisal. Requirement 5 is deferred: review these results before deciding its models/conditions. No other model was run.","",
        "## Frozen dataset and condition","",
        "All thirty questions/options/labels are unchanged released MedQA USMLE four-option **test** items, not the internal seven-plus-three mixture. The source contains 1273 items at revision `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`. Final selection is in [the frozen subset](../../medqa-benchmark-v3-20261011/manifest.json); seed 202610114, 1076 eligible rows, first 30 of a frozen 50-item order. It excludes prior prepared/used questions, duplicates, conservative missing-visual references and input-contract failures. No difficulty, correctness, answer-position or short-word filtering.","",
        "The agent graph/prompt/model/reasoning/tools/bounds are unchanged from the completed requirement-3 v3 study: Gemma `google/gemma-4-26b-a4b-it`, configured NextBit/no fallback, temperature 0, reasoning enabled/private text excluded, 16384 output tokens per call, timeout 180 seconds, three research turns/actions plus at most two final attempts, maximum five calls. Tools are PubMed search and reading found abstracts, with actual result feedback. Search remains autonomous. Research uses JSON decisions; only finalization uses native final-only JSON Schema. No MCP, medRxiv, PDF or patient memory is added.","",
        "Keys and researcher sources never enter generation. Each question starts fresh. A separate successful runner pilot outside the benchmark preceded the primary cohort; the previously confirmed native-final probe and live tool smoke are retained as provenance.","",
        "## Agent and evidence observations","",
        "| Measure | Observed primary result |","|---|---:|",
        "| Valid answers / planned | 30 / 30 |","| Exact key matches / planned | 30 / 30 |",
        "| Cases choosing search | 17 / 30 |","| Cases with at least one returned paper | 11 / 30 |",
        "| Search operations / empty searches | 22 / 11 |","| Abstract-read operations | 1 |",
        "| Successful PubMed HTTP requests | 34 |","| Model inference requests | 53 |",
        "| Answers with cited identifiers | 7 / 30 |","| Cited identifiers valid in registry | 8 / 10 |",
        "| Local normalizations / final retries / application fallbacks | 0 / 0 / 0 |","",
        "BM23 performs three near-equivalent sargramostim queries with no matches and still answers correctly in the reserved fourth model call. This demonstrates that research exhaustion no longer leaves an empty answer. It also exposes inefficient query formulation. The 13 direct cases and six cases with only empty search results answer from existing knowledge; this is allowed but does not provide external verification.","",
        "**Separate citation support:** of ten cited identifiers, **4 support the assessed central claims, 3 partially support their associated content, 1 does not support its attributed claim, and 2 cannot pass primary support assessment because their IDs are unregistered**. These are explicit unblinded assistant grades, not objective medical labels or independent clinician validation. Support is claim-specific, not a guarantee of every sentence in an answer.","",
        "**Separate retrieval relevance:** 42 unique question-paper pairs: **13 directly relevant, 23 partially relevant, 5 irrelevant and 1 not assessable** from delivered text/title only. Relevance does not establish support. Two records are title-only in the burn case; one record supplies an unavailable title. No hidden host text, full-text reading or exhaustive recall is credited.","",
        "## Important answer/citation findings","",
        "- **BM15:** correct area-postrema answer, but a cited ondansetron/5-HT3 excerpt is used for metoclopramide/D2. Another excerpt supports a central emetic role while questioning circulating substances in dog experiments; it does not establish all asserted BBB/toxin details. The independently checked D2 fact does not repair the original citation.",
        "- **BM26:** correct bacterial-translocation answer and neutrophil calculation, but the model cites `P19266595`/`P38936554` instead of exact delivered view IDs. These are genuine returned papers, not fabricated articles. Separate unique-PMID diagnostics identify their supportive excerpts without changing the preserved primary citation failure.",
        "- **BM21:** correct xanthine-oxidase option with a material explanatory limitation: dismisses colchicine marrow toxicity too broadly and poorly handles the IBD/polypharmacy interaction clue. Official labels document colchicine pancytopenia and allopurinol/thiopurine interaction. The stem does not prove a particular concomitant medication; no alternative key is invented.",
        "- **BM16:** searches successfully but says no evidence registry was provided, despite five delivered records. Its doxycycline choice is correct; actual excerpts do not verify the exact recommendation.",
        "- **BM11/BM13:** correct options with limited serology precision and overstated specificity of typhoid findings. CDC references contextualize these limits without changing dataset scores.",
        "- **BM19/BM27:** observable supported citation paths. BM19 actually reads a source explicitly associating carbamazepine with neural-tube defects; its old broader treatment advice is not adopted. BM27's excerpt explicitly gives the fragile-X physical phenotype.","",
        f"All 30 explanations were reviewed; **{len(material)}** have a material issue in at least one qualitative dimension ({', '.join(material)}). This subjective flag count is separate from 30/30 key accuracy. Full minor-limitations and claim scope are in the [answer appraisal](answer-assessment.md) and [retrieval/citation appraisal](retrieval-assessment.md).", "",
        "## Preserved integration interruption and appraisal revisions","",
        "The first prepared selection missed an explicit absent MRI reference. The first collection was stopped after two completed cases, before that item and without using answer scores/content for reselection. Its two answers, four saved response charges (one without a completed episode), possible pending request and source snapshots remain in [the interruption record](../gemma-full-agent-medqa30-20261011/interruption.json). A second prepared set was never executed after an exhibit/arrow dependency was caught during full input inspection. The final visual screen was broadened, both prepared sets excluded under the existing history rule, and every final selected input inspected before a new matching pilot. The final subset/primary cohort are distinct and all thirty final questions are retained. No interrupted scores are pooled.","",
        "The final query review uses explicit wording/topic grades. An initial draft assigned query grades from result counts; it remains preserved but is superseded by [review-notes-v2.json](review-notes-v2.json). Answer/paper/citation grades and objective scores are unchanged. Empty search results alone do not establish an inappropriate query.","",
        "## Costs and verification","",
        f"Primary cohort: **USD {summary['reported_or_estimated_cost_usd']:.10f} / 53 inference requests**. All new point-4 work, including both pilots and interrupted integration: **USD {cost['provider_reported_known_total_usd']:.10f} in 59 saved request charges**, plus a possible unrecorded in-flight request. The latest saved account observation is **USD {cost['latest_ledger']['total_usage']:.9f} cumulative**; aggregate delta agrees with known charges within rounding. The interruption's individual pending charge is not invented or assumed zero. [Cost/account audit](../../../checks/requirement-4-cost-audit-20261011.json).", "",
        "All **172 offline tests pass**, including 13 new benchmark-selection/scoring regressions. [Primary replay](../../../checks/requirement-4-main-verification-20261011.json) verifies all 30 episodes/53 saved model responses/34 HTTP responses with zero new network or credentials; the matching pilot also replays. Hashes establish the unchanged graph/prompt/profile, source/key wording, sampling, empty initial state, native closing schema, traces and numeric costs. Tests and replay verify engineering behavior, not clinical validity.","",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -X utf8 scripts/verify_clinical_benchmark.py --run-dir data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011",
        ".\\.venv\\Scripts\\python.exe -X utf8 scripts/build_clinical_benchmark_report.py",
        "```","",
        "These inspect/re-render saved artifacts without new model calls. The [hosted runner](../../../../scripts/run_clinical_benchmark.py) requires a fresh directory, matching successful pilot/smoke, current public endpoint/account checks and a conservative full-batch allowance. Do not rerun a completed cohort to improve scores.","",
        "## Scope and continuation","",
        "Requirement 4 now has an objectively scored, separately appraised full-agent benchmark subset. This is a small filtered exam sample, with possible benchmark memorization and a wide interval even at 100%. It does not demonstrate an improvement caused by tools, current clinical-care competence, complete sourcing or superiority to OpenEvidence's different 500-item study. Perfect option scores can coexist with incorrect citation associations or weak explanations.","",
        "The next discussion is whether requirement 5 should reuse this configuration/subset or introduce a declared new condition to address sourcing limits, and which additional models to include. No further model selection or hosted comparison is executed by this report. PDF ingestion and patient memory remain requirements 6-7.","",
        "[All exact questions/answers](answers.md), [raw records](answers), [scores](scores.json), [manifest/source hashes/dependencies](manifest.json), [frozen protocol](protocol.md), [evaluation rubric](rubric.json), [benchmark provenance sources](../../../research/requirement-4-benchmark-sources-20261011.json), [clinical review references/read limits](../../../research/requirement-4-clinical-review-sources-20261011.json), [all observed paper metadata](../../../research/requirement-4-observed-papers-20261011.json). No new compiled LaTeX/PDF report.","",
        "## Per-case operational results",""]+table+[""]
    for name,content in [("answers.md",exact),("answer-assessment.md",quality),("retrieval-assessment.md",evidence),("results.md",lines)]:
        (RUN/name).write_text("\n".join(content),encoding="utf-8")
    print(json.dumps({"rendered_cases":len(records),"correct":summary["correct"],"paper_pairs":assessment["unique_question_paper_pairs"],
        "material_issue_cases":material,"empty_searches":empty,"new_model_requests":0}))


if __name__=="__main__":main()
