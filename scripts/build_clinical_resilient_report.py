"""Render exact v3 answers and separate operational, medical and retrieval findings."""

import json
import re
from os.path import relpath
from pathlib import Path

from medical_copilot.evaluation import read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/retrieval-runs/gemma-resilient-main-20261011"


def main():
    summary = read_json(RUN / "summary.json")
    scores = {s["question_id"]: s for s in read_json(RUN / "scores.json")["scores"]}
    answer_review = read_json(RUN / "answer-assessments.json")
    retrieval_review = read_json(RUN / "retrieval-assessments.json")
    appraisal = read_json(RUN / "assessment-summary.json")
    reviews = {r["question_id"]: r for r in answer_review["reviews"]}
    retrieval = {r["question_id"]: r for r in retrieval_review["episodes"]}
    assert set(scores) == set(reviews) == set(retrieval)
    exact = ["# Exact v3 questions and model answers", "",
        "Seven unchanged MedQA training questions plus three project-authored primary-trial fact questions. Keys below are evaluator-only and were never supplied to generation. Original model wording is preserved; commentary/appraisal is separate.", ""]
    quality = ["# Separate answer appraisal", "", answer_review["reviewer"] + ".", "",
        "Exact option scoring is in [scores.json](scores.json); qualitative notes do not change a key. The [rubric](answer-rubric.json) and [reused clinical source metadata](../../internal-runs/gemma-baseline-restart-20261011/review-sources.json) remain distinct from agent-retrieved evidence.", ""]
    relevance = ["# Separate retrieval and citation appraisal", "", retrieval_review["reviewer"] + ".", "",
        "Only actual delivered titles/excerpts/abstracts are credited as model reading. Real IDs establish provenance, not complete claim support. [Rubric](retrieval-rubric.json), [hash-bound structured notes](retrieval-assessments.json), [full observed-source metadata](../../../research/requirement-3-resilient-observed-papers-20261011.json). No exhaustive recall, full-text review or clinician validation.", ""]
    rows = ["| Case | Key origin | Key | Selected | Status | Calls | Tool actions | Final calls |", "|---|---|---|---|---|---:|---:|---:|"]
    for identifier in sorted(scores):
        record = read_json(RUN / "answers" / (identifier + ".json"))
        response, question, score = record["response"], record["question"], scores[identifier]
        rows.append(f"| {identifier} | {'MedQA' if score['key_type'] == 'released_medqa_label' else 'Trial fact'} | {score['reference_option']} | {score['selected_option']} | {record['status']} | {len(record['model_calls'])} | {record['tool_actions']} | {record['final_attempts']} |")
        exact += [f"## {identifier}", "", question["question"], ""]
        exact += [f"- **{letter}:** {text}" for letter, text in question["options"].items()]
        exact += ["", f"Evaluator key: **{score['reference_option']}** ({score['key_type']}). Model selected **{response['selected_option']}**.", "",
                  "**Exact model answer:**", "", response["answer"], "", "**Exact model justification:**", "", response["justification"], "",
                  "**Exact model limitations:**", ""]
        exact += ["- " + value for value in response["limitations"]] or ["Empty list returned by the model."]
        exact += ["", "**Cited registry views:**", ""]
        for sid in response["citations"]:
            source = record["evidence"][sid]
            exact.append(f"- `{sid}`: [{source['title']}]({source['url']}) ({source['text_kind']}).")
        if not response["citations"]:
            exact.append("No source cited.")
        exact += ["", f"[Original full episode record](answers/{identifier}.json).", ""]
        quality += [f"## {identifier}", "", "; ".join(f"{k}: {v}" for k, v in reviews[identifier]["dimension_ratings"].items()) + ".", "", reviews[identifier]["brief_observed_notes"], ""]
        episode = retrieval[identifier]
        relevance += [f"## {identifier}", "", f"Policy observation: **{episode['search_decision']}**; tool argument/execution errors: {episode['operational_tool_errors']}.", ""]
        for query in episode["query_assessments"]:
            relevance += [f"Query: `{query['query']}` — {query['rating']}. {query['brief_reason']}", ""]
        if episode["query_assessments"] and not episode["paper_assessments"]:
            relevance += ["Zero papers returned; this is not evidence that literature is absent.", ""]
        for paper in episode["paper_assessments"]:
            relevance += [f"- PMID {paper['pmid']}, [{paper['title']}]({paper['url']}): **{paper['paper_relevance']}**. {paper['brief_reason']}"]
        relevance += ["", "Citation support:", ""]
        for citation in episode["citation_assessments"]:
            relevance.append(f"- `{citation['source_id']}`: **{citation['rating']}**. {citation['brief_reason']}")
        if not episode["citation_assessments"]:
            relevance.append("No cited view to assess.")
        if episode["reference_example_pmid"]:
            relevance += ["", f"Frozen example original PMID {episode['reference_example_pmid']} found: **{episode['reference_example_found']}**. An example is not an exhaustive reference set."]
        relevance += ["", episode["answer_evidence_notes"], ""]
    results = ["# Reserved finalization and format recovery: observed results", "",
        "The user requested operational fixes for all ten questions and future cases. This separately versioned cohort completes **10/10 valid model answers**, with **9/10 exact-key matches**, zero application fallbacks and no episode-ending format/action-limit failure. Primary cost: **USD 0.0087362775 in 22 inference requests**. Separate native-final probe plus pilot add USD 0.0005789925; all new work totals **24 requests / USD 0.0093152700**.", "",
        "## Implemented changes", "",
        "Three research turns/tool actions plus an explicitly reserved final call and one bounded final retry, at most five model requests. Finalization exposes no tools and uses the final-only native JSON Schema prevalidated on Gemma/NextBit. Deterministic unambiguous wrapper/alias/list/case normalization leaves option meaning, medical text and tool arguments intact; original responses and repair notes remain auditable. An unrecoverable episode returns an explicit application nonanswer, not a fabricated medical option. Context compaction removes duplicate tool-source text from history while retaining the complete delivered views in the registry.", "",
        "Search remains autonomous, with general evidence/query guidance. Every case starts fresh. Model/reasoning/profile/questions/keys remain unchanged; prompt, routing, per-episode budgets and final transport change together. This is an operational development condition, not an isolated causal tool-effect experiment.", "",
        "## Observed comparison", "",
        "| Condition | Valid model answers | Exact-key matches / planned | Model calls | Search episodes | Primary cost USD |",
        "|---|---:|---:|---:|---:|---:|",
        "| R2 tools disabled | 10/10 | 9/10 | 10 | 0 | 0.0034073400 |",
        "| R3 evidence-encouraged v2 | 8/10 | 7/10 | 15 | 3 | 0.0053369700 |",
        "| R3 resilient v3 | 10/10 | 9/10 | 22 | 8 | 0.0087362775 |", "",
        "Against baseline: nine both-correct cases and IC04 both wrong by the unchanged key. V2's two operational nonanswers become completed correct choices, but all three trial choices were already correct in the tools-disabled baseline. Restored coverage and observable source support do not establish higher clinical accuracy than baseline.", "",
        "**Key origins:** IC01–IC07 are unchanged released MedQA training questions (**6/7** matches). IC08–IC10 are project-authored BALANCE/PREVENT CLOT/SOUL numeric questions based on primary published results (**3/3** matches), not official MedQA questions or labels. Their keys were frozen before inference, without an LLM correctness judge.", ""] + rows + ["",
        "## The former two failure cases", "",
        "**IC08 / BALANCE:** current query finds and reads the original trial (PMID 39565030), then answers B in three calls. No live normalization is needed in this new episode. The exact earlier nested read wrapper is separately recovered in an offline regression without rewriting the old record. The answer's intention-to-treat CI confidence-level label is imprecise (95% versus delivered 95.7%); option facts are supported.", "",
        "**IC10 / SOUL:** search finds a results review (PMID 41370018), not the original trial. The first read incorrectly passes a view ID as a PMID and is rejected without HTTP execution; Gemma corrects it to a numeric PMID in the third research turn. The reserved fourth call answers C using native final-only output. This demonstrates the closing branch after all three action attempts, including a rejected one. The review supports HR/CI but omits crude event rates; those correct rates cannot be credited to retrieval. The final answer fails to disclose that evidence gap.", "",
        "No format failure or provider outage occurred in this new cohort. Local normalization count is zero and no final retry is used; their regressions are engineering evidence, not observed live rescues. Actual native final output is confirmed in the separate probe and IC10. All-model compatibility and future medical-answer uptime are not guaranteed.", "",
        "## Retrieval findings, separately assessed", "",
        "Eight searches were executed. Four read actions were requested; three executed successfully and one was rejected locally for its argument. Eighteen HTTP requests completed, with no HTTP retry. IC06 has an empty search and still gives a knowledge-based answer. Five final answers cite registered views; all seven cited view IDs are structurally valid.", "",
        f"The **32 unique question-paper pairs** comprise {appraisal['paper_relevance']['directly_relevant']} direct, {appraisal['paper_relevance']['partially_relevant']} partial and {appraisal['paper_relevance']['irrelevant']} irrelevant papers under unblinded assistant review. Six cited views only partially support their associated content; one fully supports the assessed DVT/PE claims. A real source is not sufficient evidence for every assertion. These grades follow actual delivered text, without full-text or exhaustive recall claims.", "",
        "[Separate answer appraisal](answer-assessment.md), [separate retrieval/citation appraisal](retrieval-assessment.md), [all exact answers](answers.md), [structured scores](scores.json), [baseline pairing](paired-scores.json), [raw traces/outputs](answers), [frozen configuration and source hashes](manifest.json). The prior [v2 results](../gemma-evidence-main-20261011/results.md) and all earlier failures remain preserved.", "",
        "## Checks and future use", "",
        "All **159 offline tests** pass, including 26 new robustness checks. Exact replay passes for the separate probe, pilot and ten primary cases: 24 model responses and 18 HTTP responses, with zero new inference/retrieval or credential reads. Probe uses research turns zero to force the final mode; ordinary pilot/primary share the full frozen policy. SDK retries, paid judges and medical-quality reruns remain disabled.", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -X utf8 scripts/verify_clinical_resilient.py --run-dir data/evaluation/retrieval-runs/gemma-resilient-main-20261011",
        ".\\.venv\\Scripts\\python.exe -X utf8 scripts/build_clinical_resilient_report.py", "```", "",
        "For **new questions**, [the application CLI](../../../../scripts/ask_clinical_resilient.py) takes one question object or a `questions` array and requires a fresh output directory. It uses the tested Gemma reference profile, preserves every episode and continues the batch after a handled failure. No evaluator keys are accepted, and no accuracy is automatically claimed. [Usage and failure policy](../../../../docs/implementation/requirement-3-resilience.md). The complete cohort CLI is [run_clinical_resilient.py](../../../../scripts/run_clinical_resilient.py); it requires native probe and full-policy pilot provenance before primary generation.", "",
        "[Frozen protocol](protocol.md), [readable protocol](protocol-readable.md), [new technical-source catalogue](../../../research/requirement-3-resilient-technical-sources-20261011.json), [all observed papers and read provenance](../../../research/requirement-3-resilient-observed-papers-20261011.json), [cost/account audit](../../../checks/clinical-resilient-cost-audit-20261011.json). Clinical source metadata reuse the prior journal entries. No new compiled PDF/LaTeX report.", "",
        "Freeze the next benchmark configuration separately; requirements 4–7 remain pending. These development cases and later policy changes cannot be mixed silently into a held-out benchmark or earlier paired results.", ""]
    origin = ROOT / "docs/implementation/requirement-3-resilient-protocol.md"
    def portable_link(match):
        label, target = match.group(1), match.group(2)
        if "://" in target or target.startswith("#"):
            return match.group(0)
        path, separator, anchor = target.partition("#")
        return f"[{label}]({Path(relpath((origin.parent / path).resolve(), RUN)).as_posix()}{separator}{anchor})"
    portable = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", portable_link, (RUN / "protocol.md").read_text(encoding="utf-8"))
    (RUN / "protocol-readable.md").write_text("Readable view of the unchanged [frozen protocol](protocol.md).\n\n" + portable, encoding="utf-8")
    for name, lines in [("answers.md", exact), ("answer-assessment.md", quality), ("retrieval-assessment.md", relevance), ("results.md", results)]:
        (RUN / name).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rendered_cases": len(scores), "correct": summary["correct"], "valid_answers": summary["valid_completed"], "new_model_requests": 0}))


if __name__ == "__main__":
    main()
