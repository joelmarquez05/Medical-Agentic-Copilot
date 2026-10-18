"""Render objective comparisons, exact answers and separate explicit qualitative notes."""
from collections import Counter
from pathlib import Path
from medical_copilot.clinical_model_comparison import MODELS, REFERENCE
from medical_copilot.evaluation import read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/comparison-runs/medqa30-four-model-main-20261011"


def render_model(slug):
    directory = RUN / slug
    scores = {s["question_id"]: s for s in read_json(directory / "scores.json")["scores"]}
    reviews = {r["question_id"]: r for r in read_json(directory / "answer-assessments.json")["reviews"]}
    retrieval = {r["question_id"]: r for r in read_json(directory / "retrieval-assessments.json")["episodes"]}
    records = [read_json(p) for p in sorted((directory / "answers").glob("*.json"))]
    assert len(records) == len(reviews) == len(retrieval) == 30
    exact = ["# Exact questions and model answers: " + slug, "",
             "All model answer, justification, limitation and citation text is preserved. Released keys shown here never entered generation. Qualitative findings are separate.", ""]
    quality = ["# Separate answer appraisal: " + slug, "",
               "Unblinded implementation-assistant review; no independent clinician or paid judge. Keys are unchanged. All thirty answers are reviewed under the frozen rubric.", "",
               "Reference IDs resolve to the complete [point-4 catalogue](../../../../research/requirement-4-clinical-review-sources-20261011.json) and [point-5 catalogue](../../../../research/requirement-5-clinical-review-sources-20261011.json). These were researcher checks after generation, not answering-agent inputs.", ""]
    evidence = ["# Separate retrieval/citation appraisal: " + slug, "",
                "Grades are explicit manual observations of actual delivered views. Source identity, relevance and claim support are separate; no full-text or exhaustive recall claim.", ""]
    for row in records:
        identifier, response = row["question"]["question_id"], row["response"]
        score, review, episode = scores[identifier], reviews[identifier], retrieval[identifier]
        exact += ["## " + identifier, "", row["question"]["question"], ""]
        exact += ["- **" + letter + ":** " + option for letter, option in row["question"]["options"].items()]
        exact += ["", f"Released key: **{score['reference_option']}**. Model choice: **{score['selected_option']}**. Status: `{row['status']}`.", "",
                  "**Exact model answer:**", "", response["answer"], "", "**Exact model justification:**", "", response["justification"], "",
                  "**Exact model limitations:**", ""]
        exact += ["- " + item for item in response["limitations"]] or ["Empty list returned."]
        exact += ["", "**Exact cited identifiers:**", ""]
        for sid in response["citations"]:
            source = row["evidence"].get(sid)
            exact.append(f"- `{sid}`: [PMID {source['pmid']}]({source['url']}), {source['title']} ({source['text_kind']})." if source else
                         f"- `{sid}`: unregistered source ID; original output is not repaired.")
        if not response["citations"]:
            exact.append("No source cited.")
        exact += ["", f"[Original record](answers/{identifier}.json).", ""]
        quality += ["## " + identifier, "", "; ".join(k + ": " + v for k, v in review["dimension_ratings"].items()) + ".", "",
                    review["brief_observed_notes"], "", "Researcher references: " + (", ".join(review["researcher_reference_ids"]) or "None added.") + ".", ""]
        evidence += ["## " + identifier, ""]
        for query in episode["query_assessments"]:
            evidence += [f"Query: `{query['query']}` -- **{query['rating']}**. {query['notes']}", ""]
        if not episode["query_assessments"]:
            evidence += ["Answered directly; no external search verification in this episode.", ""]
        elif not episode["paper_assessments"]:
            evidence += ["No papers returned; this does not prove absence of literature.", ""]
        for paper in episode["paper_assessments"]:
            evidence.append(f"- [PMID {paper['pmid']}]({row['evidence'][paper['generation_view_ids'][0]]['url']}), {paper['title']}: **{paper['paper_relevance']}**. {paper['brief_reason']}")
        evidence += ["", "Citation support:", ""]
        for citation in episode["citation_assessments"]:
            evidence.append(f"- `{citation['source_id']}`: **{citation['rating']}**; registry ID valid: **{citation['view_exists']}**. {citation['brief_reason']}")
        if not episode["citation_assessments"]:
            evidence.append("No cited view; external claim support remains absent/unverified.")
        for reference in episode["supplementary_inline_reference_assessments"]:
            evidence.append(f"- Supplementary inline-only `{reference['source_id']}`: **{reference['rating']}**. {reference['brief_reason']} Not pooled into the primary structured-citation counts.")
        evidence += [""]
    for name, lines in [("answers.md", exact), ("answer-assessment.md", quality), ("retrieval-assessment.md", evidence)]:
        (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    new_summary = read_json(RUN / "summary.json")
    cost = read_json(ROOT / "data/checks/requirement-5-cost-audit-20261011.json")
    appraisals = read_json(RUN / "assessment-summary.json")["models"]
    gemma = ROOT / REFERENCE
    labels = {"glm-5.3": "GLM 5.3", "kimi-k3": "Kimi K3", "mimo-v2.6-pro": "MiMo-V2.6-Pro",
              "qwen3.8-27b": "Qwen3.8 27B", "gemma-reference": "Gemma 4 26B A4B"}
    summaries = {"gemma-reference": read_json(gemma / "summary.json"), **new_summary["models"]}
    grades = {"gemma-reference": read_json(gemma / "assessment-summary-v2.json"), **appraisals}
    table = ["| Model | Released-key matches | Valid answers | Searches / reads | Searching cases | Citing answers | Primary requests | Primary cost USD |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    cites = ["| Model | Supported | Partial | Unsupported | Not assessable | Invalid registry IDs |",
             "|---|---:|---:|---:|---:|---:|"]
    relevance = ["| Model | Direct | Partial | Irrelevant | Not assessable | Paper pairs |",
                 "|---|---:|---:|---:|---:|---:|"]
    scores = {"gemma-reference": {s["question_id"]: s for s in read_json(gemma / "scores.json")["scores"]}}
    for slug, summary in summaries.items():
        names = summary["tool_names"]
        table.append(f"| {labels[slug]} | {summary['correct']} / 30 | {summary['valid_completed']} / 30 | {names.get('search_pubmed', 0)} / {names.get('read_pubmed_abstract', 0)} | {summary['episodes_with_search']} / 30 | {summary['episodes_with_citations']} / 30 | {summary['provider_requests']} | {summary['reported_or_estimated_cost_usd']:.8f} |")
        support = grades[slug]["citation_support"]
        cites.append(f"| {labels[slug]} | {support.get('supported', 0)} | {support.get('partially_supported', 0)} | {support.get('unsupported', 0)} | {support.get('not_assessable', 0)} | {grades[slug].get('unknown_citation_ids', 0)} |")
        papers = grades[slug]["paper_relevance"]
        relevance.append(f"| {labels[slug]} | {papers.get('directly_relevant', 0)} | {papers.get('partially_relevant', 0)} | {papers.get('irrelevant', 0)} | {papers.get('not_assessable', 0)} | {sum(papers.values())} |")
        if slug in MODELS:
            scores[slug] = {s["question_id"]: s for s in read_json(RUN / slug / "scores.json")["scores"]}
            render_model(slug)
    paired = ["| Question | Released key | Gemma | GLM | Kimi | MiMo | Qwen |", "|---|---|---|---|---|---|---|"]
    for identifier, gold in scores["gemma-reference"].items():
        choices = [str(scores[slug][identifier]["selected_option"]) for slug in ["gemma-reference", *MODELS]]
        paired.append("| " + " | ".join([identifier, gold["reference_option"], *choices]) + " |")
    issues = ["- **Gemma 4 26B A4B (reused appraisal):** material-issue cases: BM15, BM21, BM26."]
    issues += [f"- **{labels[slug]}:** material-issue cases: " + (", ".join(appraisals[slug]["material_issue_cases"]) or "None under this assistant rubric.") for slug in MODELS]
    links = [f"- **{labels[slug]}:** [exact answers]({slug}/answers.md), [answer appraisal]({slug}/answer-assessment.md), [retrieval/citations]({slug}/retrieval-assessment.md), [raw records]({slug}/answers)." for slug in MODELS]
    intervals = [f"- {labels[slug]}: {100*s['wilson_95_interval_planned'][0]:.2f}%-{100*s['wilson_95_interval_planned'][1]:.2f}%." for slug, s in summaries.items()]
    text = ["# Requirement 5: five-model full-agent MedQA comparison", "",
        "All four newly selected models completed thirty fresh-state benchmark episodes. Gemma's existing thirty-case cohort is reused, not regenerated. The score is code-based exact agreement with unchanged released MedQA labels; no LLM determines the correct option. Explanation, retrieval and citation grades below are separate unblinded assistant assessments.", "",
        "## Objective and operational results", "", *table, "",
        "One common thirty-question test subset, same option order, effective prompt, v3 LangGraph, PubMed tools, 16384 output-token cap, 180-second timeout and research/finalization allowance. No wrong-answer reruns or score-based replacements. Model-specific reasoning/serving differences are declared in [the frozen protocol](protocol.md): GLM/Kimi high, Qwen xhigh because high is not advertised, MiMo/Gemma enabled without equivalent effort scales. Provider/quantization and earlier Gemma execution prevent claiming identical internal compute or serving conditions.", "",
        "The frozen sampler uses seed 202610114, the first thirty of a preserved fifty-case order, after project-history/duplicate and missing-visual screening (1076 eligible of 1273 released test rows). Question wording, options and keys are unchanged. [Selection provenance](selection-manifest.json), [original source catalogue](../../../research/requirement-4-benchmark-sources-20261011.json). This holds out project development history, not model pretraining; the items are general medical examination questions, separate from the ten internal cases.", "",
        "Configured provider pins are Baidu (GLM), InferenceNet (Kimi), Xiaomi (MiMo), Near AI (Qwen) and the retained NextBit reference (Gemma), with fallbacks disabled. Public endpoint metadata advertises fp8/fp4/fp8/fp8 respectively for the four new routes. Pins, parameter support and numeric reasoning usage were checked; actual weight quantization/internal compute are not independently observable. [Profiles and public audit](../../../checks/requirement-5-panel-public-audit-20261011.json), [provider/reasoning method sources](../../../research/requirement-5-comparison-method-sources-20261011.json).", "",
        "Tool actions are validated JSON decisions dispatched by the graph, not native function calling or an MCP server. Each episode starts without evidence; the model can answer directly, search up to five 400-character excerpts, or read up to three found abstracts of 3500 characters. No model receives gold papers or another model's answer. Three research actions are followed by reserved finalization, with five model requests maximum.", "",
        "Operationally, all 120 new primary episodes produced model answers, with no application fallback, blocked budget or model/provider failure. Kimi/MiMo/Qwen used reserved finalization in 4/6/2 episodes; GLM needed none. Qwen had one local transport normalization; one Kimi PubMed HTTP retry accounts for its 65 attempts versus 64 logical requests. Recovery is bounded and changes neither clinical wording nor the chosen option. Successful coverage here is an observed result, not a future uptime guarantee.", "",
        "Wilson 95% intervals (do not account for benchmark contamination or clinical deployment):", "", *intervals, "",
        "## Separate citation support", "", *cites, "",
        "Counts concern actual structured cited identifiers and claim-specific delivered-text support. Additional inline-only reference mentions are inspected and documented separately in each model's detailed appraisal, without silently repairing the array or changing primary counts. Bare-PMID abbreviations of already-assessed structured references are not counted twice. Valid real-paper IDs are not proof that the claimed result appears in an excerpt. Supported does not certify every sentence or all patient-level inferences. Unregistered references remain failures; source text read by another model or a later researcher is never retroactive evidence.", "",
        "## Separate paper relevance", "", *relevance, "",
        "Paper pairs are model-question-paper pairs; the same paper can appear in several models. These are topical relevance grades of delivered views, not exhaustive recall, full-text review or study-quality certification. Query grades assess wording/topic rather than being assigned from whether results are empty.", "",
        "## Qualitative findings and review limits", "", *issues, "",
        "All 120 new explanations, every new returned paper pair and every cited view are reviewed. Gemma's complete previous appraisal remains separate and is reused. The exact notes show which decisive facts are used, where explanations add unsupported details, whether clinical assumptions are disclosed, and whether retrieved text supports the associated claim. Released-key disagreement is never corrected by the assistant's opinion; possible ambiguity is documented separately.", "",
        "Focused illustrations already substantiated in notes: MiMo BM04 confuses prophase-I arrest with whole-meiosis completion; Kimi BM05 attributes confirmation to a trial when its delivered view is background/methods; reading the full BM07 abstract gives MiMo actual human hemodynamic findings unavailable to models citing only the search methods snippet. These examples demonstrate why option accuracy, real-reference identity and claim support must remain distinct.", "",
        "The actual retrieval differences are substantial: MiMo searched every case and read eight times, while Qwen searched nineteen cases and read four times. MiMo supplied more supported citations in absolute count, but had three key mismatches and six material explanation flags; more searches are not automatically a better answer. Unsupported citations concern delivered-view scope, not fabricated paper existence. The one MiMo BM13 inline-only reference was separately assessed as supported and is excluded from the structured-citation totals.", "",
        "Key-disagreement cases deserve separate interpretation. In BM20, Kimi and MiMo select contrast enema before any initial plain radiograph is supplied; condition-specific diagnostic papers do not establish that first-step sequence. In BM21, GLM and MiMo select colchicine instead of the released XO-inhibitor key: colchicine marrow toxicity is real, but the IBD/unknown-drug clue suggests a thiopurine interaction; the missing drug list limits certainty. In BM28, MiMo and Qwen select intubation after assuming serotonin toxicity, while the new antiemetic points toward NMS; GLM matches dantrolene yet also assumes undocumented MDMA exposure. These observations do not reassign official labels. See the exact per-case appraisals and [researcher source catalogues](../../../research/requirement-5-clinical-review-sources-20261011.json).", "",
        "## Paired per-question choices", "", *paired, "",
        "This small filtered exam sample has a ceiling: Gemma's 30/30 cannot be improved in option score on these same cases. A difference of one to three items is not a reliable general medical ranking. Model pretraining contamination remains possible. There is no tools-disabled benchmark condition, so scores/search frequency cannot identify a causal tool benefit. No independent clinician validation is claimed.", "",
        "## Cost, checks and reproducibility", "",
        f"All new requirement-5 stages: **USD {cost['known_reported_or_estimated_total_usd']:.10f} / {cost['saved_provider_requests']} saved provider charges**, including four native probes and four full-policy pilots outside the score. Latest saved cumulative account usage: **USD {cost['latest_ledger']['total_usage']:.9f}**, below USD 6. [Cost/account audit](../../../checks/requirement-5-cost-audit-20261011.json). Earlier point-4 in-flight uncertainty remains explicit; Gemma is not billed again.", "",
        "The observed full offline suite passed **184 tests**, including twelve new scheduling/prerequisite/lagging-ledger checks. [Primary replay](../../../checks/requirement-5-main-verification-20261011.json) verifies 120 episodes/264 saved model responses/239 HTTP responses; probe/pilot replay adds 8 episodes/11 responses/5 HTTP responses, with zero new network/credentials. Replay verifies operational fidelity and scores, not clinical correctness.", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -X utf8 scripts/verify_clinical_model_comparison.py --run-dir data/evaluation/comparison-runs/medqa30-four-model-main-20261011",
        ".\\.venv\\Scripts\\python.exe -X utf8 scripts/build_clinical_model_comparison_report.py", "```", "",
        "These replay/re-render saved artifacts without paid inference. Hosted stages require fresh directories, matching probes/pilots, price/parameter checks and a shared full-cap per-request spending guard; no medical-quality retry.", "",
        "Qualitative note validation also caught an authoring-label mismatch in nine notes: material_concern was normalized to the frozen rubric's material_issue, with original notes/hash history preserved in [review revisions](manual-review-revisions). The observations and clinical judgments were unchanged; no response, key, agent or frozen protocol was modified.", "",
        "## Artifact map", "", *links, "",
        "Gemma: [original results](../../benchmark-runs/gemma-full-agent-medqa30-v3-20261011/results.md), [exact answers](../../benchmark-runs/gemma-full-agent-medqa30-v3-20261011/answers.md), [citation appraisal](../../benchmark-runs/gemma-full-agent-medqa30-v3-20261011/retrieval-assessment.md).", "",
        "[Manual notes](manual-reviews), [structured assessments](assessment-summary.json), [protocol](protocol.md), [answer rubric](rubric.json), [retrieval rubric](retrieval-rubric.json), [source/profile/data provenance](manifest.json), [method sources](../../../research/requirement-5-comparison-method-sources-20261011.json), [new clinical check sources](../../../research/requirement-5-clinical-review-sources-20261011.json), [all new primary observed papers](../../../research/requirement-5-observed-papers-20261011.json). Clinical references also reuse [the twelve point-4 sources](../../../research/requirement-4-clinical-review-sources-20261011.json); indexed/archived/read limitations are retained. These are researcher context, not answering-agent inputs.", "",
        "Ling Sante and Opus were excluded. Requirements 6-7 (PDF QA and separate patient memory) remain pending. Report is Markdown; no new compiled LaTeX/PDF.", ""]
    (RUN / "results.md").write_text("\n".join(text), encoding="utf-8")
    print({"rendered_models": 4, "exact_new_answers": 120, "new_model_requests": 0})


if __name__ == "__main__":
    main()
