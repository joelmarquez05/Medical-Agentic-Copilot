"""Render preserved baseline answers, objective scores and separate appraisal."""

import argparse
import json
from collections import Counter
from pathlib import Path

from medical_copilot.evaluation import file_hash, read_json, write_json


def render(root, run):
    manifest, summary = read_json(run / "manifest.json"), read_json(run / "summary.json")
    assert manifest["stage"] == "baseline" and manifest.get("finished_at_utc")
    keys = {item["question_id"]: item for item in read_json(run / "answer-key.json")["keys"]}
    scores = {item["question_id"]: item for item in read_json(run / "scores.json")["scores"]}
    review = read_json(run / "qualitative-review.json")
    reviews = {item["question_id"]: item for item in review["reviews"]}
    assert set(reviews) == set(keys) == set(scores)
    catalogs = ["data/research/requirement-2-restart-sources.json", "data/research/internal-pubmed-set-sources-20261011.json"]
    sources = {item["id"]: item for name in catalogs for item in read_json(root / name)["sources"]}
    used_ids = sorted({sid for item in reviews.values() for sid in item["reference_source_ids"]})
    assert set(used_ids) <= set(sources)
    write_json(run / "review-sources.json", {
        "purpose": "Evaluator references; never evidence retrieved/supplied by the baseline answering agent",
        "source_catalogue_hashes": {name: file_hash(root / name) for name in catalogs},
        "sources": [sources[sid] for sid in used_ids],
        "additional_post_generation_read_observations": [
            {"source_id": "CL05", "date": "2026-10-11", "read_level": "Indexed primary-publisher section 11 and 11.2.3 text. Direct publisher full-page fetch returned 403; PMC returned a browser challenge.", "supports": "Separates initial UFH from urgent revascularization in IC04; no alternative key assigned."},
            {"source_id": "CL06", "date": "2026-10-11", "read_level": "Indexed official ACOG recommendations and indications text. Direct full-page fetch returned 402.", "supports": "Prior affected infant supports prophylaxis in the supplied active-labor case."},
            {"source_id": "PREVENT_CLOT", "date": "2026-10-11", "read_level": "Original PubMed abstract opened again; matching previously saved NCBI EFetch abstract.", "supports": "Case IC09 population description differs from original eligibility despite correct DVT numbers."}
        ],
        "primary_abstracts": "../../../research/internal-pubmed-original-abstracts-20261011.json",
        "interpretation": "Post-generation review does not establish the answering model's source access. Published keys and primary frozen inputs unchanged."})
    answers = ["# Requirement-2 baseline: questions and preserved answers", "",
               "Ten original primary answers; no replacements, paid judge, repair or retrieval. Evaluator keys below were not supplied to Gemma. Trial-fact keys are project-authored published-fact checks, not official examination labels.", ""]
    appraisal = ["# Separate qualitative explanation review", "", review["reviewer"] + ".",
                 "", "Ratings apply the [frozen rubric](qualitative-rubric.json). Exact option scoring remains in [scores.json](scores.json); these ratings cannot change its keys.", "",
                 "A correct option can coexist with an explanation error. Reference concordance here is post-generation appraisal, not evidence grounding by the baseline model.", "",
                 "| Case | Case reasoning | Reference concordance | Calibration/provenance |",
                 "|---|---|---|---|"]
    table = ["| Case | Key type | Reference | Selected | Match | Status |", "|---|---|---|---|---|---|"]
    for identifier in manifest["question_ids"]:
        record_path = run / "answers" / (identifier + ".json")
        record, item, score, key = read_json(record_path), reviews[identifier], scores[identifier], keys[identifier]
        assert item["answer_sha256"] == file_hash(record_path)
        assert item["option_score_unchanged"] is True
        assert item["reference_source_ids"] == key["reference_source_ids"]
        question, response = record["question"], record.get("response") or {}
        ratings = item["dimension_ratings"]
        assert set(ratings.values()) <= {"consistent", "minor_limitation", "material_issue", "not_evaluable"}
        match = "Yes" if score["correct"] is True else "No" if score["correct"] is False else "Not evaluable"
        table.append(f"| [{identifier}](answers.md#{identifier.lower()}) | {key['key_type']} | {key['answer_idx']} | {score['selected_option'] or 'None'} | {match} | {record['status']} |")
        appraisal.append(f"| {identifier} | {ratings['case_reasoning']} | {ratings['reference_concordance']} | {ratings['calibration_and_provenance']} |")
        answers += [f"## {identifier}", "", question["question"], ""]
        answers += [f"- {letter}: {text}" for letter, text in question["options"].items()]
        answers += ["", f"Evaluator key: **{key['answer_idx']} — {key['answer']}** ({key['key_type']}). Selected: **{score['selected_option']}**. Exact match: **{match}**.", "",
                    "Model answer:", "", response.get("answer", "No usable answer."), "", "Model justification:", "", response.get("justification", "Not available."), "",
                    "Model limitations: " + ("; ".join(response.get("limitations", [])) or "None returned."), "",
                    f"Operational status: {record['status']}; model turns: {record['model_turns']}; tool actions: {record['tool_actions']}; evidence registry: empty.", "",
                    f"[Original JSON record](answers/{identifier}.json). Post-generation appraisal: {item['brief_observed_notes']}", ""]
    for identifier in manifest["question_ids"]:
        item = reviews[identifier]
        appraisal += ["", f"## {identifier}", "", item["brief_observed_notes"], "",
                      "Reference context: " + ", ".join(f"[{sid}]({sources[sid]['url']})" for sid in item["reference_source_ids"]) + ".",
                      "", f"Saved-answer SHA-256: `{item['answer_sha256']}`."]
    appraisal += ["", "## Limitations", "", review["limits"], "",
                  "Six responses (IC02–IC07) omit the system-requested explicit acknowledgment that no external evidence was supplied. Their empty citation lists and saved evidence registries do not contain a false search, but the missing acknowledgment is a prompt-compliance/provenance limitation. It does not affect option scores.", "",
                  "Source titles, authors/organizations, years, DOI/PMID/URLs and read/access limitations are preserved in [review-sources.json](review-sources.json). The earlier research catalogues remain unchanged.", ""]
    counts = {dimension: dict(Counter(item["dimension_ratings"][dimension] for item in reviews.values())) for dimension in next(iter(reviews.values()))["dimension_ratings"]}
    write_json(run / "assessment-summary.json", {"reviewed": len(reviews), "dimension_counts": counts,
               "option_scores_changed": 0, "paid_judge_requests": 0,
               "interpretation": "Descriptive counts from subjective unblinded review, not a validated aggregate medical-quality score."})
    cost = read_json(root / "data/checks/gemma-restart-cost-audit-20261011.json")
    pilot_dir = Path(manifest["pilot_origin"])
    pilot = read_json(pilot_dir / "summary.json")
    duration = sum(read_json(run / "answers" / (identifier + ".json"))["elapsed_seconds"] for identifier in manifest["question_ids"])
    results = ["# Restarted requirement-2 baseline results", "",
               "The new LangGraph agent completed all ten questions with Gemma 4 26B A4B and literature tools disabled. **9/10 selected options match the frozen keys:** 6/7 released MedQA labels and 3/3 project-authored primary-trial fact keys. All ten produced valid answers; one completed answer differs from its key. No provider, truncation or JSON failure was observed.", "",
               "The [separate integration pilot](../gemma-restart-pilot-20261011/answers/PILOT-01.json) completed before the primary collection and is excluded from accuracy. Historical studies are not pooled into these results.", "",
               "## Implemented setup", "",
               "A single bounded LangGraph agent initializes fresh state, obtains a model action and routes to a permitted tool or final-answer validation. The shared LangChain text backend and validated JSON action contract allow provider/model substitution. This run exposes an empty tool registry, so every question took one model call followed by validation and finish. Synthetic offline tests exercise result feedback and the tool branch; real literature-tool evaluation belongs to requirement 3.", "",
               "The unchanged reference profile uses `google/gemma-4-26b-a4b-it` through OpenRouter, configured NextBit pin/no fallback, temperature 0, reasoning enabled/private reasoning excluded, a 16384-token output cap and 180-second timeout. SDK and application retries, format repairs and self-review are disabled. Actual provider-route identity is not independently returned by the saved SDK response. This is application-dispatched JSON, not demonstrated native vendor function calling.", "",
               "The seven examination cases are MedQA training rows, not held-out external benchmark cases. The three named-trial cases ask for published numerical findings from BALANCE, PREVENT CLOT and SOUL. Trial names/answer options are visible, but no key, abstract, reference PMID/DOI or prior answer enters generation. These tasks allow memorization and elimination from options; correctness alone does not demonstrate source reading.", "",
               "## Objective option scores", "", *table, "",
               "Completion coverage: 10/10. Exact-key agreement: 9/10 planned and 9/10 completed. Operational failures: 0. Unattempted cases: 0. Tool executions/PubMed requests: 0.", "",
               "Scoring is exact code comparison with the [frozen evaluator keys](answer-key.json). It is not an LLM determination of the correct option. [All stems/options and exact model answers](answers.md), [machine scores](scores.json), [raw records](answers) and [protocol](protocol.md) are reviewable.", "",
               "## Explanation findings", "",
               "The explanations usually identify the decisive vignette facts or correct trial numbers. The assistant's [unblinded separate review](qualitative-review.md) applies the frozen three-dimension rubric and binds each assessment to the original record hash. It is not clinician validation or a second paid model run.", "",
               "**IC04:** Gemma selects thrombectomy C, while the released label is heparin B. The guideline supports UFH on diagnosis and urgent revascularization for an immediately threatened limb. The answer recognizes the latter but does not clearly prioritize the initial anticoagulation step. The vignette juxtaposes initial treatment and definitive rescue, so this remains both a recorded key mismatch and a clinical prioritization ambiguity. The original key and 9/10 headline are preserved. [2024 PAD guideline](https://www.ahajournals.org/doi/full/10.1161/CIR.0000000000001251).", "",
               "**IC09:** the selected DVT percentages and option D are correct, but Gemma describes PREVENT CLOT as a hip-fracture trial. Its original population included operatively treated extremity fractures and pelvic/acetabular fractures. This material applicability error illustrates why explanation review is kept separate from option accuracy. [Original trial abstract](https://pubmed.ncbi.nlm.nih.gov/36652352/).", "",
               "Other limits: IC02 introduces a pleuritic descriptor absent from the stem; IC03 does not discuss the medication distractor; IC02–IC07 omit the requested no-supplied-evidence acknowledgment. No baseline answer claims an actual PubMed search or uses a registered citation.", "",
               "## Execution, cost and checks", "",
               f"Ten primary inference requests: **USD {summary['reported_or_estimated_cost_usd']:.10f}** provider-reported. One pilot: **USD {pilot['reported_or_estimated_cost_usd']:.10f}**. Total new reported charges: **USD {cost['new_provider_reported_total_usd']:.10f}**, with zero unknown-charge records. Uncertainty headroom is bookkeeping, not an observed charge.", "",
               f"Latest authenticated account observation: **USD {cost['latest_ledger']['total_usage']:.9f} cumulative usage** at {cost['latest_ledger']['checked_at_utc']}, below USD 6. Immediate pilot/baseline ledgers lagged response usage. The later account delta differs from summed response costs by approximately {abs(cost['account_observed_delta_usd']-cost['new_provider_reported_total_usd']):.2g} USD, consistent within 1e-8 USD rounding tolerance; no charge is inferred as zero from an unchanged immediate balance. [Cost observations](../../../checks/gemma-restart-cost-audit-20261011.json).", "",
               f"Baseline tokens: {summary['input_tokens']} input and {summary['output_tokens']} output, including {summary['reasoning_tokens']} numerically reported reasoning tokens. Private reasoning text is excluded. Summed per-case wall time: {duration:.1f} seconds; this single-provider observation is not a latency benchmark.", "",
               "The 108-test offline suite passes, including 24 restarted graph/runner tests. The one pilot and ten primary records replay through the JSON adapter and graph without credentials, hosted inference or PubMed calls. Verification checks frozen code/input/key/profile/protocol/rubric hashes, unchanged options, fresh history/evidence, cost accounting and absence of credential-shaped strings/private reasoning fields. [Pilot verification](../../../checks/gemma-restart-pilot-verification-20261011.json), [baseline verification](../../../checks/gemma-baseline-restart-verification-20261011.json).", "",
               "## Interpretation and next step", "",
               "Requirement 2 now has an implemented provider-swappable graph, ten-case internal baseline and recorded qualitative findings. These ten selected cases cannot establish general clinical accuracy, oncology-nutrition competence, independent safety, source grounding or tool benefit. Nine correct baseline answers leave limited option-score improvement headroom, especially the 3/3 trial-fact group. Preserve every case and its key even if the tool-enabled condition reaches the same score.", "",
               "Requirement 3 should attach actual PubMed search/abstract operations to this same graph and repeat the same inputs/core profile. Record whether tools are chosen/executed, what material is returned, retrieval relevance and answer/attribution findings separately. Requirements 4–5 remain a larger separate benchmark and six-model comparison; PDF ingestion and patient memory remain later work.", "",
               "## Reproduce and inspect", "", "From the project root:", "", "```powershell",
               ".\\.venv\\Scripts\\python.exe -X utf8 scripts/verify_clinical_baseline.py --run-dir data/evaluation/internal-runs/gemma-baseline-restart-20261011",
               ".\\.venv\\Scripts\\python.exe -X utf8 scripts/build_clinical_baseline_report.py --run-dir data/evaluation/internal-runs/gemma-baseline-restart-20261011",
               "```", "", "These commands use saved artifacts only. A new authorized paid collection uses `scripts/run_clinical_baseline.py --stage pilot|baseline --run-dir <fresh-directory>` and a valid separate `--pilot-dir` for baseline; existing run directories are refused. Do not rerun primary answers to improve observed accuracy.", "",
               "Clinical reference details and actual read levels: [review-sources.json](review-sources.json). Product/architecture/model sources: [restart catalogue](../../../research/requirement-2-restart-sources.json) and [Gemma/trial catalogue](../../../research/internal-pubmed-set-sources-20261011.json). Full source/input/protocol hashes and dependency versions: [manifest.json](manifest.json) and [source snapshots](source-snapshot).", ""]
    for name, lines in [("answers.md", answers), ("qualitative-review.md", appraisal), ("results.md", results)]:
        (run / name).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rendered_cases": len(reviews), "correct": summary["correct"], "dimension_counts": counts,
                      "new_inference_requests": 0, "files": ["results.md", "answers.md", "qualitative-review.md", "review-sources.json", "assessment-summary.json"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    render(Path(__file__).resolve().parents[1], args.run_dir.resolve())


if __name__ == "__main__":
    main()
