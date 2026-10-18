"""Render saved outputs and source provenance without paid model calls."""
from __future__ import annotations

from pathlib import Path

from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.medqa_option_diagnostic import visible_option

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/benchmark-runs/openweight-medqa-high-20261011"
EXTENSION = ROOT / "data/evaluation/benchmark-runs/ling-sante-medqa-extension-20261011"
COLLECTION = ROOT / "data/evaluation/benchmark-runs/medqa-six-model-summary-20261011"

TECH_SOURCES = [
    ("MQT01", "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams", "Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, Peter Szolovits", "2020", "https://arxiv.org/abs/2009.13081", "Consulted preprint arXiv:2009.13081, DOI 10.48550/arXiv.2009.13081; dataset origin and medical multiple-choice task."),
    ("MQT02", "MedQA: Code and data for MedQA", "jind11 / MedQA authors", "Repository, accessed 2026", "https://github.com/jind11/MedQA", "Original provenance, dataset splits and four-option release."),
    ("MQT03", "MedQA-USMLE-4-options", "GBaker", "Dataset repository, accessed 2026", "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options", "Frozen practical artifact, published answer/index fields and declared CC BY 4.0 license; not a separately verified examination-board key."),
    ("MQT04", "Reasoning Tokens", "OpenRouter", "Living documentation, accessed 2026", "https://openrouter.ai/docs/guides/best-practices/reasoning-tokens", "Advertised native effort discovery, reasoning enablement, exclusion, billing and output-budget limits."),
    ("MQT05", "Get remaining credits", "OpenRouter", "Living documentation, accessed 2026", "https://openrouter.ai/docs/api/api-reference/credits/get-remaining-credits", "Read-only account ledger fields; actual observed totals are saved separately."),
    ("MQT06", "Provider Routing", "OpenRouter", "Living documentation, accessed 2026", "https://openrouter.ai/docs/guides/routing/provider-selection", "Price ceilings and parameter support filtering; routes can vary."),
    ("MQT07", "GLM 5.3: model page", "OpenRouter / Z.ai", "Model catalogue, accessed 2026", "https://openrouter.ai/z-ai/glm-5.3", "Model identity; live JSON catalogue/endpoint snapshots provide the actual advertised prices and reasoning levels used."),
    ("MQT08", "Kimi K3: model page", "OpenRouter / Moonshot AI", "Model catalogue, accessed 2026", "https://openrouter.ai/moonshotai/kimi-k3", "Selected exact model identity."),
    ("MQT09", "MiMo-V2.6-Pro: model page", "OpenRouter / Xiaomi", "Model catalogue, accessed 2026", "https://openrouter.ai/xiaomi/mimo-v2.6-pro", "Selected exact model identity; native effort scale is not advertised in the saved catalogue."),
    ("MQT10", "Qwen3.8 27B: model page", "OpenRouter / Qwen", "Model catalogue, accessed 2026", "https://openrouter.ai/qwen/qwen3.8-27b", "Selected exact identity and native-effort discrepancy."),
    ("MQT11", "Gemma 4 26B A4B: model page", "OpenRouter / Google", "Model catalogue, accessed 2026", "https://openrouter.ai/google/gemma-4-26b-a4b-it", "Selected MoE identity; distinct from the older 31B profile."),
    ("MQT12", "Qwen3.8-27B: developer model card", "Qwen", "Model repository, accessed 2026", "https://huggingface.co/Qwen/Qwen3.8-27B", "Native reasoning controls; no local inference is performed."),
    ("MQT13", "Gemma 4 Prompt Formatting", "Google", "Living documentation, accessed 2026", "https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4", "Thinking-mode formatting; no native high scale or equal compute is inferred."),
    ("MQT14", "MiMo-V2.6-Pro: Intelligence, Performance & Price Analysis", "Artificial Analysis", "Living evaluation page, accessed 2026", "https://artificialanalysis.ai/models/mimo-v2-6-pro", "Context for the user's selection rationale; the full healthcare ranking is not independently reproduced by this study."),
    ("MQT15", "Ling 3.0 Flash Sante: model page", "OpenRouter / InclusionAI", "Model catalogue, accessed 2026", "https://openrouter.ai/inclusionai/ling-3.0-flash-sante", "Exact sixth-model identity, advertised medical focus and regular endpoint pricing; medical safety claims are vendor descriptions rather than this study's validation."),
]


def main():
    summary = read_json(RUN / "summary.json")
    manifest = read_json(RUN / "manifest.json")
    extension_manifest = read_json(EXTENSION / "manifest.json")
    assert file_hash(RUN / "questions.json") == file_hash(EXTENSION / "questions.json")
    assert file_hash(RUN / "answer-key.json") == file_hash(EXTENSION / "answer-key.json")
    questions = read_json(RUN / "questions.json")["questions"]
    keys = {item["question_id"]: item for item in read_json(RUN / "answer-key.json")["keys"]}
    profiles = [item["id"] for item in manifest["profiles"]]
    extension_profiles = [item["id"] for item in extension_manifest["profiles"]]
    origins = {profile: RUN for profile in profiles}
    origins.update({profile: EXTENSION for profile in extension_profiles})
    profiles.extend(extension_profiles)
    COLLECTION.mkdir(exist_ok=True)
    sources = [{"id": identifier, "title": title, "authors_or_organization": authors, "year": year, "url": url, "support": support, "role": "Dataset or technical context"} for identifier, title, authors, year, url, support in TECH_SOURCES]
    papers = {}
    lines = ["# Saved MedQA answers", "", "Objective correctness is exact equality with the unchanged released dataset option. Justification appraisal is separate. Native reasoning controls differ as disclosed in protocol.md. All outputs are educational research artifacts.", ""]
    for question in questions:
        qid = question["id"]
        lines.extend([f"## {qid} (original test row {question['source_row_index']})", "", question["question"], ""])
        lines.extend([f"- **{letter}:** {value}" for letter, value in question["options"].items()])
        lines.extend(["", f"**Released key:** {keys[qid]['answer_idx']} - {keys[qid]['answer']}", ""])
        for mode in ("direct", "agent"):
            if mode == "agent" and qid not in manifest["agent_question_ids"]:
                continue
            lines.extend([f"### {mode.title()} condition", ""])
            for profile in profiles:
                path = origins[profile] / "models" / profile / "answers" / f"{mode}-{qid}.json"
                record = read_json(path)
                answer = record.get("answer", {})
                if record["status"] == "completed":
                    correct = answer["selected_option"] == keys[qid]["answer_idx"]
                    lines.extend([f"**{profile}: {answer['selected_option']} ({'key-concordant' if correct else 'key-discordant'})**", "", answer["justification"], ""])
                else:
                    lines.extend([f"**{profile}: {record['status']}**", "", f"Saved error type: {record.get('error_type', 'not applicable')}. No completed medical answer is inferred.", ""])
                    diagnostic = visible_option(record)
                    if diagnostic:
                        lines.extend([f"Post hoc deterministic format diagnostic: the single explicit embedded JSON object selects **{diagnostic['selected_option']}**, {'matching' if diagnostic['selected_option'] == keys[qid]['answer_idx'] else 'differing from'} the released key. The primary strict-format failure is preserved.", "", diagnostic["justification"], ""])
                if mode == "agent":
                    decision = record.get("decision", record.get("partial_state", {}).get("decision", {}))
                    lines.extend([f"Tool decision: {decision.get('use_pubmed', 'unavailable')}; query: {decision.get('query')}; reason: {decision.get('reason', 'unavailable')}", ""])
                    retrieval = record.get("retrieval", record.get("partial_state", {}).get("retrieval", {}))
                    for paper in retrieval.get("sources", []):
                        item = papers.setdefault(paper["id"], {"id": paper["id"], "title": paper["title"], "authors_or_organization": paper["authors"], "year": paper["year"], "doi": paper.get("doi"), "pmid": paper["pmid"], "url": paper["url"], "support": "Actually retrieved article, not automatic endorsement or proof of a claim", "role": "Agent-retrieved metadata/abstract", "read_level": paper["read_level"], "case_uses": []})
                        item["case_uses"].append({"profile": profile, "question_id": qid, "record_sha256": file_hash(path)})
                        lines.append(f"- Retrieved [{paper['id']}: {paper['title']}]({paper['url']}); abstract present: {paper['abstract_present']}; saved abstract truncated: {paper['abstract_truncated']}.")
                    if retrieval.get("sources"):
                        lines.append("")
                relative = "../" + origins[profile].name + f"/models/{profile}/answers/{mode}-{qid}.json"
                lines.extend([f"[Raw record]({relative})", ""])
    (COLLECTION / "answers.md").write_text("\n".join(lines), encoding="utf-8")
    sources.extend(papers.values())
    for filename, support in [
        ("medqa-posthoc-anatomy-references.json", "Surgical-route alternatives or variable nerve/artery anatomy"),
        ("medqa-posthoc-justification-references.json", "Reactive-arthritis triad and the passive-extension-pain distractor"),
        ("medqa-posthoc-hcc-reference.json", "Conditional workup of indeterminate liver observations; blanket mandatory-biopsy claims are too broad"),
    ]:
        posthoc = ROOT / "data/checks" / filename
        if posthoc.exists():
            for paper in read_json(posthoc)["sources"]:
                sources.append({"id": "MQC" + paper["pmid"], "title": paper["title"], "authors_or_organization": paper["authors"], "year": paper["year"], "doi": paper.get("doi"), "pmid": paper["pmid"], "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10663390/" if paper["pmid"] == "37199193" else paper["url"], "role": "Post hoc appraisal only; never supplied by the evaluator to generation or substituted for a key", "support": support + "; does not establish a replacement answer key", "read_level": "Relevant operative methods at publisher" if paper["pmid"] == "15006883" else ("Relevant AASLD guidance text at PMC" if paper["pmid"] == "37199193" else "PubMed metadata/abstract")})
    sources.append({"id": "MQC_BOOK_2018", "title": "Pathways for the Spread of Disease in the Abdomen and Pelvis", "authors_or_organization": "James A. Brink, Brent J. Wagner", "year": "2018", "doi": "10.1007/978-3-319-75019-4_6", "url": "https://link.springer.com/chapter/10.1007/978-3-319-75019-4_6", "role": "Post hoc anatomy appraisal only; expert textbook chapter, not a new benchmark key or model input", "read_level": "Relevant chapter text, section 6.2.1 at publisher", "support": "Distinguishes gastrohepatic and hepatoduodenal parts of lesser omentum and identifies the hepatoduodenal free edge; limits Qwen's extra anatomical label."})
    write_json(ROOT / "data/research/medqa-mini-sources.json", {"sources": sources, "source_count": len(sources), "agent_paper_count": len(papers)})
    rows = summary["rows"] + read_json(EXTENSION / "summary.json")["rows"]
    record_origins = [{"profile": record["profile"], "mode": record["mode"], "question_id": record["question_id"], "path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path)} for origin in (RUN, EXTENSION) for path in sorted(origin.glob("models/*/answers/*.json")) for record in [read_json(path)]]
    collection = {"question_sha256": file_hash(RUN / "questions.json"), "key_sha256": file_hash(RUN / "answer-key.json"), "cohorts": [{"path": origin.relative_to(ROOT).as_posix(), "manifest_sha256": file_hash(origin / "manifest.json")} for origin in (RUN, EXTENSION)], "rows": rows, "records": record_origins, "timing_limitation": "Ling was added as a later cohort at the user's request; five primary models were interleaved in seeded rotation, Ling was not.", "objective_scoring": "Unchanged released MedQA option labels, never an LLM judge", "final_budget": read_json(EXTENSION / "budget.json")}
    audit = read_json(ROOT / "data/checks/medqa-six-model-cost-audit-20261011.json")
    collection["final_account_ledger_audit"] = "data/checks/medqa-six-model-cost-audit-20261011.json"
    collection["final_observed_account_usage_usd"] = audit["ledger"]["total_usage"]
    write_json(COLLECTION / "summary.json", collection)
    overview = ["# MedQA mini evaluation results", "", "Ten original short MedQA test questions; objective released-key scoring. Three presampled cases also receive an optional-search LangGraph agent run for every model. No paid judge or compiled report.", "", "| Model | Direct correct/planned | Direct completed | Agent correct/planned | Agent completed | New reported USD |", "|---|---:|---:|---:|---:|---:|"]
    for profile in profiles:
        direct = next(row for row in rows if row["profile"] == profile and row["mode"] == "direct")
        agent = next(row for row in rows if row["profile"] == profile and row["mode"] == "agent")
        overview.append(f"| {profile} | {direct['correct']}/10 | {direct['completed']}/10 | {agent['correct']}/3 | {agent['completed']}/3 | {direct['reported_cost_usd'] + agent['reported_cost_usd']:.6f} |")
    total_cost = sum(row["reported_cost_usd"] for row in rows)
    overview.extend(["", f"New reported inference charges: **USD {total_cost:.9f}**. Final authenticated account consumption: **USD {collection['final_observed_account_usage_usd']:.9f}**, below the user's cumulative USD 6 guard. The increase matches reported new charges within rounding; the USD 0.15 reserve is bookkeeping, not an additional bill.", "", "Three direct outputs exhausted the 6144-token budget, predominantly in reasoning: GLM MQ04 and Gemma MQ04/MQ06. These lack usable final answers. Ling MQ06 is different: generation stopped normally, but prose preceded the JSON, causing the strict parser to reject it. A separate post hoc deterministic diagnostic extracts its explicit D, whereas the released key is A. That discrepancy is recorded without changing original strict scores, labels or outputs. MiMo MQ04 is the completed key-discordant direct answer. Report format compliance and completion alongside option concordance; endpoint/cap failures are not observed medical-belief errors.", "", "All six models have 3/3 key-concordant paired agent answers, equal to their direct baselines on those three cases. The lightweight agent condition adds 18 planning calls and leaves correctness unchanged in this small ceiling-limited sample. Gemma MQ01 is the only PubMed query: zero matches, including an unexpected 'to the' author mapping in automatic query translation. No retrieved papers or citation benefit were demonstrated by this small paired study.", "", "Native reasoning: GLM/Kimi high; Qwen xhigh; MiMo/Gemma/Ling enabled without a native effort scale. The user accepted the initial differences; Ling uses the same enablement convention. Hidden compute and provider routing are unequal. Ling is a later extension cohort, not interleaved with the initial five.", "", "The objective scores do not adjudicate whether every sentence in a key-concordant explanation is clinically sound. Qualitative justification and agent findings are recorded separately in qualitative-review.md. Small sample, potential public-benchmark memorization, one run per condition, general medicine and no independent clinician validation limit interpretation.", "", "[Saved questions, keys and all answers](answers.md) | [Provenance and combined numerical results](summary.json) | [Primary objective scores](../openweight-medqa-high-20261011/scores.json) | [Ling objective scores](../ling-sante-medqa-extension-20261011/scores.json) | [Separate raw-option diagnostic](raw-option-diagnostic.json) | [Primary protocol](../openweight-medqa-high-20261011/protocol.md) | [Ling addendum](../ling-sante-medqa-extension-20261011/protocol-addendum.md) | [Final cost audit](../../../checks/medqa-six-model-cost-audit-20261011.json)", "", "Earlier nutrition experiments remain preserved. Requirements 6-7 are not demonstrated by this study."])
    (COLLECTION / "results.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    print(f"Rendered {len(questions)} question sections, {len(profiles)} models and {len(sources)} registered sources. No inference calls.")


if __name__ == "__main__":
    main()
