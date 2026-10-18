"""Bind authored qualitative observations to saved answers without model judging."""
from pathlib import Path

from medical_copilot.evaluation import file_hash, read_json, write_json
from medical_copilot.medqa_option_diagnostic import visible_option

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "data/evaluation/benchmark-runs/medqa-six-model-summary-20261011"

# Authored after reading the actual saved answers. These do not assign the key.
SPECIFIC = {
    ("glm-5.3", "direct", "MQ05"): "The correct option is accompanied by a triad that substitutes prior infection for urethritis. See the shared MQ05 note.",
    ("mimo-v2.6-pro", "direct", "MQ05"): "The core enthesitis explanation fits its chosen option, but the passive-extension distractor is mislabeled dactylitis; see the post hoc tenosynovitis reference.",
    ("glm-5.3", "direct", "MQ07"): "The explanation implies suppressed renin-mediated physiology without a provided renin measurement. The option score remains determined by the released key.",
    ("kimi-k3", "direct", "MQ07"): "The explanation uses a normal abdominal exam to weaken renal artery stenosis; this negative exam is a weak basis for excluding a competing mechanism.",
    ("ling-3.0-flash-sante", "direct", "MQ07"): "Explicitly acknowledges the hydrochlorothiazide confounder before explaining the best offered aldosterone option; this is a useful case-specific qualification.",
    ("glm-5.3", "direct", "MQ06"): "The phrase reserving observation for subcentimeter nodules is too categorical compared with the post hoc AASLD conditional categories. No new category or key is assigned.",
    ("kimi-k3", "direct", "MQ06"): "The 'only' enhancement-plus-washout rule is narrower than the contemporary conditional imaging framework. This does not change the published option.",
    ("mimo-v2.6-pro", "direct", "MQ06"): "Recognizes missing washout but describes a universal biopsy requirement more strongly than the conditional contemporary workup.",
    ("qwen3.8-27b", "direct", "MQ06"): "Mentions the capsule as an additional feature, but labels observation categorically too conservative and biopsy required; the post hoc AASLD caveat still applies.",
    ("kimi-k3", "direct", "MQ08"): "Keeps nephrolithiasis explicitly possible and explains why analgesic exposure favors the offered papillary-injury choice, instead of claiming a normal examination excludes stones.",
    ("glm-5.3", "direct", "MQ08"): "Expands several weeks of use into established chronic tubulointerstitial damage, and weakly uses absent fever/mass to argue against stones.",
    ("mimo-v2.6-pro", "direct", "MQ08"): "No previous stone history does not exclude a first stone episode; the analgesic clue still explains its selected key-concordant option.",
    ("mimo-v2.6-pro", "direct", "MQ04"): "Identifies a fungal rash but selects B versus the released A. Calls griseofulvin standard and itraconazole a refractory alternative without an authoritative explanation in the selected dataset release. Plausible prose cannot override the key.",
    ("ling-3.0-flash-sante", "direct", "MQ04"): "Chooses the published A, but the assertion that oral therapy is indicated generalizes beyond the constrained option set for a small localized lesion. Relative efficacy/course claims are not supported by a supplied reference rationale.",
    ("ling-3.0-flash-sante", "direct", "MQ06"): "Prose before JSON violates the strict response format; the explicit embedded option is D versus released A. Its short-interval-imaging reasoning resembles a contemporary conditional AASLD pathway, but categorical rejection of biopsy also overstates that guidance. No diagnostic category or replacement key is assigned; benchmark disagreement is distinct from independent clinical adjudication.",
    ("ling-3.0-flash-sante", "direct", "MQ09"): "Uses 'environmental sterilization' for the room-disinfection choice. This is a terminology precision concern, with the released option unchanged.",
    ("qwen3.8-27b", "agent", "MQ01"): "Adds the claim that gastrohepatic ligament is the free edge of lesser omentum. The post hoc expert chapter identifies the hepatoduodenal free edge; this illustrates an additional anatomical imprecision despite unchanged option correctness.",
    ("gemma-4-26b-a4b", "agent", "MQ01"): "Searches for a standard anatomy fact, obtains zero matches, then gives a key-concordant answer without fabricated citations. Unlike its direct version it mentions the gastrocolic route too; with no retrieved evidence this change cannot be credited to paper support.",
    ("glm-5.3", "agent", "MQ03"): "Its exclusion language and declaration of all DSM criteria being met are stronger than the compressed vignette establishes; prefer best-fit examination reasoning.",
    ("ling-3.0-flash-sante", "agent", "MQ03"): "Says absence of an acute stressor/social fear rules out other diagnoses, rather than explaining relative likelihood under the available examination facts.",
    ("gemma-4-26b-a4b", "agent", "MQ03"): "Adds 'more days than not' as established frequency although the stem does not explicitly state it. This is justification appraisal, not an altered diagnosis label.",
}


def main():
    collection = read_json(COLLECTION / "summary.json")
    shared = read_json(ROOT / "data/evaluation/medqa-mini-20261011/appraisal-working-notes.json")["notes"]
    shared["MQ01"]["concern"] += " Qwen's agent response also labels the gastrohepatic portion as the free edge; the post hoc expert chapter distinguishes the hepatoduodenal free edge."
    shared["MQ01"]["posthoc_source_ids"].append("MQC_BOOK_2018")
    assessed = []
    for origin in collection["records"]:
        path = ROOT / origin["path"]
        record = read_json(path)
        assert file_hash(path) == origin["sha256"]
        identity = (record["profile"], record["mode"], record["question_id"])
        diagnostic = visible_option(record)
        if diagnostic:
            justification = diagnostic["justification"]
            note = SPECIFIC.get(identity, "The saved short rationale addresses the examination case and chosen option; apply the shared case caveats. No exhaustive sentence-level clinical validation or authoritative reference explanation is claimed.")
            assessment = {"status": "qualitative_appraisal" if record["status"] == "completed" else "appraisal_of_nonconforming_output", "primary_status": record["status"], "note": note, "answer_excerpt": justification, "justification_words": len(justification.split()), "exceeds_requested_80_words": len(justification.split()) > 80}
        else:
            assessment = {"status": "not_evaluable", "note": "No completed answer/justification. Preserve the paid truncated output as an operating-limit failure; no medical-belief score is inferred."}
        assessed.append({**origin, **assessment})
    write_json(COLLECTION / "qualitative-review.json", {"assessor": "Unblinded Codex implementation assistant; no paid judge or independent clinician", "scope": "Qualitative justification and agent behavior only. Dataset option correctness remains separately computed from released labels.", "shared_case_notes": shared, "records": assessed})
    lines = ["# Separate justification and agent appraisal", "", "Assessor: Codex implementation assistant, unblinded, also the implementer. No paid model judge, independent clinician or validated clinical score. All 78 saved records were inspected: 74 have valid completed outputs, one has an explicit answer in nonconforming prose-plus-JSON, and three have unusable truncated responses. Primary strict-format scores and released keys remain unchanged. The post hoc option diagnostic applies symmetrically to every model without a model judge.", "", "## Agent findings", "", "Every direct baseline on the three presampled agent cases is already key-concordant: 18/18 across six models. Agent responses also score 18/18. This ceiling prevents showing an accuracy gain; a three-case single-run ablation cannot establish causal benefit or statistical equivalence.", "", "The direct baseline for these cases costs 18 calls. The agent costs 36: one tool decision and one final answer per case. Seventeen decisions decline retrieval as established examination knowledge. Gemma MQ01 issues the only search and gets zero PubMed matches. Its natural-language query is translated with an unexpected `to the[Author]` component, making query formulation a concrete limitation. No relevant retrieved paper, citation gain or search coverage is demonstrated by this new small study. The earlier requirement-3 real retrieval evaluation remains separate.", "", "A correct agent option can accompany a weaker explanation: Qwen's MQ01 agent adds a free-edge anatomical imprecision; source-free extra wording is not evidence of improved grounding. A lightweight router can appropriately decline a tool, but here that decision still requires another paid call. Future designs could combine the initial answer/tool decision or reserve retrieval for questions needing current evidence; this suggestion was not implemented or evaluated.", "", "## Retrospective full-agent criticism", "", "The abandoned nutrition comparison used repeated large evidence contexts and multi-call planning/review/revision. The archived primary cohort made 111 provider requests across 36 attempted cases, yielding 19 completed answers. It supplied fixed old evidence, with zero live PubMed calls, so it did not measure fresh retrieval skill. Its unresolved self-review/attribution flags and separate requirement-3 false-negative attribution case show why self-review is not an authoritative correctness judge. These old observations are not MedQA results or paired experimental estimates.", "", "## Shared case observations", ""]
    catalogue = {item["id"]: item for item in read_json(ROOT / "data/research/medqa-mini-sources.json")["sources"]}
    for qid, note in shared.items():
        lines.extend([f"### {qid}", "", note["observed"], "", note["concern"], ""])
        if note.get("posthoc_source_ids"):
            lines.append("Post hoc appraisal context (not model inputs or replacement keys): " + "; ".join(f"[{catalogue[source]['title']}]({catalogue[source]['url']})" for source in note["posthoc_source_ids"]) + ".")
            lines.append("")
    lines.extend(["## Specific answer observations", "", "| Model | Condition | Question | Separate qualitative observation |", "|---|---|---|---|"])
    for record in assessed:
        if (record["profile"], record["mode"], record["question_id"]) in SPECIFIC or record["status"] == "not_evaluable":
            lines.append(f"| {record['profile']} | {record['mode']} | {record['question_id']} | {record['note']} |")
    lines.extend(["", "All completed justifications, exact excerpts and record fingerprints are retained in [qualitative-review.json](qualitative-review.json). Official option concordance and endpoint/cap failures are in [results.md](results.md). These findings do not establish global medical rankings, local inference feasibility or clinical safety."])
    (COLLECTION / "qualitative-review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Bound qualitative observations to {len(assessed)} original records; objective keys unchanged; no inference calls.")


if __name__ == "__main__":
    main()
