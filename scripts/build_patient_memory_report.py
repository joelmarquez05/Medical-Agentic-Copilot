"""Score frozen synthetic memory episodes and audit saved requests without APIs."""
import json
import math
import os
from pathlib import Path
import re
import sqlite3

from medical_copilot.evaluation import read_json, write_json
from medical_copilot.patient_memory import file_hash
from medical_copilot.patient_runtime import prior_usage

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/evaluation/patient-runs"
MAIN = RUNS / "gemma-patient-main-20261014"
CONTROL = RUNS / "gemma-patient-control-check-20261014"
FIXTURES = ROOT / "data/evaluation/patient-demo-20261014"
NOTES = {
    "PM01": "Original memory-off control has a provider connection failure, not a model answer. No accuracy or abstention credit; no interaction saved.",
    "PM02": "Initial alpha value matches the explicitly supplied dated fact. Case recall does not require a publication citation.",
    "PM03": "Beta value matches its own fact and a real read_patient_facts result. No alpha value or file was delivered.",
    "PM04": "Search and exact interaction reading recover the user's old snack statement, excluded from the recent-three context. The seed acknowledgment is an application fixture, not a generated medical answer.",
    "PM05": "Own alpha PDF passage explicitly records two cartons. Cited page/chunk view was actually delivered. This fictional count is not a prescribing recommendation.",
    "PM06": "Own beta PDF passage explicitly records one carton. Its delivered source and citation are different from alpha's.",
    "PM07": "Uses the latest observation, not a later-entered older observation. Matching the scripted chronological state is functional validation, not clinical verification.",
    "PM08": "Current and superseded values match the explicit correction chain; the old observation remains available and is correctly identified as superseded.",
    "PM09": "Returns null rather than inventing an albumin value. The explanation reports searches of facts, one case PDF and history; lexical zero-hit results do not exhaustively establish absence from all possible documents.",
    "PM10": "One actual PubMed search supplies the matching title, PMID and year. Citation supports these bibliographic claims; no separate abstract-reading call or full-text clinical recommendation is claimed. Four other returned records are not needed for this exact-title task.",
}


def equal_field(actual, expected):
    if expected is None:
        return actual is None
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(actual, expected, abs_tol=1e-9, rel_tol=0)
    return type(actual) is type(expected) and actual == expected


def score(answer, key):
    eligible = answer["status"] == "completed" and answer["output_origin"] == "model"
    try:
        parsed = json.loads(answer["response"]["answer"]) if eligible else None
    except (ValueError, TypeError):
        parsed = None
    parsed = parsed if isinstance(parsed, dict) else None
    fields = []
    for name, expected in key["expected_fields"].items():
        present = parsed is not None and name in parsed
        actual = parsed[name] if present else None
        fields.append({"field": name, "expected": expected, "actual": actual, "present": present,
                       "matched": eligible and present and equal_field(actual, expected)})
    tools = [h["action"]["name"] for h in answer["history"] if h.get("role") == "model" and h.get("action", {}).get("type") == "tool"]
    family_tools = {"history": {"search_patient_history", "read_patient_interaction"}, "facts": {"read_patient_facts"},
                    "pdf": {"search_pdf", "read_pdf_page"}, "pubmed": {"search_pubmed", "read_pubmed_abstract"}}
    family = key["required_tool_family"]
    return {"question_id": key["question_id"], "eligible_model_answer": eligible, "status": answer["status"],
            "fields": fields, "matched_fields": sum(f["matched"] for f in fields), "total_fields": len(fields),
            "all_fields_match": all(f["matched"] for f in fields), "tool_actions": tools,
            "required_tool_family": family, "required_tool_observed": family is None or bool(set(tools) & family_tools[family]),
            "citations": answer["response"]["citations"], "review": NOTES.get(key["question_id"], "Separate engineering control: actual model returns null with patient records withheld.")}


def audit_requests(directory, answer, episode):
    patient = episode["patient_id"]
    foreign = "beta" if patient == "alpha" else "alpha"
    snapshot = directory / episode["memory_snapshot"]
    with sqlite3.connect(f"file:{snapshot.resolve().as_posix()}?mode=ro", uri=True) as connection:
        owned_docs = {r[0] for r in connection.execute("SELECT document_id FROM documents WHERE patient_id=?", (patient,))}
        foreign_docs = {r[0] for r in connection.execute("SELECT document_id FROM documents WHERE patient_id=?", (foreign,))} - owned_docs
    private_foreign = [f"{foreign.upper()}-NUTRITION-", f"Synthetic {foreign.title()}", f"F-{foreign}-", f"I-{foreign}-", *sorted(foreign_docs)]
    if foreign == "alpha":
        private_foreign.append("toasted sesame crackers")
    for call in answer["model_calls"]:
        request = json.loads(call["user_prompt"])
        context = request["patient_context"]
        assert context["patient_id"] == patient and context["conversation_id"] == episode["conversation_id"]
        assert not any(token in call["user_prompt"] for token in private_foreign), "Foreign private sentinel in request"
        assert "C:\\Users\\" not in call["user_prompt"] and "C:\\\\Users\\\\" not in call["user_prompt"]
        assert set(d["document_id"] for d in request["document_catalog"]) <= owned_docs
        for tool in request["tools"]:
            assert "patient_id" not in tool["arguments_schema"].get("properties", {})
        for source_id in request["evidence_registry"]:
            if source_id.startswith("D"):
                assert source_id.split("_", 1)[0] in owned_docs
        if not episode["memory_enabled"]:
            assert context["current_facts"] == {} and context["files"] == [] and context["recent_interactions"] == []
            assert not request["document_catalog"]
            assert not any(t["name"] in {"search_patient_history", "read_patient_interaction", "read_patient_facts", "search_pdf", "read_pdf_page"} for t in request["tools"])
    assert all(s in answer["evidence"] for s in answer["response"]["citations"])
    return {"question_id": answer["question"]["question_id"], "requests_checked": len(answer["model_calls"]),
            "patient_and_conversation_match": True, "foreign_private_sentinels_absent": True,
            "pdf_ids_owned": True, "model_cannot_select_patient_in_tool_arguments": True,
            "memory_off_withholds_records_and_private_tools": not episode["memory_enabled"],
            "answer_sha256": file_hash(directory / "answers" / (answer["question"]["question_id"] + ".json"))}


def link(path):
    return Path(os.path.relpath(path, MAIN)).as_posix()


def main():
    manifest = read_json(MAIN / "manifest.json")
    keys = {k["question_id"]: k for k in read_json(FIXTURES / "answer-key.json")["keys"]}
    selection = read_json(FIXTURES / "manifest.json")
    for name, digest in selection["source_hashes"].items():
        assert file_hash(FIXTURES / name) == digest
        if (MAIN / name).exists():
            assert file_hash(MAIN / name) == digest
    rows, answers = [], []
    for subrun in manifest["subruns"]:
        directory = ROOT / subrun["run_dir"]
        answer_file = directory / "answers" / (subrun["question_id"] + ".json")
        assert file_hash(answer_file) == subrun["answer_sha256"]
        answer = read_json(answer_file)
        answers.append(answer)
        row = score(answer, keys[subrun["question_id"]])
        row.update(patient_id=answer["patient_id"], memory_enabled=answer["memory_enabled"], answer_file=link(answer_file),
                   answer_sha256=file_hash(answer_file), model_requests=len(answer["model_calls"]))
        rows.append(row)
    totals = {"planned_episodes": 10, "terminal_messages": len(answers), "valid_model_answers": sum(r["eligible_model_answer"] for r in rows),
              "matched_fields": sum(r["matched_fields"] for r in rows), "total_fields": sum(r["total_fields"] for r in rows),
              "memory_on_episodes": sum(r["memory_enabled"] for r in rows),
              "memory_on_all_fields_matched": sum(r["memory_enabled"] and r["all_fields_match"] for r in rows),
              "memory_on_matched_fields": sum(r["matched_fields"] for r in rows if r["memory_enabled"]),
              "memory_on_total_fields": sum(r["total_fields"] for r in rows if r["memory_enabled"]),
              "required_tool_cases": sum(r["required_tool_family"] is not None for r in rows),
              "required_tool_cases_passed": sum(r["required_tool_family"] is not None and r["required_tool_observed"] for r in rows)}
    assert totals["valid_model_answers"] == 9 and totals["matched_fields"] == 11 and totals["total_fields"] == 12
    write_json(MAIN / "scores.json", {"method": "Frozen scripted state / official article metadata; null requires an eligible model answer and an explicitly present field. No LLM correctness judge.", "totals": totals, "cases": rows})
    write_json(MAIN / "manual-reviews.json", {"method": "Unblinded assistant review of actual public responses, supplied facts/history and delivered sources. No independent clinical reviewer or paid judge.",
        "cases": [{"question_id": r["question_id"], "answer_file": r["answer_file"], "answer_sha256": r["answer_sha256"],
                   "note": r["review"], "cited_view_assessments": [{"source_id": s, "support": "supports_requested_claim",
                       "view_sha256": __import__("hashlib").sha256(json.dumps(a["evidence"][s], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                       "support_basis": "Explicit own PDF count" if s.startswith("D") else "Matching delivered title, PMID and year metadata"} for s in r["citations"]]}
                  for r, a in zip(rows, answers)]})
    audits, all_costs, stages = [], [], []
    for directory in sorted(RUNS.iterdir()):
        if not (directory / "usage-audit.json").exists():
            continue
        m = read_json(directory / "manifest.json")
        if m.get("requirement") != 7:
            continue
        cost_rows = read_json(directory / "usage-audit.json")["records"]
        all_costs.extend(cost_rows)
        stages.append({"run_id": directory.name, "requests": len(cost_rows), "known_cost_usd": round(sum(r["cost_usd"] or 0 for r in cost_rows), 8),
                       "unknown_charge_records": sum(r["cost_usd"] is None for r in cost_rows)})
        for episode in m["episodes"]:
            answer = read_json(directory / "answers" / (episode["question_id"] + ".json"))
            audits.append(audit_requests(directory, answer, episode))
    storage = read_json(MAIN / "storage-checks.json")
    assert all(storage.values()) and sum(a["requests_checked"] for a in audits) == len(all_costs) == 27
    write_json(ROOT / "data/checks/requirement-7-request-isolation-20261014.json", {"passed": True, "episodes": audits,
        "requests_checked": 27, "storage_checks": storage, "new_api_requests": 0,
        "interpretation": "Saved input/sentinel/ownership audit and negative storage checks; not a security proof against every adversarial prompt."})
    control = read_json(CONTROL / "answers/CONTROL-OFF.json")
    control_score = score(control, {"question_id": "CONTROL-OFF", "expected_fields": {"weight_kg": None}, "required_tool_family": None})
    assert control_score["all_fields_match"] and not control["memory_write"]["saved"]
    write_json(CONTROL / "engineering-assessment.json", {"separate_from_primary_denominator": True, "score": control_score,
        "runtime_fix": "patient-runtime-v1.1 recognizes the observed OpenAIConnectionError alias for one reserved final retry. Offline transport-failure regression tests the retry; this live control succeeded without a failure/retry.",
        "original_PM01_preserved": True})
    prior = prior_usage(ROOT)
    ledger = read_json(CONTROL / "final-ledger.json")
    known = round(sum(r["cost_usd"] or 0 for r in all_costs), 8)
    unknown = round(sum(r["estimated_maximum_usd"] for r in all_costs if r["cost_usd"] is None), 8)
    conservative = max(ledger["total_usage"], prior["account_anchor_usd"] + prior["known_cost_usd"]) + prior["unknown_reserve_usd"] + .15
    costs = {"requirement": 7, "new_requests": len(all_costs), "known_cost_usd": known,
             "unknown_charge_records": sum(r["cost_usd"] is None for r in all_costs), "new_unknown_charge_reserve_usd": unknown,
             "stages": stages, "combined_prior_ledger": prior, "latest_saved_account_observation": ledger,
             "historical_uncertainty_headroom_usd": .15, "cumulative_limit_usd": 6,
             "conservative_usage_plus_reserves_usd": round(conservative, 9), "below_cumulative_limit": conservative < 6,
             "new_requests_for_this_audit": 0, "interpretation": "Known request charges and account-wide usage are different quantities. The failed request has no reported price; its reservation is not an observed charge."}
    write_json(ROOT / "data/checks/requirement-7-cost-audit-20261014.json", costs)
    papers = answers[-1]["literature"]["events"][0]["papers"]
    write_json(ROOT / "data/research/requirement-7-observed-papers-20261014.json", {"scope": "Actual PM10 PubMed records, retrieved 14 October 2026; only PMID 33946039 cited. Model received truncated generation views, not all raw metadata/abstracts.",
        "papers": [{k: p.get(k) for k in ["title", "authors", "author_list", "year", "doi", "pmid", "url"]} | {
                   "support": "PM10 exact bibliographic identification" if p["pmid"] == "33946039" else "Returned retrieval record; not needed for the exact-title question"} for p in papers]})
    lines = ["# Requirement 7: persistent patient-memory demonstration", "", "14 October 2026. The patient-memory MVP is implemented and demonstrated with two fictional patients, registered PDF files, older public history and the existing PubMed-enabled LangGraph agent. **Nine of nine memory-enabled episodes match every requested field (11/11 fields).** Across the full ten-episode protocol, there are **9/10 model answers, 11/12 matching fields and 10/10 terminal messages**. The original memory-off control failed during provider connection and receives no answer/abstention credit.", "",
        "The implementation adapts the identity/read-use-save pattern from LangChain's MIT `memory-agent`; it is a separate SQLite application store, not a full fork, native LangGraph BaseStore or checkpoint-resume system. [Architecture and usage](" + link(ROOT / "docs/implementation/requirement-7-patient-memory.md") + "), [research/read-scope catalogue](" + link(ROOT / "data/research/requirement-7-memory-sources-20261014.json") + "), [pinned reference/license](" + link(ROOT / "data/research/requirement-7-memory-20261014/reference-memory-agent/reference-manifest.json") + ").", "",
        "## Fixed protocol", "", "The [questions](questions.json), [keys](answer-key.json), [fixture state](fixtures.json) and [selection hashes](selection-manifest.json) were frozen before inference. Patient identities are application inputs, never tool parameters controlled by the model. Each episode runs in a new Python process with a new graph/client, using the same on-disk patient database. The public interaction is saved after each memory-enabled episode. Initial private action history, counters and evidence registry are empty. Older user history is seeded explicitly by the fixture script; it is not an invented previous LLM conversation.", "",
        "All ten planned episodes were retained. Fields come from authored synthetic state; PM10 comes from the official ESPEN 2021 article metadata (Muscaritoli, Arends, Bachmann and colleagues; DOI 10.1016/j.clnu.2021.02.005, [PMID 33946039](https://pubmed.ncbi.nlm.nih.gov/33946039/)). Keys were never included in model inputs. This is a functional memory demonstration, separate from the earlier medical benchmarks. Same Gemma 4 26B A4B IT / OpenRouter NextBit configured pin, enabled reasoning, temperature 0, 16,384 output-token cap and 180-second timeout. Actual serving route is not independently exposed in responses. Research allows three calls/actions; reserved finalization allows two attempts, at most five model requests. No hidden SDK or JSON-repair retries.", "",
        "## All primary episodes", "", "| Case | Scope | Requested fields / observed result | Calls / tools | Outcome |", "|---|---|---|---|---|"]
    for row in rows:
        vals = ", ".join(f"{f['field']}={json.dumps(f['actual'], ensure_ascii=False)}" for f in row["fields"]) if row["eligible_model_answer"] else "Provider connection failure; no model answer"
        lines.append(f"| [{row['question_id']}]({row['answer_file']}) | {row['patient_id']}; {'memory on' if row['memory_enabled'] else 'memory off'} | {vals} | {row['model_requests']}; {', '.join(row['tool_actions']) or 'none'} | {row['matched_fields']}/{row['total_fields']} fields |")
    lines += ["", "[Mechanical scores](scores.json) enforce actual model output, explicit field presence, type/numeric agreement and frozen keys. An application nonanswer or missing field cannot receive null credit. [Separate unblinded reviews](manual-reviews.json) assess actual explanations and supplied sources; no clinician or paid LLM judge assigned correctness.", "",
        "Seven memory-tool calls, three PDF searches and one PubMed search were executed in the primary cohort. All five cases with a specified tool family used it. PM04 actually recovered history beyond the recent-three window; PM07 selected observation time over late entry time; PM08 identified both the corrected value and the retained superseded observation. PM09 abstained from inventing a measurement, although its zero-hit lexical searches are not exhaustive absence evidence. The three structured citations (PM05, PM06, PM10) all identify actually delivered views supporting the requested counts or bibliographic fields. PM10's five returned records include four unnecessary records for the exact-title task; retrieval/citation findings remain distinct from field accuracy. [Returned-paper metadata](" + link(ROOT / "data/research/requirement-7-observed-papers-20261014.json") + ").", "",
        "## Failure and separate engineering control", "", "PM01 preserves the original `OpenAIConnectionError`, one failed request, unknown-charge reservation and transparent application nonanswer. Its generation runtime did not recognize that observed SDK alias as recoverable. After collection, `patient-runtime-v1.1` added it to the transient classifier. Offline regression checks a real transport-layer fake failure followed by one reserved native final request, with the unknown-charge reserve retained. The old cohort is not rerun or retroactively reclassified; its original runtime snapshot and failure routing remain replayable.", "",
        "An [additional memory-off engineering control](" + link(CONTROL / "engineering-assessment.json") + ") returned `weight_kg: null` with no remembered records, private tools or persisted interaction. It succeeded in one live request, so it does not demonstrate recovery from a live transient failure. This check is outside the primary denominator. It is also not a controlled estimate of a general memory advantage: collection times differ and the primary control had no model answer.", "",
        "## Persistence, isolation and reproducibility", "", "All [four negative storage checks](storage-checks.json) passed: foreign conversation, document and interaction access were rejected, and beta's history search did not recover alpha's snack. The [saved-request audit](" + link(ROOT / "data/checks/requirement-7-request-isolation-20261014.json") + ") checks all 27 requests from 14 episodes: correct patient/conversation, no foreign private sentinels, owned PDF IDs, no model-controlled patient tool argument and withheld records/tools in both memory-off controls. This is an observed audit plus code tests, not a universal prompt-injection/security guarantee. No real patient data was used.", "",
        "The full offline suite passes **262 tests**, including **32 new memory/runtime cases**. Source restoration from older citations, tampered-file rejection, context overflow, persistence failure and bounded invalid citation handling are tested offline; not all occurred in live generation. [Test log](" + link(ROOT / "data/checks/requirement-7-pytest-20261014.txt") + "). Exact offline prompt/schema/action/tool/source/memory replay passes for all **14 episodes / 27 saved responses / 2 saved PubMed HTTP requests**, including the original failed control. It makes no inference or network requests. Probe, pilot, main and additional-control receipts are under `data/checks/requirement-7-*-replay-20261014.json`. The generation-only runtime change is explicitly recorded; other agent/store/tool sources and earlier operational source/profile/question artifacts remain frozen.", "",
        "## Charges and limits", "", f"Primary collection: **21 requests / USD {sum(r['cost_usd'] or 0 for a in answers for r in a['response_audit']):.8f} known**, plus one unknown-charge reservation. Probe, two pilots, primary collection and additional control together: **27 requests / USD {known:.8f} known**, plus **USD {unknown:.8f} reserved** for the unpriced failure. The reservation is not a claimed charge. [Exact all-stage audit](" + link(ROOT / "data/checks/requirement-7-cost-audit-20261014.json") + f") uses the saved account observation **USD {ledger['total_usage']:.9f} at {ledger['checked_at_utc']}**, historical reserves and USD 0.15 headroom; it remains below the shared USD 6 guard. Documentation/replay adds zero API requests.", "",
        "The MVP has a trusted local operator, explicit profile updates, lexical history search over the newest 1,000 interactions, an 8 KiB initial context and the previous selectable-text PDF limits. Historical assistant statements are not promoted into verified facts. It has no embeddings, background consolidation, automatic clinical extraction, graph interruption recovery or multi-user authentication. Results establish this scripted two-patient workflow, not general medical reliability. Requirement 7 is covered as a working proof of concept.", ""]
    (MAIN / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"totals": totals, "requests_audited": len(all_costs), "known_cost_usd": known, "conservative_usage_usd": costs["conservative_usage_plus_reserves_usd"]}))


if __name__ == "__main__":
    main()
