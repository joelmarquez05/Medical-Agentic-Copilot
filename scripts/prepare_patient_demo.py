"""Freeze synthetic patient inputs and expected checks before hosted generation."""
import json
from pathlib import Path

import pymupdf

from medical_copilot.clinical_graph import save_trace
from medical_copilot.patient_memory import file_hash

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "data/evaluation/patient-demo-20261014"


def main():
    if SET.exists():
        raise ValueError("Preserve the earlier fixture set; choose a new version.")
    SET.mkdir(parents=True)
    fixture = {"patients": [{"patient_id": "alpha", "display_name": "Synthetic Alpha", "weight_kg": 64.2, "cartons": 2, "marker": "ALPHA-NUTRITION-47"},
                             {"patient_id": "beta", "display_name": "Synthetic Beta", "weight_kg": 81.7, "cartons": 1, "marker": "BETA-NUTRITION-93"}],
               "alpha_snack": "toasted sesame crackers", "filler_turns": 5,
               "new_alpha_weight": 62.4, "corrected_alpha_weight": 62.1, "late_old_alpha_weight": 70.3,
               "source": "Entirely authored synthetic records; not patients, MedQA or an LLM-assigned reference."}
    save_trace(SET / "fixtures.json", fixture)
    for patient in fixture["patients"]:
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((40, 70), "SYNTHETIC PATIENT CASE FILE", fontsize=15)
            page.insert_text((40, 110), "Patient: " + patient["patient_id"] + "\nRecord marker: " + patient["marker"] +
                             "\nRecorded supplement cartons per day: " + str(patient["cartons"]) +
                             "\nThis is a fictional record for software testing, not a prescribing recommendation.", fontsize=11)
            pdf.save(SET / (patient["patient_id"] + ".pdf"))
    definitions = [
        ("PM01", "alpha", "control", False, False, "What is my registered weight? Return the answer as a JSON object with the single field weight_kg. Use null if it is not recorded in the supplied patient context.", {"weight_kg": None}),
        ("PM02", "alpha", "alpha-session-1", True, False, "What is my registered weight? Return the answer as a JSON object with the single field weight_kg. Use null if it is not recorded in the supplied patient context.", {"weight_kg": 64.2}),
        ("PM03", "beta", "beta-session-1", True, False, "What is my registered weight? Return a JSON object with weight_kg. Use only my own patient record.", {"weight_kg": 81.7}),
        ("PM04", "alpha", "alpha-session-2", True, False, "What preferred snack did I state in an earlier conversation? Recover my older history if needed. Return a JSON object with snack containing my exact original wording, or null if unavailable.", {"snack": "toasted sesame crackers"}),
        ("PM05", "alpha", "alpha-session-2", True, True, "According to my registered synthetic case PDF, how many supplement cartons per day were recorded? Read its actual text and cite the exact source view. Return a JSON object with cartons_per_day.", {"cartons_per_day": 2}),
        ("PM06", "beta", "beta-session-2", True, True, "According to my registered synthetic case PDF, how many supplement cartons per day were recorded? Read its actual text and cite the exact source view. Return a JSON object with cartons_per_day.", {"cartons_per_day": 1}),
        ("PM07", "alpha", "alpha-session-3", True, False, "What is my latest observed weight, based on observation date rather than the last entry time? Return a JSON object with weight_kg.", {"weight_kg": 62.4}),
        ("PM08", "alpha", "alpha-session-4", True, False, "Use my dated fact records to check the correction to my most recent weight observation. Return a JSON object with current_weight_kg and superseded_weight_kg. Inspect the factual history if the earlier value is not in the current profile.", {"current_weight_kg": 62.1, "superseded_weight_kg": 62.4}),
        ("PM09", "alpha", "alpha-session-4", True, False, "What is my measured serum albumin in g/dL? Return a JSON object with albumin_g_dl. Give null if no result is recorded; do not substitute a normal reference range.", {"albumin_g_dl": None}),
        ("PM10", "alpha", "alpha-session-5", True, True, "Find the PubMed record for the ESPEN practical guideline Clinical Nutrition in cancer. Return a JSON object with pmid as a string and year as an integer, and cite an actual PubMed source returned during this episode.", {"pmid": "33946039", "year": 2021}),
    ]
    questions = [{"patient_id": p, "conversation_id": c, "memory_enabled": m, "require_reference": r,
                  "question": {"question_id": qid, "question": text, "options": None}} for qid, p, c, m, r, text, _ in definitions]
    keys = [{"question_id": qid, "expected_fields": expected, "key_type": "synthetic-script-state" if qid != "PM10" else "official-fixed-article-metadata",
             "required_tool_family": "pdf" if qid in {"PM05", "PM06"} else "pubmed" if qid == "PM10" else "history" if qid == "PM04" else "facts" if qid == "PM08" else None} for qid, _, _, _, _, _, expected in definitions]
    save_trace(SET / "questions.json", {"questions": questions})
    save_trace(SET / "answer-key.json", {"keys": keys, "provenance": "Synthetic data/updates fixed by script before inference; PM10 uses ESPEN 2021 metadata, DOI 10.1016/j.clnu.2021.02.005, PMID 33946039. No LLM correct-answer assignment."})
    save_trace(SET / "manifest.json", {"requirement": 7, "version": "synthetic-patient-demo-v1", "question_ids": [row[0] for row in definitions],
               "source_hashes": {path.name: file_hash(path) for path in SET.iterdir() if path.is_file()}, "frozen_before_hosted_generation": True,
               "scope": "Ten scripted functional episodes including one memory-off control; not a new medical benchmark."})
    print("Frozen ten synthetic memory episodes, twelve field checks and two private fixture PDFs.")


if __name__ == "__main__":
    main()
