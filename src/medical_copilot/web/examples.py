"""Editable examples and explicit fictional setup; this module performs no inference."""
import json

EXAMPLES = [
    {"id": "literature", "flow": "original", "featured": True, "title": "Explore the evidence",
     "description": "Find medical papers and inspect the references.",
     "question": "Find the PubMed record for the 2021 ESPEN practical guideline on clinical nutrition in cancer. What does it cover? Cite the retrieved source."},
    {"id": "pdf", "flow": "original", "featured": True, "title": "Ask a document",
     "description": "Read a specific recommendation and its source page.",
     "question": "According to the attached ESPEN 2021 PDF, what intake thresholds trigger medical nutrition? Retrieve and cite the relevant passage, including its PDF page."},
    {"id": "memory", "flow": "original", "featured": True, "title": "Continue a patient conversation",
     "description": "Load a concise case with dated records and an owned PDF.",
     "question": "What is my most recent recorded weight? Give the observation date and say if any information is missing."},
    {"id": "continuity-evidence", "flow": "continuity", "featured": False, "title": "Find the PROTEOS study",
     "description": "Start without a PDF. Find the trial on PubMed and check what it measured.",
     "question": "Find the PubMed record for the 2023 PROTEOS trial of high-protein oral nutritional supplements during systemic anti-cancer treatment, by Dingemans and colleagues. What was the primary outcome and one important study limitation? Cite a retrieved reference."},
    {"id": "continuity-case", "flow": "continuity", "featured": False, "title": "Read the case and the paper",
     "description": "Load Alpha's one-page record and the PROTEOS PDF, ready to question.",
     "question": "From my initial case record, what food-intake problems and preferred snack were recorded? From Table 3 of the attached PROTEOS paper, report the T1 mean protein intake and number of participants with data in each group. Keep my case facts separate from the study results and cite the PDF pages."},
    {"id": "continuity-followup", "flow": "continuity", "featured": False, "title": "Continue in a new conversation",
     "description": "Add a dated follow-up and reopen Alpha in a fresh conversation.",
     "question": "What changed since my previous visit? Give both dated weights and the change in kg, recall my preferred snack, and briefly describe what we discussed in an earlier conversation. Use my saved context and case PDFs; if no earlier discussion exists, say so. Distinguish patient observations from study results and cite the case passages."},
]


def prepare_clinical_patients(store, root, model_id):
    fixtures = json.loads((root / "demo/cases/patients.json").read_text(encoding="utf-8"))
    people = []
    for profile in fixtures["patients"]:
        key = "clinical-" + profile["code"].lower()
        patient = next((p for p in store.patients() if p["demo_key"] == key), None)
        if patient is None:
            patient = store.create_patient(profile["display_name"], demo=True, demo_key=key)
            for fact_key, value, unit in [("weight_kg", profile["weight_kg"], "kg"), ("snack", profile["snack"], None)]:
                store.add_fact(patient["patient_id"], {"fact_key": fact_key, "value": value, "unit": unit,
                    "observed_at": profile["observation_date"] + "T12:00:00+00:00",
                    "source_ref": "Fictional clinical demo 2026-10-16: explicitly entered initial " + profile["code"] + " record.",
                    "origin": "user_provided"})
        owned = next((s for s in store.sessions() if s["patient_id"] == patient["patient_id"]), None)
        if owned is None:
            owned = store.create_session(patient["patient_id"], title=profile["display_name"] + " consultation", model_id=model_id)
        store.register_pdf(owned["id"], root / "demo/cases" / profile["filename"], profile["title"])
        people.append(patient)
    return fixtures, people


def prepare_example(store, root, example_id, model_id):
    example = next((item for item in EXAMPLES if item["id"] == example_id), None)
    if example is None:
        raise ValueError("This example is unavailable.")
    session = None
    notice = "Editable question ready. Send it when you are ready."
    if example_id in {"memory", "continuity-case", "continuity-followup"}:
        fixtures, people = prepare_clinical_patients(store, root, model_id)
        patient = people[0]
        # A new thread demonstrates continuity rather than resetting the patient.
        session = store.create_session(patient["patient_id"], title=example["title"], model_id=model_id)
        if example_id in {"continuity-case", "continuity-followup"}:
            store.register_pdf(session["id"], root / "demo/02_proteos_protein_2023.pdf", "PROTEOS - Protein intake trial (2023)")
        if example_id == "continuity-followup":
            followup = fixtures["followup"]
            store.register_pdf(session["id"], root / "demo/cases" / followup["filename"], followup["title"])
            source_ref = "Fictional clinical demo 2026-10-16: explicitly added Alpha follow-up on 2026-10-13."
            with store.memory() as memory:
                already_added = any(fact["fact_key"] == "weight_kg" and fact["source_ref"] == source_ref
                                    for fact in memory.fact_history(patient["patient_id"]))
            if not already_added:
                store.add_fact(patient["patient_id"], {"fact_key": "weight_kg", "value": followup["weight_kg"], "unit": "kg",
                    "observed_at": followup["observation_date"] + "T12:00:00+00:00", "source_ref": source_ref, "origin": "user_provided"})
            notice = "Alpha follow-up available. New conversation opened; earlier files, dated records and public history remain available."
        else:
            notice = "Alpha's case is ready. Existing records and corrections are preserved."
    if session is None:
        session = store.create_session(model_id=model_id)
        if example_id == "pdf":
            store.register_pdf(session["id"], root / "demo/01_espen_cancer_2021.pdf", "ESPEN practical guideline: Clinical Nutrition in cancer (2021)")
            notice = "ESPEN PDF attached to a new general conversation."
    return {"session_id": session["id"], "question": example["question"], "notice": notice,
            "setup_only": True, "model_inference_requests": 0}
