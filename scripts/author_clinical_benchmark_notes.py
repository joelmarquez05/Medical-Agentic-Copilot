"""Materialize explicit post-generation assistant notes for the fixed Gemma cohort.

These are authored observations, not an automatic medical judge or provider call.
"""
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/benchmark-runs/gemma-full-agent-medqa30-v3-20261011"

NOTES = {
 "BM01": ("consistent", "consistent", "consistent", "Correct D. Identifies worsening shock, failed fluid stabilization and supplied hemoperitoneum; prioritizes operative control over CT. No invented injury grade or organ diagnosis. WSES guidance supports the instability/operative priority; direct answer explicitly lacks external verification.", ["R4-CL01"]),
 "BM02": ("consistent", "consistent", "consistent", "Correct C. Uses S3, crackles and edema to identify the heart-failure pattern, and explicitly notes that low potassium is less certain without medications. Search returned zero; no source is invented. Its limitation handles physiologic uncertainty but does not explicitly acknowledge absent PubMed verification.", []),
 "BM03": ("consistent", "consistent", "consistent", "Correct D. Post-exercise facilitation, proximal weakness and poor anticholinesterase response support the examination's Lambert-Eaton/SCLC association. Does not assert a proven lung tumor. The very restrictive query returns zero and this is acknowledged, rather than presented as no literature exists.", []),
 "BM04": ("consistent", "consistent", "minor_limitation", "Correct B. Two actual excerpts support fetal meiosis and diplotene/prophase-I arrest. The answer's later hormonal-resumption detail is background knowledge not established by these short views; no clinical study on this infant is implied. Other returned papers are not all directly useful.", []),
 "BM05": ("consistent", "minor_limitation", "consistent", "Correct B. Recognizes simvastatin transaminase elevation and that fenofibrate can also contribute. The archived label corroborates the stable warning. Only-one wording should not imply that the other listed drugs can never cause liver injury; causation in this patient is not proven. Query conjunctively names four drugs and returns zero; absent verification is acknowledged.", ["R4-CL02"]),
 "BM06": ("consistent", "consistent", "consistent", "Correct D. Connects chronic severe renal impairment/uremic features with broad waxy casts and distinguishes the supplied alternatives. Explicit absent-verification limitation. This answers the examination's sediment question, not a complete acute management plan.", []),
 "BM07": ("consistent", "consistent", "consistent", "Correct D. Burn-related stress ulceration and mucosal hypoperfusion coherently support the choice. Actual returned records include two title-only sources and related ulcer/burn physiology; no exact mechanism is falsely credited to a read. The answer explains its evidence gap rather than citing irrelevant material.", []),
 "BM08": ("consistent", "consistent", "consistent", "Correct B. Severe obesity coherently links the osteoarthritis-like knee pattern and sleep-apnea-like symptoms. The answer describes suggested conditions, not an obtained sleep-study result. It acknowledges no PubMed verification.", []),
 "BM09": ("consistent", "consistent", "consistent", "Correct C. Identifies likely pyruvate-kinase deficiency/glycolysis and correctly distinguishes PFK-1 allosteric regulation from the indirect hepatic PKA/PFK-2 route. The independent researcher reference verifies hepatic isoenzyme regulation; it was not agent evidence. No claim that RBC PFK-1 is directly phosphorylated by PKA.", ["R4-CL03"]),
 "BM10": ("consistent", "consistent", "minor_limitation", "Correct C. Symptoms/serology support the expected celiac villous atrophy and crypt hyperplasia. Diagnostic wording is stronger than an independently confirmed biopsy, which is exactly what the question asks to predict. No histology result is invented and no source verification is claimed.", []),
 "BM11": ("consistent", "minor_limitation", "minor_limitation", "Correct C among the restricted options. Correctly distinguishes core antibody exposure from surface-antibody immunity, but does not specify IgM anti-HBc versus total anti-HBc or the antigen/antibody panel needed to characterize active infection. CDC confirms these distinctions; no key change. No explicit absent-external-verification acknowledgment.", ["R4-CL04"]),
 "BM12": ("consistent", "consistent", "consistent", "Correct D. Uses cytopenias, splenomegaly and dry tap to predict hairy-cell/TRAP findings; does not pretend an actual positive stain was obtained. Restrictive query returns zero, with an explicit evidence limitation.", []),
 "BM13": ("consistent", "minor_limitation", "minor_limitation", "Correct C. Travel, evolving fever/GI symptoms and rose spots fit the examination diagnosis. Calling hepatosplenomegaly/rose spots pathognomonic overstates specificity; CDC notes overlap and culture-based diagnosis. No external verification was claimed.", ["R4-CL05"]),
 "BM14": ("consistent", "consistent", "consistent", "Correct D for a palpable unilateral undescended testis at 11 months. EAU guidance supports early orchiopexy, ideally before 12 months and at latest 18; the model's 6-12-month target is reasonable. No drug dose or fabricated source. This is knowledge-based guideline attribution, explicitly unverified by the agent.", ["R4-CL06"]),
 "BM15": ("consistent", "minor_limitation", "material_issue", "Correct D but incomplete citation support. The area-postrema source supports a central emetic role, not every blood-brain-barrier/circulating-toxin claim; its visible text even questions circulating substances in dog experiments. The second cited excerpt concerns ondansetron/5-HT3, not metoclopramide/D2. FDA confirms the D2 fact independently, without repairing that unsupported citation. Etoposide-versus-cisplatin scope is acknowledged.", ["R4-CL12"]),
 "BM16": ("consistent", "consistent", "minor_limitation", "Correct A. Increased dyspnea/purulent sputum coherently support antibiotic consideration and the released doxycycline choice. Search supplied five records, so the limitation claiming no evidence registry was provided is false. Excerpts do not verify its exact drug recommendation or current full GOLD criteria. It cites no paper and makes no dose claim.", []),
 "BM17": ("consistent", "consistent", "minor_limitation", "Correct A. Sudden intraoperative rigidity, hyperthermia and ETCO2 85 support malignant hyperthermia over the medication distractors. The specific anesthetic is correctly acknowledged as unknown, rather than invented. The answer omits explicit acknowledgment of no external verification; it is not a complete resuscitation protocol.", []),
 "BM18": ("consistent", "consistent", "consistent", "Correct B. Groin/perineal distribution, scrotal sparing and wrestling context support tinea cruris among the choices. No obtained fungal test or reference is invented; absent verification is explicit.", []),
 "BM19": ("consistent", "consistent", "minor_limitation", "Correct B. The actually read 2002 review explicitly links carbamazepine to neural-tube defects, supporting the meningocele association. It does not quantify comparative absolute risk for this patient. Only that association is credited; the old review's broader pregnancy-treatment recommendations are not validated or adopted.", []),
 "BM20": ("consistent", "consistent", "minor_limitation", "Correct A. Distinguishes initial obstruction imaging from a later cause-specific workup; ACR's indexed primary guidance supports initial radiography for poor feeding/no meconium. Retrieved excerpts describe neonatal differentials but not this imaging sequence. Must-precede wording is broader than all possible clinical pathways; no retrieved trial is falsely cited.", ["R4-CL07"]),
 "BM21": ("minor_limitation", "material_issue", "minor_limitation", "Correct D but the explanation weakly addresses the IBD/polypharmacy clue and dismisses colchicine marrow toxicity too broadly. Its official label explicitly reports pancytopenia at therapeutic doses. Allopurinol labeling documents myelosuppression and increased thiopurine exposure; a possible IBD-related azathioprine/mercaptopurine interaction is relevant but not a proven exposure in this stem. DRESS is not established by the supplied facts. Preserve correct key alongside this material explanation issue.", ["R4-CL08", "R4-CL09"]),
 "BM22": ("consistent", "consistent", "consistent", "Correct A. Identifies DPP-4 inhibition and incretin preservation and distinguishes the alternative mechanisms. No dose, patient recommendation or source verification is invented.", []),
 "BM23": ("consistent", "consistent", "consistent", "Correct C. The independent LEUKINE label supports first-administration infusion reactions with respiratory/hemodynamic features; not proof that every symptom in this vignette uniquely diagnoses the drug. Three near-equivalent restrictive queries all return zero. The reserved fourth call produces an honest knowledge-based answer, demonstrating bounded finalization without fabricated references.", ["R4-CL10"]),
 "BM24": ("consistent", "minor_limitation", "consistent", "Correct A. Recognizes current primary-hypothyroid features and appropriately leaves thyroiditis subtype uncertain. The supplied review excerpt supports general thyroiditis/initial hormone release, but stops before the later hypothyroid phase or TSH finding. Calling the two described states triphasic is loose; complete course is not observed.", []),
 "BM25": ("consistent", "consistent", "consistent", "Correct D. Male infant, age after maternal IgG wanes and recurrent encapsulated bacterial infections support X-linked agammaglobulinemia; the distractor distinctions are coherent. Empty search is acknowledged, with no fabricated laboratory results or references.", []),
 "BM26": ("consistent", "minor_limitation", "material_issue", "Correct C with correct 1900*0.60=1140 ascitic neutrophil calculation. Two papers actually returned directly support bacterial translocation, but the model cites raw P-prefixed PMIDs rather than the exact registered view IDs. Host flags both unknown citations. Separate unique-PMID diagnostics confirm real papers/content without repairing the primary response. PMN threshold alone does not exclude secondary peritonitis; confirms is stronger than the limited stem/evidence.", []),
 "BM27": ("consistent", "consistent", "consistent", "Correct B. Actual delivered clinical-study excerpt explicitly supports the long-face/large-ear phenotype and expanded FMR1 CGG repeat association. Historical terminology in the source is not adopted as a modern clinical recommendation.", []),
 "BM28": ("consistent", "minor_limitation", "minor_limitation", "Correct B. Plausible NMS after a dopamine-antagonist antiemetic; exact drug/exposure is explicitly unknown. A returned case excerpt supports syndrome/antiemetic association, but neither successful query nor failed follow-up supplies dantrolene outcome evidence. FDA supports metoclopramide as a possible NMS trigger, not proof of this patient's drug. Calling dantrolene the specific NMS treatment is stronger than the evidence obtained.", ["R4-CL12"]),
 "BM29": ("consistent", "minor_limitation", "consistent", "Correct A. Actual lamotrigine/SJS psychiatric-case excerpt and independent boxed warning support the drug association. A single case does not establish comparative psychiatric-population risk; mucosal involvement or skin detachment is not supplied, so the vignette's exact rash syndrome is not proven. The model appropriately notes absent direct confirmation.", ["R4-CL11"]),
 "BM30": ("consistent", "consistent", "consistent", "Correct C. Sympathetic activation coherently explains increased heart rate/systolic pressure during exercise. The baroreflex contrast is simplified and does not discuss exercise-related resetting, but the selected mechanism is consistent. Knowledge-based explanation explicitly lacks external verification.", []),
}

# Explicit title/delivered-text grades, independently separated from option score.
PAPERS = {
 "37784186": ("directly_relevant", "Human fetal meiosis and diplotene/prophase-I arrest are explicit in the delivered excerpt."),
 "25652637": ("partially_relevant", "Human germ-cell development context; excerpt stops before female meiotic stage."),
 "29544632": ("directly_relevant", "Primordial-follicle excerpt explicitly states an oocyte arrested at diplotene."),
 "30611263": ("directly_relevant", "Oocyte meiotic-arrest regulation and fetal meiosis directly address the developmental topic."),
 "24806359": ("partially_relevant", "Age/cohesin/aneuploidy topic rather than newborn arrest stage."),
 "4567321": ("not_assessable", "Generic Burns title with no abstract cannot establish its stress-ulcer mechanism coverage."),
 "27048251": ("irrelevant", "Functional dyspepsia differs from acute burn-related stress bleeding."),
 "4053922": ("partially_relevant", "General acute ulcer defense/ischemia context; not burn-specific."),
 "497665": ("directly_relevant", "Title explicitly addresses gastric/duodenal ulceration after burns; no mechanism text supplied."),
 "8821097": ("partially_relevant", "Burn hypovolemia/fluid physiology, not a delivered ulcer-mechanism account."),
 "1485177": ("directly_relevant", "Area-postrema/chemotherapy emesis mechanisms directly address the target organ; scope is not all circulating-toxin claims."),
 "25295673": ("partially_relevant", "Cisplatin/Bacopa animal emesis work, not etoposide or metoclopramide mechanism."),
 "29615907": ("partially_relevant", "Cisplatin/cannabis pigeon emesis work; indirect relevance only."),
 "1838490": ("partially_relevant", "Ondansetron/5-HT3 emesis article, not evidence for metoclopramide/D2."),
 "31768887": ("partially_relevant", "Gingerol/cisplatin animal studies rather than this clinical drug/structure question."),
 "34353802": ("partially_relevant", "Antibiotic initiation protocol mentions GOLD/Anthonisen but does not supply doxycycline selection evidence."),
 "40898355": ("irrelevant", "Chronic eosinophil/ICS reduction and prophylactic azithromycin protocol, not this acute antibiotic decision."),
 "41844237": ("directly_relevant", "AECOPD pharmacological-treatment review directly concerns the clinical management topic; excerpt lacks exact drug recommendation."),
 "15293621": ("partially_relevant", "Broader atypical respiratory-infection topic; no direct COPD drug choice in the view."),
 "35321460": ("partially_relevant", "COPD hospital-care audit/GOLD comparison rather than this specific antibiotic decision."),
 "8419718": ("partially_relevant", "General epilepsy/AED treatment context; delivered excerpt lacks pregnancy/NTD data."),
 "8400940": ("partially_relevant", "Epilepsy/pregnancy title only, no carbamazepine outcome text."),
 "19330496": ("partially_relevant", "Antibipolar teratogenesis review methods; no risk finding appears in the excerpt."),
 "235422": ("partially_relevant", "Old AED-teratogenicity title only, without the needed drug association."),
 "11734102": ("directly_relevant", "Actual read explicitly links carbamazepine/valproate to neural-tube defects; historical review, not current regimen advice."),
 "37828049": ("partially_relevant", "Hirschsprung/neonatal obstruction signs; excerpt lacks initial imaging sequence."),
 "20714730": ("partially_relevant", "Neonatal anorectal-obstruction differential/diagnostic delay, not the requested first imaging step."),
 "32152637": ("irrelevant", "Older-child chronic constipation differs from acute neonatal obstruction."),
 "10807228": ("partially_relevant", "Meconium obstruction in very-low-birth-weight infants; different population and no initial radiograph in excerpt."),
 "35313185": ("partially_relevant", "Syndromic Hirschsprung case, not the nondysmorphic newborn's initial imaging decision."),
 "35604323": ("irrelevant", "Graves orbitopathy does not address this hyper-to-hypothyroid clinical sequence."),
 "34913664": ("directly_relevant", "Thyroiditis phases address the diagnostic topic; truncated excerpt does not verify final TSH."),
 "28351085": ("partially_relevant", "Subclinical thyroid/metabolic-syndrome study, not this overt clinical sequence; supplied title unavailable."),
 "17080266": ("partially_relevant", "Pituitary TSH-adenoma differential, not evidence for the specific primary-thyroid course."),
 "38936554": ("directly_relevant", "Gut bacterial translocation/SBP mechanism is explicit in the first delivered sentence."),
 "19266595": ("directly_relevant", "SBP review explicitly identifies bacterial translocation as the key pathogenesis mechanism."),
 "30930689": ("partially_relevant", "General cirrhosis immune dysfunction; excerpt does not identify the decisive mechanism."),
 "8728196": ("irrelevant", "Peritoneal dialysis/liver-virus barriers differ from spontaneous infection in this nondialysis patient."),
 "31324695": ("partially_relevant", "Cirrhotic infection/endotoxemia renal context rather than the specific ascites mechanism."),
 "10874635": ("directly_relevant", "Actual excerpt explicitly supplies FMR1/CGG and long-face/protruding-ear features."),
 "22101771": ("directly_relevant", "NMS phenotype and antiemetic association are explicit; excerpt does not supply dantrolene treatment evidence."),
 "40078262": ("directly_relevant", "Lamotrigine/SJS psychiatric case directly supports the drug association; not comparative-risk evidence."),
}

CITATIONS = {
 "37784186": ("supported", "Delivered text explicitly states fetal meiosis initiation and diplotene/prophase-I arrest; that is the assessed central claim."),
 "29544632": ("supported", "Delivered text explicitly states primordial follicles contain a diplotene-arrested oocyte. Later hormonal-resumption detail is not independently verified by this view."),
 "1485177": ("partially_supported", "Supports area-postrema involvement in chemotherapy emesis, not the BBB/circulating-toxin assertion. The supplied dog-experiment text questions a circulating-substance role."),
 "1838490": ("unsupported", "The cited view discusses ondansetron and 5-HT3, not metoclopramide or D2-receptor blockade. The D2 fact is independently true but unsupported by this citation."),
 "11734102": ("supported", "Actual read explicitly states carbamazepine is associated with neural-tube defects. Does not quantify this patient's comparative risk or validate all historical treatment advice."),
 "34913664": ("partially_supported", "Excerpt supports thyroiditis and initial hormone release; stops before hypothyroid phase/elevated TSH, so complete asserted sequence exceeds the view."),
 "19266595": ("not_assessable", "Primary cited ID is not registered. Separate unique-PMID diagnostic maps to a real supplied view supporting bacterial translocation; no primary repair/reclassification."),
 "38936554": ("not_assessable", "Primary cited ID is not registered. Separate unique-PMID diagnostic maps to a real supplied view supporting gut translocation/SBP; not a fabricated paper and not a repaired primary citation."),
 "10874635": ("supported", "Actual excerpt directly supplies the cited long-face/large-protruding-ear and expanded CGG/FMR1 phenotype association."),
 "40078262": ("partially_supported", "Supports lamotrigine/SJS association in a psychiatric case, not population-level preferential risk or proof that this vignette's exact rash is SJS."),
}

# Manual query grades address wording/topic/decision specificity. An empty result
# is never, by itself, evidence that a query is inappropriate.
QUERIES = {
 "BM02": [("partially_appropriate", "Correct heart-failure topic, but combines BNP, ADH, sodium, potassium and abnormalities in one restrictive query; no reformulation.")],
 "BM03": [("partially_appropriate", "Correct Lambert-Eaton/SCLC topic, but combines the full symptom list including bladder incontinence; no broader query after the empty result.")],
 "BM04": [("appropriate", "Directly targets fetal female meiosis and arrest stage.")],
 "BM05": [("partially_appropriate", "Conjoins four separate drugs with a liver-adverse-effect topic, rather than checking a likely candidate or each alternative.")],
 "BM07": [("appropriate", "Directly targets the named burn-related ulcer and its pathophysiology; imperfect returned papers do not invalidate that topic choice.")],
 "BM12": [("partially_appropriate", "Combines nearly every vignette feature with TRAP and hairy-cell leukemia; no simpler disease/stain query follows.")],
 "BM15": [("partially_appropriate", "Combines etoposide, chemotherapy vomiting, area postrema and metoclopramide; narrower than a general mechanism check."),
          ("appropriate", "Broadens from etoposide to chemotherapy emesis/area postrema; relevance and drug-specific support remain separate.")],
 "BM16": [("appropriate", "Targets GOLD/AECOPD antibiotic treatment, the correct clinical decision topic.")],
 "BM19": [("appropriate", "Directly targets carbamazepine, pregnancy and neural-tube-defect risk.")],
 "BM20": [("partially_appropriate", "Neonatal delayed-meconium management is related, but omits the decisive initial-imaging/workup question.")],
 "BM23": [("partially_appropriate", "Correct drug/reaction topic, but conjunctive symptom terms narrow retrieval excessively."),
          ("partially_appropriate", "Near-equivalent symptom query; little substantive broadening or GM-CSF synonym use."),
          ("partially_appropriate", "Removes one symptom but retains the same narrow approach; no broader drug/infusion-reaction query.")],
 "BM24": [("appropriate", "Targets the hyper-to-hypothyroid clinical sequence directly.")],
 "BM25": [("partially_appropriate", "Correct X-linked agammaglobulinemia topic, but adds several literal phenotype terms rather than a simpler disorder/presentation query.")],
 "BM26": [("appropriate", "Directly targets SBP/cirrhosis bacterial-translocation mechanism.")],
 "BM27": [("appropriate", "Directly targets the requested fragile-X physical phenotype.")],
 "BM28": [("partially_appropriate", "Targets plausible NMS/MH differential but uses both acronyms and lacks a specific treatment term."),
          ("partially_appropriate", "Adds the MDMA differential while retaining a conjunctive acronym query; does not focus on a named intervention.")],
 "BM29": [("partially_appropriate", "Targets lamotrigine rash/bipolar disorder but uses a narrow lesion-description combination."),
          ("appropriate", "Reformulates toward the named SJS drug association; evidence strength of a case report is separately graded.")],
}


def main():
    notes = {"reviewer": "Unblinded implementation assistant; manual post-generation observations, no paid judge",
             "answers": {}, "queries": {}, "papers": {}, "citations": {}}
    for identifier, values in NOTES.items():
        row = read_json(RUN / "answers" / (identifier+".json"))
        notes["answers"][identifier] = {"ratings": dict(zip(["case_reasoning", "reference_concordance", "calibration_and_provenance"], values[:3])),
            "notes": values[3], "reference_sources": values[4]}
        views = {v["id"]: v for e in row["literature"]["events"] for v in e["generation_view"]["sources"]}
        papers = {p["pmid"]: p for e in row["literature"]["events"] for p in e["papers"]}
        for pmid in papers:
            grade, reason = PAPERS[pmid]
            notes["papers"][identifier+":"+pmid] = {"rating": grade, "notes": reason}
        queries = [e for e in row["literature"]["events"] if e["name"] == "search_pubmed"]
        authored_queries = QUERIES.get(identifier, [])
        assert len(authored_queries) == len(queries)
        notes["queries"][identifier] = [{"rating": grade, "notes": reason} for grade, reason in authored_queries]
        for sid in row["response"].get("citations", []):
            view = views.get(sid)
            if view is None:
                assert identifier == "BM26" and sid in {"P19266595", "P38936554"}
                candidates = [v for v in views.values() if v["pmid"] == sid[1:]]
                assert len(candidates) == 1
                pmid = candidates[0]["pmid"]
            else:
                pmid = view["pmid"]
            grade, reason = CITATIONS[pmid]
            notes["citations"][identifier+":"+sid] = {"rating": grade, "notes": reason}
    notes["revision_note"] = "V2 replaces the initial draft's result-count-derived query grades with explicit wording/clinical-target grades. Keys, raw answers, paper relevance and citation grades are unchanged. Empty retrieval alone does not establish an inappropriate query."
    save_trace(RUN / "review-notes-v2.json", notes)


if __name__ == "__main__":
    main()
