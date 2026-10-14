"""Save the post-generation clinical-reference catalogue with actual read limits."""
from pathlib import Path
from medical_copilot.clinical_graph import save_trace
from medical_copilot.evaluation import read_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    checked = read_json(ROOT / "data/research/requirement-4-review-pubmed-20261011/records.json")["papers"]
    spleen = next(p for p in checked if p["pmid"] == "28828034")
    pfk = read_json(ROOT / "data/research/requirement-4-review-pubmed-20261011/pfk-record.json")["papers"][0]
    sources = []
    for identifier, paper, support, level in [
        ("R4-CL01", spleen, "BM01 hemodynamic-instability/operative priority; not a confirmed splenic injury in this case.", "Official publisher operative-management sections plus original PubMed metadata/abstract; PMC access challenged."),
        ("R4-CL03", pfk, "BM09 indirect hepatic PKA/PFK-2 regulation and tissue-specific contrast; not a claim that RBC PFK-1 is directly phosphorylated.", "Original PubMed EFetch full abstract; indexed full-text regulation excerpt. Direct PMC/institutional PDF reads failed; no full-text read claimed."),
    ]:
        sources.append({"id": identifier, "title": paper["title"], "authors_or_organization": paper["authors"],
            "year": paper["year"], "doi": paper["doi"], "pmid": paper["pmid"], "url": paper["url"],
            "supports": support, "read_level": level, "preprint": paper["preprint"]})
    pages = [
        ("R4-CL02", "SIMVASTATIN tablet, film coated: prescribing information", "DOH CENTRAL PHARMACY / DailyMed, U.S. National Library of Medicine; repackaged label", 2017,
         "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=7e3fc83f-e322-44a1-b0c3-30e6292b92d9", "BM05 stable hepatic/transaminase warning; archived label, not a complete current prescription recommendation.", "Official DailyMed hepatic-dysfunction section; label updated 2017-02-03."),
        ("R4-CL04", "Clinical Testing and Diagnosis for Hepatitis B", "Centers for Disease Control and Prevention", 2026,
         "https://www.cdc.gov/hepatitis-b/hcp/diagnosis-testing/index.html", "BM11 total anti-HBc versus IgM anti-HBc, surface antibody and multianalyte testing distinctions.", "Official testing/serologic-interpretation sections; page dated 2026-08-14."),
        ("R4-CL05", "Clinical Guidance for Typhoid Fever and Paratyphoid Fever", "Centers for Disease Control and Prevention", 2024,
         "https://www.cdc.gov/typhoid-fever/hcp/clinical-guidance/index.html", "BM13 clinical-feature overlap and culture diagnosis; rose spots/hepatosplenomegaly are not sufficient definitive identification.", "Official clinical features/diagnosis sections; page dated 2024-04-24."),
        ("R4-CL06", "EAU Guidelines on Paediatric Urology: Management of Undescended Testes", "European Association of Urology, Paediatric Urology Guidelines Panel", "Living guideline accessed 2026",
         "https://uroweb.org/guidelines/paediatric-urology/chapter/management-of-undescended-testes", "BM14 early surgical orchiopexy, ideally before 12 months/by 18 months latest, and palpable/nonpalpable distinction.", "Official management and recommendation sections read directly."),
        ("R4-CL07", "ACR Appropriateness Criteria: Vomiting in Infants", "American College of Radiology", 2020,
         "https://acsearch.acr.org/docs/69445/Narrative/", "BM20 initial radiography for poor feeding/no meconium and obstruction workup.", "Indexed official narrative/initial imaging excerpt; revised 2020. Direct narrative/PDF access returned errors, so no full-text read claimed."),
        ("R4-CL08", "ALLOPURINOL tablets: prescribing information", "Unichem Pharmaceuticals (USA), Inc. / DailyMed, U.S. National Library of Medicine", 2024,
         "https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?setid=e3face45-f920-46a0-b707-6a66aa9735bd&type=pdf", "BM21 xanthine-oxidase inhibition, marrow suppression and mercaptopurine/azathioprine interaction; no proof of this patient's unnamed concomitant drug.", "Official PDF indication, myelosuppression and drug-interaction sections; revision 2024-02."),
        ("R4-CL09", "COLCRYS (colchicine): prescribing information", "Takeda Pharmaceuticals America, Inc.; Aphena Pharma Solutions - Tennessee, LLC repackaged label / DailyMed", 2019,
         "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=a80036fe-3016-4541-aa42-6c06cf37ae55", "BM21 colchicine blood dyscrasias/pancytopenia can occur at therapeutic doses, contradicting an overly broad dismissal.", "Official section 5.2 Blood Dyscrasias; label updated 2019-12-19. Archived warning reference, not a full current dosing guide."),
        ("R4-CL10", "LEUKINE (sargramostim): prescribing information", "Partner Therapeutics, Inc. / DailyMed, U.S. National Library of Medicine", 2024,
         "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=bb9a3820-f31a-4dd9-bef4-97094c01a398", "BM23 first-administration respiratory/hemodynamic infusion reaction; does not uniquely prove every symptom/drug in this vignette.", "Official section 5.1 infusion-related reactions and adverse-reaction context; label updated 2024-12-06."),
        ("R4-CL11", "LAMOTRIGINE tablets: prescribing information", "Sun Pharmaceutical Industries, Inc. / DailyMed, U.S. National Library of Medicine", 2026,
         "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=b829f9b2-f0ac-408a-a6f4-18b72150227d", "BM29 lamotrigine serious rash/SJS/TEN association; not independent confirmation of the described rash phenotype.", "Official boxed warning/serious-rash sections; label updated 2026-06-03."),
        ("R4-CL12", "REGLAN (metoclopramide) tablets: prescribing information", "U.S. Food and Drug Administration; prescribing label", 2017,
         "https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/017854s062lbl.pdf", "BM15 D2 antagonism is independently supported despite the agent's wrong associated citation; BM28 metoclopramide is a possible NMS trigger, not a proven administered drug or evidence for dantrolene efficacy.", "Official archived PDF description (D2 antagonist) and NMS warning/adverse-event sections, revised 2017-08. No current dosing/prescribing claim."),
    ]
    for identifier, title, author, year, url, support, level in pages:
        sources.append({"id": identifier, "title": title, "authors_or_organization": author, "year": year,
            "url": url, "supports": support, "read_level": level, "preprint": False})
    for source in sources:
        source.update(accessed="2026-10-11", role="Post-generation researcher appraisal context; not agent evidence or an alternative scoring key.")
    save_trace(ROOT / "data/research/requirement-4-clinical-review-sources-20261011.json", {"sources": sorted(sources, key=lambda s: s["id"]),
        "count": len(sources), "key_changes": 0,
        "limits": "Focused support checks for flagged statements; not independent full clinical validation of all 30 cases. Older labels and inaccessible full texts are explicit.",
        "excluded_lookup": "data/research/requirement-4-review-pubmed-20261011/lookup-resolution.json"})


if __name__ == "__main__":
    main()
