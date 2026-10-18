"""Generate a self-contained English report using preserved real observations."""
from pathlib import Path
from medical_copilot.evaluation import read_json, file_hash
from build_requirement2_report import esc

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evaluation/retrieval-runs"
FINAL = BASE / "gemini-pubmed-final-20261010"
INITIAL = BASE / "gemini-pubmed-internal-20261010"


def paragraph(text):
    return esc(text) + "\n\n"


def main():
    final, initial = read_json(FINAL / "engineering-metrics.json"), read_json(INITIAL / "engineering-metrics.json")
    assessment = read_json(FINAL / "assessments.json")
    summary, first_summary = read_json(FINAL / "retrieval-summary.json"), read_json(INITIAL / "retrieval-summary.json")
    manifest = read_json(FINAL / "manifest.json")
    catalogue = read_json(ROOT / "data/research/requirement-3-source-catalogue.json")
    sources = catalogue["technical_sources"] + catalogue["papers"]
    parts = [r"""% Generated from preserved NCBI/provider observations and separate reviews.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm,headheight=15pt]{geometry}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}\setsansfont{Latin Modern Sans}\setmonofont{Latin Modern Mono}
\usepackage{booktabs,longtable,tabularx,array,enumitem,xcolor,fancyhdr,xurl,hyperref,tikz,float}
\usetikzlibrary{arrows.meta,positioning}
\definecolor{accent}{HTML}{183D55}
\hypersetup{colorlinks=true,linkcolor=accent,urlcolor=accent,citecolor=accent,pdftitle={Medical Agentic Copilot: Requirement 3 PubMed Integration and Retrieval Evaluation},pdfauthor={Medical Agentic Copilot Project},pdflang={en-US}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small\sffamily Medical Agentic Copilot}\fancyhead[R]{\small\sffamily Requirement 3}
\fancyfoot[L]{\small Implementation and observations: 10 October 2026}\fancyfoot[R]{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt plus 1pt}\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.14}
\setlist[itemize]{leftmargin=1.5em,itemsep=2pt}\setlist[enumerate]{leftmargin=1.7em,itemsep=3pt}
\begin{document}
\begin{center}{\LARGE\sffamily\bfseries Medical Agentic Copilot}\par\vspace{8pt}
{\large Requirement 3: PubMed Integration and Retrieval Evaluation}\par
{\small LangChain tools connected to LangGraph; separate paper-relevance findings}\par\vspace{5pt}10 October 2026\end{center}
\begin{abstract}
The requirement-2 agent now searches real PubMed records through NCBI ESearch and EFetch, supplies available abstracts to generation, and appends verified metadata for cited sources. Two ten-question runs preserve queries, rankings, raw responses, code snapshots, answers and separate assistant-authored relevance judgments. The final run returned 49 question--paper pairs: 13 directly useful, 17 partially related and 19 irrelevant. All ten questions had related material, but seven had partial coverage. Broader querying did not establish overall quality improvement. A material source-attribution failure remained undetected by a targeted revised-model review. The implementation is a research proof of concept; retrieval, relevance, source validity, faithful attribution and clinical correctness are distinct.
\end{abstract}
\tableofcontents\clearpage
\section{Scope and delivered behavior}
"""]
    parts.append(paragraph("Requirement 3 asks for MCP integrations and/or tools on PubMed and/or medRxiv, connection to the agent, reuse of the original internal questions, and a qualitative retrieval evaluation distinct from answer quality. This implementation uses a LangChain StructuredTool and PubMed. A dedicated MCP server or medRxiv connector is not claimed. Requirements 1 and 2 remain documented; requirements 4-7 remain pending."))
    parts.append(paragraph("The source adapter uses official E-utilities, with request pacing and batched retrieval. The PubMed User Guide explains Boolean field-tag searching and indexing/filter limitations."))
    parts.append(r"Technical basis: \cite{R3T01,R3T02,R3T03}." + "\n")
    parts.append(paragraph("The original guidance/calculation graph remains available with --evidence local. The CLI ask command now defaults to PubMed plus curated guidance. Direct search-pubmed needs no model or NCBI API key; optional NCBI_EMAIL identifies a contact without recording it in public request metadata. Model keys are not sent to NCBI."))
    parts.append(r"\section{Connected workflow and implementation}" + "\n")
    parts.append(r"""\begin{figure}[H]\centering
\begin{tikzpicture}[>=Stealth,box/.style={draw=accent,rounded corners,fill=accent!4,minimum width=22mm,minimum height=10mm,align=center,font=\small\sffamily}]
\node[box] (plan) at (0,0) {Plan};\node[box] (tools) at (3,0) {Validate\\and retrieve};
\node[box] (draft) at (6,0) {Draft};\node[box] (review) at (9,0) {Source\\review};\node[box] (end) at (12,0) {Finalize};
\node[box] (revise) at (9,-1.6) {Extra tools\\and revise};
\draw[->] (plan)--(tools);\draw[->] (tools)--(draft);\draw[->] (draft)--(review);\draw[->] (review)--(end);
\draw[->] ([xshift=4mm]review.south west)--([xshift=4mm]revise.north west);
\draw[->] ([xshift=-4mm]revise.north east)--([xshift=-4mm]review.south east);
\end{tikzpicture}
\caption{The bounded graph uses model-selected tools and at most one revision by default. A question permits two PubMed searches; retrieval/attribution limitations remain reviewable.}
\end{figure}
""")
    descriptions = [
        ("pubmed.py", "Validated short query; fixed HTTPS endpoints; URL-encoded POST; ESearch JSON followed by batch EFetch XML; preserved ESearch order; safe failures and response hashes."),
        ("literature.py", "Adds search_pubmed to the original tool registry. In-scope plans require one query. Reviews may request one more query or local guidance. Sources use P plus PMID and curated S IDs."),
        ("retrieval_evaluation.py", "Passes only question text, saves independent retrieval records and answers, preserves raw HTTP/code snapshots, and validates separately authored hash-bound relevance reviews."),
        ("cli.py", "ask --evidence pubmed/local, search-pubmed, evaluate-retrieval, and summarize-retrieval. Original evaluate remains the requirement-2 local workflow."),
    ]
    parts.append(r"\begin{longtable}{p{44mm}p{113mm}}\toprule File & Responsibility\\\midrule" + "\n")
    for name, description in descriptions:
        parts.append(esc(name) + " & " + esc(description) + r"\\" + "\n")
    parts.append(r"\bottomrule\end{longtable}" + "\n")
    parts.append(paragraph("Parameters bound queries to 600 characters and 1-8 results, normally five ordered by relevance. Each request has a 20-second timeout, a 4 MB response limit, a minimum 0.4-second interval and at most two transient retries with waits bounded to five seconds. These are per-process controls, not a guarantee about unrelated activity from the shared IP. There is no persistent cache or fabricated literature fallback."))
    parts.append(paragraph("The parser preserves titles, authors/collective names, publication date/year, DOI/PMCID, publication type, language, correction/retraction links, abstract presence/truncation and publisher copyright notices. It does not fetch full text. Abstracts are capped at 4,500 characters; empty abstracts and unparseable journal records are disclosed. An external DTD is not fetched and entity declarations are rejected."))
    parts.append(paragraph("Indexed retracted publications/notices are excluded by the query and checked again in returned metadata. Preprints and expressions of concern are labeled where supplied; no new source in these runs was indexed as a preprint. PubMed inclusion or absence of a flag does not establish quality. NCBI does not grant rights to publisher full text. The report uses original assessment rationales and bibliographic metadata rather than full articles."))
    parts.append(paragraph("Tool text is evidence, not instruction; prompts forbid following commands in titles/abstracts or including identifiers/secrets in queries. Endpoints and operations are fixed. These prompt safeguards are not a proven resistance to adversarial content. No real patient record was used. Fresh per-question state is separate from the audit logs; requirement-7 patient memory is not implemented."))
    parts.append(r"\section{Protocol, rubric and assessment limits}" + "\n")
    parts.append(paragraph("All ten original oncology-nutrition questions were reused without modification. Dataset, rubric, guidance and code hashes are in each manifest. Expected answers, reference IDs, critical failures and the retrieval rubric do not enter generation. The retrieval rubric was written before the first cohort. It does not assign an answer-correctness score."))
    parts.append(r"\begin{tabularx}{\linewidth}{p{22mm}X}\toprule Rating & Paper relevance\\\midrule 2 & Directly useful for a central population, intervention or decision in the question.\\ 1 & Related background or a peripheral component, with a population/intervention/detail mismatch.\\ 0 & No meaningful usefulness for the question in the supplied title/abstract material.\\\bottomrule\end{tabularx}" + "\n")
    parts.append(paragraph("Question coverage is separately rated 2 for substantial representation of central concepts, 1 for partial representation with important omissions, and 0 for no meaningful coverage. Direct relevance does not establish study quality, currentness, correct attribution, or a patient-specific prescription. Missing abstract is not automatically irrelevance; its reduced read level is stated."))
    parts.append(paragraph("The implementation assistant read the saved titles/available abstracts and authored one rationale for every question-paper pair. Ratings are unblinded, lack independent clinician adjudication and do not assume full-text review. The same constructed set informed development. No exhaustive relevant-paper set exists, so recall, clinical accuracy and causal improvement are not estimated. Identical ON06/ON09 abstracts were verified before reusing initial rationales."))
    parts.append(r"\section{Preserved runs and observed results}" + "\n")
    rows = [
        ("Questions", 10, 10), ("Model calls", initial["model_call_count"], final["model_call_count"]),
        ("Reported tokens", initial["reported_usage"]["total_tokens"], final["reported_usage"]["total_tokens"]),
        ("PubMed searches", 10, 10), ("HTTP requests / attempts", "19 / 19", "20 / 21"),
        ("Questions returning papers", 9, 10), ("Question-paper pairs", 27, 49),
        ("Distinct PMIDs in run", 27, 48), ("Direct / partial / irrelevant", "14 / 11 / 2", "13 / 17 / 19"),
        ("Substantial / partial / no coverage", "5 / 3 / 2", "3 / 7 / 0"),
        ("Self-review pass / review flag", "10 / 0", "9 / 1"),
    ]
    parts.append(r"\begin{longtable}{p{83mm}p{35mm}p{35mm}}\toprule Observation & Initial v1 & Revised v2\\\midrule" + "\n")
    for name, a, b in rows:
        parts.append(esc(name) + " & " + esc(a) + " & " + esc(b) + r"\\" + "\n")
    parts.append(r"\bottomrule\end{longtable}" + "\n")
    parts.append(paragraph("Both cohorts used Gemini 3.5 Flash-Lite with the unchanged provider profile: no temperature override, output limit 4,096 tokens, 60-second model timeout, SDK retries disabled, minimum four-second interval and bounded rate-limit retries. This is not a different-LLM comparison. Token counts are reported by the provider; free-tier eligibility, quota remaining, costs or billing were not inferred."))
    parts.append(paragraph("The final model stages were ten plans, ten drafts, twelve reviews, two revisions and three plan-format repairs. The final summed question time was 121.549 seconds and median 11.340 seconds, excluding the configured inter-question delay. One EFetch request returned HTTP 502 and then succeeded. No question failed with a provider/API error. No live reviewer requested an additional search; the refinement branch and two-search budget were demonstrated in offline tests."))
    parts.append(paragraph("Version 1's restrictive queries produced an empty fasting/KD result and a PCOS supplement mismatch. Version 2 requested grouped alternative concepts and source-specific attribution checks. It returned related material for every question and a directly useful source for eight questions, but also many more irrelevant pairs. ON05's ungrouped OR treatment expanded the query to over fourteen million matches. Larger result counts did not establish better precision, completeness or clinical quality."))
    parts.append(r"\section{Faithful attribution: an unresolved limitation}" + "\n")
    parts.append(paragraph("Initial ON07 retrieved a 2026 noninferiority trial in HSCT/acute-leukemia patients. Its abstract reported that the liberalized arm crossed the predefined major-infection stopping boundary and did not improve caloric intake. The answer framed the trial as supporting the nutritional/quality-of-life burden of rigid restrictions, switched the arm attributed to the calorie result, and omitted the contradictory infection finding."))
    parts.append(r"The observed source is \cite{P41406403}; this is not a universal clinical diet recommendation." + "\n")
    parts.append(paragraph("Version 2 requires per-cited-paper reported-result/support checks, and a negative or missing check triggers revision. In a targeted post hoc diagnostic, the revised reviewer received the identical original draft and actual saved evidence, with one new Gemini call and zero new PubMed requests. It correctly summarized the adverse liberalized-arm finding but still accepted the faulty original attribution. This remains an unresolved false negative. The final live ON07 search did not retrieve that trial, so its changed answer cannot prove correction of the conflict."))
    parts.append(paragraph("Final ON01 retained three missing source-specific check flags after revision and is labeled needs_human_review. The nine other self_review_passed results are internal operational statuses. Valid PMID, relevant paper, source-specific review and model self-check do not establish medical truth or faithful attribution. Separate answer-observation records preserve this failure; original outputs are never rewritten."))
    parts.append(r"\section{Reproduction and engineering verification}" + "\n")
    parts.append(paragraph("Run the following from the project root after the requirement-2 environment setup. PubMed adds no new dependency. Existing model credentials remain in environment variables/ignored files. Use a fresh run directory. Author assessments separately against the frozen rubric before summarization."))
    parts.append(r"""\begin{minipage}{\linewidth}{\small\begin{verbatim}
.\.venv\Scripts\python.exe -X utf8 -m medical_copilot `
  search-pubmed "cancer nutrition screening"
.\.venv\Scripts\python.exe -X utf8 -m medical_copilot `
  ask "Explain nutrition risk during cancer treatment."
.\.venv\Scripts\python.exe -X utf8 -m medical_copilot `
  ask "Explain cancer nutrition." --evidence local
.\.venv\Scripts\python.exe -X utf8 -m medical_copilot `
  evaluate-retrieval `
  --run-dir data/evaluation/retrieval-runs/new-run
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -X utf8 scripts/replay_retrieval_runs.py
.\.venv\Scripts\python.exe -X utf8 scripts/verify_requirement3.py
\end{verbatim}}\end{minipage}""")
    parts.append(paragraph("The 58 passing offline tests include the original 32 plus PubMed parsing, ordering, missing abstracts, truncation, retractions, safe failures, transient retries, source-controlled bibliography, tool-to-draft connection, independent question state, interrupted evidence retention, mandatory search, revision triggers and the two-search ceiling. Fictional test fixtures are never clinical run artifacts. pip check found no broken requirements."))
    parts.append(paragraph("Twenty recorded retrieval cases replay with each archived prompt version, with identical substantive model inputs, final answers and flags and no network calls. Thirteen previous requirement-2/NVIDIA cases also replay unchanged. These are engineering checks, not additional live medical evaluation. Current code matches the final snapshot; the only initial-to-final package change was literature.py. The CLI change relative to requirement 2 is documented without changing its original questions or outputs."))
    parts.append(paragraph("Run folders include manifest.json, source-snapshot, answers, retrieval, http raw JSON/XML, http-requests.json, engineering-metrics.json, assessments.json and retrieval-summary.json. Request snapshots are named by SHA-256; assessment records bind to retrieval-file hashes. The public source catalogue, source journal and implementation guide provide locators and limitations. No credentials are included in reports or logs."))
    parts.append(r"\section{Limits and next work}" + "\n")
    parts.append(paragraph("Query formulation and Boolean grouping remain fallible. PubMed ranking can include pediatric, animal, preventive, surgical or unrelated records and omit important conflicting evidence. The journal-article-only parser excludes some record types. Abstracts may be absent, trimmed or old and do not expose all methods, subgroup details, biases or adverse events. Metadata retraction checks are incomplete safeguards. Attribution review remains fallible even when the correct abstract is supplied."))
    parts.append(paragraph("The next assignment item is requirement 4: a distinct benchmark-subset evaluation chosen from the requirement-1 inventory. Cross-model comparison, PDF ingestion and patient-specific memory remain later requirements. Future retrieval work could add librarian-reviewed concept formulation, evidence/population filters, explicit disagreement handling, full-text appraisal where permitted, and independent clinical review. None is claimed here. The present work demonstrates connected real search and separately documented relevance, with known weaknesses."))
    parts.append(r"\appendix\clearpage\section{All final questions, queries and paper-relevance rationales}" + "\n")
    rows_by_id = {row["question_id"]: row for row in summary["questions"]}
    for item in assessment["assessments"]:
        qid = item["question_id"]
        record = read_json(FINAL / "retrieval" / (qid + ".json"))
        papers = {source["pmid"]: source for source in record["unique_papers"]}
        parts.append(r"\subsection{" + esc(qid) + "}\n")
        parts.append(paragraph(record["question"]))
        for search in record["searches"]:
            parts.append(r"\textbf{Observed query:} " + esc(search["arguments"]["query"]) + "\n\n")
            if search["status"] == "ok":
                parts.append(paragraph("Total indexed matches: " + str(search["output"]["total_matches"]) + "; returned journal records: " + str(len(search["output"]["sources"])) + ". Exclusions: " + str(search["output"]["excluded_records"])))
        counts = rows_by_id[qid]["pooled_relevance_counts"]
        parts.append(paragraph(f"Direct / partial / irrelevant pairs: {counts['2']} / {counts['1']} / {counts['0']}. Coverage: {item['coverage_rating']}/2. {item['coverage_rationale']}"))
        parts.append(paragraph("Search findings: " + item["search_findings"]))
        parts.append(r"\begin{enumerate}" + "\n")
        for paper_review in item["paper_assessments"]:
            source = papers[paper_review["pmid"]]
            parts.append(r"\item \textbf{Relevance " + str(paper_review["relevance"]) + r"/2.} " + esc(source["title"]) + r" \cite{" + source["id"] + "}.\n\n")
            parts.append(paragraph("Read level: " + source["read_level"] + ("; abstract truncated by tool." if source["abstract_truncated"] else ".") + " " + paper_review["rationale"]))
        parts.append(r"\end{enumerate}" + "\n")
    parts.append(r"\section{Artifact fingerprints and source record}" + "\n")
    parts.append(paragraph("The initial run and its separate 27-pair review remain in gemini-pubmed-internal-20261010. The final 49-pair review is in gemini-pubmed-final-20261010. The report does not reproduce all generated clinical answers; those original drafts, answers, source metadata, reviews and prompts are preserved in the JSON/Markdown artifacts. The bibliography includes all 67 observed paper records across both cohorts and preflight, including irrelevant matches, plus three technical sources; inclusion is not endorsement."))
    for key in ["dataset_sha256", "rubric_sha256", "guidance_sha256"]:
        parts.append(r"\textbf{" + esc(key) + r"}\par{\small\ttfamily " + esc(manifest[key]) + r"}\par" + "\n")
    parts.append(r"\begin{thebibliography}{P99999999}\small" + "\n")
    for source in sources:
        authors = source["authors"]
        if len(source.get("author_list") or []) > 3:
            authors = ", ".join(source["author_list"][:3]) + ", and colleagues"
        parts.append(r"\bibitem[" + source["id"] + "]{" + source["id"] + "} " + esc(authors) + " (" + esc(source["year"]) + "). " + esc(source["title"]) + ". ")
        if source.get("pmid"):
            parts.append("PMID " + esc(source["pmid"]) + ". ")
        if source.get("doi"):
            parts.append("DOI " + esc(source["doi"]) + ". ")
        parts.append(r"\url{" + source["url"] + "}. " + esc(source["supports"]) + "\n")
    parts.append(r"\end{thebibliography}\end{document}" + "\n")
    target = ROOT / "docs/implementation/requirement-3-report.tex"
    target.write_text("\n".join(parts), encoding="utf-8")
    print(f"Generated report source with {len(sources)} bibliography items and all 49 final paper rationales.")


if __name__ == "__main__":
    main()
