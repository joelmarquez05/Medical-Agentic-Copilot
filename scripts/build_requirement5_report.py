"""Build a detailed English LaTeX report from the actual frozen comparison."""
from __future__ import annotations

from pathlib import Path

from medical_copilot.evaluation import DIMENSIONS, file_hash, read_json
from build_requirement2_report import esc

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/comparison-runs/selected-five-comparison-20261010"
PRIMARY = ROOT / "data/evaluation/comparison-runs/selected-five-main-20261010"
RECOVERY = ROOT / "data/evaluation/comparison-runs/selected-five-recovery-20261010"
PILOT = ROOT / "data/evaluation/comparison-runs/selected-five-pilot-20261010"
OUT = ROOT / "docs/implementation/requirement-5-report.tex"
NAMES = {"openrouter-claude-opus-5.5": "Claude Opus 5.5", "openrouter-glm-5.3": "GLM-5.3", "openrouter-qwen3.8-27b": "Qwen3.8 27B", "gemini-3.8-flash": "Gemini 3.8 Flash", "gemma-4-31b": "Gemma 4 31B"}


def paragraph(text):
    return esc(text) + "\n\n"


def number(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def main():
    manifest = read_json(RUN / "manifest.json")
    engineering = read_json(RUN / "engineering-summary.json")
    primary_engineering = read_json(PRIMARY / "engineering-summary.json")
    recovery_engineering = read_json(RECOVERY / "engineering-summary.json")
    quality = read_json(RUN / "qualitative-summary.json")
    assessments = read_json(RUN / "assessments.json")
    findings = read_json(RUN / "comparative-findings.json")
    evidence_manifest = read_json(RUN / "evidence/manifest.json")
    technical = read_json(ROOT / "data/research/requirement-5-source-catalogue.json")["sources"]
    rubric = read_json(RUN / "qualitative-rubric.json")
    question_set = read_json(RUN / "question-set.json")
    complete = sum(model["completed_answers"] for model in engineering["models"])
    attempted = sum(model["attempted_cases"] for model in primary_engineering["models"])
    skipped = sum(model["unattempted_provider_unavailable_cases"] for model in primary_engineering["models"])
    primary_complete = sum(model["completed_answers"] for model in primary_engineering["models"])
    main_cost = sum(model["reported_or_estimated_cost_usd"] or 0 for model in engineering["models"])
    sources = {source["id"]: source for source in technical}
    for item in evidence_manifest["questions"]:
        evidence = read_json(RUN / "evidence" / (item["question_id"] + ".json"))
        for source in evidence["sources"]:
            sources.setdefault(source["id"], source)
    parts = [r"""% Generated from actual saved model responses and separate qualitative reviews.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm,headheight=15pt]{geometry}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}\setsansfont{Latin Modern Sans}\setmonofont{Latin Modern Mono}
\usepackage{booktabs,longtable,tabularx,array,enumitem,xcolor,fancyhdr,xurl,hyperref,tikz,float,graphicx}
\usetikzlibrary{arrows.meta,positioning}
\definecolor{accent}{HTML}{183D55}
\hypersetup{colorlinks=true,linkcolor=accent,urlcolor=accent,citecolor=accent,pdftitle={Medical Agentic Copilot: Five-Model Controlled Evaluation},pdfauthor={Medical Agentic Copilot Project},pdflang={en-US}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small\sffamily Medical Agentic Copilot}\fancyhead[R]{\small\sffamily Requirement 5}
\fancyfoot[L]{\small Internal model comparison: 10 October 2026}\fancyfoot[R]{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt plus 1pt}\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.16}
\setlist[itemize]{leftmargin=1.5em,itemsep=2pt}\setlist[enumerate]{leftmargin=1.7em,itemsep=3pt}
\begin{document}
\begin{center}{\LARGE\sffamily\bfseries Medical Agentic Copilot}\par\vspace{8pt}
{\large Five-Model Controlled Agent Evaluation}\par
{\small Requirement 5: common questions and fixed medical evidence}\par\vspace{5pt}10 October 2026\end{center}
\begin{abstract}
"""]
    parts.append(paragraph(f"The user selected Claude Opus 5.5, GLM-5.3, Qwen3.8 27B, Gemini 3.8 Flash and Gemma 4 31B to compare frontier hosted models and open-weight candidates for possible local research. The same ten synthetic oncology-nutrition questions and fixed guidance/PubMed records were offered through a bounded LangGraph workflow. The primary cohort completed {primary_complete} final answers from {attempted} attempted cases; {skipped} planned cases were not sent after repeated provider failures. A separate recovery cohort retried only noncompleted Google cases with bounded transient-server retries. Collection selects the first completed final answer for each model-question pair, yielding {complete} scored final answers; it is not a new inference cohort. Medical ratings are separately authored unblinded ordinal assessments, not model self-review, benchmark accuracy or clinical validation. Configurations, failures, visible outputs and origin hashes remain reviewable."))
    parts.append(r"\end{abstract}\tableofcontents\clearpage\section{Results and practical interpretation}" + "\n")
    for text in findings["main_findings"]:
        parts.append(paragraph(text))
    parts.append(r"\begin{longtable}{p{37mm}rrrr}\toprule Model & Final/10 & Mean total/10 & Median s & USD paid\\\midrule" + "\n")
    for operational, medical in zip(engineering["models"], quality["models"]):
        parts.append(esc(NAMES[operational["profile"]]) + " & " + str(operational["completed_answers"]) + "/10 & " + number(medical["mean_ordinal_total"]) + " & " + number(operational["median_case_elapsed_seconds"], 1) + " & " + number(operational["reported_or_estimated_cost_usd"], 4) + r"\\" + "\n")
    parts.append(r"\bottomrule\end{longtable}" + "\n")
    parts.append(paragraph("Mean totals summarize the first completed final answer per pair, never the highest-quality answer. N/A failures are not zero medical scores. The median covers selected graph executions; it excludes preceding failed graphs and includes stage pacing, repairs and recovery retries when selected. Primary and recovery reliability are shown separately below. Google has a larger recovery retry budget, so latency/completion cannot establish intrinsic model superiority. Monetary Google cost remains unknown. Self-review flags and critical failures are separate."))
    for filename, caption in [("requirement-5-quality-and-completion.pdf", "First completed answer ratings and collected coverage, with N/A explicit; no selection by medical score."), ("requirement-5-latency-and-cost.pdf", "Selected-graph latency and reported OpenRouter cost. Earlier failed graphs are separate; Google recovery has additional retries and unknown monetary cost.")]:
        parts.append(r"\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/" + filename + "}\n" + r"\caption{" + esc(caption) + r"}\end{figure}" + "\n")
    parts.append(r"\section{Selection, exact identifiers and openness}" + "\n")
    parts.append(paragraph("The selection follows the user's request for strong proprietary and open-weight comparators, plus smaller powerful candidates for potential local investigation. The labels 'best medical' and 'best open-source medical' are motivation, not established rankings. Vendor capability statements and general/coding benchmark scores do not prove medical superiority on this project."))
    parts.append(r"\begin{longtable}{p{37mm}p{71mm}p{43mm}}\toprule Model & Exact API ID and platform & Weight access\\\midrule" + "\n")
    openness = {"openrouter-claude-opus-5.5": ("OpenRouter", "Proprietary hosted", "R5T02"), "openrouter-glm-5.3": ("OpenRouter", "Open weights; custom license", "R5T03,R5T12,R5T13"), "openrouter-qwen3.8-27b": ("OpenRouter", "Apache 2.0; 27B dense", "R5T04,R5T10,R5T11"), "gemini-3.8-flash": ("Google AI Studio", "Proprietary hosted", "R5T06"), "gemma-4-31b": ("Google AI Studio", "Apache 2.0; 30.7B dense", "R5T08,R5T09")}
    for profile in manifest["profiles"]:
        platform, description, references = openness[profile["id"]]
        parts.append(esc(NAMES[profile["id"]]) + " & " + esc(profile["config"]["model"]) + r"\newline " + esc(platform) + " & " + esc(description) + r" \cite{" + references + r"}\\" + "\n")
    parts.append(r"\bottomrule\end{longtable}" + "\n")
    parts.append(paragraph("Qwen and Gemma publish Apache 2.0 model artifacts. GLM-5.3 publishes weights under its own license, including a conditional security review for specified large Model-as-a-Service businesses. This report calls it open-weight rather than automatically equating it with an unrestricted open-source license. No legal opinion or local execution test is implied."))
    parts.append(paragraph("As an arithmetic orientation, idealized four-bit dense weights for 27 billion parameters occupy 13.5 GB (about 12.6 GiB); 30.7 billion occupy 15.35 GB (about 14.3 GiB). These exclude quantization metadata, vision components, runtime buffers and KV cache. They are not tested RAM/VRAM requirements. GLM is much larger; active MoE parameter count does not eliminate storage of the other experts. Quantization, hardware and context length require their own local study."))
    parts.append(r"\section{Access checks and pilot adaptations}" + "\n")
    parts.append(paragraph("Authenticated catalog discovery found all five exact IDs. OpenRouter key/credit endpoints accepted the supplied key, with USD 10 credits and a separate USD 12 key limit at discovery. A catalog entry does not establish working generation or clinical quality. The original preflight obtained OK from Opus, GLM, Qwen and Gemma; Gemini 3.8 Flash timed out. REST, low-thinking and alternate documented transports were preserved as separate diagnostics. The low-thinking error exposed HTTP code 504; this is an endpoint observation, not a claim that the Google key is invalid."))
    parts.append(r"Transport references: \cite{R5T01,R5T06,R5T07,R5T08}." + "\n")
    pilot = read_json(PILOT / "engineering-summary.json")
    parts.append(r"\begin{longtable}{p{37mm}p{41mm}rp{57mm}}\toprule Model & ON02 pilot outcome & Calls & Main adaptation\\\midrule" + "\n")
    adaptations = {"openrouter-claude-opus-5.5": "Shared output cap increased; same default reasoning.", "openrouter-glm-5.3": "Shared cap increased; flags remain distinct from medical ratings.", "openrouter-qwen3.8-27b": "Review filled 8192-token cap; shared cap 16384 and Qwen medium effort.", "gemini-3.8-flash": "Generation unavailable in pilot/diagnostics; original default reasoning and 240 s retained.", "gemma-4-31b": "Accept null or empty attribution issues for all models; negative checks still force revision."}
    for model in pilot["models"]:
        parts.append(esc(NAMES[model["profile"]]) + " & " + esc(next(iter(model["statuses"]))) + " & " + str(model["model_calls"]) + " & " + esc(adaptations[model["profile"]]) + r"\\" + "\n")
    parts.append(r"\bottomrule\end{longtable}" + "\n")
    parts.append(paragraph("Pilot prompt/schema version 1 remains archived. Main version 2 permits a null issue for positive attribution checks and removes obsolete additional-guidance requests. All models receive the same schema. A negative support decision with a null issue still triggers correction; missing source checks remain flagged. No original answer is rewritten and no failed pilot is reclassified as successful live inference. Qwen's main medium-effort profile is a practical operating point, not its maximum-reasoning configuration."))
    parts.append(r"\subsection{Primary attempts and separate recovery}" + "\n")
    parts.append(r"\begin{longtable}{p{42mm}rrrr}\toprule Primary model & Attempted & Complete & Errors & Unattempted\\\midrule" + "\n")
    for model in primary_engineering["models"]:
        parts.append(esc(NAMES[model["profile"]]) + " & " + str(model["attempted_cases"]) + " & " + str(model["completed_answers"]) + " & " + str(model["statuses"].get("error", 0)) + " & " + str(model["unattempted_provider_unavailable_cases"]) + r"\\" + "\n")
    parts.append(r"\bottomrule\end{longtable}" + "\n")
    for model in recovery_engineering["models"]:
        parts.append(paragraph(f"Separate recovery {NAMES[model['profile']]}: {model['attempted_cases']} attempted cases, {model['completed_answers']} completed, {model['provider_requests']} provider attempts and {model['total_case_elapsed_seconds']:.1f} case seconds. Completed primary answers were ineligible for recovery. Up to two identical transport retries were added only for HTTP 500/502/503, with 30-second waits. Timeout 504 and clinical/format concerns were not retried by this wrapper."))
    parts.append(paragraph("The collected folder copies selected outputs without changing their answers and adds origin-file hashes. A primary completed answer always wins; otherwise the first completed recovery answer is selected. If neither finishes, the latest operational record remains N/A. Every failed first attempt is still available in its original cohort. Selection is independent of medical rating; differences in retry budget are a reliability confound."))
    parts.append(r"\section{Connected agent and controlled evidence}" + "\n")
    parts.append(r"""\begin{figure}[H]\centering
\begin{tikzpicture}[>=Stealth,box/.style={draw=accent,rounded corners,fill=accent!4,minimum width=22mm,minimum height=10mm,align=center,font=\small\sffamily}]
\node[box] (plan) at (0,0) {Plan};\node[box] (tools) at (2.9,0) {Read fixed\\evidence};
\node[box] (draft) at (5.8,0) {Draft};\node[box] (review) at (8.7,0) {Attribution\\review};\node[box] (end) at (11.6,0) {Finalize};
\node[box] (revise) at (8.7,-1.6) {Revise with\\same evidence};
\draw[->] (plan)--(tools);\draw[->] (tools)--(draft);\draw[->] (draft)--(review);\draw[->] (review)--(end);
\draw[->] ([xshift=4mm]review.south west)--([xshift=4mm]revise.north west);
\draw[->] ([xshift=-4mm]revise.north east)--([xshift=-4mm]review.south east);
\end{tikzpicture}\caption{Every completed in-scope case reads the same bound evidence for its question. The model may select the validated calculator. At most one revision is allowed.}\end{figure}
""")
    parts.append(paragraph("comparison.py reuses the requirement-2 bounded LangGraph and requirement-3 P/S source checks. read_question_evidence takes empty arguments: the runner binds the question's evidence, so the model cannot substitute a query, patient identifier or another question's source bundle. estimate_nutrition_targets remains optional and accepts only an explicit question weight. Clinical state is fresh per case, with no checkpointer or patient memory."))
    parts.append(paragraph("The fixed bundle is the unfiltered deduplicated union of two actual requirement-3 retrieval cohorts plus all ten curated guidance cards and nine guidance sources. ON01-ON10 have 7, 5, 6, 8, 9, 5, 10, 6, 4 and 5 PubMed records respectively. Repeated titles/abstracts must agree; original record paths and hashes remain in evidence/manifest.json. Query-specific ranks are removed. Relevance judgments are excluded. Irrelevant and contradictory records remain available to every completed in-scope agent."))
    parts.append(paragraph("This evaluates source-assisted answering, tool selection, calculator use, source interpretation and self-review. It does not measure each model's live PubMed search skill or coverage. No previously retrieved paper is presented as the result of a new query. Abstracts can be absent or truncated; full-text methods and complete subgroup results are unavailable. Source text is untrusted evidence, not executable instructions."))
    parts.append(r"\section{Settings, provenance and monetary accounting}" + "\n")
    parts.append(paragraph(manifest["design"]))
    parts.append(paragraph("The main profiles use a shared 16384-token output cap, 240-second call timeout, no SDK retries, at most one application rate-limit retry with a 30-second wait, one JSON repair per structured stage, and one revision. Temperature is omitted. Qwen uses medium reasoning; others use provider defaults. Google include_thoughts=false suppresses thought display without disabling thinking. Therefore token caps do not equalize compute, sampling, backend implementations or hidden revisions."))
    parts.append(paragraph("Cases run sequentially, using a seeded initial model permutation and balanced rotation across questions. Google request starts are spaced by at least 20 seconds per model, OpenRouter by one second. Different upstream speed, provider-default thinking, review length and revision decisions affect elapsed times. These are operational profiles, not local deployment speed estimates."))
    parts.append(paragraph("OpenRouter routes by price with require_parameters=true and ceilings of USD 4/M prompt and USD 20/M completion tokens. It can use different upstream services; this LangChain response does not reliably expose provider route, quantization or immutable weight revision, so none is inferred. Numeric usage.cost is logged without private thoughts. The local USD 8 guard reserves estimated prompt bytes plus full output cap before paid calls; unknown failed-call charges retain a reserve. It is not a provider billing guarantee."))
    parts.append(r"Routing policy: \cite{R5T05}." + "\n")
    parts.append(paragraph(f"Primary-cohort reported/estimated paid usage sums to USD {main_cost:.6f}; final local accounting includes the separately incurred preflight/pilot amount. The final primary budget record is {manifest['budget']}. Google recovery makes no additional OpenRouter requests. Google billing amounts are not supplied by the response and remain unknown. Catalogue advertised prices and exact observed costs are different records. Paid preflight and pilot are excluded from per-model cost charts."))
    parts.append(paragraph("A provider ending three consecutive cases with request errors is not called for remaining cases in that cohort. Those cases are labeled not_run_provider_unavailable, with zero requests and N/A quality, and are excluded from attempted-case latency. This is an explicit operational limit, not a measured medical failure for every remaining question. Truncation, structured-format failures and medical review flags are preserved separately."))
    parts.append(r"\section{Qualitative criteria and conflicting evidence}" + "\n")
    parts.append(paragraph(assessments["assessment_method"]))
    parts.append(paragraph("The five original dimensions are retained, each 0-2. Ratings are authored after reading saved final answers and their actual sources; excerpts and SHA-256 fingerprints bind every review to the output. Self-review decisions are not used as quality grades. An observed critical failure is separate from the total. Failed or unattempted cases are unscored, including preserved partial drafts."))
    parts.append(r"\begin{longtable}{p{37mm}p{115mm}}\toprule Dimension & Rating 2 criterion; 0 and 1 represent material failure or partial fulfillment\\\midrule" + "\n")
    for dimension in DIMENSIONS:
        parts.append(esc(dimension.replace("_", " ")) + " & " + esc(rubric["dimensions"][dimension]["2"]) + r"\\" + "\n")
    parts.append(r"\bottomrule\end{longtable}" + "\n")
    parts.append(paragraph("ON07's unchanged original key points reflect older routine-chemotherapy guidance. The fixed evidence includes Wingard and colleagues' 2026 randomized trial in HSCT/acute leukemia. Its abstract reports major infections of 31.4% in the liberalized diet versus 20.2% in the neutropenic diet, crossing the stopping boundary, without an intake benefit. Before the main run, an evaluator-only addendum required correct result direction, population boundaries and explicit conflict handling for categorical diet claims. It is absent from generation inputs. An older rubric match alone cannot justify reversing the newer abstract or claiming no benefit in any population."))
    parts.append(r"Conflicting clinical source: \cite{P41406403}; older supplied guidance: \cite{S05,P35576506}." + "\n")
    parts.append(r"\section{Observed differences and limitations}" + "\n")
    for text in findings["cross_model_observations"] + findings["limitations"]:
        parts.append(paragraph(text))
    parts.append(r"\section{Reproduction and engineering checks}" + "\n")
    parts.append(paragraph("The implementation guide documents the isolated environment, public profiles, ignored credential references and commands. Evaluation requires network access and available provider credit/quota; replay and artifact checks do not. Fresh run directories prevent overwriting observations. Public traces contain prompts, visible responses and numeric usage, never credential values or private reasoning text."))
    parts.append(r"""\begin{minipage}{\linewidth}{\small\begin{verbatim}
.\.venv\Scripts\python.exe -X utf8 -m medical_copilot `
  compare-models --run-dir data/evaluation/comparison-runs/new-run `
  --budget-usd 8 --initial-spend-usd 0
.\.venv\Scripts\python.exe -X utf8 -m medical_copilot `
  summarize-comparison `
  --run-dir $comparisonRun --assessments "$comparisonRun/assessments.json"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -X utf8 scripts/replay_comparison_run.py
.\.venv\Scripts\python.exe -X utf8 scripts/verify_requirement5.py
python scripts/render_requirement5_plots.py
.\.venv\Scripts\python.exe -X utf8 scripts/build_requirement5_report.py
\end{verbatim}}\end{minipage}""" + "\n")
    parts.append(paragraph("comparison_evaluation.py writes per-case outputs immediately, a numeric request audit, budget snapshots, engineering summaries, configurations and source-code snapshots. Evidence and assessment fingerprints are checked. Offline tests cover bound evidence identity, contradictory-study retention, source-specific revision, null issue handling, calculator validation inherited from earlier tests, interrupted finalization, monetary audit privacy, pre-request budget blocking and separate skipped-case records. Recorded-output replay is an engineering check, not a new medical experiment."))
    parts.append(paragraph("Requirements 1-3 retain their original artifacts and recorded behavior. Requirement 5 is demonstrated by this internal controlled comparison, with access/coverage limits explicit. Requirement 4 remains a distinct benchmark-subset evaluation; PDF ingestion and patient-specific memory (6-7) remain future work. This report is an output artifact, not requirement-6 PDF ingestion."))
    parts.append(r"\appendix\clearpage\section{All ten questions and case-specific findings}" + "\n")
    review_map = {(item["profile"], item["question_id"]): item for item in assessments["assessments"]}
    for question in question_set["questions"]:
        question_id = question["id"]
        entry = next(item for item in evidence_manifest["questions"] if item["question_id"] == question_id)
        parts.append(r"\subsection{" + esc(question_id + ": " + question["topic"].replace("_", " ")) + "}\n")
        parts.append(paragraph(question["question"]))
        parts.append(paragraph(f"Fixed question evidence: {len(entry['pubmed_ids'])} PubMed records plus all ten guidance cards. Five-dimension order: relevance / correctness / evidence support / actionability / safety and uncertainty."))
        for profile in manifest["profiles"]:
            item = review_map[profile["id"], question_id]
            answer = read_json(RUN / "models" / profile["id"] / "answers" / (question_id + ".json"))
            score = "N/A" if item["ratings"] is None else "/".join(str(item["ratings"][dimension]) for dimension in DIMENSIONS) + "; total " + str(sum(item["ratings"].values())) + "/10"
            parts.append(r"\subsubsection*{" + esc(NAMES[profile["id"]] + " -- " + score) + "}\n")
            parts.append(paragraph(f"Operational status: {answer['status']}; {len(answer['model_calls'])} model calls; {answer['elapsed_seconds']:.1f} seconds. " + item["case_findings"]))
            if item["ratings"] is not None:
                for dimension in DIMENSIONS:
                    parts.append(r"\textbf{" + esc(dimension.replace("_", " ")) + ":} " + esc(item["dimension_rationales"][dimension]) + "\n\n")
                if item["critical_failures_observed"]:
                    parts.append(r"\textbf{Observed critical failures:} " + esc("; ".join(item["critical_failures_observed"])) + "\n\n")
            else:
                parts.append(paragraph("No final medical answer was scored. " + item.get("operational_reason", "")))
        parts.append(paragraph("Complete original answers, intermediate drafts, reviews, prompts and source metadata are in the per-model answers directory and answers.md. The report's findings do not replace those original outputs."))
    parts.append(r"\section{Artifact fingerprints and complete supplied-source record}" + "\n")
    for title, path in [("Main manifest", RUN / "manifest.json"), ("Question set", RUN / "question-set.json"), ("Rubric", RUN / "qualitative-rubric.json"), ("Protocol", RUN / "protocol.md"), ("Evidence inventory", RUN / "evidence/manifest.json"), ("Separate medical assessments", RUN / "assessments.json")]:
        parts.append(r"\textbf{" + esc(title) + r"}\par{\small\ttfamily " + esc(file_hash(path)) + r"}\par" + "\n")
    parts.append(paragraph("The bibliography registers primary technical model/API/license sources and every supplied guidance/PubMed record, including irrelevant matches. Inclusion is not endorsement or a full-text appraisal. Complete authors, abstracts, read levels, retrieval dates, truncation flags and original provenance are preserved in JSON; earlier clinical source records are linked in the journal."))
    parts.append(r"\begin{thebibliography}{P99999999}\small" + "\n")
    for identifier, source in sources.items():
        authors = source.get("authors") or source.get("organization", "")
        if len(source.get("author_list") or []) > 3:
            authors = ", ".join(source["author_list"][:3]) + ", and colleagues"
        parts.append(r"\bibitem[" + identifier + "]{" + identifier + "} " + esc(authors) + " (" + esc(source["year"]) + "). " + esc(source["title"]) + ". ")
        if source.get("pmid"):
            parts.append("PMID " + esc(source["pmid"]) + ". ")
        if source.get("doi"):
            parts.append("DOI " + esc(source["doi"]) + ". ")
        parts.append(r"\url{" + source["url"] + "}. ")
        if source.get("support"):
            parts.append(esc(source["support"]))
        elif source.get("read_level"):
            parts.append(esc("Supplied " + source["read_level"] + "; inclusion is not clinical endorsement."))
        parts.append("\n\n")
    parts.append(r"\end{thebibliography}\end{document}" + "\n")
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"Generated {OUT.relative_to(ROOT)} from {len(assessments['assessments'])} case records and {len(sources)} sources.")


if __name__ == "__main__":
    main()
