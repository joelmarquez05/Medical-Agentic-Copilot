"""Build an English, self-contained LaTeX report from observed study artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from medical_copilot import prompts
from medical_copilot.evaluation import DIMENSIONS, read_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/runs/gemini-internal-final-20261010"
OUT = ROOT / "docs/implementation/requirement-2-report.tex"

ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "/": r"/\allowbreak{}"}


def esc(value) -> str:
    text = str(value).replace("\u202f", " ").replace("\u00a0", " ")
    # Normalize rare transcript glyphs for portable report typography.
    for original, normalized in [("【", "["), ("】", "]"), ("⁻¹", "^-1"), ("≥", ">="), ("≤", "<=")]:
        text = text.replace(original, normalized)
    return "".join(ESCAPES.get(char, char) for char in text)


def inline(value, citations=True) -> str:
    text = str(value)
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text).replace(r"\times", " x ")
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    pattern = r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[(?:[ST]\d{2})(?:,\s*[ST]\d{2})*\])"
    pieces = []
    for token in re.split(pattern, text):
        if token.startswith("**") and token.endswith("**"):
            pieces.append(r"\textbf{" + esc(token[2:-2]) + "}")
        elif token.startswith("*") and token.endswith("*"):
            pieces.append(r"\textit{" + esc(token[1:-1]) + "}")
        elif token.startswith("`") and token.endswith("`"):
            pieces.append(r"\texttt{" + esc(token[1:-1]) + "}")
        elif citations and re.fullmatch(r"\[(?:[ST]\d{2})(?:,\s*[ST]\d{2})*\]", token):
            pieces.append(r"\cite{" + token[1:-1].replace(" ", "") + "}")
        else:
            pieces.append(esc(token))
    return "".join(pieces)


def markdown(text, citations=True) -> str:
    lines = []
    list_type = None
    for raw in text.splitlines():
        stripped = raw.strip()
        bullet = re.match(r"^(?:[-*+]\s+|\d+\.\s+)(.*)", stripped)
        if bullet:
            kind = "enumerate" if re.match(r"^\d+\.", stripped) else "itemize"
            if list_type != kind:
                if list_type:
                    lines.append(r"\end{" + list_type + "}")
                lines.append(r"\begin{" + kind + "}")
                list_type = kind
            lines.append(r"\item " + inline(bullet.group(1), citations))
            continue
        if list_type:
            lines.append(r"\end{" + list_type + "}")
            list_type = None
        if not stripped or stripped == "---":
            lines.append("\n")
        elif stripped.startswith("#"):
            lines.append(r"\paragraph{" + inline(stripped.lstrip("# "), False) + "}")
        else:
            lines.append(inline(stripped, citations))
    if list_type:
        lines.append(r"\end{" + list_type + "}")
    return "\n".join(lines)


def main():
    manifest = read_json(RUN / "manifest.json")
    metrics = read_json(RUN / "engineering-metrics.json")
    assessment = read_json(RUN / "assessments.json")
    summary = read_json(RUN / "qualitative-summary.json")
    dataset = read_json(ROOT / "data/evaluation/oncology_nutrition_questions.json")
    rubric = read_json(ROOT / "data/evaluation/qualitative_rubric.json")
    sources = read_json(ROOT / "data/research/requirement-2-source-catalogue.json")["sources"]
    guidance = read_json(ROOT / "src/medical_copilot/resources/oncology_guidance.json")
    checks = read_json(ROOT / "data/checks/requirement-2-test-results.json")
    demos = read_json(ROOT / "data/checks/provider-demonstrations.json")
    answers = {question["id"]: read_json(RUN / "answers" / (question["id"] + ".json")) for question in dataset["questions"]}
    reviews = {item["question_id"]: item for item in assessment["assessments"]}
    parts = [r"""% Generated from saved research/evaluation artifacts. Compile twice with XeLaTeX.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm,headheight=15pt]{geometry}
\usepackage{fontspec}
\IfFontExistsTF{TeX Gyre Pagella}{\setmainfont{TeX Gyre Pagella}}{\setmainfont{Latin Modern Roman}}
\IfFontExistsTF{TeX Gyre Heros}{\setsansfont{TeX Gyre Heros}}{\setsansfont{Latin Modern Sans}}
\setmonofont{Latin Modern Mono}
\usepackage{booktabs,longtable,tabularx,array,enumitem,xcolor,fancyhdr,xurl,hyperref}
\usepackage{tikz}
\usepackage{float}
\usetikzlibrary{arrows.meta,positioning}
\definecolor{accent}{HTML}{183D55}
\definecolor{muted}{HTML}{566573}
\hypersetup{colorlinks=true,linkcolor=accent,urlcolor=accent,citecolor=accent,
pdftitle={Medical Agentic Copilot: Requirement 2 Implementation and Evaluation},
pdfauthor={Medical Agentic Copilot Project},pdflang={en-US},bookmarksnumbered=true}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\sffamily\color{muted}Medical Agentic Copilot}
\fancyhead[R]{\small\sffamily\color{muted}Requirement 2}
\fancyfoot[L]{\small\color{muted}Implementation and observations: 10 October 2026}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt plus 1pt}
\setlength{\emergencystretch}{3em}
\setlist[itemize]{leftmargin=1.5em,itemsep=2pt,topsep=3pt}
\setlist[enumerate]{leftmargin=1.7em,itemsep=2pt,topsep=3pt}
\renewcommand{\arraystretch}{1.16}
\setlength{\tabcolsep}{2pt}
\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcommand{\artifact}[1]{\path{#1}}
\newcommand{\cell}[1]{\begin{minipage}[t]{\linewidth}#1\end{minipage}}
\title{\sffamily\color{accent}\textbf{Medical Agentic Copilot}\\[5pt]
\Large Requirement 2: Implementation and Qualitative Evaluation\\[3pt]
\large LangGraph, interchangeable providers, and oncology nutrition}
\author{Medical Agentic Copilot Project}
\date{10 October 2026}
\begin{document}
\maketitle\thispagestyle{fancy}
\begin{abstract}
\noindent This report documents an implemented LangGraph agent with provider-independent
text-chat calls, model-selected local guidance and calculation tools, structured planning,
self-review, and a bounded revision loop. Ten synthetic adult oncology-nutrition questions
were executed with Gemini and assessed separately against a prespecified qualitative rubric.
The final live run produced ten answers with 42 application-level model calls and 78,202
reported tokens. Seven answers passed internal self-review; three retained citation-related
flags. The separate assessment identified substantive imprecision and omissions beyond those
flags. Three requested NVIDIA models were integrated and tested: GPT-OSS completed a flagged
answer, Kimi reached review before a revision timeout, and DeepSeek timed out at planning.
Code, prompts, references, configurations, original outputs, failures, reviews, and checks
are preserved. This is a functional educational proof of concept, with no independent
clinical validation or claim of benchmark accuracy.
\end{abstract}
\clearpage\tableofcontents\clearpage
"""]

    def section(title):
        parts.append(r"\section{" + esc(title) + "}")

    def subsection(title):
        parts.append(r"\subsection{" + esc(title) + "}")

    def paragraph(text):
        parts.append(inline(text) + "\n")

    def bullets(items):
        parts.append(r"\begin{itemize}")
        parts.extend(r"\item " + inline(item) for item in items)
        parts.append(r"\end{itemize}")

    def table(headers, rows, widths):
        parts.append(r"\begin{longtable}{@{}" + "".join(f"P{{{width}mm}}" for width in widths) + r"@{}}\toprule")
        parts.append(" & ".join(r"\textbf{" + esc(header) + "}" for header in headers) + r"\\\midrule\endhead")
        for row in rows:
            parts.append(" & ".join(inline(value) for value in row) + r"\\")
        parts.append(r"\bottomrule\end{longtable}")

    def code(text):
        parts.append("\\begin{small}\n\\begin{verbatim}\n" + text + "\n\\end{verbatim}\n\\end{small}")

    section("Objective, delivered scope, and acceptance evidence")
    paragraph("The professor's second requirement is to build an agent using LangChain or LangGraph, make provider/model changes practical, create approximately ten questions aligned with oncology nutrition, and run a recorded qualitative answer evaluation. The recommendation to use Gemini or Cloudflare is a provider suggestion, not a constraint. Gemini was already confirmed to generate text in the prior credential preflight; that preflight did not establish medical quality.")
    table(["Required item", "Delivered implementation/evidence"], [
        ("LangChain or LangGraph", "Compiled StateGraph, LangChain chat models, and StructuredTool instances in the installed Python package."),
        ("Interchangeable LLMs", "Validated JSON profiles, a shared text-chat client, LangChain's provider factory, and configurable OpenAI-compatible endpoints."),
        ("Approximately ten internal medical questions", "Ten project-authored adult cases, frozen before the live studies, with expected points and case-critical failure criteria."),
        ("Qualitative evaluation", "Ten real final-run Gemini answers, separately written five-dimension assessments, original excerpts, answer hashes, and reproducible summaries."),
        ("User-requested NVIDIA additions", "Exact Kimi K3, DeepSeek V4.1 Flash, and GPT-OSS 20B profiles; authenticated discovery, bounded preflights, and one-question agent demonstrations."),
        ("Detailed compiled report", "This self-contained PDF, editable LaTeX source, a generator, and documented rebuild commands.")
    ], [44, 115])
    paragraph("Requirement 1 remains documented in its earlier research inventory. Requirements 3-7 remain separate work: connected paper search/retrieval assessment, an external benchmark subset, a controlled LLM comparison, PDF ingestion/QA, and patient-specific memory. Generating this report does not demonstrate PDF ingestion. The project journal records development history; it is not patient application memory.")

    section("Architecture and actual agent decisions")
    paragraph("The system is a constrained agentic workflow: the outer graph has fixed nodes, while the model selects tools/topics, identifies missing information, critiques drafts, and controls whether further guidance and revision are needed. This uses LangGraph's graph and evaluator/optimizer concepts [T01]. It is a single running application agent, not a team of subagents.")
    parts.append(r"""\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=10mm and 22mm,>=Stealth,
box/.style={draw=accent,rounded corners,fill=accent!5,align=center,text width=47mm,minimum height=9mm,font=\small\sffamily}]
\node[box] (plan) {Plan: choose tools/topics};
\node[box,below=of plan] (tools) {Validate and execute tools};
\node[box,below=of tools] (draft) {Draft from supplied evidence};
\node[box,below=of draft] (review) {Model review + citation checks};
\node[box,below=of review] (final) {Finalize: answer + status};
\node[box,right=of review,text width=40mm] (revise) {Extra guidance if needed; revise};
\draw[->] (plan)--(tools);\draw[->] (tools)--(draft);\draw[->] (draft)--(review);
\draw[->] (review)--node[left,font=\scriptsize,align=right]{no issue or\\limit reached}(final);
\draw[->] (review)--node[below,font=\scriptsize]{revise}(revise);
\coordinate (feedback) at ([yshift=6mm]review.north);
\draw[->] (revise.north)|-node[pos=.55,above,font=\scriptsize]{new draft}(feedback)--(review.north);
\end{tikzpicture}
\caption{Implemented decision loop. Issues or a request for further guidance trigger revision while allowance remains. The default permits one revision; the actual Mermaid graph is saved in each run.}
\end{figure}""")
    table(["Node", "Responsibility", "Stored evidence"], [
        ("plan", "Use the question and a catalog to choose read-only tools. Return a validated Plan.", "Scope, focus, missing facts, tool names/arguments."),
        ("execute_tools", "Validate arguments, enforce explicit weight, and call allowed tools only.", "Inputs, outputs, timing, safe error summaries."),
        ("draft", "Answer using the question, plan, and actual tool outputs.", "Initial visible draft and model-call metadata."),
        ("review", "Check substantive concerns and citation findings against supplied evidence.", "Issues, revision decision, extra topics, uncertainty."),
        ("revise", "Optionally obtain requested guidance and rewrite within the limit.", "Additional tool results, new draft, revision count."),
        ("finalize", "Retain unresolved flags and add registry-controlled source details.", "Final answer, sources, status, complete node trace.")
    ], [27, 71, 61])
    paragraph("State stores the question, plan, tool results, every draft/review, citation issues, trace, revision count, final sources, and status. Each question starts with fresh state, without a checkpointer or cross-question conversation context. Model-call logs are audit data, not conversation memory fed into later questions.")
    table(["Status", "Exact interpretation"], [
        ("self_review_passed", "No remaining issue from internal review or structural checks; not clinical validation."),
        ("needs_human_review", "An answer exists but unresolved issues remain after the bounded revision allowance."),
        ("insufficient_evidence", "The question is outside scope or no usable source evidence was obtained."),
        ("error", "A provider or structured-output operation failed. Partial work is preserved but is not presented as a final answer.")
    ], [50, 109])

    section("Model abstraction and provider configuration")
    paragraph("All nodes call one ModelClient that accepts system/user text and returns visible answer text. Construction is isolated from graph logic using LangChain init_chat_model [T02,T03]. OpenAI-compatible providers use the same adapter with a configurable base URL and Chat Completions mode. The graph does not require a provider's native tool calling or structured-output API: JSON decisions are prompted, parsed, and validated with Pydantic. One format-only repair is permitted when needed.")
    paragraph("This makes many instruction-following LangChain models practical to swap without rewriting the graph, but schema compliance and endpoint behavior still require testing. Unknown providers may need their integration package and native credentials/configuration. A small ChatBackend protocol also permits custom backends. Compatibility is an interface design, not proof that every model obeys the prompts.")
    table(["Profile/model", "Route and observed status"], [
        ("Gemini: gemini-3.5-flash-lite", "google_genai. Stable model ID verified [T04]; complete final ten-question run."),
        ("NVIDIA: moonshotai/kimi-k3", "openai_compatible at integrate.api.nvidia.com/v1. Plan, draft, and review returned; revision timed out."),
        ("NVIDIA: deepseek-ai/deepseek-v4.1-flash", "Same NVIDIA endpoint. Advertised ID verified; planning timed out without a visible answer."),
        ("NVIDIA: openai/gpt-oss-20b", "Same NVIDIA endpoint. Full one-question graph completed with unresolved issues."),
        ("Other OpenAI-compatible endpoints", "Configurable local/hosted server template; no additional hosted inference study."),
        ("Cloudflare", "Official Workers AI Chat Completions template [T06]; account ID/token required; not live-tested."),
        ("Anthropic / Ollama", "Optional integration/profile templates; credentials or local model/server required; not inference-tested.")
    ], [73, 86])
    subsection("Final Gemini configuration")
    table(["Field", "Recorded value"], [(key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)) for key, value in manifest["model_config"].items()], [62, 97])
    paragraph("A pilot requested temperature 1.0, but the installed Gemini integration warned that this model ignores custom sampling parameters. The package implementation was inspected; the final profile sets temperature to null, omitting a custom value [T07]. Identical future wording is not guaranteed. No seed or deterministic-output guarantee is claimed.")
    paragraph("The graph's ordinary path has three application-level calls: planning, drafting, and review. Each revision adds a rewrite and another review. With one revision and at most one JSON repair for each structured stage, the nominal bound is eight application-level calls per question. The final Gemini run used 42 calls for ten questions; it needed no JSON-format repairs. SDK/network attempts are separate from these application calls.")
    subsection("Credential and operational handling")
    paragraph("Profiles contain credential references, never the actual keys. Environment values take precedence; the supplied ignored api_key.md and nvidia_api_key.md provide local fallbacks. Common literal credential fields and credential-bearing URLs are rejected. The active credential is redacted before artifact writes. Provider exceptions are summarized by type, not raw traceback or full request. Hosted tracing is disabled. No patient records were submitted; all prompts use synthetic cases [T15,T16].")
    paragraph("The initial rapid run hit GoogleRateLimitError after five completed answers. Final request starts are spaced by four seconds, SDK retries are disabled for that profile, and at most two application retries wait 30 seconds after a rate-limit error. This reflects an observed transient issue and Google's quota dimensions [T05]; actual account tier, credit balance, and quota limits were not inspected. The final Gemini run needed no retry. Timings include scheduled waits, while failed requests may have no token metadata; reported totals are not a billing-cost calculation.")

    section("Tools, clinical guidance, and citation provenance")
    paragraph("Two LangChain StructuredTool objects are available. lookup_guidance selects from ten manually curated topic cards backed by nine clinical sources. estimate_nutrition_targets performs deterministic arithmetic from an explicitly supplied kilogram weight. The allowed tool registry is fixed; invalid arguments are reported rather than silently corrected. There are no shell/network/patient-record tools in the answer workflow.")
    table(["Topic", "Sources and purpose"], [(card["topic"], ", ".join(card["source_ids"]) + ": " + card["description"]) for card in guidance["cards"]], [48, 111])
    paragraph("The cards are short project-authored paraphrases, with source IDs and recommendation/section locators. The references are ESPEN, ASCO, NICE, ESMO, and NCI guidance/evidence summaries [S01,S02,S03,S04,S05,S06,S07,S08,S09]. Read levels are recorded: some official recommendations were accessible as indexed text rather than a complete publisher document. NCI living summaries explicitly use an access-year snapshot. This is a selected local collection, not an exhaustive current review or live PubMed/medRxiv search.")
    paragraph("The calculator returns energy at weight times 25-30 kcal/day and protein strictly above weight times 1.0, up to weight times 1.5 g/day when feasible, following ESPEN [S01]. A 60 kg input therefore yields 1,500-1,800 kcal/day and protein above 60, up to 90 g/day. Nonpositive/nonfinite inputs, booleans, string weights, values above the validation ceiling, and weights invented by the planner are rejected. These are general adult estimates; the tool explicitly excludes initial refeeding prescriptions and requires clinical adjustment.")
    paragraph("Individual and grouped ASCII source markers are parsed. Only IDs actually present in tool evidence may enter the automatic bibliography. Model-written URLs are flagged. This gives identifier provenance, not proof that a sentence is supported or medically correct. Internal G-card tags, unavailable IDs, and semantic overstatement were observed despite the source registry and self-review. Those shortcomings remain visible in the findings.")

    section("Internal question set and assessment protocol")
    paragraph("The ten adult cases were authored for this project and frozen with the rubric before the main live studies. They cover common oncology-nutrition decisions and selected safety-sensitive situations. No real patient records or OpenEvidence benchmark questions were used. Question IDs ON01-ON10 remain stable for later retrieval and provider studies.")
    table(["ID", "Clinical task"], [(question["id"], question["topic"].replace("_", " ")) for question in dataset["questions"]], [24, 135])
    paragraph("Expected key points, reference IDs, and critical failures are part of the evaluator's dataset, but only each question's text enters the agent. An engineering test places a sentinel in expected criteria and verifies it never reaches model prompts. The answer workflow receives the guidance/tool catalog, not gold answers. The study still has construction bias because the internal set, selected guidance, and assessment expectations overlap.")
    paragraph("The subsequent review was authored by the same AI implementation assistant, reading saved answers, prespecified points, and reference snapshots. It was not blinded, not independent clinician adjudication, and not an automated same-model grading call. Model self-review is a generation step; this separate review examines the final saved answers. Every rating has a rationale and observed excerpts, and each assessment is tied to its answer-file SHA-256 hash.")
    table(["Dimension", "0", "1", "2"], [(dimension.replace("_", " "), rubric["dimensions"][dimension]["0"], rubric["dimensions"][dimension]["1"], rubric["dimensions"][dimension]["2"]) for dimension in DIMENSIONS], [31, 40, 43, 45])
    paragraph("A sum out of ten is only a descriptive ordinal checklist aid. It is not accuracy, an external benchmark score, a probability of safety, or evidence of clinical validation. Prespecified critical failures are recorded separately; a favorable total cannot override a critical failure. Source selection is reported as engineering behavior, while a paper-retrieval relevance evaluation remains requirement 3.")

    section("Execution history and preserved failures")
    rows = []
    for run_id, purpose in [("gemini-pilot-20261010", "Two-case SDK/graph pilot; sampling warning observed."), ("gemini-internal-20261010", "Rapid ten-case attempt: five completed, then rate-limit failures."), ("gemini-internal-paced-20261010", "Completed ten-case run; grouped-citation handling defect identified."), ("gemini-internal-final-20261010", "Final ten-case run after citation-parser correction.")]:
        run = ROOT / "data/evaluation/runs" / run_id
        old = read_json(run / "engineering-metrics.json")
        rows.append((run_id.replace("-20261010", ""), str(old["question_count"]), str(old["model_call_count"]), str(old["reported_usage"]["total_tokens"]), purpose))
    table(["Run", "Q", "Calls", "Tokens", "Purpose/outcome"], rows, [43, 8, 13, 22, 73])
    paragraph("All runs are preserved. These are development iterations, not independent replicates or a statistical estimate of model performance. The final run alone is the medical answer set scored below. The citation correction accepted grouped markers and detected invalid annotations; no earlier answer was rewritten to hide a failure.")
    paragraph("Interrupted NVIDIA requests subsequently exposed that the original invocation path did not return completed node state on failure, although model-call prompts still retained the evidence. The final driver streams completed node updates and preserves partial plans, tools, drafts, and reviews while keeping the final answer empty on error. The exact 11 modules used for the live final Gemini study were archived before this error-path improvement. Offline replay of 13 recorded cases verified unchanged successful answers, sources, traces, and statuses and recovered separately labeled partial state. This replay made zero new API calls and is not another clinical evaluation.")

    section("Observed final Gemini results")
    table(["Measurement", "Observed value"], [
        ("Run ID", manifest["run_id"]), ("UTC start", manifest["started_at_utc"]), ("UTC finish", manifest["finished_at_utc"]),
        ("Completed question count", str(metrics["question_count"])), ("Statuses", "7 self_review_passed; 3 needs_human_review"),
        ("Application-level calls", "42: 10 plan, 10 draft, 16 review, 6 revise"),
        ("Provider-reported tokens", "68,865 input; 9,337 output; 78,202 total"),
        ("Question latency", f"Median {metrics['median_question_seconds']:.3f}s; range {metrics['min_question_seconds']:.3f}-{metrics['max_question_seconds']:.3f}s"),
        ("Sum of question latencies", f"{metrics['sum_question_seconds']:.3f}s, excluding the explicit inter-question delay"),
        ("Revised questions", ", ".join(metrics["revised_questions"])), ("Successful/failed tool calls", "12 / 0"),
        ("Residual citation findings", ", ".join(metrics["questions_with_mechanical_citation_issues"])),
        ("Observed rate-limit retries in final run", "0"),
    ], [64, 95])
    subsection("Qualitative ratings")
    parts.append(r"\begin{longtable}{@{}lrrrrrr@{}}\toprule Question & Rel. & Corr. & Evid. & Action & Safety & Total\\\midrule\endhead")
    for item in assessment["assessments"]:
        scores = [str(item["ratings"][dimension]) for dimension in DIMENSIONS]
        parts.append(" & ".join([item["question_id"]] + scores + [str(sum(item["ratings"].values())) + "/10"]) + r"\\")
    parts.append(r"\bottomrule\end{longtable}")
    table(["Dimension", "Rating 0", "Rating 1", "Rating 2"], [(dimension.replace("_", " "), *[str(summary["rating_counts_by_dimension"][dimension][str(score)]) for score in [0, 1, 2]]) for dimension in DIMENSIONS], [72, 29, 29, 29])
    paragraph("All ten answers were judged relevant under this limited rubric. Five had correctness imprecision/omission, seven had weaker evidence linkage or expansion, four had practical omissions, and four had safety/uncertainty caveats needing improvement. No prespecified critical failure was identified by this assistant review. That statement is bounded by the checklist and assessor and does not establish that the answers are safe.")
    bullets([
        "ON01: correct screening concern, but cachexia/malnutrition conflation, missing repeat assessment, and unavailable S05 attribution.",
        "ON02: successful arithmetic and strict protein floor; indirect calorimetry described too broadly as exact energy expenditure.",
        "ON03: appropriate enteral escalation; hydration, symptom control, and urgency insufficiently explicit.",
        "ON04: shared PN decision framework, but irreversible-failure eligibility phrasing is too restrictive.",
        "ON05: useful cachexia guidance and 2023 update, but invalid source annotations remain and benefit claims are too categorical.",
        "ON06: core refeeding principles covered; the reviewer redundantly requested evidence already present.",
        "ON07: appropriate restriction/hygiene discussion, with internal card references weakening citation clarity.",
        "ON08: cautious supplement discussion met the prespecified core points; product-specific safety was not evaluated.",
        "ON09: correct PEI/PERT focus, with unsupported prescriber-role wording and incomplete concurrent-cause/follow-up detail.",
        "ON10: appropriate central diet advice, but no primary-source markers/bibliography and limited practical follow-through.",
    ])
    paragraph("Self-review missed semantic weaknesses in several answers and even accepted a failed citation repair in ON05. The host correctly retained that answer's mechanical flag. A valid source identifier and a passing model review therefore cannot serve as a medical endorsement. Full answers, criteria, excerpts, and rationale are included in Appendix A.")

    section("Requested NVIDIA models: integration and observed limits")
    paragraph("The three exact model IDs and common endpoint follow the user-supplied NVIDIA pages [T08,T09,T10]. NVIDIA advertises free prototype endpoints with API-key authentication; this does not establish unlimited access or account-specific quotas. Authenticated model discovery returned HTTP 200 with 80 advertised IDs, including all three. The common key produced an HTTP-200 OK response from GPT-OSS 20B; Kimi later produced actual agent responses. DeepSeek did not return a usable response within the tested deadlines.")
    table(["Model", "Preflight and agent outcome"], [
        ("moonshotai/kimi-k3", "Initial 120-second non-streaming and 180-second streaming probes timed out. With documented low reasoning effort [T11], the agent returned plan, draft, and review; revision timed out. Four application calls; 6,035 reported tokens from successful responses; 327.543s overall."),
        ("deepseek-ai/deepseek-v4.1-flash", "Both bounded probes timed out. Agent planning timed out at 120 seconds; no visible medical answer or token metadata was available. No answer-quality rating is possible [T12]."),
        ("openai/gpt-oss-20b", "Minimal probe returned HTTP 200 and OK. ON02 completed with six application calls, one JSON repair, 14,721 reported tokens, and 139.838s elapsed; final status needs_human_review [T13].")
    ], [67, 92])
    paragraph("GPT-OSS's core energy arithmetic was recognizable, but the protein floor used >=60 rather than the strict >60 requirement. Its answer used Unicode citation brackets rather than the ASCII marker contract and added unsupported details, including a 'fat-tympanic sign' and organ-dysfunction protein-reduction advice not established by the supplied card. Its second model review missed these substantive concerns. The completed graph demonstrates provider portability, not clinically acceptable output.")
    paragraph("Kimi's visible review identified unsupported meal-level protein allocation, an interpretive energy rationale, and inconsistent protein-floor wording in its initial draft. The rewrite did not complete within the configured deadline, so no final answer is claimed. Its interim transcript remains reviewable. These observations apply to one question and the tested conditions, with different output budgets and response availability. They do not establish a model ranking or fulfill the later controlled comparison requirement.")

    section("Engineering checks, reproducibility, and artifact map")
    paragraph(f"The final offline suite passed {checks['test_count']} tests with zero failures/errors. These verify software behavior, not medicine: calculator units/inequality, invalid and invented weight rejection, argument validation, bounded revisions and JSON repair, grouped/bad-label citations, scope handling, provider construction, safe exceptions/redaction, rate-limit retry bounds, gold-criteria isolation, and partial state on interruption. Scripted test backends are used only in offline tests; the medical evaluation JSON stores actual provider outputs.")
    paragraph("pip check reported no broken requirements. The tested environment uses Python 3.12.3 on Windows. The dependency snapshot records exact observed packages, while pyproject.toml pins the core integrations. A freeze file is not a platform-independent resolver or a supply-chain hash lock. The final run manifest preserves code/source/dataset/rubric hashes and package versions, and assessments verify the original answer hashes.")
    table(["Package", "Recorded version"], list(manifest["package_versions"].items()), [104, 55])
    paragraph("Main study artifacts are under data/evaluation/runs/gemini-internal-final-20261010. Earlier runs and NVIDIA demos use distinct directories. Detailed clinical source metadata are in the guidance file and 25-source catalogue; the project journal registers what each source supports.")
    table(["Artifact / module", "Purpose"], [
        ("src/medical_copilot/graph.py", "Six-node graph, tool choice, citation gates, bounded revision, and partial-state preservation."),
        ("config.py / models.py", "Validated settings, credentials, factory, redaction, pacing, retries, text/JSON handling."),
        ("schemas.py / prompts.py", "Explicit plan/review/tool contracts and versioned prompts."),
        ("tools.py / evidence.py", "Local guidance lookup, numerical estimates, and source registry."),
        ("cli.py / evaluation.py", "Ask/evaluate/summarize commands, manifests, original outputs, and reviewed summaries."),
        ("config/models/*.json", "Gemini, three NVIDIA profiles, and additional provider templates."),
        ("data/evaluation/*.json", "Ten frozen questions, expected points, critical failures, and rubric."),
        ("answers/ON01.json ... ON10.json", "Authoritative original answers with all model calls, drafts, tools, reviews, usage, and trace."),
        ("assessments.json / qualitative-assessment.md", "Separate source-based judgments, answer hashes, excerpts, weaknesses, and improvements."),
        ("manifest.json / engineering-metrics.json", "Configuration, timestamps, fingerprints, and descriptive operational measurements."),
        ("source-snapshot/*.py", "Exact study modules before the subsequent interrupted-state driver improvement."),
        ("data/checks/*.json and tests.xml", "API probes, provider observations, test results, replay, and final artifact checks."),
        ("scripts/replay_recorded_runs.py", "Offline recorded-response verification, explicitly distinct from live inference."),
        ("scripts/build_requirement2_report.py", "Generate this report source from the recorded artifacts."),
        ("scripts/verify_requirement2.py", "Validate the final data/code/report bundle without model calls."),
    ], [77, 82])
    subsection("Install and demonstrate")
    code(r"""python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,report]"
.\.venv\Scripts\python.exe -m medical_copilot show-config
.\.venv\Scripts\python.exe -X utf8 -m medical_copilot ask `
  "For a 60 kg adult with cancer, what are approximate targets?"
""")
    paragraph("These Windows commands use a PowerShell backtick for line continuation. Existing ignored credential files or the named environment variables supply the keys. ask writes no trace unless --output is specified. A provider profile can be selected with --config; only its model integration changes, not the graph. Refer to README.md for the complete usage guide.")
    code(r""".\.venv\Scripts\python.exe -X utf8 -m medical_copilot evaluate `
  --run-dir data/evaluation/runs/my-new-run
$studyRun = "data/evaluation/runs/gemini-internal-final-20261010"
.\.venv\Scripts\python.exe -X utf8 -m medical_copilot summarize `
  --run-dir $studyRun `
  --assessments "$studyRun/assessments.json"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -X utf8 scripts/replay_recorded_runs.py
.\.venv\Scripts\python.exe -X utf8 scripts/verify_requirement2.py
""")
    paragraph("Fresh evaluations require a new empty directory. The runner does not silently replace prior findings or compute medical correctness automatically. After reviewing a new run, author the assessment JSON and use summarize to validate IDs, dimensions, and answer hashes. Optional Anthropic/Ollama integrations can be installed through extras; user-selected endpoints/models must be validated in their own environment.")
    subsection("Regenerate and compile this report")
    code(r""".\.venv\Scripts\python.exe -X utf8 scripts/build_requirement2_report.py
xelatex -interaction=nonstopmode -halt-on-error `
  -output-directory=docs/implementation `
  docs/implementation/requirement-2-report.tex
""")
    paragraph("Run XeLaTeX twice to resolve citations, the contents, and page references. The bibliography is embedded; no BibTeX invocation is needed. Use PowerShell as shown, or adapt the executable paths and continuation syntax to another shell.")

    section("Limitations and next work")
    bullets(assessment["limitations"])
    bullets([
        "The nine-source local collection can omit relevant exceptions or later research; no automatic evidence update is implemented.",
        "Source availability and output shape checks do not validate clinical entailment or treatment advice. Unsupported details survived self-review.",
        "Internal G-card and S-source labels confused models. A source-only presentation or stronger deterministic attribution could reduce this failure mode.",
        "Endpoint responsiveness, reasoning/output budgets, and quotas varied. Missing token metadata is not evidence of zero consumption or zero cost.",
        "No patient-specific storage, isolation demonstration, PDF ingestion, or external benchmark run is implemented by this requirement-2 work.",
    ])
    paragraph("The next concrete assignment step is requirement 3: connect PubMed and/or medRxiv tools, reuse these question texts, preserve actual retrieved references, and assess retrieval relevance separately. Requirement 4 should use a documented benchmark subset from the earlier inventory. Requirement 5 needs comparable runs across models, with declared configuration differences and stable endpoints. Requirements 6 and 7 remain separate PDF-QA and patient-memory demonstrations. Clinical review would strengthen the evaluation, but production readiness is outside this proof-of-concept claim.")

    parts.append(r"\clearpage\appendix")
    section("All ten questions, observed answers, and assessment rationale")
    paragraph("The following are the saved final-run responses, not reference answers. Their original JSON remains authoritative. Presentation normalizes Markdown, arithmetic notation, narrow spaces, and rare Unicode glyphs; no medical content is repaired for publication. Source IDs present in this report's bibliography may still have been unavailable or misattributed within an individual answer, as explicitly flagged below.")
    for question in dataset["questions"]:
        qid = question["id"]
        result = answers[qid]
        review = reviews[qid]
        parts.append(r"\clearpage")
        subsection(qid + ": " + question["topic"].replace("_", " "))
        paragraph("Question: " + question["question"])
        paragraph(f"Observed status: {result['status']}. Elapsed: {result['elapsed_seconds']:.3f}s. Calls: {len(result['model_calls'])}. Reported tokens: {result['usage']['total_tokens']}. Revisions: {result['revision_count']}.")
        paragraph("Node trace: " + " -> ".join(result["trace"]))
        returned = sorted({source["id"] for tool in result["tool_results"] if tool["status"] == "ok" for source in tool["output"].get("sources", [])} | {source for tool in result["tool_results"] if tool["status"] == "ok" for source in tool["output"].get("source_ids", [])})
        paragraph("Returned source IDs: " + ", ".join(returned) + ". Automatically attached bibliography IDs: " + (", ".join(source["id"] for source in result["sources"]) or "none") + ".")
        parts.append(r"\subsubsection*{Observed final answer}")
        body = result["answer"].split("\n\nSources consulted from the local guidance collection:")[0]
        parts.append(markdown(body))
        parts.append(r"\subsubsection*{Prespecified assessment points}")
        bullets(question["expected_key_points"])
        paragraph("Case-critical unacceptable outcomes: " + "; ".join(question["critical_failures"]))
        parts.append(r"\subsubsection*{Separate qualitative findings}")
        paragraph(review["overall_finding"])
        table(["Dimension", "Rating", "Rationale"], [(dimension.replace("_", " "), str(review["ratings"][dimension]) + "/2", review["rationales"][dimension]) for dimension in DIMENSIONS], [32, 15, 112])
        paragraph("Descriptive total: " + str(sum(review["ratings"].values())) + "/10. Critical failures observed against the limited prespecified list: " + ("; ".join(review["critical_failures_observed"]) or "none identified") + ".")
        bullets(review["missing_or_weak_points"])
        paragraph("Improvement: " + review["suggested_improvement"])
        parts.append(r"\subsubsection*{Internal review history and structural findings}")
        for index, model_review in enumerate(result["reviews"], 1):
            paragraph(f"Review {index}: needs_revision={model_review['needs_revision']}; additional_topics={', '.join(model_review['additional_topics']) or 'none'}.")
            bullets(model_review["issues"] or ["No issue recorded by that internal model review."])
            if model_review["uncertainty_note"]:
                paragraph("Reviewer uncertainty note: " + model_review["uncertainty_note"])
        bullets(result["mechanical_issues"] or ["No remaining structural citation issue was recorded. This does not verify all claims."])

    section("NVIDIA one-question demonstrations")
    paragraph("All three demonstrations used the same ON02 text. Configurations and availability differed, so these are integration observations, not a controlled comparative study. The original JSON and limited qualitative observations are separate from the Gemini ten-question assessment.")
    for item in demos["models"]:
        subsection(item["model"])
        paragraph(f"Status: {item['status']}. Calls: {item['model_call_count']}. Elapsed: {item['elapsed_seconds']:.3f}s. Successful application stages: {', '.join(item['successful_application_stages']) or 'none'}.")
        bullets(item["qualitative_observations"])
        raw = read_json(ROOT / "data/evaluation/runs" / item["run_id"] / "answers/ON02.json")
        if raw["answer"]:
            paragraph("Saved completed answer (Unicode brackets are normalized for printing here; the original formatting mismatch remains an observed finding):")
            parts.append(markdown(raw["answer"].split("\n\nSources consulted from the local guidance collection:")[0], citations=False))
        elif "kimi" in item["model"]:
            paragraph("Initial visible draft, explicitly incomplete because revision failed:")
            draft = next(call["response"] for call in raw["model_calls"] if call["stage"] == "draft" and call["status"] == "ok")
            parts.append(markdown(draft))
            paragraph("The actual structured critique is preserved in the model-call record; its three key findings are summarized above. The final answer field stayed empty.")
        else:
            paragraph("No medical answer was returned; token consumption could not be measured from absent metadata.")

    section("Prompt and contract transparency")
    paragraph("Prompt version: " + prompts.PROMPT_VERSION + ". The model is not sent expected key points or the external rubric. Structured calls also append the exact Pydantic JSON schema. The complete actual system/user messages are preserved per model call in each original answer JSON.")
    for title, text in [("Planning", prompts.PLAN), ("Drafting", prompts.DRAFT), ("Review", prompts.REVIEW), ("Revision", prompts.REVISE)]:
        subsection(title)
        paragraph(text)
    subsection("Decision contracts")
    table(["Contract", "Fields and bound"], [
        ("Plan", "scope; up to six answer-focus/missing-information items; at most four tool requests."),
        ("ToolCall", "One of two fixed tool names plus validated arguments; arbitrary execution is unavailable."),
        ("GuidanceArgs", "One to six topics from the fixed oncology-nutrition topic enumeration."),
        ("TargetArgs", "Finite numeric weight >0 and <=500 kg; host additionally requires that weight in the question."),
        ("Review", "needs_revision; up to eight issues; up to three available extra topics; uncertainty note."),
    ], [40, 119])

    section("Fingerprint and source-read transparency")
    table(["Artifact", "SHA-256 (abbreviated; full value in manifest)"], [("Question set", manifest["dataset_sha256"][:24] + "..."), ("Rubric", manifest["rubric_sha256"][:24] + "..."), ("Guidance", manifest["guidance_sha256"][:24] + "...")] + [(name, digest[:24] + "...") for name, digest in manifest["code_sha256"].items()], [78, 81])
    paragraph("The archived graph module uses the original successful-study invocation. The current graph differs only in preserving node updates on interruption; documented offline replay verifies substantive inputs and successful outputs, with timing fields excluded from comparison. This distinction preserves exact study provenance without presenting replayed responses as fresh inference.")
    table(["Source", "Read level / status"], [(source["id"], source.get("read_level", "Official integration/model documentation consulted; living snapshot dated by access.")) for source in sources], [21, 138])
    paragraph("All nine clinical references are published guidelines or government evidence summaries. No newly cited preprint is used as a clinical authority in this requirement-2 collection. Technical catalog pages establish provider configuration/advertised availability, not medical answer quality. Access date for this collection is 10 October 2026.")

    parts.append(r"\clearpage\phantomsection\addcontentsline{toc}{section}{References}\begin{thebibliography}{T16}")
    for source in sources:
        parts.append(r"\bibitem[" + source["id"] + "]{" + source["id"] + "} " + esc(source["authors"]) + " (" + str(source["year"]) + "). " + r"\textit{" + esc(source["title"]) + "}. " + ("DOI: " + esc(source["doi"]) + ". " if source.get("doi") else "") + r"\url{" + source["url"] + "}. Accessed 10 October 2026. " + esc(source.get("year_note", "")) + " " + esc("Supports: " + source["supports"]))
    parts.append(r"\end{thebibliography}\end{document}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Generated {OUT.relative_to(ROOT)} with ten complete cases, provider demonstrations, and {len(sources)} references.")


if __name__ == "__main__":
    main()
