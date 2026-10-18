# Medical Agentic Copilot

A student project exploring how a language-model assistant can search medical literature, read PDFs, and keep context across conversations. I started with oncology nutrition and added experiments with broader medical questions.

The aim is to make the answers inspectable: the application keeps retrieved passages, citations, tool activity, and execution records alongside the response. It is an academic prototype without clinical validation. All demo patient cases are fictional.

## What it does

- Searches PubMed and retrieves article records and abstracts.
- Reads uploaded PDFs and cites the passages and PDF pages used in an answer.
- Stores dated patient facts, documents, and previous interactions in SQLite.
- Offers Gemma 4 and Kimi K3 profiles through OpenRouter in the web interface.
- Preserves evaluation runs, including recorded usage, costs, failures, and separate qualitative reviews.

The backend uses Python, LangGraph, and FastAPI. The frontend uses React and Vite.

The initial Git history is a retrospective import of existing work. [HISTORY.md](HISTORY.md) explains its approximate author dates and the preserved experiment records.

## Quick start

You need:

- **Python 3.11 or newer.** The research environment was prepared on Windows with Python 3.12.
- **Node.js with npm** to build the frontend on a fresh checkout. The local development machine uses Node.js 24.
- **An OpenRouter API key** for the models configured in the web application.
- Internet access for dependency installation, PubMed, and model requests.

Run the following commands from the directory containing `start.py`. These examples use PowerShell.

### 1. Configure the environment

```powershell
Copy-Item .env.example .env
```

Open `.env` in your editor and set:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
NCBI_EMAIL=you@example.com
```

The OpenRouter key is shared by both web model profiles. `NCBI_EMAIL` supplies a contact email to the PubMed client. Other keys in the example file are only needed for the corresponding command-line profiles.

Environment variables take precedence over `.env`. Keep credentials in `.env` or the ignored `secrets/` directory, rather than source code or model configuration JSON.

### 2. Launch the application

```powershell
python start.py
```

The launcher creates the project's `.venv`, installs the pinned Python dependencies, builds the frontend, and validates local configuration. Later launches reuse the environment and frontend build until the relevant dependency or source files change.

It then opens:

```text
http://127.0.0.1:8790
```

Press `Ctrl+C` in the terminal to stop the server. The application listens on the local loopback interface.

On macOS or Linux, use `cp .env.example .env` and `python3 start.py` if your Python executable is named `python3`.

### Launcher options

| Command | Purpose |
| --- | --- |
| `python start.py --check` | Prepare and validate the installation without starting the server or calling models. |
| `python start.py --port 8791` | Use a different local port. |
| `python start.py --no-browser` | Start the server without opening a browser. |
| `python start.py --rebuild` | Force a frontend rebuild. |
| `python start.py --reinstall` | Reinstall the Python dependencies. |
| `python start.py --data-dir runtime/sandbox` | Use a separate application state directory. |

The check command may download dependencies or build the frontend. It does not validate a key with the provider or check whether a model is currently available.

## Try the examples

The interface includes editable starting points:

1. **Explore the evidence** prepares a question about the 2021 ESPEN guideline and its PubMed record.
2. **Ask a document** attaches the ESPEN PDF to a new conversation and prepares a question about its contents.
3. **Continue a patient conversation** loads a fictional case so you can ask about its saved records.

Preparing an example does not send a model request. Sending the question runs the agent and may incur provider charges.

The continuity examples use the PROTEOS study, Alpha's initial case, and a later follow-up. They help check whether the assistant separates patient facts from study findings and recalls previous conversations.

You can also upload your own PDFs. Use fictional patient data for testing: context used to answer a question may be sent to the selected model provider.

Conversations, databases, uploaded documents, and execution traces are stored under `runtime/` by default and remain available after a restart. Missing optional demo files produce a startup notice; the core application can still run.

## Command-line use

After the launcher has prepared the environment, activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use `source .venv/bin/activate`. If PowerShell blocks activation, call `.\.venv\Scripts\python.exe` directly in place of `python`.

Show the available commands and validate a model profile without calling a provider:

```powershell
python -m medical_copilot --help
python -m medical_copilot show-config --config config/models/openrouter-gemma-4-reference.json
```

Search PubMed without using a language model:

```powershell
python -m medical_copilot search-pubmed "ESPEN practical guideline clinical nutrition cancer" --max-results 3
```

Ask a literature question and save the execution:

```powershell
python -m medical_copilot ask "What does the 2021 ESPEN practical guideline on clinical nutrition in cancer cover? Cite the retrieved sources." --config config/models/openrouter-gemma-4-reference.json --evidence pubmed --output runtime/question.json
```

`--evidence local` uses the original prototype's local guidance. The CLI and web application have different workflows; I would start with the web interface for PDF questions and patient continuity.

Extract a PDF without calling a model:

```powershell
python scripts/ingest_medical_pdf.py --pdf demo/01_espen_cancer_2021.pdf --output-dir runtime/pdf-example --title "ESPEN 2021"
```

Choose a fresh output directory. Extraction creates a page-aware `document.json`; it does not generate an answer. Scanned PDFs without selectable text need OCR before ingestion.

## Development

To set up Python manually instead of using the launcher:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For frontend development, run the backend in one terminal with the Python environment activated:

```powershell
python scripts/run_web.py
```

In another terminal:

```powershell
cd web
npm ci
npm run dev
```

Vite serves the development interface at `http://127.0.0.1:5173` and proxies `/api` requests to the backend on port `8790`. Run `npm run build` inside `web/` to compile the frontend, or use `python start.py --rebuild` from the project root.

Run the engineering checks from the root with the Python environment activated:

```powershell
python -m pytest
python -m pip check
```

The tests use controlled responses and clients to check software behavior. They do not establish the medical correctness of model-generated answers.

## Repository layout

```text
Medical_Agentic_Copilot/
|-- start.py                # Local setup, validation, and server launcher
|-- src/medical_copilot/    # Agents, tools, memory, and web backend
|-- web/                    # React frontend and Vite configuration
|-- config/                 # Model profiles and execution policies
|-- scripts/                # Queries, experiments, checks, and reports
|-- tests/                  # Engineering tests
|-- demo/                   # Articles and fictional case PDFs
|-- data/                   # Questions, documents, and recorded experiments
|-- .env.example            # Configuration template without credentials
`-- requirements*.txt       # Python dependency files
```

`data/evaluation/` contains question sets and runs for literature retrieval, model comparisons, PDF tasks, and patient memory. Depending on the experiment, records include responses, source passages, manifests, metrics, and separately authored reviews.

Some historical evaluation and report scripts refer to `docs/` files that are absent from this copy. Those workflows are not all reproducible from the current files. Saved results describe their original configuration and execution.

## Before publishing

`.gitignore` excludes local credentials, environment files, runtime state, virtual environments, and frontend build output. The frontend is rebuilt from the committed source and lockfile.

Run the local publication check before staging, then inspect the actual Git index before committing:

```powershell
python scripts/check_publication.py
python scripts/check_publication.py --staged
```

The first command also works before Git has been initialized. `--staged` requires a local Git repository and checks the complete index, including ignored files that were force-added. The check looks for known local credential values and common token formats without printing the values. It is a local safeguard, not a guarantee that every possible secret format will be detected.

## Troubleshooting

- **Missing credentials:** set `OPENROUTER_API_KEY` in `.env`. The launcher checks both configured web profiles locally before starting.
- **Port unavailable:** use `python start.py --port 8791`.
- **Frontend setup failed:** check your Node.js installation, npm, and connection, then run `python start.py --rebuild`.
- **Python dependency setup failed:** check your connection and pinned requirements, then run `python start.py --reinstall`.
- **A model request fails:** inspect the allowed profiles in `src/medical_copilot/web/models.py` and their JSON files under `config/models/`. Model IDs and provider routes are stored explicitly and their availability can change.
- **A PDF produces little text:** check whether it is a scan that needs OCR.

A citation still needs review: retrieving a source does not mean the model interpreted it correctly. Keeping the source passages and execution records makes that review easier.
