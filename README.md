# Medical Agentic Copilot

A research prototype for searching medical literature, asking questions about PDFs, and keeping patient context across conversations. The initial focus is oncology nutrition.

Built with Python, LangGraph, FastAPI, React, and SQLite.

## Features

- PubMed searches with article records and abstracts.
- PDF questions with citations to source passages and pages.
- Persistent patient facts, documents, and conversation history.
- Gemma 4 and Kimi K3 model profiles through OpenRouter.
- Saved execution traces and evaluation results.

## Quick start

Requires Python 3.11+, Node.js with npm, and an OpenRouter API key. The commands below use PowerShell.

```powershell
git clone https://github.com/joelmarquez05/Medical_Agentic_Copilot.git
cd Medical_Agentic_Copilot
Copy-Item .env.example .env
```

Edit `.env` and set your key. You can also add a contact email for PubMed:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
NCBI_EMAIL=you@example.com
```

Start the application:

```powershell
python start.py
```

The launcher prepares `.venv`, installs dependencies, builds the frontend, and opens **http://127.0.0.1:8790**. Press `Ctrl+C` to stop.

On macOS or Linux, use `cp .env.example .env` and `python3 start.py`.

## Usage

Choose a model and try **Explore the evidence**, **Ask a document**, or **Continue a patient conversation**. You can edit the example questions or upload your own PDFs.

Sending a question calls the model provider and may incur charges. Conversations and documents are saved locally in `runtime/`. Keep API keys in `.env` or `secrets/`; both are ignored by Git.

Useful launcher options:

```powershell
python start.py --check       # Check local setup without calling models
python start.py --port 8791   # Use another port
python start.py --rebuild     # Rebuild the frontend
```

## Development

After setup, activate the environment and run the tests:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

For frontend development, run `python scripts/run_web.py` from the root, then `npm ci` and `npm run dev` inside `web/` in a separate terminal.

The agents and backend live in `src/medical_copilot/`, the frontend in `web/`, and recorded experiments in `data/evaluation/`. Some historical report scripts depend on files absent from this copy.

Demo patients are fictional, and the prototype is not clinically validated. Use fictional data for testing: relevant context may be sent to the selected model provider.

