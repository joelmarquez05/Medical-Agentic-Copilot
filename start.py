"""Prepare and run Medical Agentic Copilot locally: python start.py --help."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / ".venv"
PYTHON = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DEPENDENCIES = ("requirements.txt", "requirements.lock.txt", "requirements-web.txt", "pyproject.toml")
DEMO_FILES = (
    "demo/cases/patients.json",
    "demo/01_espen_cancer_2021.pdf",
    "demo/02_proteos_protein_2023.pdf",
    "demo/cases/alpha_initial.pdf",
    "demo/cases/beta_initial.pdf",
    "demo/cases/alpha_followup.pdf",
)


class StartupError(RuntimeError):
    """An actionable message that contains no subprocess or provider output."""


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Use a port between 1 and 65535.")
    return port


def fingerprint(paths: list[Path], *, salt: str = "") -> str:
    digest = hashlib.sha256(salt.encode())
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_fingerprint(marker: Path) -> str | None:
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return data.get("fingerprint") if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def save_fingerprint(marker: Path, value: str) -> None:
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"fingerprint": value}) + "\n", encoding="utf-8")
    temporary.replace(marker)


def run_setup(command: list[str], *, cwd: Path, failure: str) -> None:
    # Installer output can contain authenticated index URLs. Keep it off the console.
    result = subprocess.run(command, cwd=cwd, capture_output=True, check=False)
    if result.returncode:
        raise StartupError(f"{failure} (exit code {result.returncode}).")


def ensure_environment(force: bool = False) -> None:
    if not PYTHON.is_file():
        print("[setup] Creating .venv", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENVIRONMENT)
    marker = ENVIRONMENT / ".copilot-dependencies.json"
    expected = fingerprint(
        [ROOT / name for name in DEPENDENCIES],
        salt=f"{ROOT}:{sys.version_info.major}.{sys.version_info.minor}",
    )
    if force or read_fingerprint(marker) != expected:
        print("[setup] Installing Python dependencies; this may take a few minutes", flush=True)
        run_setup(
            [str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements.txt"],
            cwd=ROOT,
            failure="Dependency installation failed. Check your connection and requirements.txt",
        )
        run_setup(
            [str(PYTHON), "-m", "pip", "check"], cwd=ROOT,
            failure="Dependency validation failed. Run .venv's Python with -m pip check",
        )
        save_fingerprint(marker, expected)


def frontend_fingerprint() -> str:
    files = [ROOT / "web" / name for name in ("package.json", "package-lock.json", "index.html", "vite.config.js")]
    files.extend(path for path in (ROOT / "web/src").rglob("*") if path.is_file())
    if (ROOT / "web/public").is_dir():
        files.extend(path for path in (ROOT / "web/public").rglob("*") if path.is_file())
    return fingerprint(files)


def ensure_frontend(rebuild: bool = False) -> None:
    marker = ROOT / "web/dist/.copilot-build.json"
    expected = frontend_fingerprint()
    if not rebuild and (ROOT / "web/dist/index.html").is_file() and read_fingerprint(marker) == expected:
        return
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm:
        raise StartupError("Install Node.js with npm to build the frontend, then run python start.py again.")
    print("[setup] Building the frontend", flush=True)
    for arguments in (["ci"], ["run", "build"]):
        run_setup(
            [npm, *arguments], cwd=ROOT / "web",
            failure="Frontend setup failed. Check Node.js, npm and web/package-lock.json",
        )
    save_fingerprint(marker, expected)


def check_installation() -> None:
    """Validate local files and credentials without contacting a model provider."""
    import fastapi  # noqa: F401
    import pymupdf  # noqa: F401
    import uvicorn  # noqa: F401
    from medical_copilot.config import ConfigurationError, load_config, resolve_credential
    from medical_copilot.pubmed import load_pubmed_config
    from medical_copilot.web.app import ROOT as application_root
    from medical_copilot.web.models import PROFILES

    if application_root != ROOT:
        raise StartupError("Another project copy is installed. Run python start.py --reinstall from this copy.")
    for name, profile in PROFILES.items():
        settings = load_config(ROOT / profile["path"])
        try:
            credential, _ = resolve_credential(settings, ROOT)
        except ConfigurationError:
            raise StartupError(f"Set {settings.api_key_env} in .env to use the {name} profile.") from None
        if not credential:
            raise StartupError(f"Set {settings.api_key_env} in .env to use the {name} profile.")
    load_pubmed_config(ROOT / "config/pubmed-clinical.json")
    if not (ROOT / "web/dist/index.html").is_file():
        raise StartupError("The frontend build is missing. Run python start.py --rebuild.")
    missing = [name for name in DEMO_FILES if not (ROOT / name).is_file()]
    if missing:
        print(f"[check] {len(missing)} optional demo file(s) missing; some prepared examples are unavailable.", flush=True)
    print("[check] Dependencies, frontend, configuration and local credentials: OK", flush=True)


def available_port(port: int) -> None:
    with socket.socket() as connection:
        try:
            connection.bind(("127.0.0.1", port))
        except OSError:
            raise StartupError(f"Port {port} is unavailable. Choose another with --port.") from None


def open_when_ready(url: str, stopped: threading.Event) -> None:
    deadline = time.monotonic() + 45
    while not stopped.is_set() and time.monotonic() < deadline:
        try:
            with urlopen(url + "/api/bootstrap", timeout=1) as response:
                if response.status == 200 and not stopped.is_set():
                    webbrowser.open(url)
                    return
        except OSError:
            pass
        stopped.wait(.3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=port_number, default=8790, help="Local port (default: 8790)")
    parser.add_argument("--no-browser", action="store_true", help="Start without opening a browser")
    parser.add_argument("--check", action="store_true", help="Prepare and validate the installation without starting the server or calling models")
    parser.add_argument("--reinstall", action="store_true", help="Reinstall Python dependencies")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the frontend even when its sources are unchanged")
    parser.add_argument("--data-dir", type=Path, help="Application state directory (default: runtime/ in this project)")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        raise StartupError("Python 3.11 or newer is required.")

    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.chdir(ROOT)
    if not args.check:
        available_port(args.port)
    ensure_environment(args.reinstall)
    if Path(sys.prefix).resolve() != ENVIRONMENT.resolve():
        # The child sees the validated fingerprint and does not repeat installation.
        command = [str(PYTHON), "-X", "utf8", str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        command = [part for part in command if part != "--reinstall"]
        return subprocess.call(command, cwd=ROOT)

    sys.path.insert(0, str(ROOT / "src"))
    ensure_frontend(args.rebuild)
    check_installation()
    if args.check:
        print("[check] Complete. No model requests were sent.", flush=True)
        return 0

    directory = args.data_dir.expanduser().resolve() if args.data_dir else ROOT / "runtime"
    os.environ["MEDICAL_COPILOT_DATA_DIR"] = str(directory)
    from medical_copilot.web.app import create_app
    import uvicorn

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nMedical Agentic Copilot\n  URL:   {url}\n  State: {directory}\n\nPress Ctrl+C to stop.\n", flush=True)
    stopped = threading.Event()
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url, stopped), daemon=True).start()
    try:
        uvicorn.run(create_app(directory), host="127.0.0.1", port=args.port, access_log=False)
    finally:
        stopped.set()
    return 0


def entrypoint() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        return 130
    except StartupError as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        # Third-party exceptions can contain credentials or authenticated URLs.
        print(f"Startup failed ({type(error).__name__}). Check .env, dependencies and local files; run python start.py --check.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
