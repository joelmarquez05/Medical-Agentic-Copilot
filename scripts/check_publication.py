"""Check Git publication candidates for credentials without displaying secret values.

Run before staging, then use --staged to inspect the actual index before committing.
This local check does not contact GitHub or a model provider.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERNS = (
    re.compile(rb"\bsk-(?:or-v1-|ant-api\d+-|proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\bAIza[A-Za-z0-9_-]{30,}"),
    re.compile(rb"\bnvapi-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\b(?:ghp_|gho_|ghu_|ghs_|github_pat_)[A-Za-z0-9_]{30,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{30,}"),
)
LEGACY_KEY_NAMES = {"api_key.md", "nvidia_api_key.md", "open_router_key.md"}


def git(root: Path, args: list[str], *, git_dir: Path | None = None) -> bytes:
    command = ["git", "-c", "core.quotePath=false"]
    if git_dir is not None:
        command.extend([f"--git-dir={git_dir}", f"--work-tree={root}"])
    result = subprocess.run(command + args, cwd=root, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Git inspection failed; check that Git is installed and the repository is readable.")
    return result.stdout


@contextmanager
def repository(root: Path, *, staged: bool):
    if (root / ".git").exists():
        yield None
    elif staged:
        raise RuntimeError("There is no local Git index yet. Run without --staged first.")
    else:
        # Respect .gitignore without initializing or changing the user's project.
        with tempfile.TemporaryDirectory(prefix="copilot-publication-") as temporary:
            folder = Path(temporary)
            git(folder, ["init", "--quiet"])
            yield folder / ".git"


def known_credentials(root: Path) -> set[bytes]:
    """Read local secrets into memory only; return values, never log them."""
    values = set()
    for name, value in os.environ.items():
        if re.search(r"(?:API_KEY|TOKEN|SECRET|PASSWORD)$", name, re.I) and len(value.strip()) >= 20:
            values.add(value.strip().encode())
    paths = [root / name for name in LEGACY_KEY_NAMES]
    paths.extend(path for path in root.glob(".env*") if path.name != ".env.example" and path.is_file())
    if (root / "secrets").is_dir():
        paths.extend(path for path in (root / "secrets").rglob("*") if path.is_file())
    for path in paths:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8-sig")
        if path.name.startswith(".env"):
            for line in content.splitlines():
                match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
                if match and re.search(r"(?:API_KEY|TOKEN|SECRET|PASSWORD)$", match[1], re.I):
                    value = match[2].split(" #", 1)[0].strip().strip("\"'")
                    if len(value) >= 20:
                        values.add(value.encode())
        else:
            values.update(value.encode() for value in re.findall(r"[A-Za-z0-9_.~+=-]{20,}", content))
    return values


def private_path(name: str) -> bool:
    path = Path(name)
    return (
        "secrets" in path.parts
        or (path.name.startswith(".env") and path.name != ".env.example")
        or path.name in LEGACY_KEY_NAMES
        or path.suffix.lower() in {".key", ".pem", ".p12", ".pfx"}
    )


def inspect(root: Path, *, staged: bool = False) -> tuple[int, list[tuple[str, str]]]:
    credentials = known_credentials(root)
    findings = []
    scanned = 0
    with repository(root, staged=staged) as git_dir:
        # Scan the entire index, including ignored files that were force-added.
        arguments = ["ls-files", "-z", "--cached"]
        if not staged:
            arguments.extend(["--others", "--exclude-standard"])
        names = sorted(set(name.decode("utf-8") for name in git(root, arguments, git_dir=git_dir).split(b"\0") if name))
        for name in names:
            if private_path(name):
                findings.append((name, "private credential file is included"))
            path = root / name
            if staged:
                content = git(root, ["show", ":" + name], git_dir=git_dir)
            else:
                if path.is_symlink():
                    findings.append((name, "symbolic link requires manual review"))
                    continue
                if not path.is_file():
                    continue
                content = path.read_bytes()
            scanned += 1
            if any(value in content for value in credentials):
                findings.append((name, "matches a known local credential"))
            elif any(pattern.search(content) for pattern in TOKEN_PATTERNS):
                findings.append((name, "possible token or private key"))
    return scanned, findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Scan the complete Git index, using staged file contents")
    args = parser.parse_args(argv)
    try:
        scanned, findings = inspect(ROOT, staged=args.staged)
    except Exception as error:
        # Neither Git stderr nor exception text is safe to display here.
        print(f"Publication check could not finish ({type(error).__name__}). Check Git and local file access.", file=sys.stderr)
        return 2
    for name, reason in findings:
        # Paths can themselves contain secrets. Only print a relative path after redaction.
        safe_name = name
        for credential in known_credentials(ROOT):
            safe_name = safe_name.replace(credential.decode(errors="replace"), "[REDACTED]")
        for pattern in TOKEN_PATTERNS:
            safe_name = pattern.sub(b"[REDACTED]", safe_name.encode()).decode(errors="replace")
        print(f"BLOCKED: {safe_name}: {reason}")
    print(f"Checked {scanned} {'staged' if args.staged else 'candidate'} files; {len(findings)} credential finding(s). Secret values are never displayed.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
