"""Publication checks inspect Git's index and keep credential values private."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("publication", Path(__file__).resolve().parents[1] / "scripts/check_publication.py")
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def test_uninitialized_project_respects_ignore_rules_without_mutating_it(tmp_path):
    (tmp_path / ".gitignore").write_text("secrets/\n.env\n", encoding="utf-8")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets/provider.key").write_text("synthetic-local-secret-for-test-12345", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=synthetic-local-secret-for-test-12345", encoding="utf-8")
    (tmp_path / "README.md").write_text("A clean project.", encoding="utf-8")
    count, findings = publication.inspect(tmp_path)
    assert count == 2 and not findings
    assert not (tmp_path / ".git").exists()


def test_arbitrary_known_local_key_is_found_without_provider_prefix(tmp_path, monkeypatch, capsys):
    private = "synthetic-arbitrary-local-secret-123456"
    monkeypatch.setenv("PUBLICATION_TEST_API_KEY", private)
    monkeypatch.setattr(publication, "ROOT", tmp_path)
    (tmp_path / "example.json").write_text(private, encoding="utf-8")
    assert publication.main([]) == 1
    captured = capsys.readouterr()
    assert "example.json" in captured.out
    assert private not in captured.out + captured.err


def test_staged_secret_is_found_even_after_working_copy_is_cleaned(tmp_path):
    publication.git(tmp_path, ["init", "--quiet"])
    path = tmp_path / "config.json"
    path.write_text('"sk-or-v1-"', encoding="utf-8")
    # Assemble a synthetic token so the scanner can scan its own tests without false positives.
    path.write_text("sk-or-v1-" + "X" * 40, encoding="utf-8")
    publication.git(tmp_path, ["add", "config.json"])
    path.write_text("clean working copy", encoding="utf-8")
    _, findings = publication.inspect(tmp_path, staged=True)
    assert findings == [("config.json", "possible token or private key")]


def test_force_added_ignored_credentials_are_blocked(tmp_path):
    publication.git(tmp_path, ["init", "--quiet"])
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=", encoding="utf-8")
    publication.git(tmp_path, ["add", "--force", ".env"])
    _, findings = publication.inspect(tmp_path, staged=True)
    assert findings == [(".env", "private credential file is included")]


def test_staged_mode_requires_a_repository(tmp_path):
    with pytest.raises(RuntimeError, match="no local Git index"):
        publication.inspect(tmp_path, staged=True)
