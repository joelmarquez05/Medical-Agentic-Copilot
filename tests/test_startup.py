"""Launcher boundaries: stale builds, private diagnostics, and optional demos."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("copilot_start", Path(__file__).resolve().parents[1] / "start.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def frontend(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    (web / "dist").mkdir()
    for name in ("package.json", "package-lock.json", "index.html", "vite.config.js"):
        (web / name).write_text("initial", encoding="utf-8")
    (web / "src/main.jsx").write_text("initial source", encoding="utf-8")
    (web / "dist/index.html").write_text("old build", encoding="utf-8")
    return web


def test_changed_sources_trigger_rebuild_and_then_reuse_it(tmp_path, monkeypatch):
    web = frontend(tmp_path, monkeypatch)
    marker = web / "dist/.copilot-build.json"
    launcher.save_fingerprint(marker, launcher.frontend_fingerprint())
    calls = []
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "npm")
    monkeypatch.setattr(launcher, "run_setup", lambda command, **kwargs: calls.append(command))
    launcher.ensure_frontend()
    assert not calls
    (web / "src/main.jsx").write_text("changed source", encoding="utf-8")
    launcher.ensure_frontend()
    assert [command[1:] for command in calls] == [["ci"], ["run", "build"]]
    launcher.ensure_frontend()
    assert len(calls) == 2


def test_failed_build_does_not_mark_stale_output_as_current(tmp_path, monkeypatch):
    web = frontend(tmp_path, monkeypatch)
    marker = web / "dist/.copilot-build.json"
    launcher.save_fingerprint(marker, launcher.frontend_fingerprint())
    previous = marker.read_bytes()
    (web / "src/main.jsx").write_text("changed source", encoding="utf-8")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "npm")
    def fail(*args, **kwargs):
        raise launcher.StartupError("Build failed.")
    monkeypatch.setattr(launcher, "run_setup", fail)
    with pytest.raises(launcher.StartupError):
        launcher.ensure_frontend()
    assert marker.read_bytes() == previous


def test_installer_output_cannot_disclose_credentials(tmp_path, capsys):
    private = "private-installer-value-for-test-only"
    with pytest.raises(launcher.StartupError) as error:
        launcher.run_setup(
            [sys.executable, "-c", f"import sys; print('{private}'); sys.stderr.write('{private}'); sys.exit(1)"],
            cwd=tmp_path, failure="Installation failed",
        )
    captured = capsys.readouterr()
    assert private not in str(error.value) + captured.out + captured.err
    assert "exit code 1" in str(error.value)


def test_unexpected_startup_error_never_prints_its_private_message(monkeypatch, capsys):
    private = "private-provider-value-for-test-only"
    def fail():
        raise ValueError(private)
    monkeypatch.setattr(launcher, "main", fail)
    assert launcher.entrypoint() == 1
    captured = capsys.readouterr()
    assert "ValueError" in captured.err
    assert private not in captured.out + captured.err


def test_missing_demo_does_not_prevent_core_installation_check(monkeypatch, capsys):
    from medical_copilot import config
    monkeypatch.setattr(launcher, "DEMO_FILES", ("demo/absent-optional-test-file.pdf",))
    monkeypatch.setattr(config, "resolve_credential", lambda *args: ("synthetic-test-value", "environment"))
    launcher.check_installation()
    assert "optional demo" in capsys.readouterr().out


def test_credential_resolution_failure_has_safe_actionable_message(monkeypatch):
    from medical_copilot import config
    private = "private-credential-error-for-test-only"
    def fail(*args):
        raise config.ConfigurationError(private)
    monkeypatch.setattr(config, "resolve_credential", fail)
    with pytest.raises(launcher.StartupError, match="OPENROUTER_API_KEY") as error:
        launcher.check_installation()
    assert private not in str(error.value)


@pytest.mark.parametrize("port", ["0", "65536", "-1"])
def test_invalid_port_fails_before_setup(port):
    result = subprocess.run([sys.executable, str(launcher.ROOT / "start.py"), "--port", port], capture_output=True, text=True)
    assert result.returncode == 2
    assert "[setup]" not in result.stdout
