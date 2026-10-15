from pathlib import Path

import pytest

from medical_copilot import clinical_model_comparison as comparison
from medical_copilot.evaluation import file_hash, write_json


def test_rotation_covers_every_model_case_once_without_changing_questions():
    questions = [f"BM{n:02d}" for n in range(1, 31)]
    original = list(questions)
    order = comparison.episode_order(questions)
    assert len(order) == len(set(order)) == 120
    assert {pair for pair in order} == {(q, m) for q in questions for m in comparison.MODELS}
    assert questions == original
    assert [q for q, _ in order[::4]] == questions
    assert [m for _, m in order[::4]][:5] == list(comparison.MODELS) + [comparison.MODELS[0]]


def test_empty_order_has_no_episodes():
    assert comparison.episode_order([]) == []


def test_saved_charges_and_unknown_reserves_survive_a_lagging_ledger(tmp_path):
    directory = tmp_path / "data/evaluation/comparison-runs/previous"
    write_json(directory / "manifest.json", {"requirement": 5, "protocol_origin": comparison.PROTOCOL})
    write_json(directory / "glm-5.3/usage-audit.json", {"records": [
        {"request_sent": True, "cost_usd": 0.12, "estimated_maximum_usd": 0.5},
        {"request_sent": True, "cost_usd": None, "estimated_maximum_usd": 0.3}]})
    assert comparison.prior_request_totals(tmp_path) == {
        "known_cost_usd": 0.12, "unknown_request_reserve_usd": 0.3, "saved_requests": 2}


def test_old_comparison_protocol_is_not_double_charged(tmp_path):
    directory = tmp_path / "data/evaluation/comparison-runs/old"
    write_json(directory / "manifest.json", {"requirement": 5, "protocol_origin": "historical"})
    write_json(directory / "glm-5.3/usage-audit.json", {"records": [
        {"request_sent": True, "cost_usd": 4.0, "estimated_maximum_usd": 4.0}]})
    assert comparison.prior_request_totals(tmp_path)["saved_requests"] == 0


def prerequisite(tmp_path, monkeypatch):
    write_json(tmp_path / comparison.BENCHMARK / "manifest.json", {"synthetic": True})
    protocol = tmp_path / comparison.PROTOCOL
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text("Synthetic frozen protocol", encoding="utf-8")
    monkeypatch.setattr(comparison, "profile_hashes", lambda root: {"fixture": "hash"})
    directory = tmp_path / "probes"
    manifest = {"stage": "probes", "finished_at_utc": "2026-10-11T00:00:00Z",
                "protocol_sha256": file_hash(protocol), "profile_hashes": {"fixture": "hash"},
                "source_hashes": {"fixture.py": "digest"},
                "selection_manifest_sha256": file_hash(tmp_path / comparison.BENCHMARK / "manifest.json")}
    write_json(directory / "manifest.json", manifest)
    for model in comparison.MODELS:
        write_json(directory / model / "answers/PILOT-01.json", {
            "status": "completed", "output_origin": "model", "transport_formats": [{"response_format": "json_schema"}]})
    return directory, manifest


def test_every_model_must_complete_matching_native_probe(tmp_path, monkeypatch):
    directory, _ = prerequisite(tmp_path, monkeypatch)
    comparison.validate_prerequisite(tmp_path, directory, "probes", {"fixture.py": "digest"})


@pytest.mark.parametrize("change", ["wrong_source", "failed_model", "non_native_format", "unfinished", "changed_protocol"])
def test_prerequisite_rejects_configuration_or_completion_changes(tmp_path, monkeypatch, change):
    directory, manifest = prerequisite(tmp_path, monkeypatch)
    if change == "wrong_source":
        manifest["source_hashes"] = {"fixture.py": "changed"}
    elif change == "unfinished":
        manifest.pop("finished_at_utc")
    elif change == "changed_protocol":
        (tmp_path / comparison.PROTOCOL).write_text("Changed protocol", encoding="utf-8")
    elif change == "failed_model":
        write_json(directory / "kimi-k3/answers/PILOT-01.json", {
            "status": "application_nonanswer", "output_origin": "application"})
    elif change == "non_native_format":
        write_json(directory / "kimi-k3/answers/PILOT-01.json", {
            "status": "completed", "output_origin": "model", "transport_formats": [{"response_format": "json_object"}]})
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        comparison.validate_prerequisite(tmp_path, directory, "probes", {"fixture.py": "digest"})


def test_a_fresh_stage_refuses_existing_directory_before_any_network(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError):
        comparison.run_stage(tmp_path, existing, "probes")


def test_missing_prerequisite_is_explicit(tmp_path):
    with pytest.raises(ValueError):
        comparison.validate_prerequisite(tmp_path, None, "probes", {})
