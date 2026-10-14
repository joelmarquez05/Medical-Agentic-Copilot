import json
import shutil
from pathlib import Path

import pytest

from medical_copilot import clinical_benchmark as benchmark
from medical_copilot.evaluation import file_hash, read_json, write_json


def row(question, answer="A"):
    options = {key: "Synthetic option " + key for key in "ABCD"}
    return {"question": question, "options": options, "answer_idx": answer, "answer": options[answer]}


def test_selection_excludes_seen_visual_and_duplicate_but_not_long_text():
    long = "Synthetic long question. " * 110
    rows = [row("Already seen."), row("Refer to the image."), row("Duplicate stem."),
            row("Duplicate   stem."), row(long), row("New self-contained stem.")]
    selected = benchmark.select_rows(rows, [benchmark.question_hash("Already seen.")], count=3)
    assert set(selected["selected_source_row_indices"]) == {2, 4, 5}
    assert {r["source_row_index"] for r in selected["excluded_rows"]} == {0, 1, 3}


def test_sampling_is_reproducible_without_changing_source():
    rows = [row(f"Synthetic clinical test stem {n}.", "ABCD"[n % 4]) for n in range(100)]
    original = json.dumps(rows)
    first = benchmark.select_rows(rows, [], seed=123, count=30)
    assert first == benchmark.select_rows(rows, [], seed=123, count=30)
    assert first["selected_source_row_indices"] == first["ordered_candidate_indices"][:30]
    assert len(first["ordered_candidate_indices"]) == 50
    assert first != benchmark.select_rows(rows, [], seed=124, count=30)
    assert json.dumps(rows) == original


@pytest.mark.parametrize("change", ["answer", "answer_idx", "options"])
def test_inconsistent_released_keys_are_rejected(change):
    sample = row("Synthetic stem.")
    sample[change] = {"A": "Only one option."} if change == "options" else "INVALID"
    with pytest.raises(ValueError):
        benchmark.select_rows([sample], [], count=1)


def test_history_extraction_ignores_answer_content():
    data = [{"question": {"question_id": "X", "question": "Actual case."}, "response": {"answer": "Not an input question."}},
            {"questions": [{"question": "Another actual case."}]}]
    assert benchmark.extract_questions(data) == ["Actual case.", "Another actual case."]


@pytest.mark.parametrize("description", ["An MRI of the head and neck is shown.", "An electrocardiogram is shown.", "A CT scan is shown.", "The result is shown in the exhibit with a green arrow."])
def test_explicit_missing_scan_is_excluded_without_rejecting_described_findings(description):
    rows = [row(description), row("An MRI shows described spinal compression without an accompanying image reference.")]
    # Use a self-contained text description without the literal word image.
    rows[1]["question"] = "An MRI shows described spinal compression and normal spinal cord signal."
    selected = benchmark.select_rows(rows, [], count=1)
    assert selected["selected_source_row_indices"] == [1]
    assert selected["excluded_rows"][0]["source_row_index"] == 0


def test_original_key_cannot_be_replaced_even_with_updated_file_hash(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / benchmark.BENCHMARK
    target = tmp_path / benchmark.BENCHMARK
    shutil.copytree(source, target)
    key = read_json(target / "answer-key.json")
    entry = key["keys"][0]
    questions = read_json(target / "questions.json")["questions"]
    replacement = next(letter for letter in "ABCD" if letter != entry["answer_idx"])
    entry["answer_idx"], entry["answer"] = replacement, questions[0]["options"][replacement]
    write_json(target / "answer-key.json", key)
    manifest = read_json(target / "manifest.json")
    manifest["answer_key_sha256"] = file_hash(target / "answer-key.json")
    write_json(target / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Original wording/options/key changed"):
        benchmark.validate_benchmark(tmp_path)


def test_wilson_bounds_and_empty_denominator():
    assert benchmark.wilson(0, 0) is None
    assert benchmark.wilson(30, 30)[0] == pytest.approx(0.8864866068)
    assert benchmark.wilson(0, 30)[1] == pytest.approx(0.1135133932)
    assert benchmark.wilson(1, 2) == pytest.approx([0.0945312057, 0.9054687943])


def test_nonanswer_and_unrecorded_cases_remain_in_primary_denominator(tmp_path):
    write_json(tmp_path / "manifest.json", {"stage": "benchmark", "question_ids": ["ONE", "TWO", "THREE"]})
    write_json(tmp_path / "answer-key.json", {"keys": [{"question_id": name, "key_type": "released_medqa_label", "answer_idx": "A"} for name in ["ONE", "TWO", "THREE"]]})
    for name, status, origin, selected in [("ONE", "completed", "model", "A"), ("TWO", "operationally_unanswered", "application", None)]:
        write_json(tmp_path / "answers" / (name + ".json"), {"question": {"question_id": name}, "status": status,
            "response": {"selected_option": selected, "citations": []}, "output_origin": origin, "final_attempts": 0,
            "response_audit": [], "trace": [], "literature": {"events": [], "http_requests": []}})
    # A separate qualitative assertion cannot influence the exact-key score.
    write_json(tmp_path / "answer-assessments.json", {"claim": "Every synthetic answer is excellent."})
    summary = benchmark.summarize(tmp_path)
    assert summary["correct"] == 1 and summary["valid_completed"] == 1
    assert summary["objective_score_planned"] == pytest.approx(1/3)
    assert summary["accuracy_among_valid"] == 1 and summary["unrecorded"] == 1
    assert summary["application_fallbacks"] == 1
    assert read_json(tmp_path / "scores.json")["scores"][1]["correct"] is None
