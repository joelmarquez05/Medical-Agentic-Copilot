"""Offline checks for request budgets, usage reserves, and private diagnostics."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from medical_copilot.usage import BaselineAudit
from medical_copilot.usage import BudgetLimitExceeded, RequestBudget
from medical_copilot.config import load_config
from medical_copilot.models import Redactor

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/models/gemma.json"


class Backend:
    def __init__(self, message=None, error=None):
        self.message, self.error, self.calls = message, error, 0

    def invoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return self.message


def audit_for(tmp_path, backend, budget=None):
    return BaselineAudit(backend, load_config(PROFILE), budget or RequestBudget(6, 3), Redactor(), tmp_path / "audit.json")


def test_missing_usage_retains_full_allowance_instead_of_asserting_zero_cost(tmp_path):
    audit = audit_for(tmp_path, Backend(AIMessage(content="Synthetic response.")))
    audit.invoke([HumanMessage(content="Synthetic prompt.")])
    row = audit.records[0]
    assert row["cost_usd"] is None
    assert audit.budget.spent_usd == 3
    assert audit.budget.unreported_charge_reserve_usd == row["estimated_maximum_usd"]


def test_reported_numeric_cost_wins_and_private_reasoning_is_not_saved(tmp_path):
    message = AIMessage(content="Synthetic response.", usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                        response_metadata={"token_usage": {"cost": 0.001, "completion_tokens_details": {"reasoning_tokens": 12},
                                                           "reasoning_text": "PRIVATE_REASONING_SENTINEL"}})
    audit = audit_for(tmp_path, Backend(message))
    audit.invoke([HumanMessage(content="Synthetic prompt.")])
    assert audit.records[0]["cost_usd"] == 0.001
    assert audit.records[0]["provider_usage"]["reasoning_tokens"] == 12
    assert "PRIVATE_REASONING_SENTINEL" not in (tmp_path / "audit.json").read_text(encoding="utf-8")


def test_failed_request_is_reserved_without_recording_raw_exception_body(tmp_path):
    audit = audit_for(tmp_path, Backend(error=RuntimeError("FAKE_KEY_MUST_NOT_LEAK")))
    with pytest.raises(RuntimeError):
        audit.invoke([HumanMessage(content="Synthetic prompt.")])
    assert audit.budget.unreported_charge_reserve_usd > 0
    assert "FAKE_KEY" not in json.dumps(audit.records)


def test_budget_block_happens_before_provider_call(tmp_path):
    backend = Backend(AIMessage(content="Synthetic response."))
    audit = audit_for(tmp_path, backend, RequestBudget(6, 6))
    with pytest.raises(BudgetLimitExceeded):
        audit.invoke([HumanMessage(content="Synthetic prompt.")])
    assert backend.calls == 0
