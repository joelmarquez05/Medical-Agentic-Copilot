"""Offline engineering checks. Scripted responses are never evaluation data."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from medical_copilot.config import ConfigurationError, ModelConfig, load_config, resolve_credential
from medical_copilot.models import ModelClient, ModelError, Redactor, create_backend, message_text

ROOT = Path(__file__).resolve().parents[1]


class ScriptedBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        text = json.dumps(output) if isinstance(output, dict) else output
        return AIMessage(content=text, usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}, response_metadata={"finish_reason": "STOP"})


def config(**kwargs):
    return ModelConfig(provider="google_genai", model="test-only-model", **kwargs)












def test_environment_key_takes_precedence_without_echo(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_COPILOT_KEY", "environment-value-for-unit-test")
    (tmp_path / "api_key.md").write_text("file-value-for-unit-test-123", encoding="utf-8")
    settings = config(api_key_env="TEST_COPILOT_KEY", credential_file="api_key.md")
    key, source = resolve_credential(settings, tmp_path)
    assert key == "environment-value-for-unit-test" and source == "environment"
    assert key not in json.dumps(settings.public_dict())


@pytest.mark.parametrize("options", [{"api_key": "do-not-echo-this"}, {"nested": {"authorization": "do-not-echo-this"}}, {"max_tokens": 1}])
def test_literal_credentials_or_shared_overrides_fail_safely(tmp_path, options):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"provider": "google_genai", "model": "test", "provider_options": options}), encoding="utf-8")
    with pytest.raises(ConfigurationError) as captured:
        load_config(path)
    assert "do-not-echo-this" not in str(captured.value)


def test_credential_url_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai_compatible", model="test", base_url="https://example.com/v1?key=secret")
























def test_text_blocks_exclude_reasoning():
    message = AIMessage(content=[{"type": "reasoning", "reasoning": "Do not expose this."}, {"type": "text", "text": "Visible answer."}])
    assert message_text(message) == "Visible answer."


def test_openai_compatible_factory_routing(monkeypatch, tmp_path):
    recorded = {}
    def fake_factory(**kwargs):
        recorded.update(kwargs)
        return ScriptedBackend([])
    monkeypatch.setattr("medical_copilot.models.init_chat_model", fake_factory)
    monkeypatch.setenv("TEST_CF_KEY", "fake-cloud-token-for-unit-test")
    settings = ModelConfig(provider="openai_compatible", model="example-model", api_key_env="TEST_CF_KEY", base_url="https://example.com/ai/v1")
    _, redactor, source = create_backend(settings, tmp_path)
    assert recorded["model_provider"] == "openai"
    assert recorded["base_url"] == settings.base_url
    assert recorded["use_responses_api"] is False
    assert source == "environment"
    assert "fake-cloud-token-for-unit-test" not in redactor.text(json.dumps(recorded))


def test_rate_limit_retry_is_bounded_and_auditable():
    class SyntheticRateLimitError(RuntimeError):
        pass
    client = ModelClient(ScriptedBackend([SyntheticRateLimitError("unit test"), "OK"]), config(rate_limit_retries=1, rate_limit_wait_seconds=0))
    assert client.generate("Test", "Test", "test_only") == "OK"
    assert [attempt["status"] for attempt in client.calls[0]["attempts"]] == ["error", "ok"]
    client = ModelClient(ScriptedBackend([SyntheticRateLimitError("unit test"), SyntheticRateLimitError("unit test")]), config(rate_limit_retries=1, rate_limit_wait_seconds=0))
    with pytest.raises(ModelError):
        client.generate("Test", "Test", "test_only")
    assert len(client.calls[0]["attempts"]) == 2
