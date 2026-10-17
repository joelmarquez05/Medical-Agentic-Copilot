"""One plain-text chat interface for interchangeable LangChain providers."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from medical_copilot.config import ConfigurationError, ModelConfig, resolve_credential

Schema = TypeVar("Schema", bound=BaseModel)


class ModelError(RuntimeError):
    """Only safe exception summaries cross the provider boundary."""


class ChatBackend(Protocol):
    def invoke(self, messages: list[Any]) -> AIMessage: ...


class Redactor:
    def __init__(self, secrets: list[str] | None = None):
        self._secrets = [value for value in (secrets or []) if value]

    def text(self, value: str) -> str:
        for secret in self._secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    def object(self, value: Any) -> Any:
        return json.loads(self.text(json.dumps(value, ensure_ascii=False, default=str)))


def create_backend(config: ModelConfig, project_root: Path) -> tuple[ChatBackend, Redactor, str]:
    # This run is authorized for the configured model provider, not a trace host.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    credential, credential_source = resolve_credential(config, project_root)
    options = dict(config.provider_options)
    if config.temperature is not None:
        options["temperature"] = config.temperature
    if config.provider == "ollama":
        options["num_predict"] = config.max_tokens
        options["client_kwargs"] = {"timeout": config.timeout_seconds}
    else:
        options.update(max_tokens=config.max_tokens, timeout=config.timeout_seconds, max_retries=config.max_retries)
    if credential:
        options["api_key"] = credential
    if config.base_url:
        options["base_url"] = config.base_url
    provider = "openai" if config.provider == "openai_compatible" else config.provider
    if config.provider == "openai_compatible":
        options.setdefault("use_responses_api", False)
    try:
        backend = init_chat_model(model=config.model, model_provider=provider, **options)
    except ImportError:
        raise ConfigurationError(f"Install the LangChain integration package for provider '{provider}'.") from None
    except Exception as exc:
        raise ConfigurationError(f"Model initialization failed ({type(exc).__name__}); check provider settings.") from None
    return backend, Redactor([credential] if credential else []), credential_source


def message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    # Providers can return text blocks alongside internal reasoning blocks.
    return "\n".join(block.get("text", "") for block in message.content if isinstance(block, dict) and block.get("type") == "text").strip()


def parse_object(text: str) -> dict[str, Any]:
    """Accept an object, optionally wrapped in a single Markdown code fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object.")
    return value


class ModelClient:
    def __init__(self, backend: ChatBackend, config: ModelConfig, redactor: Redactor | None = None):
        self.backend = backend
        self.config = config
        self.redactor = redactor or Redactor()
        self.calls: list[dict[str, Any]] = []
        self._last_request_started = 0.0

    def generate(self, system: str, user: str, stage: str) -> str:
        record: dict[str, Any] = {"stage": stage, "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "system_prompt": system, "user_prompt": user, "attempts": []}
        started = time.perf_counter()
        try:
            for attempt in range(self.config.rate_limit_retries + 1):
                wait = max(0.0, self.config.min_interval_seconds - (time.perf_counter() - self._last_request_started))
                if wait:
                    time.sleep(wait)
                self._last_request_started = time.perf_counter()
                attempt_record = {"number": attempt + 1, "pre_call_wait_seconds": round(wait, 3)}
                try:
                    message = self.backend.invoke([SystemMessage(content=system), HumanMessage(content=user)])
                    attempt_record["status"] = "ok"
                    record["attempts"].append(attempt_record)
                    break
                except Exception as exc:
                    attempt_record.update(status="error", error_type=type(exc).__name__)
                    is_rate_limit = "ratelimit" in type(exc).__name__.lower() or getattr(exc, "status_code", None) == 429 or getattr(exc, "code", None) == 429
                    if is_rate_limit and attempt < self.config.rate_limit_retries:
                        attempt_record["retry_wait_seconds"] = self.config.rate_limit_wait_seconds
                        record["attempts"].append(attempt_record)
                        time.sleep(self.config.rate_limit_wait_seconds)
                        continue
                    record["attempts"].append(attempt_record)
                    raise
            text = message_text(message)
            metadata = message.response_metadata or {}
            record.update(status="ok", response=text, usage=message.usage_metadata or {}, response_metadata={key: metadata[key] for key in ["finish_reason", "model_name", "model_version", "model_provider"] if key in metadata})
            reason = str(metadata.get("finish_reason", "")).upper()
            if not text or "MAX_TOKENS" in reason or reason == "LENGTH":
                record["status"] = "unusable_response"
                raise ModelError("The provider returned an empty or truncated response; check max_tokens and model settings.")
            return self.redactor.text(text)
        except ModelError:
            raise
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__)
            raise ModelError(f"Model request failed ({type(exc).__name__}); check authentication, model availability, quota, and network access.") from None
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.calls.append(self.redactor.object(record))

    def structured(self, schema: type[Schema], system: str, user: str, stage: str) -> Schema:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        instruction = system + "\nReturn only one JSON object conforming to this schema. Do not use Markdown or explanatory prose.\n" + schema_json
        raw = self.generate(instruction, user, stage)
        for attempt in range(self.config.json_repair_attempts + 1):
            try:
                return schema.model_validate(parse_object(raw))
            except (ValueError, ValidationError):
                if attempt >= self.config.json_repair_attempts:
                    raise ModelError(f"Invalid structured response at '{stage}' after the bounded repair attempt.") from None
                raw = self.generate(instruction, "Repair the JSON format and schema only, without adding new clinical claims.\nOriginal output:\n" + raw, stage + "_json_repair")
        raise AssertionError("Unreachable bounded parser state.")
