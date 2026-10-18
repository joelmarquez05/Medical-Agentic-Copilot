"""Validated provider settings and local credential resolution."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigurationError(RuntimeError):
    """A safe, actionable configuration failure without credential contents."""


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    credential_file: str | None = None
    base_url: str | None = None
    temperature: float | None = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=256, le=32768)
    timeout_seconds: float = Field(default=60, ge=1, le=300)
    max_retries: int = Field(default=1, ge=0, le=3)
    min_interval_seconds: float = Field(default=0, ge=0, le=30)
    rate_limit_retries: int = Field(default=1, ge=0, le=2)
    rate_limit_wait_seconds: float = Field(default=30, ge=0, le=60)
    max_revisions: int = Field(default=1, ge=0, le=3)
    json_repair_attempts: int = Field(default=1, ge=0, le=1)
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_options(self) -> "ModelConfig":
        reserved = {"model", "model_provider", "api_key", "google_api_key", "temperature", "max_tokens", "max_output_tokens", "num_predict", "timeout", "max_retries", "base_url"}
        sensitive = {"authorization", "token", "password", "secret", "headers", "default_headers", "credentials"}

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    normalized = key.lower()
                    if normalized in sensitive or "api_key" in normalized or "secret" in normalized or "password" in normalized:
                        raise ValueError("Credentials must use an environment variable or credential file.")
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        if reserved.intersection(self.provider_options):
            raise ValueError("Provider options cannot override the shared model settings.")
        visit(self.provider_options)
        if self.base_url:
            url = urlsplit(self.base_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("Base URL must be an HTTP(S) endpoint without credentials, query, or fragment.")
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError("The OpenAI-compatible profile requires a base_url.")
        return self

    def public_dict(self) -> dict[str, Any]:
        """The schema contains credential references, never credential values."""
        return self.model_dump(mode="json")


def load_config(path: Path) -> ModelConfig:
    try:
        return ModelConfig.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Could not load model configuration ({type(exc).__name__}).") from None


def resolve_credential(config: ModelConfig, project_root: Path) -> tuple[str | None, str]:
    """Environment values take precedence; the key is never printed."""
    load_dotenv(project_root / ".env", override=False)
    names = [config.api_key_env] if config.api_key_env else []
    if config.provider == "google_genai":
        names += ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    for name in dict.fromkeys(names):
        if name and os.environ.get(name, "").strip():
            return os.environ[name].strip(), "environment"
    if config.credential_file:
        path = project_root / config.credential_file
        try:
            text = path.read_text(encoding="utf-8-sig").strip()
        except OSError:
            raise ConfigurationError("The configured credential file could not be read.") from None
        candidates = list(dict.fromkeys(re.findall(r"[A-Za-z0-9_.~+=-]{20,}", text)))
        if len(candidates) != 1:
            raise ConfigurationError("The credential file must contain exactly one API key, preferably on its own line.")
        return candidates[0], "file"
    if config.provider == "ollama":
        return None, "not_required"
    if config.api_key_env:
        raise ConfigurationError(f"Set {config.api_key_env} or configure a credential_file.")
    return None, "provider_default"
