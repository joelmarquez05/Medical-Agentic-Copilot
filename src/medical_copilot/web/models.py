"""Server-owned, allowlisted model profiles for the interactive workspace."""
from typing import Literal

ModelId = Literal["gemma", "kimi-k3"]
DEFAULT_MODEL: ModelId = "gemma"
PROFILES = {
    "gemma": {
        "path": "config/models/openrouter-gemma-4-reference.json",
        "name": "Gemma 4", "detail": "26B A4B · Reasoning enabled",
    },
    "kimi-k3": {
        "path": "config/models/requirement-5-20261011/kimi-k3.json",
        "name": "Kimi K3", "detail": "Reasoning high",
    },
}


def public_model(model_id):
    if model_id not in PROFILES:
        raise ValueError("Choose Gemma 4 or Kimi K3.")
    profile = PROFILES[model_id]
    return {"id": model_id, "name": profile["name"], "detail": profile["detail"],
            "provider": "OpenRouter", "status": "configured"}


def profile_path(model_id):
    public_model(model_id)
    return PROFILES[model_id]["path"]
