from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from langchain_openai import ChatOpenAI


def load_llm_config(path: str) -> Dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _find_model(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    models = config.get("models", [])
    if not isinstance(models, list):
        raise ValueError("llm_config.json must contain a 'models' list")

    for model in models:
        if isinstance(model, dict) and model.get("name") == model_name:
            return model

    available = ", ".join(str(m.get("name")) for m in models if isinstance(m, dict))
    raise ValueError(f"Model '{model_name}' not found in config. Available: {available}")


def get_llm(model_name: str, config: Dict[str, Any]) -> ChatOpenAI:
    model_cfg = _find_model(model_name, config)
    provider = str(model_cfg.get("provider", "openai")).lower()
    if provider != "openai":
        raise ValueError(f"Unsupported provider for this pipeline: {provider}")

    return ChatOpenAI(
        model=model_cfg.get("model", model_name),
        temperature=float(model_cfg.get("temperature", 0.0)),
        api_key=model_cfg.get("api_key"),
        base_url=model_cfg.get("base_url"),
        timeout=model_cfg.get("timeout", 120),
        max_retries=int(model_cfg.get("max_retries", 3)),
    )
