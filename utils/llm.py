import json
import os
import time
from typing import Dict, Any, Optional


class _LLMDebugWrapper:
    def __init__(self, llm: Any):
        self._llm = llm

    def invoke(self, messages, **kwargs):
        print("\n" + "=" * 60 + "\n[LLM Request]")
        for m in messages:
            role = getattr(m, "type", type(m).__name__)
            content = getattr(m, "content", str(m))
            print(f"  {role}: {content}")
        print()
        result = _invoke_with_backoff(self._llm, messages, **kwargs)
        out = getattr(result, "content", str(result))
        print("[LLM Response]\n  ", out)
        print("=" * 60 + "\n")
        return result

    def __getattr__(self, name):
        return getattr(self._llm, name)


class _LLMBackoffWrapper:

    def __init__(self, llm: Any):
        self._llm = llm

    def invoke(self, messages, **kwargs):
        return _invoke_with_backoff(self._llm, messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self._llm, name)


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    keys = (
        "rate limit",
        "ratelimit",
        "too many requests",
        "429",
        "tokens per min",
        "requests per min",
    )
    if any(k in text for k in keys):
        return True
    cls_name = exc.__class__.__name__.lower()
    return "ratelimit" in cls_name


def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    delay = base * (2 ** max(0, attempt - 1))
    return min(delay, cap)


def _invoke_with_backoff(llm: Any, messages: Any, **kwargs: Any) -> Any:
    max_attempts = 5
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(messages, **kwargs)
        except Exception as e:
            last_exc = e
            if not _is_rate_limit_error(e) or attempt >= max_attempts:
                raise
            time.sleep(_backoff_delay(attempt))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM invocation failed")


def load_llm_config(config_path: str = "./llm_config.json") -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _cfg_display_name(model_cfg: Dict[str, Any]) -> Optional[str]:
    name = model_cfg.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    model = model_cfg.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def list_configured_models(config: Optional[Dict[str, Any]] = None) -> list[str]:
    if config is None:
        config = load_llm_config()
    return [
        str(_cfg_display_name(m))
        for m in config.get("models", [])
        if isinstance(m, dict) and _cfg_display_name(m)
    ]


def _create_llm(model_cfg: Dict[str, Any]) -> Any:
    provider = (model_cfg.get("provider") or "").strip().lower()
    model = model_cfg["model"]
    temperature = model_cfg.get("temperature", 0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = model_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        base_url = model_cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    if provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = (
            model_cfg.get("api_key")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )

    if provider in ("anthropic", "claude"):
        from langchain_anthropic import ChatAnthropic
        api_key = model_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider: {provider!r}")


def get_llm(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    debug: Optional[bool] = None,
) -> Any:
    if config is None:
        config = load_llm_config()

    requested = (model_name or "").strip()
    models = list_configured_models(config)

    for model_cfg in config.get("models", []):
        if not isinstance(model_cfg, dict):
            continue
        display = _cfg_display_name(model_cfg)
        if display == requested or model_cfg.get("model") == requested:
            llm = _create_llm(model_cfg)
            wrapped = _LLMBackoffWrapper(llm)
            return _LLMDebugWrapper(wrapped) if debug else wrapped

    avail = ", ".join(sorted(models))
    raise ValueError(f"Model {model_name} not found in config. Available: [{avail}]")
