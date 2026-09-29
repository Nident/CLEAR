from __future__ import annotations

from typing import Any, Dict
import json
import re


def traceable(*args, **kwargs):
    def decorator(func):
        return func
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator


_RE_FENCE_JSON = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def _coerce_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="ignore")
    if isinstance(text, str):
        return text
    if isinstance(text, dict):
        if isinstance(text.get("text"), str):
            return text["text"]
        if isinstance(text.get("content"), str):
            return text["content"]
        return json.dumps(text, ensure_ascii=False)
    if isinstance(text, list):
        parts: list[str] = []
        for item in text:
            s = _coerce_text(item)
            if s:
                parts.append(s)
        return "\n".join(parts)
    return str(text)


def extract_json_obj(text: Any) -> Dict[str, Any]:
    t = _coerce_text(text).strip()
    if not t:
        return {}
    m = _RE_FENCE_JSON.search(t)
    if m:
        t = m.group(1).strip()
    if not (t.startswith("{") and t.endswith("}")):
        m2 = re.search(r"(\{[\s\S]*\})", t)
        if m2:
            t = m2.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

