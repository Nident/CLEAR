from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_instances(raw: Any) -> Tuple[List[Dict[str, Any]], bool]:
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw["data"], True
    if isinstance(raw, list):
        return raw, False
    raise ValueError("Expected {'data':[...]} or a list.")


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix == ".jsonl":
        out: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    out.append(rec)
        return out
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError("JSON root must be list or {'data':[...]} for this loader.")
