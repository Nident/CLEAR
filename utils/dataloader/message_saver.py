from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from utils.results_paths import run_messages_dir


def get_messages_dir_for_model(
    model_name: str = "default",
    task_mode: str = "detection",
    results_base: Optional[Union[Path, str]] = None,
) -> str:
    return str(
        run_messages_dir(
            model_name,
            "default",
            task_mode=task_mode,
            results_base=results_base,
        )
    )


def get_messages_dir_for_model_and_version(
    model_name: str = "default",
    version: str = "default",
    task_mode: str = "detection",
    ablation: Optional[str] = None,
    results_base: Optional[Union[Path, str]] = None,
) -> str:
    return str(
        run_messages_dir(
            model_name,
            version,
            task_mode=task_mode,
            ablation=ablation,
            results_base=results_base,
        )
    )


def _keep_only_1_json(dir_path: Path) -> None:
    for p in dir_path.glob("*.json"):
        if p.stem != "1":
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


def _safe_unlink_many(paths: Iterable[Path]) -> None:
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _load_latest_json_in_dir(dir_path: Path) -> Optional[Dict[str, Any]]:
    best_n = -1
    best_p: Optional[Path] = None
    try:
        for p in dir_path.glob("*.json"):
            try:
                n = int(p.stem)
            except Exception:
                continue
            if n > best_n:
                best_n = n
                best_p = p
    except Exception:
        best_p = None

    if not best_p:
        return None
    try:
        with open(best_p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def save_agent_message(
    index: int,
    agent_type: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    messages_dir: str = "agent/messages",
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    cleanup_legacy: bool = False,
    slim: bool = True,
) -> str:
    agent_type = (agent_type or "").strip() or "unknown"

    base_dir = Path(messages_dir)
    index_dir = base_dir / str(index)
    agent_dir = index_dir / agent_type
    agent_dir.mkdir(parents=True, exist_ok=True)

    filepath = agent_dir / "1.json"

    if slim and agent_type == "claim":
        message_content = {"claims": output_data.get("claims", [])}
    elif slim and agent_type == "critic":
        message_content = {"counterclaims": output_data.get("counterclaims", [])}
    elif slim and agent_type == "judge":
        message_content = {"judgments": output_data.get("judgments", [])}
    else:
        message_content = {
            "agent_type": agent_type,
            "model": model_name,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "input": input_data,
            "output": output_data,
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(message_content, f, ensure_ascii=False, indent=2)

    _keep_only_1_json(agent_dir)

    if cleanup_legacy:
        _safe_unlink_many(index_dir.glob(f"{agent_type}-*.txt"))
        _safe_unlink_many([index_dir / f"{agent_type}-1.txt"])

        if model_name:
            legacy_index_dir = Path(f"agent/{model_name}/messages") / str(index)
            _safe_unlink_many(legacy_index_dir.glob(f"{agent_type}-*.txt"))
            _safe_unlink_many([legacy_index_dir / f"{agent_type}-1.txt"])

    return str(filepath)


def save_all_field(
    *,
    index: int,
    messages_dir: str,
    messages: Dict[str, Dict[str, Any]],
    model_name: Optional[str] = None,
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
) -> str:
    base_dir = Path(messages_dir)
    index_dir = base_dir / str(index)
    all_dir = index_dir / "all_field"
    all_dir.mkdir(parents=True, exist_ok=True)

    filepath = all_dir / "1.json"

    payload = {
        "index": index,
        "model": model_name,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "messages": messages,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _keep_only_1_json(all_dir)
    return str(filepath)


def load_agent_messages(
    index: int,
    messages_dir: str = "agent/messages",
) -> Dict[str, Dict[str, Any]]:
    base_dir = Path(messages_dir)
    index_dir = base_dir / str(index)

    if not index_dir.exists():
        return {}

    messages: Dict[str, Dict[str, Any]] = {}

    for agent_type in ("collector", "claim", "critic", "judge"):
        agent_dir = index_dir / agent_type
        if agent_dir.exists() and agent_dir.is_dir():
            obj = _load_latest_json_in_dir(agent_dir)
            if isinstance(obj, dict):
                messages[agent_type] = obj

    if messages:
        return messages

    candidates = list(index_dir.glob("*.txt"))

    def _priority(p: Path) -> Tuple[int, float, str]:
        name = p.name
        is_legacy = name.endswith("-1.txt")
        try:
            mtime = float(p.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return (1 if is_legacy else 0, -mtime, name)

    candidates.sort(key=_priority)
    seen_types = set()
    for filepath in candidates:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
            if not isinstance(content, dict):
                continue
            agent_type = content.get("agent_type")
            if agent_type and agent_type not in seen_types:
                messages[agent_type] = content
                seen_types.add(agent_type)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Failed to load message file {filepath}: {e}")
            continue

    return messages
