from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.dataloader.results_saver import save_result
from utils.dataloader.message_saver import load_agent_messages
from utils.results_paths import run_messages_dir


def _as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _canon_bool01(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if int(v) == 1 else 0
    s = str(v).strip().lower()
    return 1 if s in ("1", "true", "yes", "y") else 0


def _canon_cwe_id(v: Any) -> str:
    s = ("" if v is None else str(v)).strip()
    return s if s else "N/A"


def derive_vulnerability_decision(judgment: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(judgment, dict):
        return {"is_vulnerable": 0, "cwe_id": "N/A"}

    if isinstance(judgment.get("root_causes"), list):
        for rc in judgment["root_causes"]:
            if not isinstance(rc, dict):
                continue

            if rc.get("status") not in ("PRESENT", "POTENTIALLY_PRESENT"):
                continue

            preconds = rc.get("required_preconditions", [])
            if not isinstance(preconds, list):
                continue

            if all(
                isinstance(p, dict) and p.get("status") in ("SUPPORTED", "POTENTIALLY_SUPPORTED")
                for p in preconds
            ):
                return {
                    "is_vulnerable": 1,
                    "cwe_id": judgment.get("cwe_id", "N/A"),
                }

        return {"is_vulnerable": 0, "cwe_id": "N/A"}

    root_status = judgment.get("root_cause_status", "UNCLEAR")
    if root_status in ("PRESENT", "POTENTIALLY_PRESENT"):
        precond_status = judgment.get("status") or judgment.get("precondition_status") or ""
        if precond_status in ("SUPPORTED", "POTENTIALLY_SUPPORTED"):
            return {
                "is_vulnerable": 1,
                "cwe_id": judgment.get("cwe_id", "N/A"),
            }

    if root_status not in ("PRESENT", "POTENTIALLY_PRESENT"):
        return {"is_vulnerable": 0, "cwe_id": "N/A"}

    preconditions = judgment.get("preconditions", [])
    if not isinstance(preconditions, list) or not preconditions:
        return {"is_vulnerable": 0, "cwe_id": "N/A"}

    if all(
        isinstance(p, dict) and p.get("status") in ("SUPPORTED", "POTENTIALLY_SUPPORTED")
        for p in preconditions
    ):
        return {
            "is_vulnerable": 1,
            "cwe_id": judgment.get("cwe_id", "N/A"),
        }

    return {"is_vulnerable": 0, "cwe_id": "N/A"}


def normalize_judgments_from_llm_output(llm_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    obj = _as_dict(llm_output)
    if isinstance(obj.get("judgments"), list):
        return [x for x in obj["judgments"] if isinstance(x, dict)]
    if isinstance(obj.get("judgment"), dict):
        return [obj["judgment"]]
    return []


def evaluate_judge_llm_output(llm_output: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = normalize_judgments_from_llm_output(llm_output)
    judgments: List[Dict[str, Any]] = []
    for j in raw:
        jj = dict(j)
        jj.setdefault("claim_id", "")
        if not isinstance(jj.get("preconditions"), list):
            jj["preconditions"] = []
        jj.setdefault("root_cause_status", "UNCLEAR")
        jj.setdefault("cwe_id", "N/A")
        final = derive_vulnerability_decision(jj)
        jj["is_vulnerable"] = _canon_bool01(final.get("is_vulnerable"))
        jj["cwe_id"] = final.get("cwe_id", "N/A") if jj["is_vulnerable"] == 1 else "N/A"
        jj["cwe"] = jj["cwe_id"]
        judgments.append(jj)
    any_vuln = any(int(j.get("is_vulnerable") or 0) == 1 for j in judgments if isinstance(j, dict))
    overall_cwe = "N/A"
    if any_vuln:
        for j in judgments:
            if int(j.get("is_vulnerable") or 0) == 1:
                overall_cwe = _canon_cwe_id(j.get("cwe_id"))
                break
    overall = {"is_vulnerable": 1 if any_vuln else 0, "cwe_id": overall_cwe, "cwe": overall_cwe}
    return judgments, overall


def evaluate_and_save_results(
    *,
    output_dir: str,
    index: Any,
    llm_output: Dict[str, Any],
) -> Dict[str, Any]:
    _, overall = evaluate_judge_llm_output(llm_output)
    save_result(output_dir, index, overall)
    return overall


def _extract_llm_output_from_saved_judge_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    m = _as_dict(msg)
    if isinstance(m.get("judgments"), list) or isinstance(m.get("judgment"), dict):
        return m
    out = _as_dict(m.get("output"))
    raw = out.get("raw")
    if isinstance(raw, dict):
        return raw
    if isinstance(out.get("judgments"), list) or isinstance(out.get("judgment"), dict):
        return out
    return {}


def _discover_indices_with_judge(messages_dir: str) -> List[Any]:
    base = Path(messages_dir)
    if not base.exists() or not base.is_dir():
        return []
    indices: List[Any] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        try:
            idx_val = int(p.name) if p.name.isdigit() else p.name
        except Exception:
            idx_val = p.name
        msgs = load_agent_messages(index=idx_val, messages_dir=messages_dir)
        if msgs.get("judge"):
            indices.append(idx_val)
    return indices


def run_decision_from_messages(
    model: str,
    version: str,
    messages_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    task_mode: str = "detection",
) -> str:
    messages_dir = messages_dir or str(run_messages_dir(model, version, task_mode=task_mode))
    output_dir = output_dir or str(run_messages_dir(model, version, task_mode=task_mode))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    indices = _discover_indices_with_judge(messages_dir)
    if not indices:
        raise FileNotFoundError(f"No judge messages found under {messages_dir}")
    for idx in indices:
        msgs = load_agent_messages(index=idx, messages_dir=messages_dir)
        judge_msg = _as_dict(msgs.get("judge"))
        if not judge_msg:
            continue
        llm_output = _extract_llm_output_from_saved_judge_message(judge_msg)
        if not llm_output:
            continue
        evaluate_and_save_results(output_dir=output_dir, index=idx, llm_output=llm_output)
    print(f"Decision: {len(indices)} samples → {output_dir}")
    return output_dir