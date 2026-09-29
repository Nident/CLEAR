from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agent.decision import evaluate_judge_llm_output
from utils.dataloader.message_saver import load_agent_messages

_RE_JSON_FENCE = re.compile(r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```", re.IGNORECASE)
_RE_CWE_ID = re.compile(r"CWE-\d+", re.IGNORECASE)


def _collect_numeric_json_files(results_path: Path) -> tuple[List[int], List[Path]]:
    stems: List[int] = []
    files: List[Path] = []
    for result_file in results_path.glob("*.json"):
        if not result_file.stem.isdigit():
            continue
        stems.append(int(result_file.stem))
        files.append(result_file)
    return stems, files


def _need_order_remap(
    stems: List[int],
    ordered_indices: Optional[List[int]],
    ground_truth_keys: Optional[Set[int]],
    require_non_negative: bool = True,
) -> bool:
    if not ordered_indices or not stems:
        return False
    if max(stems) >= len(ordered_indices):
        return False
    if require_non_negative and min(stems) < 0:
        return False
    if ground_truth_keys and not any(s in ground_truth_keys for s in stems):
        return True
    if not ground_truth_keys and min(stems) == 0:
        return True
    return False


def _remap_pred_by_order(pred: Dict[int, Any], ordered_indices: Optional[List[int]], gt_keys: Optional[Set[int]]) -> Dict[int, Any]:
    if not ordered_indices or not gt_keys or not pred:
        return pred
    keys = set(pred.keys())
    if (keys & gt_keys) or min(keys) < 0 or max(keys) >= len(ordered_indices):
        return pred
    return {ordered_indices[i]: pred[i] for i in pred if i < len(ordered_indices)}


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            out: List[Dict[str, Any]] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            return out
        data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            data = data["data"]
        return [x for x in (data if isinstance(data, list) else []) if isinstance(x, dict)]


def to_int_id(item: Dict[str, Any], fallback: int) -> int:
    v = item.get("index", item.get("idx"))
    if v is None:
        return fallback
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def load_indices_in_input_order(input_file: str, target_eq: Optional[int] = None) -> List[int]:
    records = load_records(Path(input_file))
    if target_eq is not None:
        records = [r for r in records if r.get("target") == target_eq]
    return [to_int_id(item, i) for i, item in enumerate(records)]


def load_ground_truth(input_file: str, target_eq: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
    data = load_records(Path(input_file))
    if target_eq is not None:
        data = [r for r in data if r.get("target") == target_eq]
    gt: Dict[int, Dict[str, Any]] = {}
    for item in data:
        idx = item.get("index", item.get("idx"))
        if idx is not None:
            gt[int(idx)] = {"target": item.get("target", 0), "cwe": item.get("cwe", "")}
    return gt


def normalize_cwe(cwe: Any) -> str:
    if cwe is None:
        return ""
    s = str(cwe[0]).strip() if isinstance(cwe, list) and cwe else str(cwe).strip()
    if not s:
        return ""
    m = _RE_CWE_ID.search(s)
    return m.group(0) if m else ""


def load_ground_truth_cwe(input_file: str, target_eq: Optional[int] = None) -> Dict[int, str]:
    gt = load_ground_truth(input_file, target_eq=target_eq)
    return {idx: normalize_cwe(v.get("cwe", "")) for idx, v in gt.items()}


def load_predictions_cwe_cot(
    results_dir: str,
    ordered_indices: Optional[List[int]] = None,
    ground_truth_keys: Optional[set] = None,
) -> Dict[int, str]:
    results_path = Path(results_dir)
    pred: Dict[int, str] = {}
    stems, files = _collect_numeric_json_files(results_path)
    map_by_order = _need_order_remap(stems, ordered_indices, ground_truth_keys, require_non_negative=True)
    for result_file in files:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        stem_i = int(result_file.stem)
        idx = ordered_indices[stem_i] if (map_by_order and ordered_indices) else stem_i
        if isinstance(data, dict) and ("raw" in data or "judgments" in data):
            raw_obj = data.get("raw") if "raw" in data else data
            final_cwe = raw_obj.get("final_cwe")
            if not final_cwe:
                _, overall = evaluate_judge_llm_output(raw_obj)
                final_cwe = overall.get("cwe_id") or overall.get("cwe") or ""
            raw = (final_cwe or "").strip()
        else:
            raw = (data.get("final_cwe") or data.get("cwe_id") or data.get("cwe") or "").strip()
        pred[int(idx)] = raw if raw and raw.upper() != "N/A" else "CWE-0"
    return pred


def load_predictions_cwe_gptlens(
    results_dir: str,
    ordered_indices: Optional[List[int]] = None,
    ground_truth_keys: Optional[set] = None,
    threshold: Optional[float] = None,
    ground_truth_cwe: Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    results_path = Path(results_dir)
    pred: Dict[int, str] = {}
    stems, files = _collect_numeric_json_files(results_path)
    map_by_order = _need_order_remap(stems, ordered_indices, ground_truth_keys, require_non_negative=False)
    for result_file in files:
        stem_i = int(result_file.stem)
        idx = ordered_indices[stem_i] if (map_by_order and ordered_indices) else stem_i
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = [x for x in (data if isinstance(data, list) else [data]) if isinstance(x, dict)]
        if threshold is not None:
            above = [x for x in items if float(x.get("final_score", 0)) >= float(threshold)]
            raw_cwe_list = [(x.get("cwe") or x.get("cwe_id") or "").strip() for x in above]
            raw_cwe_list = [c for c in raw_cwe_list if c and c.upper() != "CWE-0"]
            valid_cwe = [c for c in (normalize_cwe(x) for x in raw_cwe_list) if c]
            cwe_set = set(valid_cwe)
            if ground_truth_cwe and idx in ground_truth_cwe:
                gt_val = ground_truth_cwe[idx]
                pred[int(idx)] = gt_val if normalize_cwe(gt_val) in cwe_set else (valid_cwe[0] if valid_cwe else "CWE-0")
            else:
                pred[int(idx)] = valid_cwe[0] if valid_cwe else "CWE-0"
        else:
            best = max(items, key=lambda b: float(b.get("final_score", 0))) if items else {}
            raw = (best.get("cwe") or best.get("cwe_id") or "CWE-0").strip() or "CWE-0"
            pred[int(idx)] = normalize_cwe(raw) or "CWE-0"
    return pred


def _vultrial_extract_cwe_list(text: str) -> List[str]:
    if not (text or "").strip():
        return ["CWE-0"]
    raw = text.strip()
    m = _RE_JSON_FENCE.search(raw)
    cand: Optional[str] = m.group(1).strip() if m else None
    if not cand and (raw.startswith("[") or raw.startswith("{")):
        cand = raw
    if cand:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            cwes = [(item.get("cwe") or item.get("cwe_id") or "").strip() for item in data if isinstance(item, dict)]
            cwes = [c for c in cwes if c and c.upper() != "CWE-0"]
            return list(dict.fromkeys(cwes)) if cwes else ["CWE-0"]
        if isinstance(data, dict):
            cwe = (data.get("cwe") or data.get("cwe_id") or "").strip()
            return [cwe] if cwe and cwe.upper() != "CWE-0" else ["CWE-0"]
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines if lines else ["CWE-0"]


def load_predictions_cwe_vultrial(results_dir: str) -> Dict[int, List[str]]:
    results_path = Path(results_dir)
    pred: Dict[int, List[str]] = {}
    for txt_file in results_path.glob("*.txt"):
        try:
            idx = int(txt_file.stem)
        except ValueError:
            continue
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        pred[idx] = _vultrial_extract_cwe_list(text)
    return pred


def resolve_vultrial_pred_for_metrics(
    gt_cwe: Dict[int, str],
    pred_multi: Dict[int, List[str]],
) -> Dict[int, str]:
    resolved: Dict[int, str] = {}
    for idx, cwe_list in pred_multi.items():
        if not cwe_list:
            resolved[idx] = "CWE-0"
            continue
        gt = (gt_cwe.get(idx) or "").strip()
        resolved[idx] = gt if gt and gt in cwe_list else (cwe_list[0] or "CWE-0").strip()
    return resolved


def load_predictions_json_results(
    results_dir: str,
    ordered_indices: Optional[List[int]] = None,
    ground_truth_keys: Optional[set] = None,
) -> Dict[int, int]:
    results_path = Path(results_dir)
    predictions: Dict[int, int] = {}
    stems, files = _collect_numeric_json_files(results_path)
    map_by_order = _need_order_remap(stems, ordered_indices, ground_truth_keys, require_non_negative=True)
    for result_file in files:
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        stem_i = int(result_file.stem)
        idx = ordered_indices[stem_i] if (map_by_order and ordered_indices) else stem_i
        if isinstance(data, dict) and ("raw" in data or "judgments" in data):
            llm_out = data.get("raw") if "raw" in data else data
            _, overall = evaluate_judge_llm_output(llm_out)
            pred: Any = overall.get("is_vulnerable", 0)
        elif isinstance(data, dict) and "is_vulnerable" in data:
            pred = data.get("is_vulnerable", 0)
        else:
            result = data.get("result", {}) if isinstance(data, dict) else {}
            judgment = result.get("judgment", {}) if isinstance(result, dict) else {}
            pred = judgment.get("is_vulnerable", 0) if isinstance(judgment, dict) else 0
        if pred is None:
            pred = 0
        if isinstance(pred, str):
            pred = 1 if pred.strip() in ("1", "yes", "YES") else 0
        else:
            pred = int(pred) if pred else 0
        predictions[int(idx)] = int(pred)
    return predictions


def load_predictions_vulrag(
    results_file: str,
    input_file: Optional[str] = None,
) -> Dict[int, int]:
    """Load per-row predictions from VUL-RAG detect output (one JSON list of pairs)."""
    path = Path(results_file)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}

    after_by_before: Dict[int, int] = {}
    if input_file:
        gt = load_ground_truth(input_file)
        from utils.metrics import load_pairs_auto

        after_by_before = {
            int(before_idx): int(after_idx)
            for before_idx, after_idx in load_pairs_auto(input_file, gt)
        }

    predictions: Dict[int, int] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        before_idx = int(item["id"])
        before = item.get("detect_result_before", {}).get("final_result", 0)
        after = item.get("detect_result_after", {}).get("final_result", 0)
        predictions[before_idx] = 1 if before == 1 else 0
        after_idx = after_by_before.get(before_idx, before_idx + 1)
        predictions[after_idx] = 1 if after == 1 else 0
    return predictions


def _judge_llm_output_from_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    m = msg if isinstance(msg, dict) else {}
    if isinstance(m.get("judgments"), list) or isinstance(m.get("judgment"), dict):
        return m
    out = m.get("output") if isinstance(m.get("output"), dict) else {}
    raw = out.get("raw") if isinstance(out.get("raw"), dict) else None
    if raw is not None:
        return raw
    if isinstance(out.get("judgments"), list) or isinstance(out.get("judgment"), dict):
        return out
    return {}


def get_existing_indices_from_messages(messages_dir: str) -> Set[int]:
    base = Path(messages_dir)
    if not base.exists() or not base.is_dir():
        return set()
    indices: Set[int] = set()
    for p in base.iterdir():
        if not p.is_dir():
            continue
        try:
            idx_val = int(p.name)
        except ValueError:
            continue
        judge_dir = p / "judge"
        if judge_dir.is_dir() and any(judge_dir.glob("*.json")):
            indices.add(idx_val)
    return indices


def load_predictions_from_messages(
    messages_dir: str,
    ordered_indices: Optional[List[int]] = None,
    ground_truth_keys: Optional[Set[int]] = None,
) -> Dict[int, int]:
    base = Path(messages_dir)
    if not base.exists() or not base.is_dir():
        return {}
    preds: Dict[int, int] = {}
    for p in sorted(base.iterdir(), key=lambda x: (not x.name.isdigit(), int(x.name) if x.name.isdigit() else 0)):
        if not p.is_dir():
            continue
        try:
            idx = int(p.name)
        except ValueError:
            continue
        judge_msg = (load_agent_messages(idx, messages_dir) or {}).get("judge")
        llm_out = _judge_llm_output_from_message(judge_msg) if judge_msg else {}
        if not llm_out:
            continue
        _, overall = evaluate_judge_llm_output(llm_out)
        pv = overall.get("is_vulnerable", 0)
        if isinstance(pv, str):
            pv = 1 if pv.strip().lower() in ("1", "yes") else 0
        else:
            pv = int(pv) if pv else 0
        preds[idx] = int(pv)
    return _remap_pred_by_order(preds, ordered_indices, ground_truth_keys)


def load_predictions_cwe_from_messages(
    messages_dir: str,
    ordered_indices: Optional[List[int]] = None,
    ground_truth_keys: Optional[Set[int]] = None,
    ground_truth_cwe: Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    base = Path(messages_dir)
    if not base.exists() or not base.is_dir():
        return {}
    pred: Dict[int, str] = {}
    for p in sorted(base.iterdir(), key=lambda x: (not x.name.isdigit(), int(x.name) if x.name.isdigit() else 0)):
        if not p.is_dir():
            continue
        try:
            idx = int(p.name)
        except ValueError:
            continue
        judge_msg = (load_agent_messages(idx, messages_dir) or {}).get("judge")
        llm_out = _judge_llm_output_from_message(judge_msg) if judge_msg else {}
        if not llm_out:
            continue
        judgments, overall = evaluate_judge_llm_output(llm_out)
        cwe_list = [(j.get("cwe_id") or j.get("cwe") or "").strip() for j in judgments if isinstance(j, dict) and int(j.get("is_vulnerable") or 0) == 1]
        cwe_list = [c for c in cwe_list if c and c.upper() != "N/A"]
        cwe_set = {normalize_cwe(c) for c in cwe_list}
        if ground_truth_cwe and idx in ground_truth_cwe:
            gt_val = ground_truth_cwe[idx]
            pred[idx] = gt_val if normalize_cwe(gt_val) in cwe_set else (cwe_list[0] if cwe_list else "CWE-0")
        else:
            first = cwe_list[0] if cwe_list else (overall.get("cwe_id") or overall.get("cwe") or "")
            pred[idx] = first if first and first.strip().upper() != "N/A" else "CWE-0"
    return _remap_pred_by_order(pred, ordered_indices, ground_truth_keys)


def load_predictions_gptlens_rank(
    results_dir: str,
    threshold: float,
    ordered_indices: Optional[List[int]] = None,
    ground_truth_keys: Optional[set] = None,
) -> Dict[int, int]:
    results_path = Path(results_dir)
    predictions: Dict[int, int] = {}
    stems, files = _collect_numeric_json_files(results_path)
    map_by_order = _need_order_remap(stems, ordered_indices, ground_truth_keys, require_non_negative=False)
    for result_file in files:
        stem_i = int(result_file.stem)
        idx = ordered_indices[stem_i] if (map_by_order and ordered_indices) else stem_i
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        scores: List[float] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "final_score" in item:
                    try:
                        scores.append(float(item["final_score"]))
                    except (TypeError, ValueError):
                        continue
        elif isinstance(data, dict) and "final_score" in data:
            try:
                scores.append(float(data["final_score"]))
            except (TypeError, ValueError):
                pass
        predictions[int(idx)] = 1 if (scores and max(scores) >= float(threshold)) else 0
    return predictions


def _vultrial_condition(k: Dict[str, Any], status: str) -> bool:
    if status == "only_valid_high":
        return k.get("decision") == "valid" and k.get("severity") == "high"
    if status == "action_and_valid_high":
        return (
            k.get("recommended_action") == "fix immediately"
            and k.get("decision") == "valid"
            and k.get("severity") == "high"
        )
    return False


def load_predictions_vultrial_condition(results_dir: str, status: str = "action_and_valid_high") -> Dict[int, int]:
    results_path = Path(results_dir)
    predictions: Dict[int, int] = {}
    for txt_file in results_path.glob("*.txt"):
        try:
            idx = int(txt_file.stem)
        except ValueError:
            continue
        text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
        if text in ("1", "0"):
            predictions[int(idx)] = int(text)
            continue
        if "</think>" in text:
            text = text.split("</think>")[-1]
        if "```" in text:
            for part in text.replace("```json", "```").split("```"):
                part = part.strip()
                if part.startswith("[") or part.startswith("{"):
                    text = part
                    break
        text = text.replace("```json\n", "").replace("\n```", "").strip()
        pred = 0
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            for k in parsed:
                if isinstance(k, dict) and _vultrial_condition(k, status):
                    pred = 1
                    break
        elif isinstance(parsed, dict) and "verdicts" in parsed:
            for k in parsed["verdicts"]:
                if isinstance(k, dict) and _vultrial_condition(k, status):
                    pred = 1
                    break
        elif text.split("\n")[-1].strip().lower().find("mitigate") >= 0:
            pred = 1
        predictions[int(idx)] = int(pred)
    return predictions
