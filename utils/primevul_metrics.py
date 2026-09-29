from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from . import metrics_common as C


def load_pairs_by_commit_hash(
    input_file: str,
    ground_truth: Optional[Dict[int, Dict[str, Any]]] = None,
    commit_key_candidates: Tuple[str, ...] = ("commit_id", "commit_hash", "commit"),
) -> List[Tuple[int, int]]:
    path = Path(input_file)
    records = C._load_records(path)

    groups: Dict[str, List[int]] = {}
    for i, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        cid = None
        for k in commit_key_candidates:
            if item.get(k):
                cid = str(item.get(k))
                break
        if not cid:
            continue
        rid = C._to_int_id(item, i)
        groups.setdefault(cid, []).append(rid)

    pairs: List[Tuple[int, int]] = []
    for _, ids in groups.items():
        if ground_truth is not None:
            ids = [i for i in ids if i in ground_truth]
        for j in range(0, len(ids) - 1, 2):
            pairs.append((ids[j], ids[j + 1]))
    return pairs


def load_pairs_by_commit_and_secondary_key(
    input_file: str,
    ground_truth: Optional[Dict[int, Dict[str, Any]]] = None,
    secondary_key: str = "func_hash",
    commit_key_candidates: Tuple[str, ...] = ("commit_id", "commit_hash", "commit"),
) -> List[Tuple[int, int]]:
    path = Path(input_file)
    records = C._load_records(path)

    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        cid = None
        for k in commit_key_candidates:
            if item.get(k):
                cid = str(item.get(k))
                break
        if not cid:
            continue
        sk = item.get(secondary_key)
        if sk is None:
            continue
        rid = C._to_int_id(item, i)
        groups.setdefault((cid, str(sk)), []).append(rid)

    pairs: List[Tuple[int, int]] = []
    for _, ids in groups.items():
        if ground_truth is not None:
            ids = [i for i in ids if i in ground_truth]
        for j in range(0, len(ids) - 1, 2):
            pairs.append((ids[j], ids[j + 1]))
    return pairs


def load_pairs_auto(input_file: str, ground_truth: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int]]:
    path = Path(input_file)
    records = C._load_records(path)
    n = len(records)
    if n == 0:
        return []

    has_commit = sum(
        1
        for it in records
        if isinstance(it, dict) and (it.get("commit_id") or it.get("commit_hash") or it.get("commit"))
    )
    commit_ratio = has_commit / max(1, n)

    expected = n // 2
    if commit_ratio >= 0.8:
        pairs_commit_func = load_pairs_by_commit_and_secondary_key(input_file, ground_truth=ground_truth, secondary_key="func_hash")
        if len(pairs_commit_func) >= int(expected * 0.8):
            return pairs_commit_func

        pairs_commit = load_pairs_by_commit_hash(input_file, ground_truth=ground_truth)
        if len(pairs_commit) >= int(expected * 0.8):
            return pairs_commit

    ordered = C.load_indices_in_input_order(input_file)
    ids = [i for i in ordered if i in ground_truth]
    return [(ids[i], ids[i + 1]) for i in range(0, len(ids) - 1, 2)]


def calculate_metrics(input_file: str, results_dir: str) -> Dict[str, float]:
    gt = C.load_ground_truth(input_file)
    ordered = C.load_indices_in_input_order(input_file)
    preds = C.load_predictions_json_results(results_dir, ordered_indices=ordered, ground_truth_keys=set(gt.keys()))
    pairs = load_pairs_auto(input_file, gt)
    return C.calculate_pairwise_metrics(gt, preds, pairs)


def calculate_metrics_gptlens(input_file: str, results_dir: str, threshold: float = 5.0) -> Dict[str, float]:
    gt = C.load_ground_truth(input_file)
    ordered = C.load_indices_in_input_order(input_file)
    preds = C.load_predictions_gptlens_rank(results_dir, threshold, ordered_indices=ordered, ground_truth_keys=set(gt.keys()))
    pairs = load_pairs_auto(input_file, gt)
    return C.calculate_pairwise_metrics(gt, preds, pairs)


def calculate_metrics_vultrial(
    input_file: str,
    results_dir: str,
    status: str = "action_and_valid_high",
) -> Dict[str, float]:
    gt = C.load_ground_truth(input_file)
    preds = C.load_predictions_vultrial_condition(results_dir, status=status)
    pairs = load_pairs_auto(input_file, gt)
    return C.calculate_pairwise_metrics(gt, preds, pairs, skip_missing_pairs=True)


def calculate_metrics_vulrag(input_file: str, results_file: str) -> Dict[str, float]:
    gt = C.load_ground_truth(input_file)
    preds = C.load_predictions_vulrag(results_file, input_file=input_file)
    pairs = load_pairs_auto(input_file, gt)
    return C.calculate_pairwise_metrics(gt, preds, pairs, skip_missing_pairs=True)

