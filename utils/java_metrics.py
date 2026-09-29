from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from . import metrics_common as C


def load_pairs_java(input_file: str, ground_truth: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int]]:
    ordered = C.load_indices_in_input_order(input_file)
    ids = [i for i in ordered if i in ground_truth]
    return [(ids[i], ids[i + 1]) for i in range(0, len(ids) - 1, 2)]


def calculate_metrics(input_file: str, results_dir: str) -> Dict[str, float]:
    gt = C.load_ground_truth(input_file)
    ordered = C.load_indices_in_input_order(input_file)
    preds = C.load_predictions_json_results(results_dir, ordered_indices=ordered, ground_truth_keys=set(gt.keys()))
    pairs = load_pairs_java(input_file, gt)
    return C.calculate_pairwise_metrics(gt, preds, pairs)


def calculate_metrics_gptlens(input_file: str, results_dir: str, threshold: float = 5.0) -> Dict[str, float]:
    gt = C.load_ground_truth(input_file)
    ordered = C.load_indices_in_input_order(input_file)
    preds = C.load_predictions_gptlens_rank(results_dir, threshold, ordered_indices=ordered, ground_truth_keys=set(gt.keys()))
    pairs = load_pairs_java(input_file, gt)
    return C.calculate_pairwise_metrics(gt, preds, pairs)


def calculate_metrics_vultrial(
    input_file: str,
    results_dir: str,
    status: str = "action_and_valid_high",
) -> Dict[str, float]:
    gt = C.load_ground_truth(input_file)
    preds = C.load_predictions_vultrial_condition(results_dir, status=status)
    pairs = load_pairs_java(input_file, gt)
    return C.calculate_pairwise_metrics(gt, preds, pairs, skip_missing_pairs=True)


def calculate_metrics_vulrag(input_file: str, results_file: str) -> Dict[str, float]:
    gt = C.load_ground_truth(input_file)
    preds = C.load_predictions_vulrag(results_file, input_file=input_file)
    pairs = load_pairs_java(input_file, gt)
    return C.calculate_pairwise_metrics(gt, preds, pairs, skip_missing_pairs=True)

