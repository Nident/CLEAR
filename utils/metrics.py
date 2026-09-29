from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import metrics_common as C
from . import java_metrics as J
from . import primevul_metrics as P


def _is_primevul_input(input_file: str) -> bool:
    p = Path(input_file)
    if p.suffix == ".jsonl":
        return True
    s = str(p)
    return ("PrimeVul" in s) or ("primevul" in s.lower())


load_ground_truth = C.load_ground_truth
load_indices_in_input_order = C.load_indices_in_input_order
calculate_pairwise_metrics = C.calculate_pairwise_metrics
calculate_classification_metrics = C.calculate_classification_metrics
format_metrics_for_display = C.format_metrics_for_display
print_metrics_report = C.print_metrics_report

load_predictions_vultrial_condition = C.load_predictions_vultrial_condition
load_predictions_gptlens_rank = C.load_predictions_gptlens_rank

get_existing_indices_from_messages = C.get_existing_indices_from_messages
load_predictions_from_messages = C.load_predictions_from_messages
load_predictions_cwe_from_messages = C.load_predictions_cwe_from_messages

load_ground_truth_cwe = C.load_ground_truth_cwe
load_predictions_cwe_cot = C.load_predictions_cwe_cot
load_predictions_cwe_gptlens = C.load_predictions_cwe_gptlens
load_predictions_cwe_vultrial = C.load_predictions_cwe_vultrial
resolve_vultrial_pred_for_metrics = C.resolve_vultrial_pred_for_metrics
calculate_cwe_classification_metrics = C.calculate_cwe_classification_metrics
print_cwe_metrics_report = C.print_cwe_metrics_report

load_pairs_by_commit_hash = P.load_pairs_by_commit_hash
load_pairs_by_commit_and_secondary_key = P.load_pairs_by_commit_and_secondary_key

__all__ = [
    "load_ground_truth",
    "load_indices_in_input_order",
    "calculate_pairwise_metrics",
    "calculate_classification_metrics",
    "format_metrics_for_display",
    "print_metrics_report",
    "load_predictions_vultrial_condition",
    "load_predictions_gptlens_rank",
    "get_existing_indices_from_messages",
    "load_predictions_from_messages",
    "load_predictions_cwe_from_messages",
    "load_ground_truth_cwe",
    "load_predictions_cwe_cot",
    "load_predictions_cwe_gptlens",
    "load_predictions_cwe_vultrial",
    "resolve_vultrial_pred_for_metrics",
    "calculate_cwe_classification_metrics",
    "print_cwe_metrics_report",
    "load_pairs_by_commit_hash",
    "load_pairs_by_commit_and_secondary_key",
    "load_pairs_auto",
    "calculate_metrics",
    "calculate_metrics_gptlens",
    "calculate_metrics_vultrial",
    "calculate_metrics_vulrag",
]


def load_pairs_auto(input_file: str, ground_truth: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int]]:
    return P.load_pairs_auto(input_file, ground_truth) if _is_primevul_input(input_file) else J.load_pairs_java(input_file, ground_truth)


def calculate_metrics(input_file: str, results_dir: str) -> Dict[str, float]:
    return P.calculate_metrics(input_file, results_dir) if _is_primevul_input(input_file) else J.calculate_metrics(input_file, results_dir)


def calculate_metrics_gptlens(input_file: str, results_dir: str, threshold: float = 5.0) -> Dict[str, float]:
    return P.calculate_metrics_gptlens(input_file, results_dir, threshold=threshold) if _is_primevul_input(input_file) else J.calculate_metrics_gptlens(input_file, results_dir, threshold=threshold)


def calculate_metrics_vultrial(input_file: str, results_dir: str, status: str = "action_and_valid_high") -> Dict[str, float]:
    return P.calculate_metrics_vultrial(input_file, results_dir, status=status) if _is_primevul_input(input_file) else J.calculate_metrics_vultrial(input_file, results_dir, status=status)


def calculate_metrics_vulrag(input_file: str, results_file: str) -> Dict[str, float]:
    return P.calculate_metrics_vulrag(input_file, results_file) if _is_primevul_input(input_file) else J.calculate_metrics_vulrag(input_file, results_file)
