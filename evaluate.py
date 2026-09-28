from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev, variance
from typing import Any, Dict, List, Optional, Sequence

from scipy.stats import t as student_t

_ROOT = Path(__file__).resolve().parent
_BASELINE = _ROOT / "baseline"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_BASELINE) not in sys.path:
    sys.path.insert(0, str(_BASELINE))

from baseline.results_paths import (
    dl_run_dir,
    method_run_dir,
    storage_version,
    vulrag_result_file,
)
from utils.eval_common import (
    default_input_path,
    eval_output_dir,
    has_messages_results,
    resolve_run_messages_dir,
)
from utils.metrics import (
    calculate_cwe_classification_metrics,
    calculate_metrics,
    calculate_metrics_gptlens,
    calculate_metrics_vultrial,
    calculate_metrics_vulrag,
    calculate_pairwise_metrics,
    format_metrics_for_display,
    load_ground_truth,
    load_ground_truth_cwe,
    load_indices_in_input_order,
    load_pairs_auto,
    load_predictions_cwe_cot,
    load_predictions_cwe_from_messages,
    load_predictions_cwe_gptlens,
    load_predictions_cwe_vultrial,
    load_predictions_from_messages,
    print_metrics_report,
    resolve_vultrial_pred_for_metrics,
)
from utils.results_paths import (
    FIXED_RESULTS_ROOT as VCKG_RESULTS_ROOT,
    java_pair_storage_versions,
    storage_version_to_run_tag,
)

METRIC_KEYS = [
    "P-C",
    "P-V",
    "P-B",
    "P-R",
    "VPS",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
]
PAIR_PERCENT_KEYS = {"P-C", "P-V", "P-B", "P-R", "VPS"}

METHOD_CHOICES = ["CoT", "DL", "GPTLens", "VulTrial", "VUL-RAG"]
TARGET_CHOICES = ["java-pair", "primevul"]

METHOD_FOLDER = {
    "cot": "CoT",
    "gptlens": "GPTLens",
    "vultrial": "VulTrial",
    "vulrag": "VUL-RAG",
}
METHOD_LABEL = {v: k for k, v in METHOD_FOLDER.items()}
METHOD_LABEL["codebert"] = "DL"
METHOD_LABEL["unixcoder"] = "DL"

TABLE_METRICS = ["P-C", "P-V", "P-B", "P-R", "VPS"]
ROW_DISPLAY: Dict[str, str] = {
    "codebert": "CodeBERT",
    "unixcoder": "UnixCoder",
    "cot": "CoT",
    "gptlens": "GPTLens",
    "vultrial": "VulTrial",
    "vulrag": "Vul-RAG",
    "ours": "Ours",
}
ROW_ORDER = ["codebert", "unixcoder", "cot", "gptlens", "vultrial", "vulrag", "ours"]

ABLATION_TYPES = [
    "no_evidence",
    "no_entrypoint",
    "no_precondition",
    "no_root_cause",
    "no_fix_intent",
]
ABLATION_DISPLAY = {
    "no_evidence": "No_Evidence",
    "no_entrypoint": "No_Entrypoint",
    "no_precondition": "No_Precondition",
    "no_root_cause": "No_Root_Cause",
    "no_fix_intent": "No_Fix_Intent",
    "full": "Full",
}
ABLATION_ROW_ORDER = [*ABLATION_TYPES, "full"]

CLASSIFICATION_METRICS = ["Macro-F1", "Weighted-F1", "MCC"]
CLASSIFICATION_BASELINE_METHODS = ["cot", "gptlens", "vultrial", "codebert", "unixcoder"]

TARGET_TITLE = {"primevul": "PrimeVul", "java-pair": "Java-Pair"}
LANGUAGE = {"primevul": "C/C++", "java-pair": "Java"}

AGENTS_FOR_METHOD = {
    "codebert": "DL",
    "unixcoder": "DL",
    "cot": "Single-Agent",
    "vulrag": "Single-Agent",
    "gptlens": "Multi-Agent",
    "vultrial": "Multi-Agent",
    "rag_instance": "Multi-Agent",
    "rag_chunk": "Multi-Agent",
    "ours": "Multi-Agent",
}

DETECTION_TABLE_METHODS = [
    ("codebert", "CodeBERT"),
    ("unixcoder", "UnixCoder"),
    ("cot", "CoT"),
    ("gptlens", "GPTLens"),
    ("vultrial", "VulTrial"),
    ("vulrag", "Vul-RAG"),
    ("ours", "Ours"),
]
CLASSIFICATION_TABLE_METHODS = [
    ("cot", "CoT"),
    ("gptlens", "GPTLens"),
    ("vultrial", "VulTrial"),
    ("ours", "Ours"),
]
RETRIEVAL_TABLE_METHODS = [
    ("rag_instance", "Vanilla RAG"),
    ("rag_chunk", "Chunk"),
    ("ours", "Ours"),
]

DEFAULT_BASELINE_MODEL = "gpt-4o-mini"
TABLE_TARGET_ORDER = ["primevul", "java-pair"]
_VCKG_NON_MODEL_DIRS = frozenset({"eval", "analysis"})
_OTHER_MODELS_ORDER = [
    "gemini-3.1-flash-lite",
    "gpt-5-mini",
    "gemini-2.5-flash",
    "gpt-4o-mini",
    "gpt-4.1-nano",
]

_CI_LEVEL: Optional[float] = None


def _set_ci_level(level: Optional[float]) -> None:
    global _CI_LEVEL
    _CI_LEVEL = level


def _active_ci_level() -> Optional[float]:
    return _CI_LEVEL


def _t_critical(confidence_percent: float, df: int) -> float:
    if df < 1:
        return 0.0
    alpha = 1.0 - confidence_percent / 100.0
    return float(student_t.ppf(1.0 - alpha / 2.0, df))


def _ci_half_width(stats: Dict[str, Any], confidence_percent: float) -> float:
    n = int(stats.get("n", 0))
    if n <= 1:
        return 0.0
    if n == 10:
        return 2.262 * float(stats["std"]) / math.sqrt(10.0)
    return _t_critical(confidence_percent, n - 1) * float(stats["std"]) / math.sqrt(n)


def _floor_to_decimals(value: float, ndigits: int = 2) -> float:
    """Floor toward -inf at `ndigits` decimal places (display only)."""
    factor = 10.0 ** ndigits
    return math.floor(value * factor + 1e-9) / factor


def _format_uncertainty(
    stats: Dict[str, Any],
    *,
    scale: float,
    suffix: str = "",
    show_std: bool = True,
) -> str:
    mean_val = float(stats["mean"]) * scale
    n = int(stats.get("n", 0))
    mean_disp = _floor_to_decimals(mean_val, 2)
    if n <= 1:
        return f"{mean_disp:.2f}{suffix}"

    ci_level = _active_ci_level()
    if ci_level is not None:
        half = _floor_to_decimals(_ci_half_width(stats, ci_level) * scale, 2)
        return f"{mean_disp:.2f}{suffix} ± {half:.2f}{suffix} ({ci_level:.0f}% CI)"

    if not show_std:
        return f"{mean_disp:.2f}{suffix}"

    std_val = _floor_to_decimals(float(stats["std"]) * scale, 2)
    return f"{mean_disp:.2f}{suffix} ± {std_val:.2f}{suffix}"


def _normalize_method(name: str) -> str:
    key = name.strip().replace("_", "-").replace(" ", "")
    aliases = {
        "cot": "cot",
        "CoT": "cot",
        "dl": "dl",
        "DL": "dl",
        "codebert": "codebert",
        "unixcoder": "unixcoder",
        "gptlens": "gptlens",
        "GPTLens": "gptlens",
        "vultrial": "vultrial",
        "VulTrial": "vultrial",
        "vulrag": "vulrag",
        "VUL-RAG": "vulrag",
        "VULRAG": "vulrag",
    }
    if key not in aliases:
        raise ValueError(f"Unknown method: {name!r}. Choose from {METHOD_CHOICES} or codebert/unixcoder.")
    return aliases[key]


def _expand_methods(method: Optional[str]) -> List[str]:
    if method is None:
        return ["cot", "gptlens", "vultrial", "vulrag", "codebert", "unixcoder"]
    norm = _normalize_method(method)
    if norm == "dl":
        return ["codebert", "unixcoder"]
    return [norm]


def _aggregate_metrics(entries: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> Dict[str, Any]:
    metric_keys = keys or METRIC_KEYS
    out: Dict[str, Any] = {}
    for key in metric_keys:
        values = [float(e[key]) for e in entries if isinstance(e.get(key), (int, float))]
        if not values:
            continue
        if len(values) == 1:
            out[key] = {"mean": values[0], "std": 0.0, "n": 1, "values": values}
        else:
            out[key] = {
                "mean": mean(values),
                "std": stdev(values),
                "variance": variance(values),
                "n": len(values),
                "values": values,
            }
    return out


def _format_table_value(
    stats: Dict[str, Any],
    *,
    use_pct_sign: bool = False,
    show_std: bool = True,
) -> str:
    """Format aggregated stat for paper tables (values scaled to 0–100)."""
    suffix = "%" if use_pct_sign else ""
    return _format_uncertainty(stats, scale=100.0, suffix=suffix, show_std=show_std)


def _format_table_metric(
    stats: Dict[str, Any],
    key: str,
    *,
    use_pct_sign: bool = False,
    show_std: bool = True,
) -> str:
    if key not in stats:
        return "—"
    return _format_table_value(stats[key], use_pct_sign=use_pct_sign, show_std=show_std)


def _format_class_metric(stats: Dict[str, Any], key: str, *, show_std: bool = False) -> str:
    if key not in stats:
        return "—"
    if show_std:
        return _format_table_metric(stats, key, use_pct_sign=False, show_std=True)
    return f"{_floor_to_decimals(float(stats[key]['mean']) * 100.0, 2):.2f}"


def _format_mean_std(stats: Dict[str, Any], *, as_percent: bool = False) -> str:
    scale = 100.0 if as_percent else 1.0
    suffix = "%" if as_percent else ""
    return _format_uncertainty(stats, scale=scale, suffix=suffix, show_std=True)


def _print_aggregate(label: str, stats: Dict[str, Any], *, n_runs: int, keys: Optional[List[str]] = None) -> None:
    print(f"\n== {label} (n={n_runs}) ==")
    metric_keys = keys or [k for k in METRIC_KEYS if k in stats] or list(stats.keys())
    for key in metric_keys:
        if key not in stats:
            continue
        as_pct = key in PAIR_PERCENT_KEYS
        print(f"  {key:12s}  {_format_mean_std(stats[key], as_percent=as_pct)}")


def _cell(stats: Dict[str, Any], key: str, *, as_percent: bool = True) -> str:
    if key not in stats:
        return "—"
    return _format_mean_std(stats[key], as_percent=as_percent)


def _write_results_md(
    path: Path,
    *,
    model: str,
    sections: Dict[str, List[Dict[str, Any]]],
    task: str = "detection",
    doc_title: str = "Evaluation Results",
) -> None:
    """sections: {target: [{method_id, display_name?, metrics, n_runs}, ...]}"""
    if task == "classification":
        columns = CLASSIFICATION_METRICS
        header = "| **Method** | **Macro-F1** | **Weighted-F1** | **MCC** |"
        sep = "| --- | --- | --- | --- |"
        as_percent = False
    else:
        columns = TABLE_METRICS
        header = "| **Method** | **P-C(↑)** | **P-V(↓)** | **P-B(↓)** | **P-R(↓)** | **VPS(↑)** |"
        sep = "| --- | --- | --- | --- | --- | --- |"
        as_percent = True

    lines: List[str] = [f"# {doc_title}", ""]
    for target, rows in sections.items():
        title = TARGET_TITLE.get(target, target)
        suffix = " — Classification" if task == "classification" else ""
        if task == "ablation":
            lines.append(f"### {title}")
        else:
            lines.append(f"## {title}{suffix} — {model}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for row in rows:
            stats = row["metrics"]
            name = row.get("display_name") or ABLATION_DISPLAY.get(row["method_id"]) or ROW_DISPLAY.get(row["method_id"], row["method_id"])
            cells = [_cell(stats, k, as_percent=as_percent) for k in columns]
            lines.append(f"| {name} | {' | '.join(cells)} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print("Saved:", path)


def _rows_from_report(
    *,
    vckg_report: Optional[Dict[str, Any]],
    baseline_report: Optional[Dict[str, Any]],
    target: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if baseline_report:
        for method_id in ROW_ORDER:
            if method_id == "ours":
                continue
            method_data = baseline_report.get("methods", {}).get(method_id, {})
            block = method_data.get(target)
            if block and block.get("metrics"):
                rows.append(
                    {"method_id": method_id, "metrics": block["metrics"], "n_runs": block.get("n_runs", 1)}
                )
    if vckg_report:
        block = vckg_report.get("targets", {}).get(target)
        if block and block.get("metrics"):
            rows.append(
                {"method_id": "ours", "metrics": block["metrics"], "n_runs": block.get("n_runs", 1)}
            )
    order = {k: i for i, k in enumerate(ROW_ORDER)}
    rows.sort(key=lambda r: order.get(r["method_id"], 999))
    return rows


def _pair_only_metrics(metrics: dict) -> dict:
    keys = {
        "P-C", "P-V", "P-B", "P-R", "VPS",
        "total_pairs", "p_c_count", "p_v_count", "p_b_count", "p_r_count",
        "pair_count", "code_total",
        "Accuracy", "Precision", "Recall", "F1-Score",
    }
    return {k: v for k, v in metrics.items() if k in keys}


def _eval_vckg_once(
    *,
    model: str,
    target: str,
    storage_ver: str,
    results_root: Path,
    ablation: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    run_tag = storage_version_to_run_tag(target, storage_ver)
    input_path = default_input_path(target, _ROOT)
    model_dir = resolve_run_messages_dir(model, target, "detection", ablation, run_tag)
    if not model_dir.is_dir():
        return None
    if not has_messages_results(model_dir):
        return None

    gt = load_ground_truth(str(input_path))
    ordered = load_indices_in_input_order(str(input_path))
    pairs = load_pairs_auto(str(input_path), gt)
    preds = load_predictions_from_messages(
        str(model_dir),
        ordered_indices=ordered,
        ground_truth_keys=set(gt.keys()),
    )
    metrics = calculate_pairwise_metrics(gt, preds, pairs, skip_missing_pairs=True)
    display = format_metrics_for_display(metrics)
    return {
        "model": model,
        "storage_version": storage_ver,
        "ablation": ablation,
        **metrics,
        **display,
    }


def _eval_vckg_classification_once(
    *,
    model: str,
    target: str,
    storage_ver: str,
    ablation: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    run_tag = storage_version_to_run_tag(target, storage_ver)
    input_path = default_input_path(target, _ROOT)
    model_dir = resolve_run_messages_dir(model, target, "classification", ablation, run_tag)
    if not model_dir.is_dir() or not has_messages_results(model_dir):
        return None

    ordered = load_indices_in_input_order(str(input_path), target_eq=1)
    gt_cwe = load_ground_truth_cwe(str(input_path), target_eq=1)
    pred_cwe = load_predictions_cwe_from_messages(
        str(model_dir),
        ordered_indices=ordered,
        ground_truth_keys=set(gt_cwe.keys()),
        ground_truth_cwe=gt_cwe,
    )
    if not pred_cwe:
        return None
    metrics = calculate_cwe_classification_metrics(gt_cwe, pred_cwe)
    return {"model": model, "storage_version": storage_ver, **metrics}


def _eval_baseline_classification_once(
    *,
    method: str,
    model: str,
    target: str,
    storage_ver: str,
    results_root: Path,
    threshold: float,
    status: str,
) -> Optional[Dict[str, Any]]:
    input_path = default_input_path(target, _ROOT)
    run_tag = storage_version_to_run_tag(target, storage_ver)
    ordered = load_indices_in_input_order(str(input_path), target_eq=1)
    gt_cwe = load_ground_truth_cwe(str(input_path), target_eq=1)
    ground_truth_keys = set(gt_cwe.keys())

    if method == "cot":
        rdir = method_run_dir(model, "CoT", target, task="classification", results_base=run_tag, results_root=results_root)
        if not rdir.is_dir():
            return None
        pred_cwe = load_predictions_cwe_cot(str(rdir), ordered, ground_truth_keys)
    elif method == "gptlens":
        rdir = method_run_dir(model, "GPTLens", target, task="classification", results_base=run_tag, results_root=results_root) / "rank"
        if not rdir.is_dir():
            return None
        pred_cwe = load_predictions_cwe_gptlens(
            str(rdir), ordered_indices=ordered, ground_truth_keys=ground_truth_keys,
            threshold=threshold, ground_truth_cwe=gt_cwe,
        )
    elif method == "vultrial":
        rdir = method_run_dir(model, "VulTrial", target, task="classification", results_base=run_tag, results_root=results_root) / "final_record"
        if not rdir.is_dir():
            return None
        pred_multi = load_predictions_cwe_vultrial(str(rdir))
        pred_cwe = resolve_vultrial_pred_for_metrics(gt_cwe, pred_multi)
    elif method in ("codebert", "unixcoder"):
        dl_name = "CodeBERT" if method == "codebert" else "UnixCoder"
        rdir = dl_run_dir(dl_name, target, task="classification", results_base=run_tag, results_root=results_root)
        if not rdir.is_dir():
            return None
        pred_cwe = load_predictions_cwe_cot(str(rdir), ordered, ground_truth_keys)
    else:
        return None

    if not pred_cwe:
        return None
    metrics = calculate_cwe_classification_metrics(gt_cwe, pred_cwe)
    return {"method": method, "model": model, "storage_version": storage_ver, **metrics}


def _eval_baseline_once(
    *,
    method: str,
    model: str,
    target: str,
    storage_ver: str,
    results_root: Path,
    threshold: float,
    status: str,
    quiet: bool = False,
) -> Optional[Dict[str, Any]]:
    input_path = default_input_path(target, _ROOT)
    run_tag = storage_version_to_run_tag(target, storage_ver)

    if method == "vulrag":
        result_file = vulrag_result_file(model, target, run_tag, results_root=results_root)
        if not result_file.is_file():
            return None
        metrics = _pair_only_metrics(calculate_metrics_vulrag(str(input_path), str(result_file)))
        if not quiet:
            print_metrics_report(metrics, title=f"VUL-RAG — {model} ({storage_ver})", results_dir=str(result_file), include_code_level=False)
    elif method == "cot":
        rdir = method_run_dir(model, METHOD_FOLDER[method], target, task="detection", results_base=run_tag, results_root=results_root)
        if not rdir.is_dir():
            return None
        metrics = calculate_metrics(str(input_path), str(rdir))
        if not quiet:
            print_metrics_report(metrics, title=f"CoT — {model} ({storage_ver})", results_dir=str(rdir))
    elif method in ("codebert", "unixcoder"):
        dl_name = "CodeBERT" if method == "codebert" else "UnixCoder"
        rdir = dl_run_dir(dl_name, target, results_base=run_tag, results_root=results_root)
        if not rdir.is_dir():
            return None
        metrics = calculate_metrics(str(input_path), str(rdir))
        if not quiet:
            print_metrics_report(metrics, title=f"{dl_name} — {model} ({storage_ver})", results_dir=str(rdir))
    elif method == "gptlens":
        rdir = method_run_dir(model, METHOD_FOLDER[method], target, task="detection", results_base=run_tag, results_root=results_root) / "rank"
        if not rdir.is_dir():
            return None
        metrics = calculate_metrics_gptlens(str(input_path), str(rdir), threshold=threshold)
        if not quiet:
            print_metrics_report(metrics, title=f"GPTLens — {model} ({storage_ver})", results_dir=str(rdir), threshold=threshold)
    elif method == "vultrial":
        rdir = method_run_dir(model, METHOD_FOLDER[method], target, task="detection", results_base=run_tag, results_root=results_root) / "final_record"
        if not rdir.is_dir():
            return None
        metrics = calculate_metrics_vultrial(str(input_path), str(rdir), status=status)
        if not quiet:
            print_metrics_report(metrics, title=f"VulTrial — {model} ({storage_ver})", results_dir=str(rdir), status=status)
    elif method in ("rag_instance", "rag_chunk"):
        rdir = method_run_dir(model, method, target, results_base=run_tag, results_root=results_root)
        if not rdir.is_dir():
            return None
        metrics = calculate_metrics(str(input_path), str(rdir))
        if not quiet:
            print_metrics_report(metrics, title=f"{method} — {model} ({storage_ver})", results_dir=str(rdir))
    else:
        return None

    display = format_metrics_for_display(metrics, include_code_level=(method != "vulrag"))
    return {"method": method, "model": model, "storage_version": storage_ver, **metrics, **display}


def _storage_versions_for_target(target: str) -> List[str]:
    if target == "java-pair":
        return java_pair_storage_versions()
    return [target]


def _evaluate_vckg(
    *,
    model: str,
    targets: Sequence[str],
    results_root: Path,
    save: bool,
    quiet: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "vckg", "model": model, "targets": {}}
    for target in targets:
        entries: List[Dict[str, Any]] = []
        for storage_ver in _storage_versions_for_target(target):
            payload = _eval_vckg_once(
                model=model, target=target, storage_ver=storage_ver,
                results_root=results_root, ablation=None,
            )
            if payload is not None:
                entries.append(payload)

        if not entries:
            print(f"\n[SKIP] VCKG / {model} / {target}: no results found under {results_root}", file=sys.stderr)
            continue

        if target == "java-pair" and len(entries) > 1:
            stats = _aggregate_metrics(entries)
            if not quiet:
                _print_aggregate(f"VCKG / {model} / {target}", stats, n_runs=len(entries))
            report["targets"][target] = {"n_runs": len(entries), "metrics": stats, "runs": entries}
        else:
            stats = _aggregate_metrics([entries[0]])
            if not quiet:
                _print_aggregate(f"VCKG / {model} / {target}", stats, n_runs=1)
            report["targets"][target] = {"n_runs": 1, "metrics": stats, "runs": entries}

        if save:
            out_dir = eval_output_dir("detection", model)
            out_dir.mkdir(parents=True, exist_ok=True)
            if target == "java-pair":
                agg_path = out_dir / "java-pair_aggregate.json"
                agg_path.write_text(json.dumps(report["targets"][target], ensure_ascii=False, indent=2), encoding="utf-8")
                print("Saved:", agg_path)
            else:
                out_path = out_dir / f"{target}.json"
                out_path.write_text(json.dumps(entries[0], ensure_ascii=False, indent=2), encoding="utf-8")
                print("Saved:", out_path)

    return report


def _evaluate_baseline(
    *,
    model: str,
    targets: Sequence[str],
    methods: List[str],
    results_root: Path,
    save: bool,
    threshold: float,
    status: str,
    quiet: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "baseline", "model": model, "methods": {}}
    for method in methods:
        label = METHOD_LABEL.get(method, method)
        method_report: Dict[str, Any] = {}
        for target in targets:
            entries: List[Dict[str, Any]] = []
            storage_versions = _storage_versions_for_target(target)
            quiet_runs = quiet or (target == "java-pair" and len(storage_versions) > 1)
            for storage_ver in storage_versions:
                payload = _eval_baseline_once(
                    method=method,
                    model=model,
                    target=target,
                    storage_ver=storage_ver,
                    results_root=results_root,
                    threshold=threshold,
                    status=status,
                    quiet=quiet_runs,
                )
                if payload is not None:
                    entries.append(payload)

            if not entries:
                print(f"\n[SKIP] baseline / {label} / {target}: no results", file=sys.stderr)
                continue

            if target == "java-pair" and len(entries) > 1:
                stats = _aggregate_metrics(entries)
                if not quiet:
                    _print_aggregate(f"baseline / {label} / {target}", stats, n_runs=len(entries))
                method_report[target] = {"n_runs": len(entries), "metrics": stats, "runs": entries}
            else:
                stats = _aggregate_metrics([entries[0]])
                if not quiet:
                    _print_aggregate(f"baseline / {label} / {target}", stats, n_runs=1)
                method_report[target] = {"n_runs": 1, "metrics": stats, "runs": entries}

        if method_report:
            report["methods"][method] = method_report

        if save and method_report:
            out_dir = results_root / "eval" / "detection"
            out_dir.mkdir(parents=True, exist_ok=True)
            agg_path = out_dir / f"{method}_aggregate.json"
            agg_path.write_text(json.dumps(method_report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Saved:", agg_path)

    return report


def _evaluate_ablation_study(
    *,
    model: str,
    targets: Sequence[str],
    results_root: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return md sections: {target: [rows]}"""
    sections: Dict[str, List[Dict[str, Any]]] = {}
    for target in targets:
        rows: List[Dict[str, Any]] = []
        for abl_id in ABLATION_ROW_ORDER:
            ablation = None if abl_id == "full" else abl_id
            entries: List[Dict[str, Any]] = []
            for storage_ver in _storage_versions_for_target(target):
                payload = _eval_vckg_once(
                    model=model,
                    target=target,
                    storage_ver=storage_ver,
                    results_root=results_root,
                    ablation=ablation,
                )
                if payload is not None:
                    entries.append(payload)
            if not entries:
                print(f"[SKIP] ablation / {ABLATION_DISPLAY[abl_id]} / {target}", file=sys.stderr)
                continue
            stats = _aggregate_metrics(entries)
            n_runs = len(entries)
            _print_aggregate(f"ablation / {ABLATION_DISPLAY[abl_id]} / {target}", stats, n_runs=n_runs)
            rows.append({"method_id": abl_id, "metrics": stats, "n_runs": n_runs})
        if rows:
            sections[target] = rows
    return sections


def _rows_from_classification_report(
    *,
    vckg_report: Optional[Dict[str, Any]],
    baseline_report: Optional[Dict[str, Any]],
    target: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if baseline_report:
        for method_id in CLASSIFICATION_BASELINE_METHODS:
            block = baseline_report.get("methods", {}).get(method_id, {}).get(target)
            if block and block.get("metrics"):
                rows.append({"method_id": method_id, "metrics": block["metrics"], "n_runs": block.get("n_runs", 1)})
    if vckg_report:
        block = vckg_report.get("targets", {}).get(target)
        if block and block.get("metrics"):
            rows.append({"method_id": "ours", "metrics": block["metrics"], "n_runs": block.get("n_runs", 1)})
    order = {**{m: i for i, m in enumerate(CLASSIFICATION_BASELINE_METHODS)}, "ours": 99}
    rows.sort(key=lambda r: order.get(r["method_id"], 50))
    return rows


def _evaluate_vckg_classification(
    *,
    model: str,
    targets: Sequence[str],
    quiet: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "vckg-classification", "model": model, "targets": {}}
    for target in targets:
        entries: List[Dict[str, Any]] = []
        for storage_ver in _storage_versions_for_target(target):
            payload = _eval_vckg_classification_once(model=model, target=target, storage_ver=storage_ver)
            if payload is not None:
                entries.append(payload)
        if not entries:
            print(f"[SKIP] classification / {model} / {target}", file=sys.stderr)
            continue
        stats = _aggregate_metrics(entries, CLASSIFICATION_METRICS)
        if not quiet:
            _print_aggregate(f"classification / {model} / {target}", stats, n_runs=len(entries), keys=CLASSIFICATION_METRICS)
        report["targets"][target] = {"n_runs": len(entries), "metrics": stats, "runs": entries}
    return report


def _method_display(method_id: str, target: str) -> str:
    if method_id == "vultrial" and target == "java-pair":
        return "VulTrial"
    return ROW_DISPLAY.get(method_id, method_id)


def _discover_vckg_models(results_root: Path) -> List[str]:
    if not results_root.is_dir():
        return []
    models: List[str] = []
    for path in sorted(results_root.iterdir()):
        if not path.is_dir() or path.name in _VCKG_NON_MODEL_DIRS:
            continue
        if (path / "primevul").is_dir() or (path / "java-pair").is_dir():
            models.append(path.name)
    known = [m for m in _OTHER_MODELS_ORDER if m in models]
    extra = [m for m in models if m not in _OTHER_MODELS_ORDER]
    return known + extra


def _baseline_methods_for_detection() -> List[str]:
    return ["codebert", "unixcoder", "cot", "gptlens", "vultrial"]


def _baseline_methods_for_retrieval() -> List[str]:
    return ["rag_instance", "rag_chunk"]


def _stats_for_baseline_method(
    *,
    method: str,
    target: str,
    model: str,
    baseline_root: Path,
    threshold: float,
    status: str,
) -> Optional[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for storage_ver in _storage_versions_for_target(target):
        payload = _eval_baseline_once(
            method=method,
            model=model,
            target=target,
            storage_ver=storage_ver,
            results_root=baseline_root,
            threshold=threshold,
            status=status,
            quiet=True,
        )
        if payload is not None:
            entries.append(payload)
    if not entries:
        return None
    if target == "java-pair" and len(entries) > 1:
        return _aggregate_metrics(entries)
    return _aggregate_metrics([entries[0]])


def _stats_for_vckg(
    *,
    model: str,
    target: str,
    vckg_root: Path,
    ablation: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for storage_ver in _storage_versions_for_target(target):
        payload = _eval_vckg_once(
            model=model,
            target=target,
            storage_ver=storage_ver,
            results_root=vckg_root,
            ablation=ablation,
        )
        if payload is not None:
            entries.append(payload)
    if not entries:
        return None
    if target == "java-pair" and len(entries) > 1:
        return _aggregate_metrics(entries)
    return _aggregate_metrics([entries[0]])


def _stats_for_baseline_classification(
    *,
    method: str,
    target: str,
    model: str,
    baseline_root: Path,
    threshold: float,
    status: str,
) -> Optional[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for storage_ver in _storage_versions_for_target(target):
        payload = _eval_baseline_classification_once(
            method=method,
            model=model,
            target=target,
            storage_ver=storage_ver,
            results_root=baseline_root,
            threshold=threshold,
            status=status,
        )
        if payload is not None:
            entries.append(payload)
    if not entries:
        return None
    if target == "java-pair" and len(entries) > 1:
        return _aggregate_metrics(entries, CLASSIFICATION_METRICS)
    return _aggregate_metrics([entries[0]], CLASSIFICATION_METRICS)


def _stats_for_vckg_classification(
    *,
    model: str,
    target: str,
) -> Optional[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for storage_ver in _storage_versions_for_target(target):
        payload = _eval_vckg_classification_once(model=model, target=target, storage_ver=storage_ver)
        if payload is not None:
            entries.append(payload)
    if not entries:
        return None
    if target == "java-pair" and len(entries) > 1:
        return _aggregate_metrics(entries, CLASSIFICATION_METRICS)
    return _aggregate_metrics([entries[0]], CLASSIFICATION_METRICS)


def _build_detection_table_rows(
    *,
    ours_model: str,
    vckg_root: Path,
    baseline_root: Path,
    threshold: float,
    status: str,
) -> List[List[str]]:
    rows: List[List[str]] = []
    for target in TABLE_TARGET_ORDER:
        language = LANGUAGE[target]
        for method_id, _ in DETECTION_TABLE_METHODS:
            if method_id == "ours":
                stats = _stats_for_vckg(model=ours_model, target=target, vckg_root=vckg_root)
            else:
                stats = _stats_for_baseline_method(
                    method=method_id,
                    target=target,
                    model=DEFAULT_BASELINE_MODEL,
                    baseline_root=baseline_root,
                    threshold=threshold,
                    status=status,
                )
            if stats is None:
                continue
            show_std = target == "java-pair"
            rows.append([
                AGENTS_FOR_METHOD[method_id],
                _method_display(method_id, target),
                *[_format_table_metric(stats, k, show_std=show_std) for k in TABLE_METRICS],
                language,
            ])
    return rows


def _build_classification_table_rows(
    *,
    ours_model: str,
    vckg_root: Path,
    baseline_root: Path,
    threshold: float,
    status: str,
) -> List[List[str]]:
    rows: List[List[str]] = []
    for target in TABLE_TARGET_ORDER:
        language = LANGUAGE[target]
        for method_id, _ in CLASSIFICATION_TABLE_METHODS:
            if method_id == "ours":
                stats = _stats_for_vckg_classification(model=ours_model, target=target)
            else:
                stats = _stats_for_baseline_classification(
                    method=method_id,
                    target=target,
                    model=DEFAULT_BASELINE_MODEL,
                    baseline_root=baseline_root,
                    threshold=threshold,
                    status=status,
                )
            if stats is None:
                continue
            show_std = target == "java-pair"
            rows.append([
                AGENTS_FOR_METHOD[method_id],
                _method_display(method_id, target),
                *[_format_class_metric(stats, k, show_std=show_std) for k in CLASSIFICATION_METRICS],
                language,
            ])
    return rows


def _build_retrieval_table_rows(
    *,
    ours_model: str,
    vckg_root: Path,
    baseline_root: Path,
    threshold: float,
    status: str,
) -> List[List[str]]:
    rows: List[List[str]] = []
    for target in TABLE_TARGET_ORDER:
        language = LANGUAGE[target]
        for method_id, display in RETRIEVAL_TABLE_METHODS:
            if method_id == "ours":
                stats = _stats_for_vckg(model=ours_model, target=target, vckg_root=vckg_root)
                name = f"**{display}**"
            else:
                stats = _stats_for_baseline_method(
                    method=method_id,
                    target=target,
                    model=DEFAULT_BASELINE_MODEL,
                    baseline_root=baseline_root,
                    threshold=threshold,
                    status=status,
                )
                name = display
            if stats is None:
                continue
            show_std = target == "java-pair"
            rows.append([
                name,
                *[_format_table_metric(stats, k, show_std=show_std) for k in TABLE_METRICS],
                language,
            ])
    return rows


def _build_ablation_table_rows(
    *,
    ours_model: str,
    vckg_root: Path,
) -> List[List[str]]:
    rows: List[List[str]] = []
    for target in TABLE_TARGET_ORDER:
        dataset = TARGET_TITLE[target]
        use_pct = target == "java-pair"
        for abl_id in ABLATION_ROW_ORDER:
            ablation = None if abl_id == "full" else abl_id
            stats = _stats_for_vckg(
                model=ours_model,
                target=target,
                vckg_root=vckg_root,
                ablation=ablation,
            )
            if stats is None:
                continue
            rows.append([
                ABLATION_DISPLAY[abl_id],
                *[
                    _format_table_metric(
                        stats,
                        k,
                        use_pct_sign=use_pct,
                        show_std=use_pct,
                    )
                    for k in TABLE_METRICS
                ],
                dataset,
            ])
    return rows


def _build_other_models_rows(
    *,
    models: Sequence[str],
    vckg_root: Path,
) -> List[List[str]]:
    rows: List[List[str]] = []
    for model in models:
        for target in TABLE_TARGET_ORDER:
            stats = _stats_for_vckg(model=model, target=target, vckg_root=vckg_root)
            if stats is None:
                continue
            use_pct = target == "java-pair"
            show_std = target == "java-pair"
            rows.append([
                model,
                *[
                    _format_table_metric(
                        stats,
                        k,
                        use_pct_sign=use_pct,
                        show_std=show_std,
                    )
                    for k in TABLE_METRICS
                ],
                TARGET_TITLE[target],
            ])
    return rows


def _render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    header_line = "| " + " | ".join(f"**{h}**" if not h.startswith("**") else h for h in headers) + " |"
    lines = [header_line, sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _write_paper_tables_md(
    path: Path,
    *,
    ours_model: str,
    vckg_root: Path,
    baseline_root: Path,
    threshold: float,
    status: str,
) -> None:
    detection_rows = _build_detection_table_rows(
        ours_model=ours_model,
        vckg_root=vckg_root,
        baseline_root=baseline_root,
        threshold=threshold,
        status=status,
    )
    classification_rows = _build_classification_table_rows(
        ours_model=ours_model,
        vckg_root=vckg_root,
        baseline_root=baseline_root,
        threshold=threshold,
        status=status,
    )
    retrieval_rows = _build_retrieval_table_rows(
        ours_model=ours_model,
        vckg_root=vckg_root,
        baseline_root=baseline_root,
        threshold=threshold,
        status=status,
    )
    ablation_rows = _build_ablation_table_rows(ours_model=ours_model, vckg_root=vckg_root)
    other_models = _discover_vckg_models(vckg_root)
    other_rows = _build_other_models_rows(models=other_models, vckg_root=vckg_root)

    ci_note = ""
    if _active_ci_level() is not None:
        ci_note = f" · Uncertainty: mean ± half-width ({_active_ci_level():.0f}% CI)"
    lines: List[str] = [
        "# Evaluation Tables",
        "",
        f"Ours model: `{ours_model}` · Baseline model: `{DEFAULT_BASELINE_MODEL}`{ci_note}",
        "",
        "## 1. Detection",
        "",
    ]
    lines.extend(_render_markdown_table(
        ["Agents", "Method", "P-C(↑)", "P-V(↓)", "P-B(↓)", "P-R(↓)", "VPS(↑)", "Language"],
        detection_rows,
    ))
    lines.extend(["", "## 2. Classification", ""])
    lines.extend(_render_markdown_table(
        ["Agents", "Method", "Macro-F1", "Weighted-F1", "MCC", "Language"],
        classification_rows,
    ))
    lines.extend(["", "## 3. Comparison of retrieval strategies", ""])
    lines.extend(_render_markdown_table(
        ["Method", "P-C(↑)", "P-V(↓)", "P-B(↓)", "P-R(↓)", "VPS(↑)", "Language"],
        retrieval_rows,
    ))
    lines.extend(["", "## 4. Ablation", ""])
    lines.extend(_render_markdown_table(
        ["Method", "P-C(↑)", "P-V(↓)", "P-B(↓)", "P-R(↓)", "VPS(↑)", "Dataset"],
        ablation_rows,
    ))
    if other_rows:
        lines.extend(["", "## Other Models", ""])
        lines.extend(_render_markdown_table(
            ["Model", "P-C(↑)", "P-V(↓)", "P-B(↓)", "P-R(↓)", "VPS(↑)", "Dataset"],
            other_rows,
        ))
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", path)


def _run_table_mode(
    *,
    ours_model: str,
    vckg_root: Path,
    baseline_root: Path,
    md_path: Path,
    threshold: float,
    status: str,
) -> None:
    print(f"Building tables (Ours={ours_model}, baseline={DEFAULT_BASELINE_MODEL})...")
    _write_paper_tables_md(
        md_path,
        ours_model=ours_model,
        vckg_root=vckg_root,
        baseline_root=baseline_root,
        threshold=threshold,
        status=status,
    )


def _evaluate_baseline_classification(
    *,
    model: str,
    targets: Sequence[str],
    methods: List[str],
    results_root: Path,
    threshold: float,
    status: str,
    quiet: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"mode": "baseline-classification", "model": model, "methods": {}}
    for method in methods:
        if method not in CLASSIFICATION_BASELINE_METHODS:
            continue
        method_report: Dict[str, Any] = {}
        for target in targets:
            entries: List[Dict[str, Any]] = []
            storage_versions = _storage_versions_for_target(target)
            for storage_ver in storage_versions:
                payload = _eval_baseline_classification_once(
                    method=method, model=model, target=target, storage_ver=storage_ver,
                    results_root=results_root, threshold=threshold, status=status,
                )
                if payload is not None:
                    entries.append(payload)
            if not entries:
                continue
            stats = _aggregate_metrics(entries, CLASSIFICATION_METRICS)
            if not quiet:
                _print_aggregate(f"classification / {method} / {target}", stats, n_runs=len(entries), keys=CLASSIFICATION_METRICS)
            method_report[target] = {"n_runs": len(entries), "metrics": stats, "runs": entries}
        if method_report:
            report["methods"][method] = method_report
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate VCKG (run.py) or baseline results. java-pair uses java-pair~java-pair10 mean ± std.",
    )
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name (default: gpt-4o-mini)")
    p.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Results root (default: ./results or ./baseline/results with --baseline)",
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        choices=TARGET_CHOICES,
        help="Dataset: java-pair or primevul (default: both)",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate baseline results instead of run.py (VCKG) outputs",
    )
    p.add_argument(
        "--method",
        type=str,
        default=None,
        choices=METHOD_CHOICES,
        help="Baseline method (default: all). DL = CodeBERT + UnixCoder",
    )
    p.add_argument(
        "--table",
        action="store_true",
        help="Aggregate all results into paper tables (Detection, Classification, Retrieval, Ablation) and save results/results.md",
    )
    p.add_argument(
        "--md-path",
        type=Path,
        default=None,
        help="Markdown output path for --table (default: results/results.md)",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Evaluate VCKG (Ours) and all baselines in one run for comparison table",
    )
    p.add_argument(
        "--task",
        type=str,
        default="detection",
        choices=["detection", "classification"],
        help="Task type (default: detection)",
    )
    p.add_argument(
        "--ablation-study",
        action="store_true",
        help="Evaluate VCKG ablation variants and write results.md",
    )
    p.add_argument("--threshold", type=float, default=5.0, help="GPTLens threshold")
    p.add_argument("--status", type=str, default="action_and_valid_high", choices=["only_valid_high", "action_and_valid_high"])
    p.add_argument(
        "--ci",
        type=float,
        default=95.0,
        metavar="PCT",
        help="Show mean ± CI half-width as 'X.X ± Y.Y (PCT%% CI)' instead of mean ± std (default: 95).",
    )
    args = p.parse_args()

    if args.ci is not None:
        if not (0.0 < args.ci < 100.0):
            p.error("--ci must be between 0 and 100 (exclusive)")
        _set_ci_level(args.ci)

    targets = [args.target] if args.target else list(TARGET_CHOICES)

    vckg_root = (
        args.results_path.resolve()
        if args.results_path is not None
        else VCKG_RESULTS_ROOT.resolve()
    )
    baseline_root = (_BASELINE / "results").resolve()
    md_path = (args.md_path or (VCKG_RESULTS_ROOT / "results.md")).resolve()

    if args.table:
        if args.baseline:
            print("Warning: --baseline ignored with --table", file=sys.stderr)
        _run_table_mode(
            ours_model=args.model,
            vckg_root=vckg_root,
            baseline_root=baseline_root,
            md_path=md_path,
            threshold=args.threshold,
            status=args.status,
        )
        return

    if args.ablation_study:
        if args.task != "detection":
            print("Warning: ablation study only supports detection", file=sys.stderr)
        if args.baseline or args.compare:
            print("Warning: --baseline/--compare ignored with --ablation-study", file=sys.stderr)
        md_sections = _evaluate_ablation_study(model=args.model, targets=targets, results_root=vckg_root)
        return

    if args.task == "classification":
        vckg_report: Optional[Dict[str, Any]] = None
        baseline_report: Optional[Dict[str, Any]] = None
        run_vckg = not args.baseline or args.compare
        run_baseline = args.baseline or args.compare
        if run_vckg:
            vckg_report = _evaluate_vckg_classification(model=args.model, targets=targets)
        if run_baseline:
            methods = [m for m in _expand_methods(args.method) if m in CLASSIFICATION_BASELINE_METHODS]
            baseline_report = _evaluate_baseline_classification(
                model=args.model, targets=targets, methods=methods,
                results_root=baseline_root,
                threshold=args.threshold, status=args.status,
            )
        return

    vckg_report: Optional[Dict[str, Any]] = None
    baseline_report: Optional[Dict[str, Any]] = None

    run_vckg = not args.baseline or args.compare
    run_baseline = args.baseline or args.compare

    if args.compare and args.results_path is not None:
        vckg_root = args.results_path.resolve()

    if run_vckg:
        if args.method and not args.compare:
            print("Warning: --method is ignored for VCKG evaluation", file=sys.stderr)
        vckg_report = _evaluate_vckg(
            model=args.model,
            targets=targets,
            results_root=vckg_root,
            save=False,
        )

    if run_baseline:
        methods = _expand_methods(args.method)
        if args.results_path is not None and args.baseline and not args.compare:
            baseline_root = args.results_path.resolve()
        baseline_report = _evaluate_baseline(
            model=args.model,
            targets=targets,
            methods=methods,
            results_root=baseline_root,
            save=False,
            threshold=args.threshold,
            status=args.status,
        )


if __name__ == "__main__":
    main()
