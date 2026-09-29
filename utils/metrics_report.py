from __future__ import annotations

from typing import Any, Dict, Optional


def format_metrics_for_display(metrics: Dict[str, Any], *, include_code_level: bool = True) -> Dict[str, Any]:
    total = metrics.get("total_pairs", 0)
    div = (
        metrics.get("p_c_count", 0)
        + metrics.get("p_v_count", 0)
        + metrics.get("p_b_count", 0)
        + metrics.get("p_r_count", 0)
    )
    out: Dict[str, Any] = {
        "pair_count": total,
        "pc": metrics.get("p_c_count", 0),
        "pv": metrics.get("p_v_count", 0),
        "pb": metrics.get("p_b_count", 0),
        "pr": metrics.get("p_r_count", 0),
        "divider": div,
        "pc_pct": (metrics.get("p_c_count", 0) / div * 100) if div else 0.0,
        "pv_pct": (metrics.get("p_v_count", 0) / div * 100) if div else 0.0,
        "pb_pct": (metrics.get("p_b_count", 0) / div * 100) if div else 0.0,
        "pr_pct": (metrics.get("p_r_count", 0) / div * 100) if div else 0.0,
        "VPS": metrics.get("VPS", 0.0),
    }
    if include_code_level:
        out["code_total"] = total * 2
        out["Accuracy"] = metrics.get("Accuracy", 0.0)
        out["Precision"] = metrics.get("Precision", 0.0)
        out["Recall"] = metrics.get("Recall", 0.0)
        out["F1-Score"] = metrics.get("F1-Score", 0.0)
    return out


def print_metrics_report(
    metrics: Dict[str, Any],
    title: str = "Evaluation Results",
    results_dir: Optional[str] = None,
    status: Optional[str] = None,
    threshold: Optional[float] = None,
    include_code_level: bool = True,
) -> None:
    out = format_metrics_for_display(metrics, include_code_level=include_code_level)
    print("=" * 60)
    print(title)
    print("=" * 60)
    if results_dir is not None:
        print("Results dir:", results_dir)
    if status is not None:
        print("Status:", status)
    if threshold is not None:
        print("Threshold:", threshold)
    print("Pair count:", out["pair_count"])
    if include_code_level:
        print("Code total:", out["code_total"])
    if out["divider"] == 0:
        print("(no pairs matched)")
    print("--- Pair-level ---")
    print("P-C: {:.2f}% ({})".format(out["pc_pct"], out["pc"]))
    print("P-V: {:.2f}% ({})".format(out["pv_pct"], out["pv"]))
    print("P-B: {:.2f}% ({})".format(out["pb_pct"], out["pb"]))
    print("P-R: {:.2f}% ({})".format(out["pr_pct"], out["pr"]))
    print("VPS: {:.2f}%".format(out["VPS"] * 100))
    if include_code_level:
        print("--- Code-level ---")
        print("Accuracy:  {:.4f}".format(out["Accuracy"]))
        print("Precision: {:.4f}".format(out["Precision"]))
        print("Recall:    {:.4f}".format(out["Recall"]))
        print("F1-Score:  {:.4f}".format(out["F1-Score"]))
    print("=" * 60)


def print_cwe_metrics_report(
    metrics: Dict[str, Any],
    title: str = "CWE Classification Results",
    results_dir: Optional[str] = None,
) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    if results_dir is not None:
        print("Results dir:", results_dir)
    print("n_eval:     ", metrics.get("n_eval", 0))
    print("Macro-F1:   {:.4f}".format(metrics.get("Macro-F1", 0.0)))
    print("Weighted-F1: {:.4f}".format(metrics.get("Weighted-F1", 0.0)))
    print("MCC:        {:.4f}".format(metrics.get("MCC", 0.0)))
    print("=" * 60)
