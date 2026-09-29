from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _empty_metrics() -> Dict[str, Any]:
    return {
        "P-C": 0.0,
        "P-V": 0.0,
        "P-B": 0.0,
        "P-R": 0.0,
        "VPS": 0.0,
        "Accuracy": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1-Score": 0.0,
        "total_pairs": 0,
        "p_c_count": 0,
        "p_v_count": 0,
        "p_b_count": 0,
        "p_r_count": 0,
    }


def calculate_classification_metrics(true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    if not true_labels or not pred_labels:
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0}
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)
    n = len(true_labels)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}


def calculate_pairwise_metrics(
    ground_truth: Dict[int, Dict[str, Any]],
    predictions: Dict[int, int],
    pairs: Optional[List[Tuple[int, int]]] = None,
    skip_missing_pairs: bool = False,
) -> Dict[str, float]:
    if not pairs:
        return _empty_metrics()

    p_c = p_v = p_b = p_r = 0
    all_true: List[int] = []
    all_pred: List[int] = []

    for idx1, idx2 in pairs:
        if skip_missing_pairs and (idx1 not in predictions or idx2 not in predictions):
            continue
        gt1 = int(ground_truth.get(idx1, {}).get("target", 0) or 0)
        gt2 = int(ground_truth.get(idx2, {}).get("target", 0) or 0)
        pred1 = int(predictions.get(idx1, 0) or 0)
        pred2 = int(predictions.get(idx2, 0) or 0)
        all_true.extend([gt1, gt2])
        all_pred.extend([pred1, pred2])
        if pred1 == gt1 and pred2 == gt2:
            p_c += 1
        elif pred1 == 1 and pred2 == 1:
            p_v += 1
        elif pred1 == 0 and pred2 == 0:
            p_b += 1
        elif pred1 == 0 and pred2 == 1:
            p_r += 1

    total_pairs = len(all_true) // 2
    if total_pairs == 0:
        return _empty_metrics()

    divider = p_c + p_v + p_b + p_r
    p_c_ratio = p_c / divider if divider > 0 else 0.0
    p_v_ratio = p_v / divider if divider > 0 else 0.0
    p_b_ratio = p_b / divider if divider > 0 else 0.0
    p_r_ratio = p_r / divider if divider > 0 else 0.0
    vps = p_c_ratio - p_r_ratio
    cls = calculate_classification_metrics(all_true, all_pred)
    return {
        "P-C": p_c_ratio,
        "P-V": p_v_ratio,
        "P-B": p_b_ratio,
        "P-R": p_r_ratio,
        "VPS": vps,
        "Accuracy": cls["Accuracy"],
        "Precision": cls["Precision"],
        "Recall": cls["Recall"],
        "F1-Score": cls["F1-Score"],
        "total_pairs": total_pairs,
        "p_c_count": p_c,
        "p_v_count": p_v,
        "p_b_count": p_b,
        "p_r_count": p_r,
    }


def calculate_cwe_classification_metrics(
    gt_cwe: Dict[int, str], pred_cwe: Dict[int, str]
) -> Dict[str, Any]:
    common = sorted(set(gt_cwe.keys()) & set(pred_cwe.keys()))
    if not common:
        return {
            "Macro-F1": 0.0,
            "Weighted-F1": 0.0,
            "MCC": 0.0,
            "per_cwe_f1": {},
            "n_eval": 0,
        }
    y_true = [gt_cwe[i] for i in common]
    y_pred = [pred_cwe[i] for i in common]
    classes = sorted(set(y_true) | set(y_pred))
    n = len(common)

    tp_fp_fn: Dict[str, Tuple[int, int, int]] = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        tp_fp_fn[c] = (tp, fp, fn)

    f1_list: List[float] = []
    per_cwe_f1: Dict[str, float] = {}
    support: Dict[str, int] = {}
    for c in classes:
        tp, fp, fn = tp_fp_fn[c]
        support[c] = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_cwe_f1[c] = f1
        f1_list.append(f1)
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    weighted_f1 = sum(support[c] * per_cwe_f1[c] for c in classes) / n if n > 0 else 0.0

    conf: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        conf[t][p] = conf[t].get(p, 0) + 1
    n_kk = sum(conf[c][c] for c in classes)
    n_k_star = {c: sum(conf[c].values()) for c in classes}
    n_star_k = {c: sum(conf[c2][c] for c2 in classes) for c in classes}
    cov = n_kk * n - sum(n_k_star[c] * n_star_k[c] for c in classes)
    var_t = n * n - sum(n_k_star[c] ** 2 for c in classes)
    var_p = n * n - sum(n_star_k[c] ** 2 for c in classes)
    mcc = cov / (var_t * var_p) ** 0.5 if (var_t * var_p) > 0 else 0.0
    mcc = max(-1.0, min(1.0, mcc))
    return {
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "MCC": mcc,
        "per_cwe_f1": per_cwe_f1,
        "n_eval": n,
    }
