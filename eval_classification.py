from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.metrics import (
    calculate_cwe_classification_metrics,
    load_ground_truth_cwe,
    load_indices_in_input_order,
    load_predictions_cwe_cot,
    load_predictions_cwe_from_messages,
    print_cwe_metrics_report,
)
from utils.eval_common import (
    has_messages_results,
    default_input_path,
    eval_output_dir,
    resolve_run_messages_dir,
)
from utils.results_paths import storage_version

MODEL_NAME = "gpt-4o-mini"


def _storage_key(version: str, run_tag: Optional[str], ablation: Optional[str]) -> str:
    ver = storage_version(version, run_tag)
    if ablation:
        return f"{ver}_{ablation}"
    return ver


def _eval_one_model_classification(
    input_path: Path,
    model_dir: Path,
    model_name: str,
    output_dir: Optional[Path],
    version: str,
) -> Optional[dict]:
    ordered_indices = load_indices_in_input_order(str(input_path), target_eq=1)
    gt_cwe = load_ground_truth_cwe(str(input_path), target_eq=1)
    if has_messages_results(model_dir):
        pred_cwe = load_predictions_cwe_from_messages(
            str(model_dir),
            ordered_indices=ordered_indices,
            ground_truth_keys=set(gt_cwe.keys()),
            ground_truth_cwe=gt_cwe,
        )
    else:
        if not any(model_dir.glob("*.json")):
            return None
        pred_cwe = load_predictions_cwe_cot(
            str(model_dir),
            ordered_indices=ordered_indices,
            ground_truth_keys=set(gt_cwe.keys()),
        )
    if not pred_cwe:
        return None
    metrics = calculate_cwe_classification_metrics(gt_cwe, pred_cwe)
    title = f"CWE Classification Results — {model_name}"
    print_cwe_metrics_report(metrics, title=title, results_dir=str(model_dir))
    payload = {"model": model_name, **metrics}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{version}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved:", out_file)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate CWE classification (java-pair / primevul)")
    p.add_argument("--version", required=True, choices=["java-pair", "primevul"], help="Dataset version")
    p.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["no_evidence", "no_entrypoint", "no_precondition", "no_root_cause", "no_fix_intent"],
        help="Evaluate ablation results under results/{model}/{version}/ablation/{ablation}/",
    )
    p.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    p.add_argument("--input", type=Path, default=None, help="Override input file")
    p.add_argument(
        "--results-base",
        type=str,
        default=None,
        help="Run tag suffix (e.g. results2 -> java-pair2)",
    )
    p.add_argument("--output", type=Path, default=None, help="Output dir for eval JSON")
    args = p.parse_args()

    run_tag = args.results_base
    storage_ver = _storage_key(args.version, run_tag, args.ablation)

    input_path = args.input or default_input_path(args.version, _ROOT)
    if not input_path.exists():
        print("Error: input not found:", input_path, file=sys.stderr)
        raise SystemExit(1)

    model_dir = resolve_run_messages_dir(
        args.model, args.version, "classification", args.ablation, run_tag
    )
    if not model_dir.is_dir():
        print("Error: results dir not found:", model_dir, file=sys.stderr)
        raise SystemExit(1)
    if not has_messages_results(model_dir) and not any(model_dir.glob("*.json")):
        print("Error: no result files found under:", model_dir, file=sys.stderr)
        raise SystemExit(1)

    output_dir = (
        args.output.resolve()
        if args.output is not None
        else eval_output_dir("classification", args.model)
    )

    payload = _eval_one_model_classification(
        input_path,
        model_dir,
        args.model,
        output_dir,
        storage_ver,
    )
    if payload is None:
        raise SystemExit(1)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"{storage_ver}_summary.json"
        summary_path.write_text(json.dumps({"models": [payload]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()
