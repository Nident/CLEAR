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
    calculate_metrics,
    calculate_metrics_vultrial,
    load_ground_truth,
    load_pairs_auto,
    calculate_pairwise_metrics,
    load_predictions_from_messages,
    load_indices_in_input_order,
    format_metrics_for_display,
    print_metrics_report,
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


def _eval_one_model(
    input_path: Path,
    model_dir: Path,
    model_name: str,
    status: str,
    output_dir: Optional[Path],
    version: str,
) -> Optional[dict]:
    input_str = str(input_path)
    has_txt = False
    if has_messages_results(model_dir):
        gt = load_ground_truth(input_str)
        ordered = load_indices_in_input_order(input_str)
        pairs = load_pairs_auto(input_str, gt)
        preds = load_predictions_from_messages(
            str(model_dir),
            ordered_indices=ordered,
            ground_truth_keys=set(gt.keys()),
        )
        metrics = calculate_pairwise_metrics(gt, preds, pairs, skip_missing_pairs=True)
        title = f"Evaluation Results — {model_name} (messages)"
    else:
        has_json = any(model_dir.glob("*.json"))
        has_txt = any(model_dir.glob("*.txt"))
        if not has_json and not has_txt:
            return None
        if has_json:
            metrics = calculate_metrics(input_str, str(model_dir))
            title = f"Evaluation Results — {model_name} (JSON/CoT)"
        else:
            metrics = calculate_metrics_vultrial(input_str, str(model_dir), status=status)
            title = f"VulTrial Evaluation Results — {model_name}"
    print_metrics_report(
        metrics, title=title, results_dir=str(model_dir), status=status if has_txt else None,
    )
    out = format_metrics_for_display(metrics)
    payload = {"model": model_name, **metrics, **out}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{version}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved:", out_file)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate detection (java-pair / primevul)")
    p.add_argument("--version", required=True, choices=["java-pair", "primevul"], help="Dataset version")
    p.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["no_evidence", "no_entrypoint", "no_precondition", "no_root_cause", "no_fix_intent", "evidence_only"],
        help="Evaluate ablation results under results/{model}/{version}/ablation/{ablation}/",
    )
    p.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    p.add_argument("--run-decision", action="store_true", help="Run decision from agent messages before evaluating")
    p.add_argument("--input", type=Path, default=None, help="Override input file")
    p.add_argument(
        "--results-base",
        type=str,
        default=None,
        help="Run tag suffix (e.g. results2 -> java-pair2)",
    )
    p.add_argument("--output", type=Path, default=None, help="Output dir for eval JSON")
    p.add_argument("--status", type=str, default="action_and_valid_high", choices=["only_valid_high", "action_and_valid_high"])
    args = p.parse_args()

    run_tag = args.results_base
    storage_ver = _storage_key(args.version, run_tag, args.ablation)

    if args.run_decision:
        from agent.decision import run_decision_from_messages
        run_decision_from_messages(
            model=args.model,
            version=args.version,
            messages_dir=str(
                resolve_run_messages_dir(
                    args.model, args.version, "detection", args.ablation, run_tag
                )
            ),
        )

    input_path = args.input or default_input_path(args.version, _ROOT)
    if not input_path.exists():
        print("Error: input not found:", input_path, file=sys.stderr)
        raise SystemExit(1)

    model_dir = resolve_run_messages_dir(
        args.model, args.version, "detection", args.ablation, run_tag
    )
    if not model_dir.is_dir():
        print("Error: results dir not found:", model_dir, file=sys.stderr)
        raise SystemExit(1)
    if not has_messages_results(model_dir) and not any(model_dir.glob("*.json")) and not any(model_dir.glob("*.txt")):
        print("Error: no result files found under:", model_dir, file=sys.stderr)
        raise SystemExit(1)

    output_dir = (
        args.output.resolve()
        if args.output is not None
        else eval_output_dir("detection", args.model)
    )

    payload = _eval_one_model(
        input_path,
        model_dir,
        args.model,
        args.status,
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
