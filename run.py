from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import subprocess
import json
import time
import traceback

sys.path.append(str(Path(__file__).parent))

from utils.llm import load_llm_config, list_configured_models
from utils.kg_retriever import load_kg
from utils.metrics_loaders import load_records, to_int_id
from utils.dataloader.message_saver import get_messages_dir_for_model_and_version
from agent.agentController import AgentController

MAX_LLM_RETRIES = 3
RETRY_DELAY_SEC = 3


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    keys = (
        "rate limit",
        "ratelimit",
        "too many requests",
        "429",
        "tokens per min",
        "requests per min",
    )
    if any(k in text for k in keys):
        return True
    return "ratelimit" in exc.__class__.__name__.lower()


def _retry_delay(attempt: int, is_rate_limit: bool) -> float:
    if not is_rate_limit:
        return float(RETRY_DELAY_SEC)
    return min(float(RETRY_DELAY_SEC) * (2 ** max(0, attempt - 1)), 60.0)


def _paths_for_version(version: str, task_mode: str = "detection") -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    if version == "java-pair":
        input_path = root / "dataset" / "java-pair" / "java_test_paired.json"
        kg_path = root / "knowledge_graph" / "java-pair" / "java_final_knowledge_graph.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not kg_path.exists():
            raise FileNotFoundError(f"KG not found: {kg_path}")
        return input_path, kg_path
    if version == "primevul":
        input_path = root / "dataset" / "primevul" / "primevul_test_paired.jsonl"
        kg_path = root / "knowledge_graph" / "primevul" / "primevul_final_knowledge_graph.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not kg_path.exists():
            raise FileNotFoundError(f"KG not found: {kg_path}")
        return input_path, kg_path
    raise ValueError(f"Invalid version: {version}")


def _normalize_run_tag(results_base: Optional[Path]) -> Optional[str]:
    if results_base is None:
        return None
    name = results_base.name
    if not name or name == "results":
        return None
    return name


def _resolve_run_message_dirs(
    model_name: str,
    version: str,
    task_mode: str,
    ablation: Optional[str],
    run_tag: Optional[str],
) -> tuple[Optional[str], str]:
    if ablation:
        base_messages_dir = get_messages_dir_for_model_and_version(
            model_name,
            version,
            task_mode="detection",
            ablation=None,
            results_base=run_tag,
        )
        messages_dir = get_messages_dir_for_model_and_version(
            model_name,
            version,
            task_mode="detection",
            ablation=ablation,
            results_base=run_tag,
        )
        return base_messages_dir, messages_dir

    messages_dir = get_messages_dir_for_model_and_version(
        model_name,
        version,
        task_mode=task_mode,
        ablation=None,
        results_base=run_tag,
    )
    return None, messages_dir


def _completed_indices(messages_dir: str) -> set[int]:
    """Return indices with a readable final judge result."""
    completed: set[int] = set()
    base = Path(messages_dir)
    if not base.is_dir():
        return completed
    for index_dir in base.iterdir():
        if not index_dir.is_dir() or not index_dir.name.isdigit():
            continue
        judge_path = index_dir / "judge" / "1.json"
        if not judge_path.is_file():
            continue
        try:
            payload = json.loads(judge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("judgments"), list):
            completed.add(int(index_dir.name))
    return completed


def analyze_single_code(
    code: str,
    kg_retriever,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    index: int = 0,
    version: str = "java-pair",
    task_mode: str = "detection",
    debug: bool = True,
    ablation: Optional[str] = None,
    reuse_messages_dir: Optional[str] = None,
    results_base: Optional[str] = None,
) -> Dict[str, Any]:
    controller = AgentController(kg_retriever=kg_retriever, model_name=model_name, config=config, debug=debug)
    result = controller.run(
        code=code,
        index=index,
        version=version,
        claims_count=3,
        task_mode=task_mode,
        ablation=ablation,
        reuse_messages_dir=reuse_messages_dir,
        results_base=results_base,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run vulnerability analysis using agents")
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["java-pair", "primevul"],
        help="Dataset/KG preset. Must be specified.",
    )
    parser.add_argument("--config", type=str, default="./llm_config.json", help="LLM config file path")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Override the test dataset path (must already be in CLEAR format)",
    )
    parser.add_argument(
        "--knowledge-graph", type=Path, default=None,
        help="Override the final knowledge graph path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from llm_config.json (default: first model in config)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detection",
        choices=["detection", "classification"],
        help="Task mode (default: detection)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=["no_evidence", "no_entrypoint", "no_precondition", "no_root_cause", "no_fix_intent"],
        help="Ablation study: save under results/{model}/{version}/ablation/{ablation}/",
    )
    parser.add_argument(
        "--results-base",
        type=Path,
        default=None,
        help="Run tag for output suffix (e.g. results2 -> java-pair2). Default: no suffix.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip indices that already have a valid judge/1.json result",
    )
    args = parser.parse_args()
    llm_config = load_llm_config(args.config)
    configured_models = list_configured_models(llm_config)
    if args.model:
        model_name = args.model
    elif configured_models:
        model_name = configured_models[0]
    else:
        model_name = "gpt-4o-mini"
    config = llm_config

    task_mode = args.task
    run_tag = _normalize_run_tag(args.results_base)

    if args.input is None or args.knowledge_graph is None:
        default_input_path, default_kg_path = _paths_for_version(args.version, task_mode)
    else:
        default_input_path = default_kg_path = None
    input_path = args.input.resolve() if args.input is not None else default_input_path
    kg_path = args.knowledge_graph.resolve() if args.knowledge_graph is not None else default_kg_path
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not kg_path.is_file():
        raise FileNotFoundError(f"KG not found: {kg_path}")

    base_messages_dir, messages_dir = _resolve_run_message_dirs(
        model_name, args.version, task_mode, args.ablation, run_tag
    )
    if args.ablation:
        print(f"[info] ablation read (collector): {base_messages_dir}")
    print(f"[info] output messages: {messages_dir}")
    kg_retriever = load_kg(str(kg_path))

    records = load_records(input_path)
    if task_mode == "classification":
        records = [r for r in records if r.get("target") == 1]
    indexed_records = list(enumerate(records))
    error_dir = Path(messages_dir) / "error"
    error_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_indices(messages_dir) if args.resume else set()
    if args.resume:
        print(f"[resume] completed indices found: {len(completed)}")

    for pos, rec in indexed_records:
        func = rec.get("func", "") if isinstance(rec, dict) else ""
        if not func:
            continue
        idx = to_int_id(rec, pos)
        if idx in completed:
            continue

        last_error: Optional[Exception] = None
        last_traceback = ""
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                analyze_single_code(
                    func,
                    kg_retriever,
                    model_name=model_name,
                    config=config,
                    index=idx,
                    version=args.version,
                    task_mode=task_mode,
                    debug=True,
                    ablation=args.ablation,
                    reuse_messages_dir=base_messages_dir if args.ablation else None,
                    results_base=run_tag,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                last_traceback = traceback.format_exc()
                print(f"[warn] idx={idx} attempt={attempt}/{MAX_LLM_RETRIES} failed: {e}")
                if attempt < MAX_LLM_RETRIES:
                    time.sleep(_retry_delay(attempt, _is_rate_limit_error(e)))

        if last_error is not None:
            err_path = error_dir / f"{idx}-err.json"
            err_payload = {
                "index": idx,
                "model": model_name,
                "version": args.version,
                "task_mode": task_mode,
                "attempts": MAX_LLM_RETRIES,
                "error": str(last_error),
                "traceback": last_traceback,
            }
            err_path.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[skip] idx={idx} (error saved: {err_path})")
            continue
        (error_dir / f"{idx}-err.json").unlink(missing_ok=True)
    _run_post_eval(
        task_mode=task_mode,
        version=args.version,
        model=model_name,
        ablation=args.ablation,
        run_tag=run_tag,
        input_path=input_path,
    )


def _run_post_eval(
    task_mode: str,
    version: str,
    model: str,
    ablation: Optional[str] = None,
    run_tag: Optional[str] = None,
    input_path: Optional[Path] = None,
) -> None:
    evaluate_script = Path(__file__).parent / ("eval_classification.py" if task_mode == "classification" else "eval_detection.py")
    cmd = [sys.executable, str(evaluate_script), "--version", version, "--model", model]
    if ablation:
        cmd.extend(["--ablation", ablation])
    if run_tag:
        cmd.extend(["--results-base", run_tag])
    if input_path is not None:
        cmd.extend(["--input", str(input_path)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
