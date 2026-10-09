from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.llm import get_llm, load_llm_config
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge_graph.io_common import read_json_or_jsonl, write_json

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else (lambda x: x)

_CWE_RE = re.compile(r"^cwe[-_: ]?(\d+)$", re.IGNORECASE)
_CONSOLE_LOCK = Lock()


def log_stage(message: str) -> None:
    """Print one complete progress line, even when worker threads run concurrently."""
    with _CONSOLE_LOCK:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        print(f"[{timestamp}] [knowledge_data] {message}", flush=True)

def norm_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_label(s: str) -> str:
    return norm_space(s).lower()

def norm_cwe_id(s: str) -> str:
    s = norm_space(s)
    if not s:
        return ""
    m = _CWE_RE.match(s)
    if m:
        return f"CWE-{int(m.group(1))}"
    if s.upper().startswith("CWE-"):
        tail = s.split("-", 1)[1].strip()
        return f"CWE-{tail}" if tail else ""
    return s


SYSTEM_PROMPT = """
You are a knowledge data extractor for software vulnerabilities.
Given code, patch diff, and a CVE description, extract ONE evidence-grounded knowledge data, do not add facts not supported by the inputs.
All the field must be abstract, do not include any specific code comments.
Output:
Return valid JSON only. No markdown, no extra text.
""".strip()

USER_PROMPT = """
[meta]
{meta_json}

[code_before]
{code_before}

[patch_diff]
{diff}

[cve_description]
{cve_desc}

Extract a single knowledge data for this vulnerability.
Output JSON schema (return ONLY this JSON):
{{
  "nodes": [
    {{
      "type": "Entrypoint",
      "summary": "specific entrypoint label (1–5 words)",
      "details": "why this entrypoint applies"
    }},
    {{
      "type": "vulnerability_precondition",
      "summary": "specific precondition (1–5 words)",
      "details": "1–2 sentences grounded in inputs"
    }},
    {{
      "type": "vulnerability_root_cause",
      "summary": "specific root cause (1–5 words)",
      "details": "1–2 sentences grounded in inputs"
    }},
    {{
      "type": "fix_intent",
      "summary": "specific fix intent (1–5 words)",
      "details": "1–2 sentences grounded in inputs (use diff)"
    }},
    {{
      "type": "CWE",
      "CWE_ID": "CWE-XXX",
      "CWE_ID_NAME": "CWE name"
    }}
  ]
}}
""".strip()


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None

    if "```" in t:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", t, re.IGNORECASE)
        if m:
            t = m.group(1).strip()

    if not (t.startswith("{") and t.endswith("}")):
        m = re.search(r"(\{[\s\S]*\})", t)
        if m:
            t = m.group(1).strip()

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


REQUIRED_TYPES = (
    "Entrypoint",
    "vulnerability_precondition",
    "vulnerability_root_cause",
    "fix_intent",
)
OPTIONAL_TYPES = ("CWE",)
ALL_TYPES = set(REQUIRED_TYPES) | set(OPTIONAL_TYPES)

def normalize_and_validate_nodes(nodes: List[Dict[str, Any]], meta_cwe_present: bool) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(nodes, list):
        return None

    cleaned: List[Dict[str, Any]] = []
    seen_type: set = set()

    for n in nodes:
        if not isinstance(n, dict):
            continue
        t = norm_space(n.get("type", ""))
        if t not in ALL_TYPES:
            continue
        if t in seen_type:
            continue

        if t == "CWE":
            if not meta_cwe_present:
                continue
            cwe_id = norm_cwe_id(n.get("CWE_ID", "") or n.get("cwe_id", ""))
            cwe_name = norm_space(n.get("CWE_ID_NAME", "") or n.get("CWE_NAME", "") or n.get("cwe_name", ""))
            if not cwe_id or not cwe_name:
                continue
            cleaned.append({"type": "CWE", "CWE_ID": cwe_id, "CWE_ID_NAME": cwe_name})
            seen_type.add("CWE")
            continue

        summary_raw = norm_space(n.get("summary", "") or n.get("label", ""))
        details = norm_space(n.get("details", ""))

        if not summary_raw or not details:
            continue

        summary = summary_raw if t == "Entrypoint" else norm_label(summary_raw)
        cleaned.append({"type": t, "summary": summary, "details": details})
        seen_type.add(t)

    present = {x.get("type") for x in cleaned}
    if not set(REQUIRED_TYPES).issubset(present):
        return None

    order = {t: i for i, t in enumerate(list(REQUIRED_TYPES) + list(OPTIONAL_TYPES))}
    cleaned.sort(key=lambda x: order.get(x.get("type", ""), 10**9))

    return cleaned


def _detect_dataset_format(input_path: str) -> str:
    path_lower = input_path.lower()
    if "primevul" in path_lower:
        return "primevul"
    return "java-pair"


def _validate_required_fields(item: Dict[str, Any], dataset_format: str) -> tuple[bool, List[str]]:
    def _is_valid(val: Any) -> bool:
        if val is None:
            return False
        if val == "":
            return False
        if isinstance(val, str) and not val.strip():
            return False
        return True

    missing: List[str] = []
    if not _is_valid(item.get("cve_id")):
        missing.append("cve_id")
    if not _is_valid(item.get("cwe_id")):
        missing.append("cwe_id")
    if not _is_valid(item.get("func_before")):
        missing.append("func_before")
    if not _is_valid(item.get("func_diff")):
        missing.append("func_diff")
    if dataset_format == "primevul":
        if not _is_valid(item.get("cve_desc")):
            missing.append("cve_desc")
    else:
        if not _is_valid(item.get("cve_description") or item.get("cve_desc")):
            missing.append("cve_description")

    return (len(missing) == 0, missing)


def llm_extract_subgraph(
    llm,
    item: Dict[str, Any],
    dataset_format: str,
    usage: Optional[Dict[str, int]] = None,
    usage_lock: Optional[Lock] = None,
) -> Optional[Dict[str, Any]]:
    idx = item.get("index", item.get("idx"))
    meta = {
        "index": item.get("index"),
        "cve_id": item.get("cve_id"),
        "cwe_ids": item.get("cwe_id")
    }
    code_before = item.get("func_before")
    diff = item.get("func_diff")
    if dataset_format == "primevul":
        cve_desc = item.get("cve_desc")
    else:
        cve_desc = item.get("cve_description") or item.get("cve_desc")

    prompt = USER_PROMPT.format(
        meta_json=json.dumps(meta, ensure_ascii=False),
        code_before=code_before,
        diff=diff,
        cve_desc=cve_desc,
    )

    log_stage(
        f"index={idx} stage=llm_request_sent "
        f"model={getattr(llm, 'model_name', 'unknown')} prompt_chars={len(prompt)}"
    )
    request_started = time.monotonic()
    resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
    request_elapsed = time.monotonic() - request_started
    content = resp.content if hasattr(resp, "content") else str(resp)
    log_stage(
        f"index={idx} stage=llm_response elapsed_seconds={request_elapsed:.1f} "
        f"chars={len(content or '')}"
    )
    if usage is not None and usage_lock is not None:
        token_usage = (getattr(resp, "response_metadata", {}) or {}).get("token_usage", {})
        with usage_lock:
            usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + token_usage.get("prompt_tokens", 0)
            usage["completion_tokens"] = usage.get("completion_tokens", 0) + token_usage.get("completion_tokens", 0)
            usage["calls"] = usage.get("calls", 0) + 1
    obj = extract_json_from_text(content)
    if obj and isinstance(obj.get("nodes"), list):
        log_stage(f"index={idx} stage=json_parsed nodes={len(obj['nodes'])}")
        return obj
    preview = norm_space(content)[:300]
    log_stage(f"index={idx} stage=json_parse_failed response_preview={preview!r}")
    return None


def _log_error_to_file(
    error_path: Path,
    error_lock: Lock,
    index: Any,
    error_msg: str,
) -> None:
    try:
        with error_lock:
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_path, "a", encoding="utf-8") as f:
                f.write(f"index={index}\t{error_msg}\n")
    except Exception:
        pass


def process_one(
    model_name: Optional[str],
    config: Dict[str, Any],
    item: Dict[str, Any],
    dataset_format: str,
    usage: Optional[Dict[str, int]] = None,
    usage_lock: Optional[Lock] = None,
    error_path: Optional[Path] = None,
    error_lock: Optional[Lock] = None,
) -> Dict[str, Any]:
    idx = item.get("index", item.get("idx"))
    item_started = time.monotonic()
    err_path = error_path
    err_lock = error_lock or Lock()
    log_stage(f"index={idx} stage=start")

    is_valid, missing = _validate_required_fields(item, dataset_format)
    if not is_valid:
        msg = f"missing_required_fields: {','.join(missing)}"
        log_stage(f"index={idx} stage=input_validation_failed error={msg}")
        if err_path:
            _log_error_to_file(err_path, err_lock, idx, msg)
        return {
            "index": idx,
            "cve_id": item.get("cve_id"),
            "status": "error",
            "error": msg,
        }

    try:
        log_stage(f"index={idx} stage=input_validated")
        llm = get_llm(model_name, config)
        raw = llm_extract_subgraph(llm, item, dataset_format, usage=usage, usage_lock=usage_lock)
        if raw is None:
            msg = "llm_parse_failed"
            log_stage(f"index={idx} stage=failed error={msg}")
            if err_path:
                _log_error_to_file(err_path, err_lock, idx, msg)
            return {
                "index": idx,
                "cve_id": item.get("cve_id"),
                "status": "error",
                "error": msg,
            }

        meta_cwe = item.get("cwe_id")
        meta_cwe_present = bool(meta_cwe)
        nodes = normalize_and_validate_nodes(raw.get("nodes", []), meta_cwe_present=meta_cwe_present)
        if nodes is None:
            msg = "missing_required_nodes(entrypoint/precondition/root_cause/fix_intent) or invalid_fields"
            returned_types = [n.get("type") for n in raw.get("nodes", []) if isinstance(n, dict)]
            log_stage(f"index={idx} stage=node_validation_failed types={returned_types} error={msg}")
            if err_path:
                _log_error_to_file(err_path, err_lock, idx, msg)
            return {
                "index": idx,
                "cve_id": item.get("cve_id"),
                "status": "error",
                "error": msg,
            }

        log_stage(
            f"index={idx} stage=success elapsed_seconds={time.monotonic() - item_started:.1f} "
            f"nodes={len(nodes)}"
        )
        return {
            "index": idx,
            "cve_id": item.get("cve_id"),
            "status": "success",
            "data": {"nodes": nodes},
        }
    except Exception as e:
        msg = f"{type(e).__name__}: {e!s}"
        log_stage(
            f"index={idx} stage=exception elapsed_seconds={time.monotonic() - item_started:.1f} "
            f"error={msg}"
        )
        if err_path:
            _log_error_to_file(err_path, err_lock, idx, msg)
        return {
            "index": idx,
            "cve_id": item.get("cve_id"),
            "status": "error",
            "error": msg,
        }


def build_knowledge_data(
    input_path: str,
    output_path: str,
    llm_config_path: str,
    model_name: str,
    threads: int,
    limit: Optional[int],
) -> None:
    log_stage(f"stage=load_input path={input_path}")
    data = read_json_or_jsonl(input_path)
    total_input = len(data)
    if limit is not None:
        data = data[: max(0, int(limit))]
    log_stage(f"stage=input_loaded total={total_input} selected={len(data)} limit={limit}")

    log_stage(f"stage=load_config path={llm_config_path}")
    config = load_llm_config(llm_config_path)
    if not model_name:
        models = config.get("models", [])
        if not models or not isinstance(models[0], dict) or not models[0].get("name"):
            raise ValueError("No model specified and llm_config.json has no configured model")
        model_name = str(models[0]["name"])
    get_llm(model_name, config)
    log_stage(f"stage=config_validated model={model_name}")

    dataset_format = _detect_dataset_format(input_path)
    output_parent = Path(output_path).resolve().parent
    if dataset_format in str(output_parent):
        error_dir = output_parent / "error"
    else:
        error_dir = output_parent / dataset_format / "error"
    error_dir.mkdir(parents=True, exist_ok=True)
    error_path = error_dir / "knowledge_data.err"
    log_stage(f"stage=paths output={output_path} errors={error_path}")
    error_lock = Lock()

    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    usage_lock = Lock()

    results: List[Dict[str, Any]] = []
    log_stage(f"stage=workers_start threads={max(1, int(threads))} items={len(data)}")
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        pending = {
            ex.submit(
                process_one,
                model_name,
                config,
                item,
                dataset_format,
                usage,
                usage_lock,
                error_path,
                error_lock,
            )
            for item in data
        }
        with tqdm(total=len(pending), desc="Extracting knowledge data") as progress:
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                if not done:
                    running = sum(1 for fut in pending if fut.running())
                    queued = len(pending) - running
                    log_stage(
                        f"stage=waiting completed={len(results)}/{len(data)} "
                        f"running={running} queued={queued}"
                    )
                    continue
                for fut in done:
                    results.append(fut.result())
                    progress.update(1)

    results.sort(key=lambda x: (x.get("index") is None, x.get("index", 10**18)))
    ok = [r for r in results if r.get("status") == "success"]
    err = [r for r in results if r.get("status") != "success"]

    out_data: List[Dict[str, Any]] = []
    for r in ok:
        entry = {
            "index": r.get("index"),
            "nodes": (r.get("data") or {}).get("nodes", []),
        }
        out_data.append(entry)

    out: Dict[str, Any] = {"data": out_data}
    if err:
        out["errors"] = err[:200]

    write_json(output_path, out)
    log_stage(f"stage=output_written path={output_path} success={len(ok)} errors={len(err)}")

    PRICE_PER_1M = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4": (30.00, 60.00)}
    inp, out_p = PRICE_PER_1M.get(model_name, (0.15, 0.60))
    total_cost = (usage["prompt_tokens"] * inp + usage["completion_tokens"] * out_p) / 1_000_000.0
    n_calls = usage.get("calls", 0)
    avg_cost = (total_cost / n_calls) if n_calls else 0.0
    print("\nSummary:")
    print(f"  llm_calls:              {usage.get('calls', 0)}")
    print(f"  prompt_tokens:          {usage.get('prompt_tokens', 0)}")
    print(f"  completion_tokens:      {usage.get('completion_tokens', 0)}")
    print(f"  total API call cost ($): {total_cost:.6f}")
    print(f"  average API call cost ($): {avg_cost:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract knowledge data.")
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--llm_config", required=True, type=str, default="../llm_config.json")
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name from llm_config.json (default: first configured model)",
    )
    ap.add_argument("--threads", required=True, type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    build_knowledge_data(
        input_path=args.input,
        output_path=args.output,
        llm_config_path=args.llm_config,
        model_name=args.model,
        threads=args.threads,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
