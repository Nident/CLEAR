from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent

METHOD_ALIASES = {
    "codebert": "codebert",
    "unixcoder": "unixcoder",
    "cot": "cot",
    "gptlens": "gptlens",
    "vultrial": "vultrial",
    "rag-instance": "rag-instance",
    "rag_instance": "rag-instance",
    "rag-chunk": "rag-chunk",
    "rag_chunk": "rag-chunk",
    "vulrag": "vulrag",
    "vul-rag": "vulrag",
    "vul_rag": "vulrag",
}

DEFAULT_METHODS = [
    "codebert",
    "unixcoder",
    "cot",
    "gptlens",
    "vultrial",
    "rag-instance",
    "rag-chunk",
    "vulrag",
]


@dataclass(frozen=True)
class Cmd:
    argv: List[str]
    cwd: Path


def _run(cmd: Cmd) -> int:
    print(f"[run_baseline] $ (cd {cmd.cwd}) {' '.join(cmd.argv)}")
    return subprocess.run(cmd.argv, cwd=str(cmd.cwd)).returncode


def _require_file(path: Path, *, strict: bool, label: str) -> bool:
    if path.is_file():
        return True
    msg = f"[run_baseline] SKIP {label}: not found: {path}"
    if strict:
        raise FileNotFoundError(msg)
    print(msg, file=sys.stderr)
    return False


def _normalize_method(name: str) -> str:
    key = name.strip().lower().replace(" ", "")
    if key not in METHOD_ALIASES:
        raise ValueError(
            f"Unknown method: {name!r}. Choose from: {', '.join(DEFAULT_METHODS)}"
        )
    return METHOD_ALIASES[key]


def _parse_methods(raw: Optional[str], flag_methods: Sequence[str]) -> List[str]:
    selected: List[str] = []
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if part:
                selected.append(_normalize_method(part))
    for m in flag_methods:
        selected.append(_normalize_method(m))
    out: List[str] = []
    seen = set()
    for m in selected:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _java_pair_run_tags(last: int = 10) -> List[Optional[str]]:
    """None -> java-pair, results2 -> java-pair2, ..."""
    tags: List[Optional[str]] = [None]
    tags.extend(f"results{i}" for i in range(2, last + 1))
    return tags


def _iter_version_runs(
    versions: Sequence[str],
    *,
    java_pair_runs: int,
) -> Iterable[tuple[str, Optional[str]]]:
    for version in versions:
        if version == "primevul":
            yield "primevul", None
        elif version == "java-pair":
            for tag in _java_pair_run_tags(java_pair_runs):
                yield "java-pair", tag
        else:
            raise ValueError(f"Unknown version: {version!r} (expected primevul/java-pair)")


def _with_results_base(argv: List[str], results_base: Optional[str]) -> List[str]:
    if results_base:
        argv += ["--results-base", results_base]
    return argv


def _dl_cmd(model: str, *, version: str, results_base: Optional[str]) -> Cmd:
    entry = ROOT / "baseline" / "DL" / "run_dl.py"
    argv = [sys.executable, str(entry), "--model", model, "--version", version]
    return Cmd(argv=_with_results_base(argv, results_base), cwd=ROOT)


def _cot_cmd(*, version: str, results_base: Optional[str], task: str = "detection") -> Cmd:
    entry = ROOT / "baseline" / "CoT" / "run_cot.py"
    argv = [sys.executable, str(entry), "--version", version, "--task", task]
    return Cmd(argv=_with_results_base(argv, results_base), cwd=ROOT)


def _gptlens_cmd(*, version: str, results_base: Optional[str], task: str = "detection") -> Cmd:
    entry = ROOT / "baseline" / "GPTLens" / "run_gptlens.py"
    argv = [sys.executable, str(entry), "--version", version, "--task", task]
    return Cmd(argv=_with_results_base(argv, results_base), cwd=ROOT)


def _vultrial_cmd(*, version: str, results_base: Optional[str], task: str = "detection") -> Cmd:
    entry = ROOT / "baseline" / "VulTrial" / "run_vultrial.py"
    argv = [sys.executable, str(entry), "--version", version, "--task", task]
    return Cmd(argv=_with_results_base(argv, results_base), cwd=ROOT)


def _rag_cmd(kind: str, *, version: str, results_base: Optional[str]) -> Cmd:
    entry = ROOT / "baseline" / "RAG" / f"run_{kind}.py"
    argv = [sys.executable, str(entry), "--version", version]
    return Cmd(argv=_with_results_base(argv, results_base), cwd=ROOT)


def _vulrag_detect_cmd(
    *,
    version: str,
    model_name: str,
    summary_model_name: str,
    retrieval_top_k: int,
    max_knowledge: int,
    model_settings: str,
    thread_pool_size: int,
) -> Cmd:
    cwd = ROOT / "baseline" / "VUL-RAG"
    entry = cwd / "src" / "vulnerability_detect.py"

    if version == "primevul":
        input_file_name = "primevul_test.json"
        knowledge_file_name = "primevul_knowledge.json"
        output_file_name = f"primevul_detect_{model_name}.json"
    elif version == "java-pair":
        input_file_name = "java_pair_test.json"
        knowledge_file_name = "java_pair_knowledge.json"
        output_file_name = f"java_pair_detect_{model_name}.json"
    else:
        raise ValueError(f"Invalid version for VUL-RAG detect: {version!r}")

    knowledge_path = cwd / "output" / "knowledge" / knowledge_file_name
    detect_output = cwd / "output" / "detect" / output_file_name

    argv = [
        sys.executable,
        str(entry),
        "--input_file_name",
        input_file_name,
        "--knowledge_file_name",
        knowledge_file_name,
        "--output_file_name",
        output_file_name,
        "--model_name",
        model_name,
        "--summary_model_name",
        summary_model_name,
        "--retrieval_top_k",
        str(int(retrieval_top_k)),
        "--max_knowledge",
        str(int(max_knowledge)),
        "--model_settings",
        model_settings,
        "--thread_pool_size",
        str(int(thread_pool_size)),
        "--knowledge_path",
        str(knowledge_path),
        "--detect_output",
        str(detect_output),
    ]
    return Cmd(argv=argv, cwd=cwd)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run baselines under ./baseline. "
            "By default runs: CodeBERT, UnixCoder, CoT, GPTLens, VulTrial, "
            "rag-instance, rag-chunk, VUL-RAG over primevul + java-pair~java-pair10."
        )
    )
    p.add_argument(
        "--method",
        type=str,
        default=None,
        help=(
            "CSV of methods: codebert,unixcoder,cot,gptlens,vultrial,"
            "rag-instance,rag-chunk,vulrag. Default: all."
        ),
    )
    p.add_argument(
        "--versions",
        type=str,
        default="primevul,java-pair",
        help="CSV: primevul,java-pair (default: both). java-pair expands to java-pair~java-pair10.",
    )
    p.add_argument(
        "--java-pair-runs",
        type=int,
        default=10,
        help="Number of java-pair runs (1..N). N>=2 uses results2..resultsN (default: 10).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="detection",
        choices=["detection", "classification"],
        help="Task for CoT/GPTLens/VulTrial (default: detection).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a requested baseline runner is missing.",
    )

    p.add_argument("--codebert", action="store_true")
    p.add_argument("--unixcoder", action="store_true")
    p.add_argument("--cot", action="store_true")
    p.add_argument("--gptlens", action="store_true")
    p.add_argument("--vultrial", action="store_true")
    p.add_argument("--rag-instance", action="store_true")
    p.add_argument("--rag-chunk", action="store_true")
    p.add_argument("--vulrag", action="store_true")

    p.add_argument("--vulrag-model-name", type=str, default="gpt-4o-mini")
    p.add_argument("--vulrag-summary-model-name", type=str, default=None)
    p.add_argument("--vulrag-retrieval-top-k", type=int, default=20)
    p.add_argument("--vulrag-max-knowledge", type=int, default=3)
    p.add_argument("--vulrag-detect-thread-pool-size", type=int, default=5)
    p.add_argument("--vulrag-detect-model-settings", type=str, default="temperature=0.01")

    args = p.parse_args()

    if args.java_pair_runs < 1:
        p.error("--java-pair-runs must be >= 1")

    flag_methods = [
        name
        for name, enabled in [
            ("codebert", args.codebert),
            ("unixcoder", args.unixcoder),
            ("cot", args.cot),
            ("gptlens", args.gptlens),
            ("vultrial", args.vultrial),
            ("rag-instance", args.rag_instance),
            ("rag-chunk", args.rag_chunk),
            ("vulrag", args.vulrag),
        ]
        if enabled
    ]
    methods = _parse_methods(args.method, flag_methods) or list(DEFAULT_METHODS)

    versions = [v.strip() for v in (args.versions or "").split(",") if v.strip()]
    if not versions:
        p.error("--versions is empty")
    for v in versions:
        if v not in {"primevul", "java-pair"}:
            p.error(f"Unknown version in --versions: {v!r}")

    jobs = list(_iter_version_runs(versions, java_pair_runs=args.java_pair_runs))
    print(f"[run_baseline] methods={methods}")
    print(
        "[run_baseline] jobs="
        + ", ".join(
            ("primevul" if v == "primevul" else (f"java-pair" if t is None else f"java-pair{t.replace('results', '')}"))
            for v, t in jobs
        )
    )

    cmds: List[Cmd] = []
    for method in methods:
        if method in ("codebert", "unixcoder"):
            entry = ROOT / "baseline" / "DL" / "run_dl.py"
            if not _require_file(entry, strict=args.strict, label=method):
                continue
            for version, tag in jobs:
                cmds.append(_dl_cmd(method, version=version, results_base=tag))
            continue

        if method == "cot":
            entry = ROOT / "baseline" / "CoT" / "run_cot.py"
            if not _require_file(entry, strict=args.strict, label="CoT"):
                continue
            for version, tag in jobs:
                cmds.append(_cot_cmd(version=version, results_base=tag, task=args.task))
            continue

        if method == "gptlens":
            entry = ROOT / "baseline" / "GPTLens" / "run_gptlens.py"
            if not _require_file(entry, strict=args.strict, label="GPTLens"):
                continue
            for version, tag in jobs:
                cmds.append(_gptlens_cmd(version=version, results_base=tag, task=args.task))
            continue

        if method == "vultrial":
            entry = ROOT / "baseline" / "VulTrial" / "run_vultrial.py"
            if not _require_file(entry, strict=args.strict, label="VulTrial"):
                continue
            for version, tag in jobs:
                cmds.append(_vultrial_cmd(version=version, results_base=tag, task=args.task))
            continue

        if method == "rag-instance":
            entry = ROOT / "baseline" / "RAG" / "run_rag_instance.py"
            if not _require_file(entry, strict=args.strict, label="RAG (instance)"):
                continue
            for version, tag in jobs:
                cmds.append(_rag_cmd("rag_instance", version=version, results_base=tag))
            continue

        if method == "rag-chunk":
            entry = ROOT / "baseline" / "RAG" / "run_rag_chunk.py"
            if not _require_file(entry, strict=args.strict, label="RAG (chunk)"):
                continue
            for version, tag in jobs:
                cmds.append(_rag_cmd("rag_chunk", version=version, results_base=tag))
            continue

        if method == "vulrag":
            entry = ROOT / "baseline" / "VUL-RAG" / "src" / "vulnerability_detect.py"
            if not _require_file(entry, strict=args.strict, label="VUL-RAG"):
                continue
            (ROOT / "baseline" / "VUL-RAG" / "output" / "detect").mkdir(parents=True, exist_ok=True)
            summary_model = args.vulrag_summary_model_name or args.vulrag_model_name
            for version in versions:
                cmds.append(
                    _vulrag_detect_cmd(
                        version=version,
                        model_name=args.vulrag_model_name,
                        summary_model_name=summary_model,
                        retrieval_top_k=args.vulrag_retrieval_top_k,
                        max_knowledge=args.vulrag_max_knowledge,
                        model_settings=args.vulrag_detect_model_settings,
                        thread_pool_size=args.vulrag_detect_thread_pool_size,
                    )
                )
            continue

        raise AssertionError(f"Unhandled method: {method}")

    if not cmds:
        print("[run_baseline] Nothing to run.", file=sys.stderr)
        return

    print(f"[run_baseline] queued {len(cmds)} commands")
    for cmd in cmds:
        rc = _run(cmd)
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
