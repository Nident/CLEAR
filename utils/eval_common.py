from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union

from utils.dataloader.message_saver import get_messages_dir_for_model_and_version
from utils.results_paths import FIXED_RESULTS_ROOT, run_messages_dir, storage_version


def has_result_files(d: Path) -> bool:
    if not d.is_dir():
        return False
    return any(d.glob("*.json")) or any(d.glob("*.txt"))


def default_input_path(version: str, root: Path) -> Path:
    if version == "java-pair":
        return root / "dataset" / "java-pair" / "java_test_paired.json"
    if version == "primevul":
        return root / "dataset" / "primevul" / "primevul_test_paired.jsonl"
    raise ValueError(f"Invalid version: {version}")


def has_messages_results(d: Path) -> bool:
    if not d.is_dir():
        return False
    for p in d.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / "judge").is_dir() and any((p / "judge").glob("*.json")):
            return True
    return False


def _legacy_message_dirs(
    results_root: Path,
    version: str,
    ablation: Optional[str],
) -> list[Path]:
    found: list[Path] = []
    if ablation:
        ablation_leaf = f"{version}_{ablation}"
        candidates = [
            results_root / "messages" / "ablation" / ablation_leaf,
        ]
        for model_dir in results_root.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                candidates.append(model_dir / "messages" / "ablation" / ablation_leaf)
        for model_dir in FIXED_RESULTS_ROOT.iterdir():
            if not model_dir.is_dir():
                continue
            for ver_dir in model_dir.iterdir():
                if not ver_dir.is_dir():
                    continue
                candidates.append(ver_dir / "ablation" / ablation)
                candidates.append(
                    ver_dir / "detection" / "messages" / "ablation" / ablation_leaf
                )
                candidates.append(
                    model_dir / "ablation" / "detection" / "messages" / "ablation" / ablation_leaf
                )
    else:
        candidates = [results_root / "messages" / version]
        for model_dir in results_root.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                candidates.append(model_dir / "messages" / version)
        for model_dir in FIXED_RESULTS_ROOT.iterdir():
            if not model_dir.is_dir():
                continue
            ver_dir = model_dir / version
            candidates.append(ver_dir / "detection" / "messages" / version)
            candidates.append(ver_dir / "classification" / "messages" / version)
            candidates.append(ver_dir / "detection")
            candidates.append(ver_dir / "classification")

    for path in candidates:
        if path.exists() and has_messages_results(path):
            found.append(path)
    return found


def find_result_dirs(
    results_root: Path,
    version: Optional[str] = None,
    ablation: Optional[str] = None,
) -> list[Path]:
    results_root = results_root.resolve()
    if version:
        found = _legacy_message_dirs(results_root, version, ablation)
        if not found and not ablation:
            ver_dir = results_root / version
            if ver_dir.exists() and has_result_files(ver_dir):
                found.append(ver_dir)
        if not found and not ablation:
            for dirpath, _, filenames in os.walk(results_root):
                p = Path(dirpath)
                if p.name == version and any(n.endswith(".json") or n.endswith(".txt") for n in filenames):
                    found.append(p)
        return sorted(set(found))
    if has_result_files(results_root):
        return [results_root]
    found = []
    for dirpath, dirnames, filenames in os.walk(results_root):
        if any(name.endswith(".json") or name.endswith(".txt") for name in filenames):
            found.append(Path(dirpath))
            dirnames[:] = []
    return sorted(set(found))


def model_display_name(d: Path) -> str:
    if d.parent.name in ("detection", "classification", "ablation"):
        return d.parent.parent.name
    if d.parent.name == "messages" and d.name in ("primevul", "java-pair") or d.name.startswith("java-pair") or d.name.startswith("primevul"):
        return d.parent.parent.parent.name if d.parent.parent.name not in ("messages", "detection", "classification") else d.parent.parent.name
    if d.name in ("primevul", "java-pair") or d.name.startswith("java-pair") or d.name.startswith("primevul"):
        return d.parent.name
    return d.name


def dedupe_model_names(model_dirs: Iterable[Path], results_root: Path) -> dict[Path, str]:
    model_dirs = list(model_dirs)
    leaf_names = [model_display_name(d) for d in model_dirs]
    counts: dict[str, int] = {}
    for n in leaf_names:
        counts[n] = counts.get(n, 0) + 1
    out: dict[Path, str] = {}
    for d in model_dirs:
        base = model_display_name(d)
        if counts.get(base, 0) <= 1:
            out[d] = base
        else:
            rel = d.relative_to(results_root).as_posix().replace("/", "__")
            out[d] = rel
    return out


def resolve_messages_dirs_for_run(
    model_name: str,
    version: str,
    task_mode: str,
    ablation: Optional[str],
    results_base: Optional[Union[Path, str]] = None,
) -> tuple[Optional[str], str]:
    def _dir(ablation_val: Optional[str] = None, task: Optional[str] = None) -> str:
        return get_messages_dir_for_model_and_version(
            model_name,
            version,
            task_mode=task or task_mode,
            ablation=ablation_val,
            results_base=results_base,
        )

    if ablation:
        return (_dir(ablation_val=None, task="detection"), _dir(ablation_val=ablation, task="detection"))
    return (None, _dir())


def resolve_run_messages_dir(
    model_name: str,
    version: str,
    task_mode: str,
    ablation: Optional[str],
    results_base: Optional[Union[Path, str]] = None,
) -> Path:
    return run_messages_dir(
        model_name,
        version,
        task_mode=task_mode,
        ablation=ablation,
        results_base=results_base,
    )


def eval_output_dir(task_mode: str, model_name: str) -> Path:
    return FIXED_RESULTS_ROOT / "eval" / task_mode / model_name


def eval_output_file(
    task_mode: str,
    model_name: str,
    version: str,
    results_base: Optional[Union[Path, str]] = None,
) -> Path:
    ver = storage_version(version, results_base)
    return eval_output_dir(task_mode, model_name) / f"{ver}.json"
