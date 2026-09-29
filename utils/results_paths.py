from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXED_RESULTS_ROOT = _PROJECT_ROOT / "results"

_RUN_TAG_RE = re.compile(r"^results(\d+)$")
_VALID_DATASET_VERSIONS = frozenset({"java-pair", "primevul"})


def parse_run_suffix(results_base: Optional[Union[Path, str]] = None) -> str:
    if results_base is None:
        return ""
    name = Path(results_base).name
    if not name or name == "results":
        return ""
    match = _RUN_TAG_RE.match(name)
    if match:
        return match.group(1)
    raise ValueError(
        f"Invalid results run tag: {results_base!r}. "
        "Expected 'results', 'results2', 'results3', ... or a path ending with those names."
    )


def storage_version(version: str, results_base: Optional[Union[Path, str]] = None) -> str:
    suffix = parse_run_suffix(results_base)
    return f"{version}{suffix}" if suffix else version


def dataset_version(storage_version_name: str) -> str:
    match = re.match(r"^(java-pair|primevul)(\d+)$", storage_version_name)
    if match:
        return match.group(1)
    if storage_version_name in _VALID_DATASET_VERSIONS:
        return storage_version_name
    raise ValueError(f"Invalid storage version: {storage_version_name!r}")


def safe_model_name(model: str) -> str:
    return model.replace("/", "-")


def default_run_tags(last: int = 10) -> list[str]:
    tags = ["results"]
    tags.extend(f"results{i}" for i in range(2, last + 1))
    return tags


def java_pair_storage_versions(last: int = 10) -> list[str]:
    versions = ["java-pair"]
    versions.extend(f"java-pair{i}" for i in range(2, last + 1))
    return versions


def storage_version_to_run_tag(dataset: str, storage_ver: str) -> Optional[str]:
    if storage_ver == dataset:
        return None
    if storage_ver.startswith(dataset):
        suffix = storage_ver[len(dataset) :]
        if suffix.isdigit():
            return f"results{suffix}"
    raise ValueError(f"Cannot map storage version {storage_ver!r} to run tag for dataset {dataset!r}")


def run_messages_dir(
    model_name: str,
    version: str,
    *,
    task_mode: str = "detection",
    ablation: Optional[str] = None,
    results_base: Optional[Union[Path, str]] = None,
) -> Path:
    """results/{model}/{storage_version}/{task|ablation/{name}}"""
    model = safe_model_name(model_name)
    ver = storage_version(version, results_base)
    root = FIXED_RESULTS_ROOT / model / ver
    if ablation:
        return root / "ablation" / ablation
    return root / task_mode
