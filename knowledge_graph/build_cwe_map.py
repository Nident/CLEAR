from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from mitre_cwe_api import MitreApiConfig, fetch_cwe_weakness

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


_RE_CWE_ID = re.compile(r'"CWE_ID"\s*:\s*"CWE-(\d+)"')
_RE_CWE_CANON = re.compile(r"^CWE-(\d+)$", re.IGNORECASE)
_RE_CWE_ANY = re.compile(r"\bCWE-(\d+)\b", re.IGNORECASE)


def _canon_cwe_id(s: str) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip()
    m = _RE_CWE_CANON.match(s)
    if not m:
        return None
    n = m.group(1)
    return f"CWE-{int(n)}"


def _canon_any_cwe(v: Any) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        return None
    m = _RE_CWE_ANY.search(v.strip())
    return f"CWE-{int(m.group(1))}" if m else None


def _extract_allowed_cwes_from_dataset(path: Path) -> Set[str]:
    allowed: Set[str] = set()

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                cwe_val = obj.get("cwe", obj.get("cwe_id", obj.get("CWE_ID")))
                if isinstance(cwe_val, list):
                    for x in cwe_val:
                        cid = _canon_any_cwe(x)
                        if cid:
                            allowed.add(cid)
                else:
                    cid = _canon_any_cwe(cwe_val)
                    if cid:
                        allowed.add(cid)
        return allowed

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return allowed
    rows = raw if isinstance(raw, list) else (raw.get("data") if isinstance(raw, dict) else [])
    if not isinstance(rows, list):
        return allowed
    for it in rows:
        if not isinstance(it, dict):
            continue
        cwe_val = it.get("cwe", it.get("cwe_id", it.get("CWE_ID")))
        if isinstance(cwe_val, list):
            for x in cwe_val:
                cid = _canon_any_cwe(x)
                if cid:
                    allowed.add(cid)
        else:
            cid = _canon_any_cwe(cwe_val)
            if cid:
                allowed.add(cid)
    return allowed


def _pick_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _extract_unique_cwe_ids(path: Path) -> List[str]:
    cwes: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        instances = raw.get("data") if isinstance(raw, dict) else raw
        if not isinstance(instances, list):
            instances = []
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            for node in inst.get("nodes", []) or []:
                if not isinstance(node, dict):
                    continue
                if node.get("type") != "CWE":
                    continue
                cid = _canon_cwe_id(node.get("CWE_ID", ""))
                if cid:
                    cwes.add(cid)
    except Exception:
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in _RE_CWE_ID.finditer(text):
            try:
                cwes.add(f"CWE-{int(m.group(1))}")
            except Exception:
                pass
    return sorted(cwes, key=lambda x: int(x.split("-")[1]))


def _related_cwes_from_payload(payload: Dict[str, Any]) -> List[str]:
    related: Set[str] = set()
    weaknesses = payload.get("Weaknesses") if isinstance(payload, dict) else None
    if not isinstance(weaknesses, list) or not weaknesses:
        return []
    w0 = weaknesses[0] if isinstance(weaknesses[0], dict) else {}
    rels = w0.get("RelatedWeaknesses")
    if not isinstance(rels, list):
        return []
    for r in rels:
        if not isinstance(r, dict):
            continue
        rid = r.get("CweID")
        if rid is None:
            continue
        try:
            related.add(f"CWE-{int(str(rid).strip())}")
        except Exception:
            continue
    return sorted(related, key=lambda x: int(x.split("-")[1]))


def build_cwe_map(
    cwe_ids: Iterable[str],
    *,
    cache_dir: Path,
    sleep_sec: float,
    cfg: MitreApiConfig,
    max_cwes: Optional[int] = None,
    threads: int = 32,
    write_cache: bool = True,
    show_progress: bool = True,
    allowed_cwes: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    unique: List[str] = []
    seen: Set[str] = set()
    for cid in cwe_ids:
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(cid)
        if max_cwes is not None and len(unique) >= int(max_cwes):
            break

    def _one(cid: str) -> Dict[str, Any]:
        num = cid.split("-", 1)[1]
        cache_path = cache_dir / f"{num}.json"
        payload: Dict[str, Any]
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = fetch_cwe_weakness(num, cfg=cfg)
            if write_cache:
                try:
                    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
            if sleep_sec > 0:
                time.sleep(float(sleep_sec))
        related = _related_cwes_from_payload(payload)
        if allowed_cwes is not None:
            related = [x for x in related if x in allowed_cwes]
        return {"cwe_id": cid, "related_cwes": related}

    threads = int(threads) if threads is not None else 32
    if threads <= 1 or len(unique) <= 1:
        it = unique
        if show_progress and tqdm is not None:
            it = tqdm(unique, total=len(unique), desc="Fetch CWE RelatedWeaknesses", unit="cwe")
        rows = [_one(cid) for cid in it]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = {ex.submit(_one, cid): cid for cid in unique}
            done_iter = as_completed(futs)
            if show_progress and tqdm is not None:
                done_iter = tqdm(done_iter, total=len(futs), desc="Fetch CWE RelatedWeaknesses", unit="cwe")
            for fut in done_iter:
                try:
                    rows.append(fut.result())
                except Exception:
                    cid = futs.get(fut)
                    if cid:
                        rows.append({"cwe_id": cid, "related_cwes": []})

    rows.sort(key=lambda r: int(str(r.get("cwe_id", "CWE-0")).split("-")[1]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MITRE CWE related map from knowledge_data.json")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--java-pair", action="store_true", help="Use knowledge_graph/java/knowledge_data.json")
    g.add_argument("--primevul", action="store_true", help="Use knowledge_graph/primevul/primevul_knowledge_data.json")
    ap.add_argument("--input", type=Path, help="Knowledge-base JSON path")
    ap.add_argument("--output", type=Path, help="Output cwe_map.json path")
    ap.add_argument("--threads", type=int, default=32, help="Worker threads for API calls (default: 32)")
    ap.add_argument("--max-retries", type=int, default=3, help="HTTP retry count")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout (seconds)")
    ap.add_argument("--backoff", type=float, default=1.0, help="Retry backoff base (seconds)")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache writes (still reads if exists)")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override dataset path used to filter allowed CWE IDs",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    kg_dir = root / "knowledge_graph"

    if args.input:
        in_path = args.input.resolve()
        out_path = (args.output or (in_path.parent / "cwe_map.json")).resolve()
        cache_dir = out_path.parent / ".cache_mitre_cwe"
    elif args.java_pair:
        in_path = kg_dir / "java-pair" / "java_knowledge_data.json"
        out_path = kg_dir / "java" / "cwe_map.json"
        cache_dir = kg_dir / "java" / ".cache_mitre_cwe"
    elif args.primevul:
        in_path = kg_dir / "primevul" / "primevul_knowledge_data.json"
        out_path = kg_dir / "primevul" / "cwe_map.json"
        cache_dir = kg_dir / "primevul" / ".cache_mitre_cwe"
    else:
        ap.error("Specify --input or one of --java-pair/--primevul")

    if not in_path.exists():
        raise SystemExit(f"Not found: {in_path}")

    cwe_ids = _extract_unique_cwe_ids(in_path)
    print(f"[OK] extracted unique CWE IDs (KG): {len(cwe_ids)} from {in_path}")

    default_ds_candidates = (
        [
            root / "dataset" / "java-pair" / "java_test_paired.json",
        ]
        if args.java_pair
        else [
            root / "dataset" / "primevul" / "primevul_test_paired.jsonl",
        ]
    )
    ds_path = args.dataset.resolve() if args.dataset else _pick_existing(default_ds_candidates)
    if not ds_path:
        tried = ", ".join(str(p) for p in default_ds_candidates)
        raise SystemExit(f"Not found: dataset for filtering. Tried: {tried}")
    allowed = _extract_allowed_cwes_from_dataset(ds_path)
    if allowed:
        print(f"[OK] allowed CWE IDs (dataset): {len(allowed)} from {ds_path}")
        print("[OK] will post-filter only related_cwes (root cwe_id list unchanged)")
    else:
        allowed = None
        print(f"[WARN] no allowed CWE IDs extracted from dataset: {ds_path} (skip related_cwes filtering)")

    cfg = MitreApiConfig(
        timeout_sec=float(args.timeout),
        max_retries=int(args.max_retries),
        backoff_sec=float(args.backoff),
    )

    rows = build_cwe_map(
        cwe_ids,
        cache_dir=cache_dir,
        sleep_sec=0.1,
        cfg=cfg,
        max_cwes=None,
        threads=int(args.threads),
        write_cache=not bool(args.no_cache),
        show_progress=True,
        allowed_cwes=allowed if isinstance(allowed, set) else None,
    )

    payload = {"cwe_ids": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
