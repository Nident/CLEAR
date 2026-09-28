from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_graph.constants import (
    REL_CLASSIFIED_AS,
    REL_CONNECTED_WITH,
    REL_ENABLES,
    REL_HAS_PROPERTY,
    REL_MITIGATED_BY,
    TYPE_CWE,
    TYPE_EP,
    TYPE_FIX,
    TYPE_PRE,
    TYPE_RC,
)
from knowledge_graph.io_common import extract_instances, read_json, write_json


CORE_TYPES = (
    TYPE_EP,
    TYPE_PRE,
    TYPE_RC,
    TYPE_FIX,
    TYPE_CWE,
)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

TYPE_PREFIX = {
    TYPE_EP: "EP",
    TYPE_PRE: "PRE",
    TYPE_RC: "RC",
    TYPE_FIX: "FIX",
    TYPE_CWE: "CWE",
}

def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def assign_local_ids(instances: List[Dict[str, Any]]) -> None:
    for inst_i, inst in enumerate(instances):
        idx_raw = inst.get("index", inst_i)
        try:
            idx = int(idx_raw)
        except Exception:
            idx = inst_i

        counter = defaultdict(int)
        for node in (inst.get("nodes", []) or []):
            t = node.get("type")
            if t not in TYPE_PREFIX:
                continue
            counter[t] += 1
            node["id"] = f"L:{TYPE_PREFIX[t]}:{idx}:{counter[t]}"



def collect_global_cwe_nodes(instances: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    global_cwe = {}

    for inst in instances:
        new_nodes = []
        for node in inst.get("nodes", []):
            if node.get("type") != "CWE":
                new_nodes.append(node)
                continue

            cwe_id = node.get("CWE_ID")
            if not cwe_id:
                continue
            inst["root_cwe_id"] = str(cwe_id).strip()

            if cwe_id not in global_cwe:
                global_cwe[cwe_id] = {
                    "id": f"CWE:{cwe_id}",
                    "type": "CWE",
                    "CWE_ID": cwe_id,
                    "CWE_ID_NAME": node.get("CWE_ID_NAME", "")
                }
        inst["nodes"] = new_nodes

    return global_cwe


def _load_cwe_related_map(cwe_map_path: Path) -> Dict[str, set]:
    try:
        raw = json.loads(cwe_map_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = raw.get("cwe_ids") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return {}
    out: Dict[str, set] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        root = r.get("cwe_id")
        if not isinstance(root, str) or not root.strip():
            continue
        rel = r.get("related_cwes")
        if not isinstance(rel, list):
            rel = []
        out[root.strip()] = {str(x).strip() for x in rel if isinstance(x, str) and str(x).strip()}
    return out

def build_neighbors(
    instances: List[Dict[str, Any]],
    model: SentenceTransformer,
    topk: int,
    batch_size: int,
    skip_types: Optional[set] = None,
    cwe_related_map: Optional[Dict[str, set]] = None,
    root_cwe_field: str = "root_cwe_id",
) -> None:
    skip_types = skip_types or set()
    refs_by_type: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for ii, inst in enumerate(instances):
        for ni, node in enumerate(inst.get("nodes", []) or []):
            t = node.get("type")
            if t in CORE_TYPES and t not in skip_types:
                refs_by_type[t].append((ii, ni))

    for t, refs in refs_by_type.items():
        if len(refs) < 2:
            continue

        texts: List[str] = []
        for (ii, ni) in refs:
            n = instances[ii]["nodes"][ni]
            texts.append(_norm_text(n.get("details", "")))

        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sim = cosine_similarity(emb)

        cwe_related_map = cwe_related_map or {}
        ref_root_cwes: List[str] = []
        for (ii, _ni) in refs:
            v = instances[ii].get(root_cwe_field)
            ref_root_cwes.append(str(v).strip() if v else "")

        for i, (ii, ni) in enumerate(refs):
            node = instances[ii]["nodes"][ni]
            neighbors: Dict[str, Dict[str, Any]] = {}

            scored: List[Tuple[float, int]] = []
            root_cwe = ref_root_cwes[i]
            allowed = cwe_related_map.get(root_cwe, set()) if root_cwe else set()

            candidate_js: List[int] = []
            if allowed:
                for j in range(len(refs)):
                    if j == i:
                        continue
                    if ref_root_cwes[j] in allowed:
                        candidate_js.append(j)
            else:
                candidate_js = [j for j in range(len(refs)) if j != i]

            for j in candidate_js:
                s = float(sim[i][j])
                scored.append((s, j))

            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:topk]

            for s, j in scored:
                jj_inst, jj_node = refs[j]
                nb = instances[jj_inst]["nodes"][jj_node]

                nb_key = _norm_text(nb.get("summary", ""))
                nb_text = _norm_text(nb.get("details", ""))
                if not nb_text:
                    continue

                key = nb_key
                if key in neighbors:
                    key = f"{key} [{nb.get('id')}]"

                neighbors[key] = {
                    "id": nb.get("id"),
                    "details": nb.get("details", ""),
                    "similarity": round(s, 4),
                }

            if neighbors:
                node["neighbors"] = neighbors


def build_causal_edges(instances: List[Dict[str, Any]], global_cwe: Dict[str, Dict[str, Any]]) -> None:
    for inst in instances:
        nodes = inst.get("nodes", []) or []
        edges: List[Dict[str, Any]] = inst.get("edges", []) or []

        by_type: Dict[str, Dict[str, Any]] = {}
        cwe_id: Optional[str] = None

        for n in nodes:
            t = n.get("type")
            if t in CORE_TYPES and t != "CWE" and t not in by_type:
                by_type[t] = n

        for n in inst.get("nodes", []):
            if n.get("type") == "CWE":
                cwe_id = n.get("CWE_ID")

        if "Entrypoint" in by_type and "vulnerability_precondition" in by_type:
            edges.append({
                "src": by_type["Entrypoint"]["id"],
                "rel": REL_HAS_PROPERTY,
                "dst": by_type["vulnerability_precondition"]["id"]
            })

        if "vulnerability_precondition" in by_type and "vulnerability_root_cause" in by_type:
            edges.append({
                "src": by_type["vulnerability_precondition"]["id"],
                "rel": REL_ENABLES,
                "dst": by_type["vulnerability_root_cause"]["id"]
            })

        if "vulnerability_root_cause" in by_type and "fix_intent" in by_type:
            edges.append({
                "src": by_type["vulnerability_root_cause"]["id"],
                "rel": REL_MITIGATED_BY,
                "dst": by_type["fix_intent"]["id"]
            })

        if "vulnerability_root_cause" in by_type and cwe_id and cwe_id in global_cwe:
            edges.append({
                "src": by_type["vulnerability_root_cause"]["id"],
                "rel": REL_CLASSIFIED_AS,
                "dst": global_cwe[cwe_id]["id"]
            })

        if edges:
            inst["edges"] = edges


def build_neighbor_edges(instances: List[Dict[str, Any]]) -> None:
    id_exists = set()
    for inst in instances:
        for node in inst.get("nodes", []) or []:
            nid = node.get("id")
            if nid:
                id_exists.add(nid)

    for inst in instances:
        edges: List[Dict[str, Any]] = inst.get("edges", []) or []
        for node in inst.get("nodes", []) or []:
            src_id = node.get("id")
            neigh = node.get("neighbors") or {}
            if not src_id or not neigh:
                continue

            for neighbor_text, info in neigh.items():
                dst_id = info.get("id")
                if not dst_id or dst_id not in id_exists:
                    continue
                edges.append({
                    "src": src_id,
                    "rel": REL_CONNECTED_WITH,
                    "dst": dst_id,
                    "meta": {
                        "similarity": info.get("similarity"),
                        "neighbor_text": neighbor_text,
                    }
                })

        if edges:
            inst["edges"] = edges


def _write_error_log(output_path: str, step: str, error_msg: str) -> None:
    try:
        error_dir = Path(output_path).resolve().parent / "error"
        error_dir.mkdir(parents=True, exist_ok=True)
        err_file = error_dir / "knowledge_graph.err"
        with open(err_file, "a", encoding="utf-8") as f:
            f.write(f"step={step}\t{error_msg}\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./knowledge_data.json")
    ap.add_argument("--output", default="./knowledge_graph.json")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    try:
        raw = read_json(args.input)
        instances, wrapped = extract_instances(raw)

        assign_local_ids(instances)
        global_cwe = collect_global_cwe_nodes(instances)

        skip_types = {"CWE"}
        st_model = SentenceTransformer(EMBED_MODEL_NAME)

        cwe_map_path = Path(args.input).resolve().parent / "cwe_map.json"
        cwe_related_map = _load_cwe_related_map(cwe_map_path) if cwe_map_path.exists() else {}

        print("Building semantic neighbors...")
        build_neighbors(
            instances=instances,
            model=st_model,
            topk=int(args.topk),
            batch_size=int(args.batch_size),
            skip_types=skip_types,
            cwe_related_map=cwe_related_map,
            root_cwe_field="root_cwe_id",
        )

        print("Building causal edges...")
        build_causal_edges(instances, global_cwe)

        print("Building neighbor edges...")
        build_neighbor_edges(instances)

        total_edges = sum(len(inst.get("edges", []) or []) for inst in instances)
        neighbor_edges = sum(
            len([e for e in (inst.get("edges", []) or []) if e.get("rel") == REL_CONNECTED_WITH])
            for inst in instances
        )
        causal_edges = total_edges - neighbor_edges

        print(f"Edges created: causal={causal_edges}, neighbor={neighbor_edges}, total={total_edges}")

        out_obj = {
            "data": instances,
            "global_nodes": {
                "CWE": list(global_cwe.values())
            }
        }
        write_json(args.output, out_obj)
        print(f"[OK] saved: {args.output}")
    except Exception as e:
        step = "main"
        _write_error_log(args.output, step, f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()