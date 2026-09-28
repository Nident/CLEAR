from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REL_CLASSIFIED_AS = "classified_as"
TYPE_RC = "vulnerability_root_cause"
TYPE_CWE = "CWE"


def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _extract_instances(raw: Any) -> Tuple[List[Dict[str, Any]], bool]:
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw["data"], True
    if isinstance(raw, list):
        return raw, False
    raise ValueError("Expected list or {'data': [...]}")


def _get_global_cwe(pre_raw: Any) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    gn = pre_raw.get("global_nodes") if isinstance(pre_raw, dict) else {}
    cwe_list = gn.get("CWE") if isinstance(gn, dict) else []
    if not isinstance(cwe_list, list):
        return by_id
    for node in cwe_list:
        if not isinstance(node, dict):
            continue
        cwe_id = _norm(node.get("CWE_ID"))
        if cwe_id:
            n = dict(node)
            if not _norm(n.get("details")):
                n["details"] = _norm(n.get("CWE_ID_NAME", "")) or cwe_id
            by_id[cwe_id] = n
    return by_id


def _rc_ids_and_cwe_node(
    inst: Dict[str, Any],
    pre_inst: Optional[Dict[str, Any]],
    global_cwe: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    root_cwe_id = _norm(
        inst.get("root_cwe_id")
        or (pre_inst.get("root_cwe_id") if pre_inst else None)
    )
    if not root_cwe_id or root_cwe_id not in global_cwe:
        return [], None
    cwe_node = global_cwe[root_cwe_id]
    rc_ids: List[str] = []
    for n in inst.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        if _norm(n.get("type")) == TYPE_RC:
            nid = n.get("id")
            if isinstance(nid, str) and nid:
                rc_ids.append(nid)
    return rc_ids, cwe_node


def add_cwe_edges_to_instances(
    instances: List[Dict[str, Any]],
    pre_instances: List[Dict[str, Any]],
    global_cwe: Dict[str, Dict[str, Any]],
) -> int:
    added = 0
    for i, inst in enumerate(instances):
        pre_inst = pre_instances[i] if i < len(pre_instances) else None
        rc_ids, cwe_node = _rc_ids_and_cwe_node(inst, pre_inst, global_cwe)
        if not rc_ids or not cwe_node:
            continue
        cwe_id = cwe_node.get("id")
        if not cwe_id:
            continue
        nodes = inst.get("nodes")
        if nodes is None:
            inst["nodes"] = []
            nodes = inst["nodes"]
        node_ids = {n.get("id") for n in nodes if isinstance(n, dict) and n.get("id")}
        if cwe_id not in node_ids:
            nodes.append(dict(cwe_node))
            node_ids.add(cwe_id)
        edges = inst.get("edges")
        if edges is None:
            inst["edges"] = []
            edges = inst["edges"]
        edge_set = {(e.get("src"), e.get("rel"), e.get("dst")) for e in edges if isinstance(e, dict)}
        for rc_id in rc_ids:
            key = (rc_id, REL_CLASSIFIED_AS, cwe_id)
            if key not in edge_set:
                edges.append({"src": rc_id, "rel": REL_CLASSIFIED_AS, "dst": cwe_id})
                edge_set.add(key)
                added += 1
    return added


def run_version(base_dir: Path, final_name: str, pre_name: str, output_name: str) -> None:
    base_dir = base_dir.resolve()
    final_path = base_dir / final_name
    pre_path = base_dir / pre_name
    out_path = base_dir / output_name
    if not pre_path.exists():
        raise FileNotFoundError(f"Pre-final graph not found: {pre_path}")
    if not final_path.exists():
        raise FileNotFoundError(f"Final graph not found: {final_path}")

    pre_raw = _load_json(pre_path)
    final_raw = _load_json(final_path)
    pre_instances, _ = _extract_instances(pre_raw)
    instances, wrapped = _extract_instances(final_raw)
    global_cwe = _get_global_cwe(pre_raw)
    print(f"[{base_dir.name}] global CWE nodes: {len(global_cwe)}")
    print(f"[{base_dir.name}] instances: {len(instances)}, pre: {len(pre_instances)}")

    added = add_cwe_edges_to_instances(instances, pre_instances, global_cwe)
    print(f"[{base_dir.name}] RC→CWE edges added: {added}")

    out_obj = {"data": instances} if wrapped else instances
    _save_json(out_path, out_obj)
    print(f"[{base_dir.name}] Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add RC→CWE nodes and classified_as edges to final knowledge graph."
    )
    ap.add_argument("--java-pair", action="store_true", help="Process java-pair graph")
    ap.add_argument("--primevul", action="store_true", help="Process primevul graph")
    ap.add_argument("--input", type=Path, help="Final graph JSON path")
    ap.add_argument("--pre-input", type=Path, help="Pre-final graph JSON path")
    ap.add_argument("--output", type=Path, help="Output graph JSON path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    ran = False
    if args.input or args.pre_input or args.output:
        if not (args.input and args.pre_input and args.output):
            ap.error("--input, --pre-input and --output must be specified together")
        final_path = args.input.resolve()
        pre_path = args.pre_input.resolve()
        out_path = args.output.resolve()
        pre_raw = _load_json(pre_path)
        final_raw = _load_json(final_path)
        pre_instances, _ = _extract_instances(pre_raw)
        instances, wrapped = _extract_instances(final_raw)
        global_cwe = _get_global_cwe(pre_raw)
        added = add_cwe_edges_to_instances(instances, pre_instances, global_cwe)
        out_obj = {"data": instances} if wrapped else instances
        _save_json(out_path, out_obj)
        print(f"[custom] RC→CWE edges added: {added}")
        print(f"[custom] Saved: {out_path}")
        ran = True
    if args.java_pair:
        run_version(
            root / "java-pair",
            "java_final_knowledge_graph.json",
            "java_knowledge_graph.json",
            "java_final_knowledge_graph.json",
        )
        ran = True
    if args.primevul:
        run_version(
            root / "primevul",
            "primevul_final_knowledge_graph.json",
            "primevul_knowledge_graph.json",
            "primevul_final_knowledge_graph.json",
        )
        ran = True
    if not ran:
        ap.error("Specify at least one of --java-pair or --primevul.")


if __name__ == "__main__":
    main()
