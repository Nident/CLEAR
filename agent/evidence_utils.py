from __future__ import annotations

from typing import Any, Dict, List

NODE_TYPE_FULL_NAMES = {
    "EP": "Entrypoint",
    "PRE": "Vulnerability_Precondition",
    "RC": "Vulnerability_Root_Cause",
    "FIX": "Fix_Intent",
}


def to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return str(v)


def node_type_full_name(node_id: str) -> str:
    if not node_id:
        return "Other"
    s = str(node_id).upper()
    if ":PRE:" in s or "__PRE__" in s or s.startswith("L:PRE"):
        return NODE_TYPE_FULL_NAMES["PRE"]
    if ":RC:" in s or "__RC__" in s or s.startswith("L:RC"):
        return NODE_TYPE_FULL_NAMES["RC"]
    if ":FIX:" in s or "__FIX__" in s or s.startswith("L:FIX"):
        return NODE_TYPE_FULL_NAMES["FIX"]
    if ":EP:" in s or "__EP__" in s or "CROSS_EP" in s or s.startswith("L:EP"):
        return NODE_TYPE_FULL_NAMES["EP"]
    return "Other"


def node_type_for_filter(node: dict) -> str:
    t = (node.get("type") or "").strip()
    if t:
        return t.lower()
    return (node_type_full_name(node.get("id") or "") or "").lower()


def evidences_with_full_type(nodes: list) -> list:
    return [{**n, "type": node_type_full_name(n.get("id") or "")} for n in (nodes or []) if isinstance(n, dict)]


def ensure_evidence_struct(evidence: Any) -> Dict[str, Any]:
    if not isinstance(evidence, dict):
        return {"nodes": [], "edges": []}
    nodes = evidence.get("nodes") or []
    edges = evidence.get("edges") or []
    return {
        "nodes": nodes if isinstance(nodes, list) else [],
        "edges": edges if isinstance(edges, list) else [],
    }


def filter_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not evidence or not isinstance(evidence, dict):
        return {"nodes": [], "edges": []}
    filtered_nodes = []
    for node in evidence.get("nodes", []) or []:
        if isinstance(node, dict) and node.get("id") is not None:
            filtered_nodes.append({
                "id": node.get("id"),
                "summary": to_str(node.get("summary", "")),
                "details": to_str(node.get("details", "")),
            })
    filtered_edges = []
    for e in evidence.get("edges", []) or []:
        if isinstance(e, dict) and e.get("src") is not None and e.get("dst") is not None:
            filtered_edges.append({"src": e.get("src"), "dst": e.get("dst"), "rel": e.get("rel", "")})
    return {"nodes": filtered_nodes, "edges": filtered_edges}


def evidence_without_bridge_entity(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not evidence or not isinstance(evidence, dict):
        return {"nodes": [], "edges": []}
    nodes = evidence.get("nodes") or []
    edges = evidence.get("edges") or []
    keep = [n for n in nodes if isinstance(n, dict) and (n.get("type") or "").strip() != "BridgeEntity"]
    return {"nodes": keep, "edges": edges if isinstance(edges, list) else []}


def evidence_without_fix_intent(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not evidence or not isinstance(evidence, dict):
        return {"nodes": [], "edges": []}
    nodes = evidence.get("nodes") or []
    edges = evidence.get("edges") or []
    keep = [n for n in nodes if isinstance(n, dict) and (n.get("type") or "").lower() != "fix_intent"]
    return {"nodes": keep, "edges": edges if isinstance(edges, list) else []}


def evidence_only_ep_pre_rc(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not evidence or not isinstance(evidence, dict):
        return {"nodes": [], "edges": []}
    nodes = evidence.get("nodes") or []
    edges = evidence.get("edges") or []
    allowed = {"entrypoint", "vulnerability_precondition", "vulnerability_root_cause"}
    keep = [n for n in nodes if isinstance(n, dict) and node_type_for_filter(n) in allowed]
    return {"nodes": keep, "edges": edges if isinstance(edges, list) else []}


def evidence_for_claim(evidence: Dict[str, Any], task_mode: str) -> Dict[str, Any]:
    base = evidence_without_bridge_entity(evidence or {})
    if not base or not isinstance(base.get("nodes"), list):
        return {"nodes": [], "edges": base.get("edges", []) if isinstance(base, dict) else []}
    allowed = {"entrypoint", "vulnerability_precondition", "vulnerability_root_cause"}
    if (task_mode or "").strip().lower() == "classification":
        allowed = allowed | {"cwe"}
    nodes = [n for n in base.get("nodes", []) if isinstance(n, dict) and node_type_for_filter(n) in allowed]
    return {"nodes": nodes, "edges": base.get("edges", []) or []}


def evidence_without_precondition(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not evidence or not isinstance(evidence, dict):
        return {"nodes": [], "edges": []}
    nodes = evidence.get("nodes") or []
    edges = evidence.get("edges") or []
    keep = [n for n in nodes if isinstance(n, dict) and node_type_for_filter(n) != "vulnerability_precondition"]
    return {"nodes": keep, "edges": edges if isinstance(edges, list) else []}


def evidence_without_root_cause(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not evidence or not isinstance(evidence, dict):
        return {"nodes": [], "edges": []}
    nodes = evidence.get("nodes") or []
    edges = evidence.get("edges") or []
    keep = [n for n in nodes if isinstance(n, dict) and node_type_for_filter(n) != "vulnerability_root_cause"]
    return {"nodes": keep, "edges": edges if isinstance(edges, list) else []}


def evidence_without_entrypoint(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not evidence or not isinstance(evidence, dict):
        return {"nodes": [], "edges": []}
    nodes = evidence.get("nodes") or []
    edges = evidence.get("edges") or []
    keep = [n for n in nodes if isinstance(n, dict) and node_type_for_filter(n) != "entrypoint"]
    return {"nodes": keep, "edges": edges if isinstance(edges, list) else []}


def get_all_fix_intents(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = (evidence or {}).get("nodes") or []
    out = []
    for n in nodes:
        if not isinstance(n, dict) or n.get("id") is None:
            continue
        if (n.get("type") or "").lower() != "fix_intent":
            continue
        out.append({
            "id": n.get("id"),
            "summary": to_str(n.get("summary", "")),
            "details": to_str(n.get("details", "")),
            "type": NODE_TYPE_FULL_NAMES["FIX"],
        })
    return out
