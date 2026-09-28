from typing import Dict, List, Any, Optional
import numpy as np


_SBERT_MODEL = None
_SBERT_NAME = "all-MiniLM-L6-v2"

def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer(_SBERT_NAME)
    return _SBERT_MODEL


def kg_seed_retrieve(
    kg,
    node_type: str,
    query: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    nodes = kg.search_nodes_by_type(node_type)
    if not nodes:
        return {"nodes": [], "count": 0}

    if not query:
        return {"nodes": nodes[:top_k], "count": len(nodes)}

    model = _get_sbert()
    texts, idx = [], []

    for i, n in enumerate(nodes):
        d = n.get("details", "")
        if isinstance(d, str) and d.strip():
            texts.append(d.strip())
            idx.append(i)

    if not texts:
        return {"nodes": [], "count": 0}

    q_emb = model.encode([query], normalize_embeddings=True)
    n_emb = model.encode(texts, normalize_embeddings=True)

    sims = np.dot(n_emb, q_emb.T).flatten()
    ranked = sorted(
        [(float(sims[i]), nodes[idx[i]]) for i in range(len(idx))],
        key=lambda x: x[0],
        reverse=True,
    )

    return {
        "nodes": [n for _, n in ranked[:top_k]],
        "count": len(ranked),
    }


def kg_neighbors_expand(
    kg,
    node_ids: List[str],
    min_sim: float = 0.7,
    max_hops: int = 2,
) -> Dict[str, Any]:
    visited = set()
    collected_nodes = {}
    collected_edges = []

    def dfs(nid: str, hop: int):
        if hop > max_hops or nid in visited:
            return
        visited.add(nid)

        node = kg.get_node_by_id(nid)
        if not node:
            return

        collected_nodes[nid] = node

        for info in node.get("neighbors", {}).values():
            sim = info.get("similarity", 0.0)
            dst = info.get("id")
            if dst and sim >= min_sim:
                collected_edges.append({
                    "src": nid,
                    "dst": dst,
                    "rel": "connected_with",
                    "similarity": sim,
                })
                dfs(dst, hop + 1)

        for e in kg.get_out_edges(nid):
            dst = e.get("dst")
            if dst:
                collected_edges.append(e)
                dfs(dst, hop + 1)

    for nid in node_ids:
        dfs(nid, 0)

    return {
        "nodes": list(collected_nodes.values()),
        "edges": collected_edges,
        "expanded_count": len(collected_nodes),
    }


def get_bridges_from_nodes(
    kg,
    node_ids: List[str],
) -> Dict[str, Any]:
    paths: List[Dict[str, Any]] = []
    seen_bridge: set = set()
    for nid in node_ids:
        for e in kg.get_out_edges(nid):
            dst = e.get("dst")
            if not dst or dst in seen_bridge:
                continue
            node_dst = kg.get_node_by_id(dst)
            if not node_dst or (node_dst.get("type") or "").strip().lower() != "bridgeentity":
                continue
            seen_bridge.add(dst)
            target_id = (node_dst.get("meta") or {}).get("target_id")
            if not target_id:
                continue
            target_node = kg.get_node_by_id(target_id)
            if not target_node:
                continue
            edge_to_dst = None
            for e2 in kg.get_out_edges(dst):
                if e2.get("dst") == target_id:
                    edge_to_dst = e2
                    break
            if not edge_to_dst:
                continue
            paths.append({
                "bridge": node_dst,
                "target": target_node,
                "edge_to_bridge": e,
                "edge_to_dst": edge_to_dst,
            })
    return {"paths": paths}


def kg_node_get(
    kg,
    node_id: str,
) -> Dict[str, Any]:
    node = kg.get_node_by_id(node_id)
    return {
        "node": node,
        "found": node is not None,
    }


def execute_tool(
    tool: str,
    kg_retriever: Any,
    code: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    kg = kg_retriever
    if tool == "kg_seed_retrieve":
        out = kg_seed_retrieve(
            kg,
            node_type=kwargs.get("node_type", ""),
            query=kwargs.get("query"),
            top_k=kwargs.get("neighbor_top_k", 5),
        )
        return {"nodes": out.get("nodes", []), "edges": []}
    if tool == "kg_neighbors_expand":
        out = kg_neighbors_expand(
            kg,
            node_ids=kwargs.get("node_ids", []),
            min_sim=kwargs.get("min_sim", 0.7),
            max_hops=kwargs.get("max_hops", 2),
        )
        return {"nodes": out.get("nodes", []), "edges": out.get("edges", [])}
    if tool == "get_bridges_from_nodes":
        out = get_bridges_from_nodes(kg, node_ids=kwargs.get("node_ids", []))
        return out
    if tool == "kg_node_get":
        node_id = kwargs.get("node_id", "")
        out = kg_node_get(kg, node_id)
        node = out.get("node")
        return {"nodes": [node] if node else [], "edges": []}
    return {"nodes": [], "edges": []}