from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

import numpy as np

from tools.tools import (
    _get_sbert,
    get_bridges_from_nodes,
    kg_neighbors_expand,
    kg_node_get,
)
from utils.kg_retriever import KGRetriever


ALLOWED_SLOT_TYPES = {
    "Entrypoint",
    "vulnerability_precondition",
    "vulnerability_root_cause",
    "fix_intent",
    "CWE",
}


def _text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "\n".join(str(item) for item in value)
    rendered = str(value).strip()
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 1] + "…"


def _compact_node(node: dict[str, Any]) -> dict[str, Any]:
    """Return only fields useful to the agent and bound their token cost."""
    result: dict[str, Any] = {
        "id": str(node.get("id", "")),
        "type": str(node.get("type", "")),
        "summary": _text(node.get("summary"), 800),
        "details": _text(node.get("details"), 3000),
    }
    if node.get("category"):
        result["category"] = _text(node["category"], 300)
    meta = node.get("meta")
    if isinstance(meta, dict):
        result["meta"] = {
            str(k): _text(v, 500)
            for k, v in meta.items()
            if k in {"segment", "source_id", "target_id", "source_type", "target_type"}
        }
    return result


def _compact_edge(edge: dict[str, Any] | None) -> dict[str, Any] | None:
    if not edge:
        return None
    result = {
        "src": str(edge.get("src", "")),
        "dst": str(edge.get("dst", "")),
        "rel": str(edge.get("rel", "")),
    }
    if edge.get("similarity") is not None:
        result["similarity"] = float(edge["similarity"])
    return result


class VCKGService:
    """Validated, size-bounded facade around CLEAR's existing retriever."""

    def __init__(self, graph_path: str | Path):
        self.graph_path = Path(graph_path).expanduser().resolve()
        if not self.graph_path.is_file():
            raise FileNotFoundError(f"VCKG file not found: {self.graph_path}")
        self.kg = KGRetriever(str(self.graph_path))
        self.graph_id = self.graph_path.stem
        self._embedding_lock = threading.RLock()
        self._seed_cache: dict[str, tuple[list[dict[str, Any]], Any]] = {}

    def info(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "graph_path": str(self.graph_path),
            "node_count": len(self.kg.nodes),
            "edge_count": len(self.kg.edges),
            "node_types": {
                key: len(value)
                for key, value in sorted(self.kg.nodes_by_type.items())
            },
        }

    def seed_retrieve(self, node_type: str, query: str, top_k: int = 5) -> dict[str, Any]:
        if node_type not in ALLOWED_SLOT_TYPES:
            raise ValueError(
                f"Unsupported node_type={node_type!r}; expected one of {sorted(ALLOWED_SLOT_TYPES)}"
            )
        top_k = max(1, min(int(top_k), 10))
        query = _text(query, 2000)
        if not query:
            raise ValueError("query must be a non-empty semantic description derived from target code")
        # The original CLEAR helper re-encodes every node on every query. An
        # MCP server handles many tasks, so cache the static node embeddings
        # once per Collector slot and only encode each new query.
        with self._embedding_lock:
            cached = self._seed_cache.get(node_type)
            if cached is None:
                candidates = []
                texts = []
                for node in self.kg.search_nodes_by_type(node_type):
                    details = _text(node.get("details"), 10000)
                    if details:
                        candidates.append(node)
                        texts.append(details)
                model = _get_sbert()
                embeddings = model.encode(texts, normalize_embeddings=True)
                cached = (candidates, embeddings)
                self._seed_cache[node_type] = cached
            candidates, embeddings = cached
            query_embedding = _get_sbert().encode([query], normalize_embeddings=True)
            similarities = np.dot(embeddings, query_embedding.T).flatten()

        ranked = np.argsort(-similarities)[:top_k]
        selected = [candidates[int(index)] for index in ranked]
        nodes = [_compact_node(node) for node in selected]
        return {
            "graph_id": self.graph_id,
            "node_type": node_type,
            "query": query,
            "nodes": nodes,
            "returned": len(nodes),
            "candidate_count": len(candidates),
        }

    def get_bridges(
        self,
        node_ids: list[str],
        target_type: str | None = None,
        max_paths: int = 15,
    ) -> dict[str, Any]:
        clean_ids = list(dict.fromkeys(str(value) for value in node_ids if value))[:10]
        if not clean_ids:
            raise ValueError("node_ids must contain at least one VCKG node id")
        if target_type is not None and target_type not in ALLOWED_SLOT_TYPES:
            raise ValueError(f"Unsupported target_type={target_type!r}")
        max_paths = max(1, min(int(max_paths), 30))
        out = get_bridges_from_nodes(self.kg, clean_ids)
        paths: list[dict[str, Any]] = []
        for path in out.get("paths", []):
            target = path.get("target") or {}
            if target_type and target.get("type") != target_type:
                continue
            paths.append({
                "bridge": _compact_node(path.get("bridge") or {}),
                "target": _compact_node(target),
                "edge_to_bridge": _compact_edge(path.get("edge_to_bridge")),
                "edge_to_target": _compact_edge(path.get("edge_to_dst")),
            })
            if len(paths) >= max_paths:
                break
        return {
            "graph_id": self.graph_id,
            "source_node_ids": clean_ids,
            "target_type": target_type,
            "paths": paths,
            "returned": len(paths),
        }

    def expand_nodes(
        self,
        node_ids: list[str],
        min_similarity: float = 0.7,
        max_hops: int = 1,
        max_nodes: int = 30,
    ) -> dict[str, Any]:
        clean_ids = list(dict.fromkeys(str(value) for value in node_ids if value))[:10]
        if not clean_ids:
            raise ValueError("node_ids must contain at least one VCKG node id")
        min_similarity = max(0.0, min(float(min_similarity), 1.0))
        max_hops = max(0, min(int(max_hops), 2))
        max_nodes = max(1, min(int(max_nodes), 50))
        out = kg_neighbors_expand(
            self.kg,
            node_ids=clean_ids,
            min_sim=min_similarity,
            max_hops=max_hops,
        )
        nodes = [_compact_node(node) for node in out.get("nodes", [])[:max_nodes]]
        visible = {node["id"] for node in nodes}
        edges = [
            compact
            for edge in out.get("edges", [])
            if edge.get("src") in visible and edge.get("dst") in visible
            if (compact := _compact_edge(edge)) is not None
        ][: max_nodes * 2]
        return {
            "graph_id": self.graph_id,
            "nodes": nodes,
            "edges": edges,
            "returned_nodes": len(nodes),
            "expanded_count": int(out.get("expanded_count", len(nodes))),
            "truncated": int(out.get("expanded_count", len(nodes))) > len(nodes),
        }

    def get_node(self, node_id: str) -> dict[str, Any]:
        out = kg_node_get(self.kg, str(node_id))
        node = out.get("node")
        return {
            "graph_id": self.graph_id,
            "found": node is not None,
            "node": _compact_node(node) if node else None,
        }
