from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from starlette.responses import JSONResponse

from .service import VCKGService


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = (
    ROOT
    / "knowledge_graph"
    / "project-lvl-train"
    / "project_lvl_final_knowledge_graph.json"
)
GRAPH_PATH = Path(os.environ.get("VCKG_PATH", str(DEFAULT_GRAPH)))
HOST = os.environ.get("VCKG_HOST", "127.0.0.1")
PORT = int(os.environ.get("VCKG_PORT", "4250"))

service = VCKGService(GRAPH_PATH)
mcp = FastMCP(
    "CLEAR VCKG",
    instructions=(
        "Read-only retrieval over the CLEAR Vulnerability Causal Knowledge Graph. "
        "VCKG nodes are general prior knowledge, never proof about the target commit. "
        "Verify every hypothesis against target code with CodeBadger/CPG."
    ),
)


@mcp.tool(
    name="vckg_seed_retrieve",
    description=(
        "Semantic seed retrieval for one Collector slot. Query must describe a fact or "
        "hypothesis derived from the target code. Supported slots: Entrypoint, "
        "vulnerability_precondition, vulnerability_root_cause, fix_intent, CWE."
    ),
)
def vckg_seed_retrieve(node_type: str, query: str, top_k: int = 5) -> dict[str, Any]:
    return service.seed_retrieve(node_type=node_type, query=query, top_k=top_k)


@mcp.tool(
    name="vckg_get_bridges",
    description=(
        "Follow causal bridge entities from selected VCKG nodes toward the next Collector "
        "slot. Set target_type to keep only paths ending in the desired next slot."
    ),
)
def vckg_get_bridges(
    node_ids: list[str],
    target_type: str | None = None,
    max_paths: int = 15,
) -> dict[str, Any]:
    return service.get_bridges(node_ids, target_type=target_type, max_paths=max_paths)


@mcp.tool(
    name="vckg_expand_nodes",
    description=(
        "Expand selected VCKG nodes through similarity and directed causal edges. Use only "
        "when bridge traversal is insufficient; output and depth are strictly bounded."
    ),
)
def vckg_expand_nodes(
    node_ids: list[str],
    min_similarity: float = 0.7,
    max_hops: int = 1,
    max_nodes: int = 30,
) -> dict[str, Any]:
    return service.expand_nodes(
        node_ids,
        min_similarity=min_similarity,
        max_hops=max_hops,
        max_nodes=max_nodes,
    )


@mcp.tool(
    name="vckg_get_node",
    description="Fetch one VCKG node by exact id for evidence citation or inspection.",
)
def vckg_get_node(node_id: str) -> dict[str, Any]:
    return service.get_node(node_id)


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    return JSONResponse({"status": "up", "mcp": "vckg", **service.info()})


@mcp.custom_route("/", methods=["GET"])
async def root(_request):
    return JSONResponse({
        "service": "vckg",
        "description": "CLEAR Vulnerability Causal Knowledge Graph MCP server",
        "endpoints": {"health": "/health", "mcp": "/mcp"},
        **service.info(),
    })


if __name__ == "__main__":
    asyncio.run(mcp.run_http_async(host=HOST, port=PORT))
