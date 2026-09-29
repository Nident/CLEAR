from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from langchain_core.messages import SystemMessage, HumanMessage

import json
import re

SLOT_TYPES_DETECTION = [
    "Entrypoint",
    "vulnerability_precondition",
    "vulnerability_root_cause",
    "fix_intent",
]
SLOT_TYPES_CLASSIFICATION = [
    "Entrypoint",
    "vulnerability_precondition",
    "vulnerability_root_cause",
    "CWE",
]

MAX_STEPS = 15

SLOT_QUERY_PROMPTS = {
    "Entrypoint": """Write one sentence describing the entrypoint function's purpose in the code. Return JSON only: {"query": "<sentence>"}""",
    "vulnerability_precondition": """Write one sentence describing the vulnerability precondition implied by the code. Return JSON only: {"query": "<sentence>"}""",
    "vulnerability_root_cause": """Write one sentence describing the vulnerability root cause implied by the code. Return JSON only: {"query": "<sentence>"}""",
    "fix_intent": """Write one sentence describing the fix intent that would mitigate the vulnerability. Return JSON only: {"query": "<sentence>"}""",
    "CWE": """Write one sentence describing which CWE (weakness) type this vulnerability might relate to (e.g. CWE-78, CWE-89). Return JSON only: {"query": "<sentence>"}""",
}

FILTER_PROMPT = """You are a Collector Agent. Given code, target_type, and candidate KG nodes, select ONLY nodes that are relevant to the code and the current slot type. Return JSON only: { "keep_node_ids": ["..."], "reason": "..." }"""

BRIDGE_PRIORITY_PROMPT = """You are a Collector Agent. Given code and a list of bridge evidence (each has bridge_id, summary, details). These bridges connect the current evidence to the next conceptual step. Rank them by usefulness for the given code: which bridge is most likely to lead to relevant evidence for this code? Return JSON only: { "priority_order": ["<bridge_id_1>", "<bridge_id_2>", ...] } with the most useful first. Include every bridge_id exactly once."""

SLOT_FALLBACK_PROMPT = """The filter step found no relevant nodes for slot type "{target_type}". Write ONE sentence describing what kind of concept should be retrieved for this slot, based strictly on the code. Return JSON only: {{"query": "<sentence>"}}"""

@dataclass
class Memory:
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def ids(self) -> List[str]:
        return list(self.nodes.keys())

def _parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.I)
    try:
        return json.loads(m.group(1) if m else text)
    except Exception:
        return {}


def _coerce_llm_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("text"), str):
            return raw["text"]
        if isinstance(raw.get("content"), str):
            return raw["content"]
        return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, list):
        parts: List[str] = []
        for item in raw:
            s = _coerce_llm_text(item)
            if s:
                parts.append(s)
        return "\n".join(parts)
    return str(raw)


def _invoke(llm: Any, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    resp = llm.invoke(messages)
    raw = resp.content if hasattr(resp, "content") else resp
    text = _coerce_llm_text(raw)
    return _parse_json(text)


def collect_evidence(
    code: str,
    kg_retriever: Any,
    execute_tool: Any,
    llm: Any,
    min_sim: float = 0.7,
    neighbor_top_k: int = 15,
    max_hops: int = 1,
    debug: bool = False,
    task_mode: str = "detection",
) -> Dict[str, Any]:
    mem = Memory()
    step = 0
    current_keep_ids: Set[str] = set()
    slot_types = SLOT_TYPES_CLASSIFICATION if task_mode == "classification" else SLOT_TYPES_DETECTION

    for i, slot in enumerate(slot_types):
        if step >= MAX_STEPS:
            break

        if debug:
            print(f"\n[{slot}] Collecting evidence")

        if slot == "Entrypoint":
            q_payload: Dict[str, Any] = {"code": code}
            q_out = _invoke(llm, SLOT_QUERY_PROMPTS[slot], q_payload)
            query = q_out.get("query", "")
            if debug:
                print(f"[{slot}] query = {query}")

            out = execute_tool(
                "kg_seed_retrieve",
                kg_retriever=kg_retriever,
                code=code,
                node_type=slot,
                query=query,
                min_sim=min_sim,
                neighbor_top_k=neighbor_top_k,
                max_hops=max_hops,
            )
            candidates = [
                n for n in out.get("nodes", [])
                if isinstance(n, dict) and n.get("type") == slot
            ]
        else:
            candidates = [
                n for n in mem.nodes.values()
                if isinstance(n, dict) and n.get("type") == slot
            ]
            query = ""

        if not candidates:
            if slot != "Entrypoint" and debug:
                print(f"[{slot}] no candidates from expansion, fallback query")
            if slot == "Entrypoint" or not candidates:
                fb_payload: Dict[str, Any] = {"code": code}
                fb = _invoke(
                    llm,
                    SLOT_FALLBACK_PROMPT.format(target_type=slot),
                    fb_payload,
                )
                fallback_query = fb.get("query", "")
                if debug:
                    print(f"[Fallback query] {fallback_query}")
                out = execute_tool(
                    "kg_seed_retrieve",
                    kg_retriever=kg_retriever,
                    code=code,
                    node_type=slot,
                    query=fallback_query,
                    min_sim=min_sim,
                    neighbor_top_k=neighbor_top_k,
                    max_hops=max_hops,
                )
                candidates = [
                    n for n in out.get("nodes", [])
                    if isinstance(n, dict) and n.get("type") == slot
                ]
                query = fallback_query
            if not candidates:
                if debug:
                    print(f"[{slot}] no candidates after fallback, skip")
                mem.trace.append({
                    "step": step,
                    "slot": slot,
                    "query": query or "(fallback)",
                    "kept_nodes": [],
                    "reason": "skipped (no candidates)",
                })
                step += 1
                continue

        filt_payload: Dict[str, Any] = {"code": code, "target_type": slot, "candidates": candidates}
        filt = _invoke(llm, FILTER_PROMPT, filt_payload)
        keep_ids = set(filt.get("keep_node_ids", []))
        if debug:
            print(f"[Filtered] {filt}")

        if not keep_ids:
            if debug:
                print(f"[{slot}] filter returned no nodes, skip")
            mem.trace.append({
                "step": step,
                "slot": slot,
                "query": query or "(from expansion)",
                "kept_nodes": [],
                "reason": "skipped (filter returned no nodes)",
            })
            step += 1
            continue

        if slot == "Entrypoint":
            for n in out.get("nodes", []):
                if n.get("id") in keep_ids:
                    mem.nodes[n["id"]] = n
        else:
            for n in candidates:
                if n.get("id") in keep_ids:
                    mem.nodes[n["id"]] = n

        next_slot = slot_types[i + 1] if i + 1 < len(slot_types) else None
        bridge_out = execute_tool(
            "get_bridges_from_nodes",
            kg_retriever=kg_retriever,
            node_ids=list(keep_ids),
        )
        paths = bridge_out.get("paths", [])
        path_added = False
        if paths:
            bridge_list = [
                {
                    "bridge_id": p["bridge"].get("id"),
                    "summary": (p["bridge"].get("summary") or "").strip(),
                    "details": (p["bridge"].get("details") or "").strip(),
                }
                for p in paths
            ]
            pri_payload: Dict[str, Any] = {"code": code, "bridges": bridge_list}
            pri_out = _invoke(
                llm,
                BRIDGE_PRIORITY_PROMPT,
                pri_payload,
            )
            priority_order = pri_out.get("priority_order", [])
            if not isinstance(priority_order, list):
                priority_order = []
            by_bridge_id = {p["bridge"].get("id"): p for p in paths}
            for bridge_id in priority_order:
                p = by_bridge_id.get(bridge_id)
                if not p:
                    continue
                target = p["target"]
                if next_slot and (target.get("type") or "").strip() != next_slot:
                    continue
                bridge_node = p["bridge"]
                bid = bridge_node["id"]
                tid = target["id"]
                e1 = p.get("edge_to_bridge")
                e2 = p.get("edge_to_dst")
                mem.nodes[bid] = bridge_node
                mem.nodes[tid] = target
                added_edges: List[Dict[str, Any]] = []
                if e1 and e1.get("src") in mem.nodes and e1.get("dst") in mem.nodes:
                    mem.edges.append(e1)
                    added_edges.append(e1)
                if e2 and e2.get("src") in mem.nodes and e2.get("dst") in mem.nodes:
                    mem.edges.append(e2)
                    added_edges.append(e2)
                if not next_slot:
                    path_added = True
                    break
                next_bridges = execute_tool(
                    "get_bridges_from_nodes",
                    kg_retriever=kg_retriever,
                    node_ids=[tid],
                )
                if next_bridges.get("paths"):
                    path_added = True
                    break
                for e in added_edges:
                    if e in mem.edges:
                        mem.edges.remove(e)
                if bid in mem.nodes:
                    del mem.nodes[bid]
                if tid in mem.nodes:
                    del mem.nodes[tid]
        if not path_added and next_slot:
            next_candidates = [
                n for n in mem.nodes.values()
                if isinstance(n, dict) and (n.get("type") or "").strip() == next_slot
            ]
            if not next_candidates:
                fb_payload = {"code": code}
                fb = _invoke(
                    llm,
                    SLOT_FALLBACK_PROMPT.format(target_type=next_slot),
                    fb_payload,
                )
                fallback_query = fb.get("query", "")
                if debug:
                    print(f"[Fallback next slot {next_slot}] {fallback_query}")
                out_fb = execute_tool(
                    "kg_seed_retrieve",
                    kg_retriever=kg_retriever,
                    code=code,
                    node_type=next_slot,
                    query=fallback_query,
                    min_sim=min_sim,
                    neighbor_top_k=neighbor_top_k,
                    max_hops=max_hops,
                )
                for n in out_fb.get("nodes", []):
                    if isinstance(n, dict) and n.get("id") and (n.get("type") or "").strip() == next_slot:
                        mem.nodes[n["id"]] = n

        mem.trace.append({
            "step": step,
            "slot": slot,
            "query": query or "(from expansion)",
            "kept_nodes": list(keep_ids),
            "reason": filt.get("reason", ""),
        })
        step += 1

    return {
        "evidence": {
            "nodes": list(mem.nodes.values()),
            "edges": mem.edges,
        },
        "trace": mem.trace,
        "slot_order": list(slot_types),
        "meta": {
            "node_count": len(mem.nodes),
            "edge_count": len(mem.edges),
        },
    }