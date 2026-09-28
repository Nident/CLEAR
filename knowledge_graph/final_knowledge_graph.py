from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.llm import get_llm, load_llm_config
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge_graph.constants import (
    REL_CLASSIFIED_AS as REL_RC_TO_CWE,
    REL_CONNECTED_WITH as REL_NEIGHBOR,
    REL_ENABLES as REL_PRE_TO_RC,
    REL_HAS_PROPERTY as REL_EP_TO_PRE,
    REL_MITIGATED_BY as REL_RC_TO_FIX,
    TYPE_CWE,
    TYPE_EP,
    TYPE_FIX,
    TYPE_PRE,
    TYPE_RC,
)
from knowledge_graph.io_common import extract_instances, read_json, write_json

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return it


BRIDGE_TYPE = "BridgeEntity"

ALLOWED_RELATIONS = {
    "INVOLVES", "REQUIRES", "LACKS", "USES", "VALIDATES", "SANITIZES",
    "PASSES_TO", "REACHES", "TRIGGERS", "CAUSES", "PREVENTS", "DEPENDS_ON",
}

DEFAULT_MAX_CANDIDATES_IN_PROMPT = 260
MODEL_NAME = None

@dataclass(frozen=True)
class SegmentSpec:
    name: str
    src_type: str
    dst_type: str
    intra_rel: str
    hint_from: str
    hint_to: str


SEGMENTS: List[SegmentSpec] = [
    SegmentSpec("CROSS_EP__PRE", TYPE_EP,  TYPE_PRE, REL_EP_TO_PRE, "REQUIRES", "TRIGGERS"),
    SegmentSpec("CROSS_PRE__RC", TYPE_PRE, TYPE_RC,  REL_PRE_TO_RC, "REQUIRES", "CAUSES"),
    SegmentSpec("CROSS_RC__FIX", TYPE_RC,  TYPE_FIX, REL_RC_TO_FIX, "REQUIRES", "PREVENTS"),
]


@dataclass(frozen=True)
class NodeView:
    id: str
    type: str
    summary: str
    details: str
    inst_idx: int

SYSTEM_PROMPT_GROUP = """
You are a relation typer for cross-CVE bridges in vulnerability knowledge graphs.

You will receive:
- One SRC node (summary/details)
- Multiple DST candidates collected by a fixed structural pattern (neighbor -> next-hop)

Task:
- Select ONLY the links that are strongly justified by the given node texts.
- Prefer a minimal set (avoid redundant / weakly-related links).
- For each accepted link, create ONE BridgeEntity (summary/details) and choose two relations:
    rel_from_src: SRC -> Bridge
    rel_to_dst:   Bridge -> DST

Return JSON only.
""".strip()


def build_user_prompt_group(
    segment: str,
    src_type: str,
    src_id: str,
    src_sum: str,
    src_det: str,
    candidates_block: str,
    allowed_relations: str,
    hint_from: str,
    hint_to: str,
) -> str:
    return f"""
[SEGMENT]
{segment}

[SRC]
type: {src_type}
id: {src_id}
summary: {src_sum}
details: {src_det}

[CANDIDATES] (via neighbor -> next-hop)
{candidates_block}

Allowed relations: {allowed_relations}
Hints (optional): rel_from_src={hint_from}, rel_to_dst={hint_to}

Constraints:
- dst_id must exactly match a candidate.
- rel_from_src / rel_to_dst must be from Allowed relations.

Return JSON ONLY:
{{
  "accepted": [
    {{
      "dst_id": "...",
      "bridge_summary": "1~5 words",
      "bridge_details": "1~2 sentences",
      "rel_from_src": "REQUIRES",
      "rel_to_dst": "TRIGGERS",
      "reason": "<= 2 sentences grounded"
    }}
  ]
}}
""".strip()

def norm(s: Any) -> str:
    s = ("" if s is None else str(s)).strip()
    return re.sub(r"\s+", " ", s)


def sanitize_for_prompt(s: Any) -> str:
    if s is None:
        return ""
    try:
        t = str(s)
    except Exception:
        return ""
    t = t.encode("utf-8", errors="replace").decode("utf-8")
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    return t.strip()

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def edge_key(src: str, rel: str, dst: str) -> Tuple[str, str, str]:
    return (str(src), str(rel), str(dst))

def sanitize_relation(r: Any) -> Optional[str]:
    rr = norm(r).upper().replace("-", "_")
    return rr if rr in ALLOWED_RELATIONS else None

def stable_bridge_id(segment: str, src_id: str, dst_id: str, bsum: str, bdet: str) -> str:
    payload = f"{segment}||{src_id}||{dst_id}||{norm(bsum).lower()}||{norm(bdet).lower()}"
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"BRIDGE::{segment}::{h}"

def extract_json_obj(text: str) -> Dict[str, Any]:
    t = norm(text)
    if not t:
        return {}
    if "```" in t:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", t, flags=re.IGNORECASE)
        if m:
            t = m.group(1).strip()
    if not (t.startswith("{") and t.endswith("}")):
        m = re.search(r"(\{[\s\S]*\})", t)
        if m:
            t = m.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def build_node_index(instances: List[Dict[str, Any]]) -> Dict[str, NodeView]:
    idx: Dict[str, NodeView] = {}
    for inst_i, inst in enumerate(instances):
        for n in inst.get("nodes", []) or []:
            nid = n.get("id")
            if isinstance(nid, str) and nid:
                idx[nid] = NodeView(
                    id=nid,
                    type=norm(n.get("type")),
                    summary=norm(n.get("summary")),
                    details=norm(n.get("details")),
                    inst_idx=inst_i,
                )
    return idx

def build_out_adj(instances: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for inst in instances:
        for e in inst.get("edges", []) or []:
            if not isinstance(e, dict):
                continue
            src = e.get("src")
            if isinstance(src, str) and src:
                out.setdefault(src, []).append(e)
    return out

def build_global_edge_set(instances: List[Dict[str, Any]]) -> set:
    s = set()
    for inst in instances:
        for e in inst.get("edges", []) or []:
            if isinstance(e, dict):
                s.add(edge_key(e.get("src", ""), e.get("rel", ""), e.get("dst", "")))
    return s


def collect_next_hop_candidates(
    src_id: str,
    seg: SegmentSpec,
    node_idx: Dict[str, NodeView],
    out_adj: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if src_id not in node_idx or src_id not in out_adj:
        return []

    neighbors: List[Tuple[float, str]] = []
    for e in out_adj[src_id]:
        if norm(e.get("rel")) != REL_NEIGHBOR:
            continue
        nb_id = norm(e.get("dst"))
        if not nb_id or nb_id not in node_idx:
            continue
        if node_idx[nb_id].type != seg.src_type:
            continue
        sim = safe_float((e.get("meta") or {}).get("similarity"), 0.0)
        neighbors.append((sim, nb_id))

    if not neighbors:
        return []

    neighbors.sort(key=lambda x: x[0], reverse=True)

    best: Dict[str, Tuple[float, str]] = {}
    for sim, nb_src in neighbors:
        for e2 in out_adj.get(nb_src, []):
            if norm(e2.get("rel")) != seg.intra_rel:
                continue
            dst_id = norm(e2.get("dst"))
            if not dst_id or dst_id not in node_idx:
                continue
            if node_idx[dst_id].type != seg.dst_type:
                continue
            if (dst_id not in best) or (sim > best[dst_id][0]):
                best[dst_id] = (sim, nb_src)

    ranked = sorted(
        ((dst, sc, sup) for dst, (sc, sup) in best.items()),
        key=lambda x: x[1],
        reverse=True,
    )

    out: List[Dict[str, Any]] = []
    for dst_id, sim, sup in ranked:
        out.append({
            "dst_id": dst_id,
            "sim": float(sim),
            "support_neighbor_src_id": sup,
            "support_neighbor_src_summary": node_idx[sup].summary if sup in node_idx else "",
        })
    return out


def llm_group_type_bridges(
    llm,
    seg: SegmentSpec,
    src_view: NodeView,
    candidates: List[Dict[str, Any]],
    node_idx: Dict[str, NodeView],
    counters: Dict[str, Any],
    counters_lock: Lock,
    max_candidates_in_prompt: int = DEFAULT_MAX_CANDIDATES_IN_PROMPT,
) -> List[Dict[str, Any]]:
    candidates_for_prompt = candidates[:max_candidates_in_prompt]
    valid_dst = {c["dst_id"] for c in candidates_for_prompt if c.get("dst_id")}

    lines: List[str] = []
    for i, c in enumerate(candidates_for_prompt, 1):
        dv = node_idx.get(c["dst_id"])
        if not dv:
            continue
        lines.append(
            f"- cand#{i}\n"
            f"  dst_id: {sanitize_for_prompt(dv.id)}\n"
            f"  dst_summary: {sanitize_for_prompt(dv.summary)}\n"
            f"  dst_details: {sanitize_for_prompt(dv.details)}\n"
            f"  support_neighbor_src_id: {sanitize_for_prompt(c.get('support_neighbor_src_id',''))}\n"
            f"  support_neighbor_src_summary: {sanitize_for_prompt(c.get('support_neighbor_src_summary',''))}\n"
            f"  neighbor_similarity: {round(float(c.get('sim',0.0)), 4)}\n"
        )

    candidates_block = "".join(lines) if lines else "(none)"
    allowed_relations_str = ", ".join(sorted(ALLOWED_RELATIONS))

    src_sum_safe = sanitize_for_prompt(src_view.summary)
    src_det_safe = sanitize_for_prompt(src_view.details)

    prompt = build_user_prompt_group(
        segment=seg.name,
        src_type=seg.src_type,
        src_id=sanitize_for_prompt(src_view.id),
        src_sum=src_sum_safe,
        src_det=src_det_safe,
        candidates_block=candidates_block,
        allowed_relations=allowed_relations_str,
        hint_from=seg.hint_from,
        hint_to=seg.hint_to,
    )
    prompt = sanitize_for_prompt(prompt)

    try:
        resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT_GROUP), HumanMessage(content=prompt)])
    except Exception as e:
        err_text = str(e).lower()
        if ("invalid" in err_text and "json" in err_text) or "could not parse" in err_text:
            return []
        raise
    usage = getattr(resp, "response_metadata", {}) or {}
    usage = usage.get("token_usage", {}) or usage
    with counters_lock:
        counters["prompt_tokens"] = counters.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
        counters["completion_tokens"] = counters.get("completion_tokens", 0) + usage.get("completion_tokens", 0)
    obj = extract_json_obj(resp.content if hasattr(resp, "content") else str(resp))

    accepted = obj.get("accepted", [])
    if not isinstance(accepted, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for it in accepted:
        if not isinstance(it, dict):
            continue

        dst_id = norm(it.get("dst_id"))
        if dst_id not in valid_dst:
            continue

        bsum = norm(it.get("bridge_summary"))
        bdet = norm(it.get("bridge_details"))
        r1 = sanitize_relation(it.get("rel_from_src"))
        r2 = sanitize_relation(it.get("rel_to_dst"))
        reason = norm(it.get("reason"))

        if not (dst_id and bsum and bdet and r1 and r2):
            continue

        cleaned.append({
            "dst_id": dst_id,
            "bridge_summary": bsum,
            "bridge_details": bdet,
            "rel_from_src": r1,
            "rel_to_dst": r2,
            "reason": reason,
        })

    return cleaned


def ensure_node(inst: Dict[str, Any], node: Dict[str, Any]) -> None:
    nodes = inst.get("nodes") or []
    if not isinstance(nodes, list):
        nodes = []
    nid = node.get("id")
    if nid and any(isinstance(n, dict) and n.get("id") == nid for n in nodes):
        inst["nodes"] = nodes
        return
    nodes.append(node)
    inst["nodes"] = nodes

def append_edge_if_new(
    inst: Dict[str, Any],
    edge: Dict[str, Any],
    global_edge_set: set,
    lock: Lock,
) -> bool:
    edges = inst.get("edges") or []
    if not isinstance(edges, list):
        edges = []
    k = edge_key(edge.get("src", ""), edge.get("rel", ""), edge.get("dst", ""))
    with lock:
        if k in global_edge_set:
            inst["edges"] = edges
            return False
        global_edge_set.add(k)
    edges.append(edge)
    inst["edges"] = edges
    return True


def process_instance(
    inst_idx: int,
    inst: Dict[str, Any],
    segments: List[SegmentSpec],
    node_idx: Dict[str, NodeView],
    out_adj: Dict[str, List[Dict[str, Any]]],
    global_edge_set: set,
    edge_lock: Lock,
    model_name: str,
    config: Dict[str, Any],
    counters: Dict[str, int],
    counters_lock: Lock,
    done_set: Optional[set] = None,
    max_candidates_in_prompt: int = DEFAULT_MAX_CANDIDATES_IN_PROMPT,
) -> int:
    done_set = done_set or set()
    llm = get_llm(model_name, config)
    nodes = inst.get("nodes") or []
    if not isinstance(nodes, list):
        return 0

    added = 0
    for seg in segments:
        src_ids = [
            n.get("id") for n in nodes
            if isinstance(n, dict)
            and norm(n.get("type")) == seg.src_type
            and isinstance(n.get("id"), str)
        ]

        for src_id in filter(None, src_ids):
            if (inst_idx, seg.name, src_id) in done_set:
                continue
            if src_id not in node_idx:
                continue
            src_view = node_idx[src_id]

            cands = collect_next_hop_candidates(src_id, seg, node_idx, out_adj)
            cands = [
                c for c in cands
                if node_idx.get(c["dst_id"])
                and node_idx[c["dst_id"]].inst_idx != src_view.inst_idx
            ]
            if not cands:
                continue

            accepted = llm_group_type_bridges(llm, seg, src_view, cands, node_idx, counters, counters_lock)
            with counters_lock:
                counters["llm_calls"] += 1

            if not accepted:
                continue

            for it in accepted:
                dst_id = it["dst_id"]
                dst_view = node_idx[dst_id]
                cand = next((c for c in cands if c["dst_id"] == dst_id), None) or {}
                support_nb = cand.get("support_neighbor_src_id", "")
                support_nb_sum = cand.get("support_neighbor_src_summary", "")
                sim = float(cand.get("sim", 0.0))

                bridge_id = stable_bridge_id(seg.name, src_id, dst_id, it["bridge_summary"], it["bridge_details"])

                ensure_node(inst, {
                    "type": BRIDGE_TYPE,
                    "id": bridge_id,
                    "summary": it["bridge_summary"],
                    "details": it["bridge_details"],
                    "meta": {
                        "segment": seg.name,
                        "source_id": src_id,
                        "target_id": dst_id,
                        "source_type": seg.src_type,
                        "target_type": seg.dst_type,
                    },
                })

                e1 = {
                    "src": src_id,
                    "rel": it["rel_from_src"],
                    "dst": bridge_id,
                    "meta": {
                        "segment": seg.name,
                        "role": "SRC_TO_BRIDGE",
                        "method": "neighbors_next_hop_group",
                        "support_neighbor_src_id": support_nb,
                        "support_neighbor_src_summary": support_nb_sum,
                        "neighbor_similarity": round(sim, 4),
                        "verifier_reason": it.get("reason", ""),
                        "target_id": dst_id,
                        "target_instance": dst_view.inst_idx,
                    },
                }
                e2 = {
                    "src": bridge_id,
                    "rel": it["rel_to_dst"],
                    "dst": dst_id,
                    "meta": {
                        "segment": seg.name,
                        "role": "BRIDGE_TO_DST",
                        "method": "neighbors_next_hop_group",
                        "support_neighbor_src_id": support_nb,
                        "support_neighbor_src_summary": support_nb_sum,
                        "neighbor_similarity": round(sim, 4),
                        "verifier_reason": it.get("reason", ""),
                        "source_id": src_id,
                        "source_instance": src_view.inst_idx,
                    },
                }

                if append_edge_if_new(inst, e1, global_edge_set, edge_lock):
                    added += 1
                if append_edge_if_new(inst, e2, global_edge_set, edge_lock):
                    added += 1

    return added

def build_done_set_from_output(instances: List[Dict[str, Any]]) -> set:
    done: set = set()
    for inst_i, inst in enumerate(instances):
        for e in inst.get("edges", []) or []:
            seg = (e.get("meta") or {}).get("segment")
            src = e.get("src")
            if seg and src:
                done.add((inst_i, seg, src))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="knowledge_graph.json")
    ap.add_argument("--output", default="final_knowledge_graph.json")
    ap.add_argument("--llm_config", default="../llm_config.json")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Resume: load output, skip (inst, segment, src) already present, then merge and save")
    ap.add_argument("--save_every", type=int, default=1, help="Write output file every N instances (default: 1 = after each)")
    args = ap.parse_args()

    out_path = args.output or args.input
    save_every = max(1, args.save_every)
    done_set: set = set()
    if args.resume and Path(out_path).is_file():
        print(f"[Resume] Loading existing output: {out_path}")
        raw = read_json(out_path)
        instances, wrapped = extract_instances(raw)
        done_set = build_done_set_from_output(instances)
        print(f"[Resume] Skipping {len(done_set)} (inst, segment, src) already done.")
    else:
        raw = read_json(args.input)
        instances, wrapped = extract_instances(raw)

    source_instances = instances[: max(0, args.limit)] if args.limit is not None else instances

    config = load_llm_config(args.llm_config)
    models = config.get("models", [])
    if not models or not isinstance(models[0], dict) or not models[0].get("name"):
        raise ValueError("llm_config.json has no configured model")
    model_name = str(models[0]["name"])
    print(f"[Config] model={model_name}", flush=True)

    node_idx = build_node_index(instances)
    out_adj = build_out_adj(instances)

    global_edge_set = build_global_edge_set(instances)
    edge_lock = Lock()


    counters = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    counters_lock = Lock()

    out_obj = {"data": instances} if wrapped else instances
    total_added = 0
    n_futs = len(source_instances)
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        futs = [
            ex.submit(
                process_instance,
                inst_idx,
                inst,
                SEGMENTS,
                node_idx,
                out_adj,
                global_edge_set,
                edge_lock,
                model_name,
                config,
                counters,
                counters_lock,
                done_set,
            )
            for inst_idx, inst in enumerate(source_instances)
        ]
        for i, fut in enumerate(tqdm(as_completed(futs), total=n_futs, desc="Cross-CVE bridges")):
            total_added += int(fut.result() or 0)
            if (i + 1) % save_every == 0 or (i + 1) == n_futs:
                write_json(out_path, out_obj)

    PRICE_PER_1M = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4": (30.00, 60.00)}
    inp, out = PRICE_PER_1M.get(model_name, (0.15, 0.60))
    total_cost = (counters["prompt_tokens"] * inp + counters["completion_tokens"] * out) / 1_000_000.0
    avg_cost = total_cost / counters["llm_calls"] if counters["llm_calls"] else 0.0

    print("\nSummary:")
    print(f"  full_instances_loaded:      {len(instances)}")
    print(f"  source_instances_processed: {len(source_instances)}")
    print(f"  cross_edges_added:          {total_added} (2 edges per accepted link)")
    print(f"  llm_calls:                  {counters['llm_calls']}")
    print(f"  prompt_tokens:              {counters.get('prompt_tokens', 0)}")
    print(f"  completion_tokens:         {counters.get('completion_tokens', 0)}")
    print(f"  total API call cost ($):     {total_cost:.6f}")
    print(f"  average API call cost ($):   {avg_cost:.6f}")

if __name__ == "__main__":
    main()
