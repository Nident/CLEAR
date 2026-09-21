#!/usr/bin/env python3
"""Build a CLEAR-style Vulnerability Causal Knowledge Graph."""
from __future__ import annotations

import json, os, re, zipfile
from pathlib import Path
from typing import Any, Iterator

import networkx as nx
from dotenv import load_dotenv


class VCKGBuilder:
    ROLES = ("EP", "PC", "RC", "FI")

    def __init__(self, top_k=5, min_similarity=0.20, embedding="tfidf"):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.embedding = embedding

    def load_records(self, path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
        if path.is_dir():
            for file_path in path.rglob("*.json"):
                yield file_path.stem, json.loads(file_path.read_text(encoding="utf-8"))
            return
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith(".json") and not name.startswith("__MACOSX/"):
                    yield Path(name).stem, json.loads(archive.read(name))

    def record_text(self, record):
        parts = [record.get("summary", ""), record.get("details", ""), record.get("project", "")]
        for key in ("before", "after"):
            value = record.get(key, {})
            if isinstance(value, dict):
                parts.extend([str(value.get("context", "")), str(value.get("snippet", ""))])
        for update in record.get("updates", []):
            if isinstance(update, dict):
                parts.extend([str(update.get("snippet", "")), str(update.get("context", ""))])
        return "\n".join(part for part in parts if part)

    def extract_entities(self, record):
        text = self.record_text(record)
        summary = record.get("summary", "") or "unknown vulnerability"
        cwe = ", ".join(record.get("cwe_ids", []) or ["CWE-Unknown"])
        patch = " ".join(str(u.get("patch", "")) for u in record.get("updates", []) if isinstance(u, dict))
        return {
            "EP": f"{record.get('language', 'unknown')} input reaches {record.get('project', 'unknown')} code",
            "PC": summary,
            "RC": re.sub(r"\s+", " ", (summary + " " + text)[:900]),
            "FI": re.sub(r"\s+", " ", patch[:900]) if patch else f"Apply the security fix for {summary}",
            "CWE": cwe,
        }

    def create_local_graph(self, records):
        graph = nx.MultiDiGraph(name="CLEAR VCKG")
        nodes = []
        for record_id, record in records:
            cwe = "|".join(record.get("cwe_ids", []) or ["CWE-Unknown"])
            for role, value in self.extract_entities(record).items():
                node_id = f"{record_id}:{role}"
                graph.add_node(node_id, role=role, text=value, record_id=record_id, cwe=cwe)
                nodes.append((node_id, role, value, cwe))
            for source, target in zip(self.ROLES, self.ROLES[1:]):
                graph.add_edge(f"{record_id}:{source}", f"{record_id}:{target}", kind="local")
            for role in self.ROLES:
                graph.add_edge(f"{record_id}:{role}", f"{record_id}:CWE", kind="local")
        return graph, nodes

    def build_similarity(self, texts):
        if self.embedding == "minilm":
            from sentence_transformers import SentenceTransformer
            import numpy as np
            vectors = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, normalize_embeddings=True)
            return lambda i, j: float(np.dot(vectors[i], vectors[j]))
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectors = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
        return lambda i, j: float(cosine_similarity(vectors[i], vectors[j])[0, 0])

    def add_neighborhood_edges(self, graph, nodes):
        similarity = self.build_similarity([node[2] for node in nodes])
        clusters = {}
        for index, (_, role, _, cwe) in enumerate(nodes):
            clusters.setdefault((role, cwe), []).append(index)
        for indexes in clusters.values():
            for index in indexes:
                ranked = sorted(((similarity(index, other), other) for other in indexes if other != index), reverse=True)[:self.top_k]
                for score, other in ranked:
                    if score >= self.min_similarity:
                        graph.add_edge(nodes[index][0], nodes[other][0], kind="neighborhood", score=round(score, 6))

    def build(self, records):
        if not records:
            raise ValueError("No JSON records found")
        graph, nodes = self.create_local_graph(records)
        self.add_neighborhood_edges(graph, nodes)
        return graph

    def save(self, graph, output_path: Path, record_count):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graph, output_path)
        summary = {"records": record_count, "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "output": str(output_path)}
        output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


def main():
    config_path = Path(__file__).parent / "config" / ".env"
    load_dotenv(config_path)

    input_path = Path(os.getenv("VCKG_INPUT", "data/sampled.zip"))
    output_path = Path(os.getenv("VCKG_OUTPUT", "outputs/vckg.graphml"))
    top_k = int(os.getenv("VCKG_TOP_K", "5"))
    min_similarity = float(os.getenv("VCKG_MIN_SIMILARITY", "0.20"))
    embedding = os.getenv("VCKG_EMBEDDING", "tfidf")

    if embedding not in ("tfidf", "minilm"):
        raise ValueError("VCKG_EMBEDDING must be either 'tfidf' or 'minilm'")

    builder = VCKGBuilder(top_k, min_similarity, embedding)
    records = list(builder.load_records(input_path))
    graph = builder.build(records)
    builder.save(graph, output_path, len(records))


if __name__ == "__main__":
    main()
