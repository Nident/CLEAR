from __future__ import annotations

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class KGRetriever:
    def __init__(self, kg_path: str):
        with open(kg_path, 'r', encoding='utf-8') as f:
            self.kg_data = json.load(f)
        
        if "data" in self.kg_data:
            all_nodes = []
            all_edges = []
            for instance in self.kg_data.get("data", []):
                instance_nodes = instance.get("nodes", [])
                instance_edges = instance.get("edges", [])
                all_nodes.extend(instance_nodes)
                all_edges.extend(instance_edges)
            self.nodes = all_nodes
            self.edges = all_edges
        else:
            self.nodes = self.kg_data.get("nodes", [])
            self.edges = self.kg_data.get("edges", [])
        
        self.node_by_id = {node["id"]: node for node in self.nodes}
        self.nodes_by_type = {}
        for node in self.nodes:
            node_type = node.get("type")
            if node_type:
                if node_type not in self.nodes_by_type:
                    self.nodes_by_type[node_type] = []
                self.nodes_by_type[node_type].append(node)
    
    def search_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        return self.nodes_by_type.get(node_type, [])
    
    def search_nodes_by_category(self, node_type: str, category: str) -> List[Dict[str, Any]]:
        nodes = self.search_nodes_by_type(node_type)
        return [n for n in nodes if n.get("category", "").lower() == category.lower()]
    
    def search_nodes_by_keyword(self, keyword: str, node_type: Optional[str] = None) -> List[Dict[str, Any]]:
        keyword_lower = keyword.lower()
        nodes_to_search = self.nodes
        if node_type:
            nodes_to_search = self.search_nodes_by_type(node_type)
        
        results = []
        for node in nodes_to_search:
            summary = node.get("summary", "").lower()
            category = node.get("category", "").lower()
            details = " ".join(node.get("details", [])).lower()
            
            if keyword_lower in summary or keyword_lower in category or keyword_lower in details:
                results.append(node)
        
        return results
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.node_by_id.get(node_id)

    def get_out_edges(self, node_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.edges if e.get("src") == node_id]

    def get_related_nodes(self, node_id: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        related_node_ids = set()
        
        for edge in self.edges:
            src = edge.get("src")
            dst = edge.get("dst")
            rel = edge.get("rel")
            
            if src == node_id:
                if relation_type is None or rel == relation_type:
                    related_node_ids.add(dst)
            elif dst == node_id:
                if relation_type is None or rel == relation_type:
                    related_node_ids.add(src)
        
        return [self.node_by_id[nid] for nid in related_node_ids if nid in self.node_by_id]
    
    def search_by_pattern(self, pattern_type: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        nodes = self.search_nodes_by_type(pattern_type)
        if category:
            nodes = [n for n in nodes if n.get("category", "").lower() == category.lower()]
        return nodes
    
    def get_cwe_for_root_cause(self, root_cause_category: str) -> List[Dict[str, Any]]:
        root_cause_nodes = self.search_nodes_by_category("RootCausePattern", root_cause_category)
        
        cwe_nodes = []
        for rc_node in root_cause_nodes:
            related = self.get_related_nodes(rc_node["id"], "IMPLIES_CWE")
            cwe_nodes.extend(related)
        
        return cwe_nodes


def load_kg(kg_path: str) -> KGRetriever:
    return KGRetriever(kg_path)
