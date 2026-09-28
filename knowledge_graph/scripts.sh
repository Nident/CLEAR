#!/bin/bash
python knowledge_data.py --input ../dataset/java-pair/java_data_paired.json --output ./java-pair/java_knowledge_base.json --llm_config ../llm_config.json --threads 16
python build_cwe_map.py --java-pair
python knowledge_graph.py --input ./java-pair/java_knowledge_base.json --output ./java-pair/java_knowledge_graph.json --topk 5
python final_knowledge_graph.py --input ./java-pair/java_knowledge_graph.json --output ./java-pair/java_final_knowledge_graph.json --threads 32
python add_cwe_edges.py --java-pair

python knowledge_data.py  --input ../dataset/primevul/primevul_knowledge_data_pair.json --output ./primevul/primevul_knowledge_base.json --llm_config ../llm_config.json --threads 16
python build_cwe_map.py --primevul
python knowledge_graph.py --input ./primevul/primevul_knowledge_base.json --output ./primevul/primevul_knowledge_graph.json --topk 5
python final_knowledge_graph.py --input ./primevul/primevul_knowledge_graph.json --output ./primevul/primevul_final_knowledge_graph.json --threads 16
python add_cwe_edges.py --primevul
