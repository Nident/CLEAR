#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
INPUT="$ROOT_DIR/dataset/java-pair/java_data_paired_examples_FILE_LVL.json"
OUT_DIR="$SCRIPT_DIR/file-lvl-50"
CONFIG="$ROOT_DIR/llm_config.json"

mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"

echo "[1/5] Extracting knowledge data"
"$PYTHON" knowledge_data.py --input "$INPUT" --output "$OUT_DIR/file_lvl_knowledge_base.json" --llm_config "$CONFIG" --threads 4 --limit 50

echo "[2/5] Building CWE relation map"
"$PYTHON" build_cwe_map.py --input "$OUT_DIR/file_lvl_knowledge_base.json" --output "$OUT_DIR/cwe_map.json" --dataset "$INPUT" --threads 8

echo "[3/5] Building semantic and causal graph"
"$PYTHON" knowledge_graph.py --input "$OUT_DIR/file_lvl_knowledge_base.json" --output "$OUT_DIR/file_lvl_knowledge_graph.json" --topk 5

echo "[4/5] Building cross-CVE bridges"
"$PYTHON" final_knowledge_graph.py --input "$OUT_DIR/file_lvl_knowledge_graph.json" --output "$OUT_DIR/file_lvl_final_knowledge_graph.pre_cwe.json" --llm_config "$CONFIG" --threads 4 --save_every 1 --resume

echo "[5/5] Adding CWE edges"
"$PYTHON" add_cwe_edges.py --input "$OUT_DIR/file_lvl_final_knowledge_graph.pre_cwe.json" --pre-input "$OUT_DIR/file_lvl_knowledge_graph.json" --output "$OUT_DIR/file_lvl_final_knowledge_graph.json"

echo "Done: $OUT_DIR/file_lvl_final_knowledge_graph.json"
