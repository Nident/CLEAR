#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
INPUT="${INPUT:-$ROOT_DIR/dataset/java-pair/java_train_paired_PROJECT_LVL.json}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/project-lvl-train}"
CONFIG="${CONFIG:-$ROOT_DIR/llm_config.json}"
THREADS="${THREADS:-4}"
CWE_THREADS="${CWE_THREADS:-8}"
TOPK="${TOPK:-5}"
SAVE_EVERY="${SAVE_EVERY:-1}"
START_STEP="${START_STEP:-1}"

if [[ ! "$START_STEP" =~ ^[1-5]$ ]]; then
  echo "START_STEP must be an integer from 1 to 5" >&2
  exit 2
fi
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "PROJECT_LVL dataset not found: $INPUT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "LLM config not found: $CONFIG" >&2; exit 1; }

mkdir -p "$OUT_DIR"

KNOWLEDGE_BASE="$OUT_DIR/project_lvl_knowledge_base.json"
CWE_MAP="$OUT_DIR/cwe_map.json"
KNOWLEDGE_GRAPH="$OUT_DIR/project_lvl_knowledge_graph.json"
PRE_CWE_GRAPH="$OUT_DIR/project_lvl_final_knowledge_graph.pre_cwe.json"
FINAL_GRAPH="$OUT_DIR/project_lvl_final_knowledge_graph.json"

cd "$SCRIPT_DIR"

if (( START_STEP <= 1 )); then
  echo "[1/5] Extracting PROJECT_LVL knowledge data"
  "$PYTHON" knowledge_data.py \
    --input "$INPUT" \
    --output "$KNOWLEDGE_BASE" \
    --llm_config "$CONFIG" \
    --threads "$THREADS"
fi

if (( START_STEP <= 2 )); then
  [[ -f "$KNOWLEDGE_BASE" ]] || { echo "Missing step 1 output: $KNOWLEDGE_BASE" >&2; exit 1; }
  echo "[2/5] Building CWE relation map"
  "$PYTHON" build_cwe_map.py \
    --input "$KNOWLEDGE_BASE" \
    --output "$CWE_MAP" \
    --dataset "$INPUT" \
    --threads "$CWE_THREADS"
fi

if (( START_STEP <= 3 )); then
  [[ -f "$KNOWLEDGE_BASE" ]] || { echo "Missing step 1 output: $KNOWLEDGE_BASE" >&2; exit 1; }
  echo "[3/5] Building semantic and causal graph"
  "$PYTHON" knowledge_graph.py \
    --input "$KNOWLEDGE_BASE" \
    --output "$KNOWLEDGE_GRAPH" \
    --topk "$TOPK"
fi

if (( START_STEP <= 4 )); then
  [[ -f "$KNOWLEDGE_GRAPH" ]] || { echo "Missing step 3 output: $KNOWLEDGE_GRAPH" >&2; exit 1; }
  echo "[4/5] Building cross-CVE bridges"
  final_graph_cmd=(
    "$PYTHON" final_knowledge_graph.py
    --input "$KNOWLEDGE_GRAPH"
    --output "$PRE_CWE_GRAPH"
    --llm_config "$CONFIG"
    --threads "$THREADS"
    --save_every "$SAVE_EVERY"
  )
  if [[ -f "$PRE_CWE_GRAPH" ]]; then
    final_graph_cmd+=(--resume)
  fi
  "${final_graph_cmd[@]}"
fi

if (( START_STEP <= 5 )); then
  [[ -f "$PRE_CWE_GRAPH" ]] || { echo "Missing step 4 output: $PRE_CWE_GRAPH" >&2; exit 1; }
  [[ -f "$KNOWLEDGE_GRAPH" ]] || { echo "Missing step 3 output: $KNOWLEDGE_GRAPH" >&2; exit 1; }
  echo "[5/5] Adding CWE edges"
  "$PYTHON" add_cwe_edges.py \
    --input "$PRE_CWE_GRAPH" \
    --pre-input "$KNOWLEDGE_GRAPH" \
    --output "$FINAL_GRAPH"
fi

echo "Done: $FINAL_GRAPH"
