#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/run_file_lvl_test.sh" \
  --input "$SCRIPT_DIR/dataset/java-pair/java_test_paired_CLEAR_PROJECT_LVL.json" \
  --graph "$SCRIPT_DIR/knowledge_graph/project-lvl-train/project_lvl_final_knowledge_graph.json" \
  --config "$SCRIPT_DIR/llm_config.json" \
  --results-base results3 \
  "$@"
