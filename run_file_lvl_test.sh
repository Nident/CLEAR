#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
INPUT=""
GRAPH=""
CONFIG="$SCRIPT_DIR/llm_config.json"
MODEL=""
TASK="detection"
RESULTS_BASE="results2"
RESUME=1

usage() {
  cat <<'EOF'
Usage:
  ./run_file_lvl_test.sh --input CLEAR_TEST.json --graph FINAL_GRAPH.json [options]

Required:
  --input PATH          Test dataset converted to CLEAR format
  --graph PATH          Existing final knowledge graph

Options:
  --config PATH         LLM config (default: ./llm_config.json)
  --model NAME          Model from the LLM config (default: first configured)
  --task MODE           detection or classification (default: detection)
  --results-base NAME   Result run tag: results, results2, ... (default: results2)
  --python PATH         Python executable (default: ./.venv/bin/python)
  --restart             Reprocess every index instead of resuming
  -h, --help            Show this help

This script only runs testing. It never builds or modifies the knowledge graph.
EOF
}

while (($#)); do
  case "$1" in
    --input) INPUT="${2:?Missing value for --input}"; shift 2 ;;
    --graph) GRAPH="${2:?Missing value for --graph}"; shift 2 ;;
    --config) CONFIG="${2:?Missing value for --config}"; shift 2 ;;
    --model) MODEL="${2:?Missing value for --model}"; shift 2 ;;
    --task) TASK="${2:?Missing value for --task}"; shift 2 ;;
    --results-base) RESULTS_BASE="${2:?Missing value for --results-base}"; shift 2 ;;
    --python) PYTHON="${2:?Missing value for --python}"; shift 2 ;;
    --restart) RESUME=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$INPUT" ]] || { echo "--input is required" >&2; usage >&2; exit 2; }
[[ -n "$GRAPH" ]] || { echo "--graph is required" >&2; usage >&2; exit 2; }
[[ "$TASK" == "detection" || "$TASK" == "classification" ]] || {
  echo "--task must be detection or classification" >&2
  exit 2
}
[[ "$RESULTS_BASE" =~ ^results([2-9][0-9]*)?$ ]] || {
  echo "--results-base must be results, results2, results3, ..." >&2
  exit 2
}
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "CLEAR test dataset not found: $INPUT" >&2; exit 1; }
[[ -f "$GRAPH" ]] || { echo "Final graph not found: $GRAPH" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "LLM config not found: $CONFIG" >&2; exit 1; }

cmd=(
  "$PYTHON" "$SCRIPT_DIR/run.py"
  --version java-pair
  --task "$TASK"
  --input "$INPUT"
  --knowledge-graph "$GRAPH"
  --config "$CONFIG"
  --results-base "$RESULTS_BASE"
)
if [[ -n "$MODEL" ]]; then
  cmd+=(--model "$MODEL")
fi
if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi

printf 'Starting CLEAR test with %s\n' "$INPUT"
printf 'Using existing graph: %s\n' "$GRAPH"
exec "${cmd[@]}"
