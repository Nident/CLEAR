# CLEAR

## Setup

```bash
conda create -n CLEAR python=3.12 -y
conda activate CLEAR
pip install -r requirements.txt
```


## Set API key in `llm_config.json`.(Project Root)
```json
{
  "models": [
    {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "api_key": "<OPENAI_API_KEY>"
    },
    {
      "provider": "openai",
      "model": "gpt-4.1-nano",
      "api_key": "<OPENAI_API_KEY>"
    },
    {
      "provider": "google",
      "model": "gemini-2.5-flash",
      "api_key": "<GEMINI_API_KEY>"
    },
    {
      "provider": "openai",
      "model": "gpt-5-mini",
      "api_key": "<OPENAI_API_KEY>"
    },
    {
      "provider": "google",
      "model": "gemini-3.1-flash-lite",
      "api_key": "<GEMINI_API_KEY>"
    }
  ]
}

```

## Vulnerability Causal Knowledge Graph
```bash
cd knowledge_graph && chmod +x scripts.sh
./scripts.sh
```

## VCKG MCP server

The read-only MCP adapter exposes the existing `KGRetriever` to an external
agent. It loads the graph once and provides bounded semantic retrieval and
causal traversal tools; it does not call an LLM itself.

```bash
./vckg_mcp_ctl.sh start
./vckg_mcp_ctl.sh status
./vckg_mcp_ctl.sh logs
./vckg_mcp_ctl.sh stop
```

Defaults:

- MCP endpoint: `http://127.0.0.1:4250/mcp`
- health endpoint: `http://127.0.0.1:4250/health`
- graph: `knowledge_graph/project-lvl-train/project_lvl_final_knowledge_graph.json`

Select another graph before starting the server:

```bash
export VCKG_PATH="$PWD/knowledge_graph/file-lvl-train/file_lvl_final_knowledge_graph.json"
./vckg_mcp_ctl.sh restart
```

Exposed tools: `vckg_seed_retrieve`, `vckg_get_bridges`,
`vckg_expand_nodes`, and `vckg_get_node`.

## Start
```bash
python run.py --version primevul --task detection --model gpt-4o-mini
python run.py --version java-pair --task detection --model gpt-4o-mini
python run.py --version primevul --task classification --model gpt-4o-mini
python run.py --version java-pair --task classification --model gpt-4o-mini

# ablation
python run.py --version primevul --ablation no_evidence --model gpt-4o-mini
python run.py --version primevul --ablation no_entrypoint --model gpt-4o-mini
python run.py --version primevul --ablation no_precondition --model gpt-4o-mini
python run.py --version primevul --ablation no_root_cause --model gpt-4o-mini
python run.py --version primevul --ablation no_fix_intent --model gpt-4o-mini
```

## Baseline
```bash
chmod +x ./baseline/baseline.sh
./baseline/baseline.sh
```

## Evaluation

```bash
python evaluate.py
python evaluate.py --table
```
