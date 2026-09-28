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
