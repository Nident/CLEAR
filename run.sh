# RQ1. Detection / Classification
```bash
python run.py --version primevul --task detection --mod el gpt-4o-mini
python run.py --version java-pair --task detection --model gpt-4o-mini
python run.py --version java-pair --task detection --model gpt-4o-mini --results-base results2
```


# RQ2. Classification
```bash
python run.py --version primevul --task classification --model gpt-4o-mini
python run.py --version java-pair --task classification --model gpt-4o-mini
```

# RQ4. Ablation Study (run.py default task = detection)
```bash
python run.py --version java-pair --ablation no_evidence --model gpt-4o-mini
python run.py --version java-pair --ablation no_entrypoint --model gpt-4o-mini
python run.py --version java-pair --ablation no_precondition --model gpt-4o-mini
python run.py --version java-pair --ablation no_root_cause --model gpt-4o-mini
python run.py --version java-pair --ablation no_fix_intent --model gpt-4o-mini
```


```bash
python run.py --version primevul --ablation no_evidence --model gpt-4o-mini
python run.py --version primevul --ablation no_entrypoint --model gpt-4o-mini
python run.py --version primevul --ablation no_precondition --model gpt-4o-mini
python run.py --version primevul --ablation no_root_cause --model gpt-4o-mini
python run.py --version primevul --ablation no_fix_intent --model gpt-4o-mini
```

# Other Models
python run.py --version primevul --task detection --model gpt-4o-mini
python run.py --version primevul --task detection --model gemini-2.5-flash

# Evaluation
```bash
python evaluate.py
python evaluate.py --table
```