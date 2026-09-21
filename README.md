# CLEAR-style VCKG construction

`build_vckg.py` reads either a directory of JSON files or the supplied
`data/sampled.zip` / `data/examples.zip` archives and writes a GraphML VCKG.
Runtime parameters are stored in `config/.env` and loaded with `os.getenv()`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fast, fully local baseline
python3 build_vckg.py

# Embeddings used in CLEAR (downloads all-MiniLM-L6-v2 on first run)
pip install sentence-transformers
# Set VCKG_EMBEDDING=minilm in config/.env first, then run:
python3 build_vckg.py
```

The graph contains five node roles per vulnerability instance: `EP`, `PC`,
`RC`, `FI`, and `CWE`. Local edges form `EP -> PC -> RC -> FI` and connect
causal nodes to CWE. Neighborhood edges are restricted to the same role and
CWE cluster, with Top-k filtering (`--top-k 5`). A JSON summary is written next
to the GraphML file.

The deterministic extractor is intentionally offline and reproducible. It uses
the available advisory, patch, and before/after fields. For an exact CLEAR
replication, replace `heuristic()` with an LLM structured-output extractor for
the five causal entities; the graph integration and export stages remain the
same.
# CLEAR
