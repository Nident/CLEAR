
import json
from pathlib import Path


def load_test_data(data_path: str = "../dataset/test.json"):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    if path.suffix == ".jsonl":
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        data = items
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            data = data["data"]

    if not isinstance(data, list):
        raise ValueError(f"Unsupported input format (expected list): {path}")

    return [item.get("func", "") for item in data if isinstance(item, dict)]