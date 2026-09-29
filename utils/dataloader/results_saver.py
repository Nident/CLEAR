import json
from pathlib import Path
from typing import Dict, Any, Set


def get_existing_indices(output_dir: str) -> Set[int]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return set()
    
    indices = set()
    for file in output_path.glob("*.json"):
        try:
            index = int(file.stem)
            indices.add(index)
        except ValueError:
            continue
    return indices


def save_result(output_dir: str, index: int, result: Dict[str, Any]):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / f"{index}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
