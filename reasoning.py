from __future__ import annotations

import json
from pathlib import Path

from llm_client import LLMClient


class ReasoningStage:
    def __init__(self, client: LLMClient, prompt_dir: Path, run_dir: Path):
        self.client = client
        self.prompt_dir = prompt_dir
        self.run_dir = run_dir

    def prepare_question(self, record: dict, source_file: str = "") -> dict:
        variables = {
            "source_file": source_file,
            "record": json.dumps(record, ensure_ascii=False, indent=2),
            "task": "Extract EP, PC, RC, FI and CWE and explain their causal links.",
        }
        result = self.client.ask(self.prompt_dir / "causal_extraction.yaml", variables)
        path = self.run_dir / f"04_reasoning_{record.get('id', 'unknown')}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def prepare_all(self, records: list[dict]) -> list[dict]:
        results = [self.prepare_question(record) for record in records]
        (self.run_dir / "04_reasoning_all.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results
