from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from graph_stage import GraphStage
from llm_client import LLMClient
from reasoning import ReasoningStage


def main() -> None:
    root = Path(__file__).parent
    load_dotenv(root / "config" / ".env")
    run_dir = root / os.getenv("VCKG_RUNS_DIR", "outputs/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = root / os.getenv("VCKG_INPUT", "data/sampled.zip")

    stage = GraphStage(dataset, run_dir)
    records = stage.read_dataset()
    stage.clone_projects(records)
    prepared = stage.prepare_records(records)

    client = LLMClient(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.8-27b:free"),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
    )
    ReasoningStage(client, root / "prompt", run_dir).prepare_all(prepared)
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
