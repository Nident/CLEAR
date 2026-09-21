from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_openai import ChatOpenAI


class LLMClient:
    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0.0):
        self.model = model
        self.llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url,
                              temperature=temperature)

    def load_prompt(self, path: Path) -> dict:
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def ask(self, prompt_path: Path, variables: dict) -> dict:
        prompt = self.load_prompt(prompt_path)
        system = prompt.get("system", "")
        user = prompt.get("user", "").format(**variables)
        response = self.llm.invoke([("system", system), ("human", user)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_response": content}
