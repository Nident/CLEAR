from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

from utils.llm import get_llm
from agent.llm_agent_utils import extract_json_obj
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT_CRITIC = """
You are a Critic Agent that refutes vulnerability claims by strictly relying on explicit code-level evidence.   
your role is to rigorously refute the Claim vulnerability hypothesis by using evidence and logical reasoning, in order to persuasively demonstrate to the Judge Agent that the code is secure.
You MUST refer to the given code and evidences to construct the counterclaim.
reference must be the most relevant evidence from the privided evidences.
Return JSON only in the following format:
{
  "counterclaims": [
    {
      "claim_id": "<claim_id>",
      "counterclaim": "<concise refutation summary>",
      "reference": "<FIX_REFERENCE>"
    }
  ]
}
""".strip()

SYSTEM_PROMPT_CRITIC_CLASSIFICATION = """
You are a Critic Agent that refutes vulnerability claims by strictly relying on explicit code-level evidence.   
your role is to rigorously refute the Claim vulnerability hypothesis by using evidence and logical reasoning, in order to persuasively demonstrate to the Judge Agent that the code is secure.
You MUST refer to the given code and evidences to construct the counterclaim.
reference must be the most relevant evidence from the privided evidences.
Return JSON only in the following format:
{
  "counterclaims": [
    {
      "claim_id": "<claim_id>",
      "counterclaim": "<concise refutation summary>",
    }
  ]
}
""".strip()


def generate_counterclaims(
    code: str,
    claims: List[Dict[str, Any]],
    fix_intents: List[Dict[str, Any]],
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    task_mode: str = "detection",
) -> Dict[str, Any]:
    llm = get_llm(model_name, config)

    payload: Dict[str, Any] = {"code": code, "claims": claims, "fix_intents": fix_intents}
    use_classification = task_mode == "classification"
    system_prompt = SYSTEM_PROMPT_CRITIC_CLASSIFICATION if use_classification else SYSTEM_PROMPT_CRITIC
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    if debug:
        print("[Critic] request:", messages)

    resp = llm.invoke(messages)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    if debug:
        print("[Critic] response:", raw)

    obj = extract_json_obj(raw)

    if use_classification:
        opinions = obj.get("opinions", [])
        if not isinstance(opinions, list):
            opinions = []
        counterclaims = []
        for op in opinions:
            if not isinstance(op, dict):
                continue
            cid = op.get("claim_id", "")
            counterclaims.append({
                "claim_id": cid,
                "target_claim_id": cid,
                "counterclaim": op.get("opinion", ""),
                "reasoning": op.get("reasoning", ""),
            })
    else:
        counterclaims = obj.get("counterclaims", [])
        if not isinstance(counterclaims, list):
            counterclaims = []

    return {"counterclaims": counterclaims}

