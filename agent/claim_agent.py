from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

from utils.llm import get_llm
from agent.llm_agent_utils import extract_json_obj
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT_CLAIM = """
You are a Claim Agent that constructs plausible vulnerability claims grounded in provided evidences.
Your role is to propose PLAUSIBLE vulnerability causal chains by interpreting the given evidences in the MOST SUSPICIOUS way that is still grounded in the code.

For EACH claim, you MUST explicitly construct a causal structure with:
1. Entrypoint: The code's entrypoint that code inputs are passed to
2. Preconditions: The vulnerability precondition that must be met for the vulnerability to occur
3. Root Cause: The vulnerability root cause that is the core of the vulnerability
4. References: The references that support the claim. describe the evidence that supports the claim.
You MUST produce exactly {claims_count} distinct claims, each with a unique claim_id (C1, C2, C3, ...).

Return JSON only in the following format:
{{
  "claims": [
    {{
      "claim_id": "C1",
      "claim": "<concise vulnerability hypothesis>",
      "cwe_id": "<CWE-XXX>",
      "entrypoint": "<how attacker-controlled data could enter>",
      "preconditions": "<vulnerability precondition>",
      "root_cause": "<vulnerability root cause>",
      "references": [
        "<EP_REFERENCE>",
        "<PRE_REFERENCE>",
        "<RC_REFERENCE>"
      ]
    }}
  ]
}}
""".strip()

SYSTEM_PROMPT_CLAIM_CLASSIFICATION = """
You are a Claim Agent that constructs plausible vulnerability claims grounded in provided evidences.
Your role is to propose PLAUSIBLE vulnerability causal chains by interpreting the given evidences in the MOST SUSPICIOUS way that is still grounded in the code.

For EACH claim, you MUST explicitly construct a causal structure with:
1. Entrypoint: The code's entrypoint that code inputs are passed to
2. Preconditions: The vulnerability precondition that must be met for the vulnerability to occur
3. Root Cause: The vulnerability root cause that is the core of the vulnerability
4. CWE: The most appropriate CWE-ID that best matches the vulnerability type

You MUST produce exactly {claims_count} distinct claims, each with a unique claim_id (C1, C2, C3, ...).

Return JSON only in the following format:
{{
  "claims": [
    {{
      "claim_id": "C1",
      "claim": "<concise vulnerability hypothesis>",
      "cwe_id": "<CWE-XXX>",
      "entrypoint": "<how attacker-controlled data could enter>",
      "preconditions": "<vulnerability precondition>",
      "root_cause": "<vulnerability root cause>",
      "references": [
        "<reference>"
      ]
    }}
  ]
}}
""".strip()


def generate_claims(
    code: str,
    evidences: List[Dict[str, Any]],
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    claims_count: int = 3,
    task_mode: str = "detection",
) -> Dict[str, Any]:
    llm = get_llm(model_name, config)


    payload: Dict[str, Any] = {"code": code, "evidences": evidences, "claims_count": claims_count}

    use_classification = task_mode == "classification"
    system_content = (SYSTEM_PROMPT_CLAIM_CLASSIFICATION if use_classification else SYSTEM_PROMPT_CLAIM).format(claims_count=claims_count)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    if debug:
        print("[Claim] request:", messages)

    resp = llm.invoke(messages)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    if debug:
        print("[Claim] response:", raw)

    obj = extract_json_obj(raw)
    claims = obj.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    for i, c in enumerate(claims):
        if isinstance(c, dict):
            c["claim_id"] = f"C{i + 1}"
    return {"claims": claims}

