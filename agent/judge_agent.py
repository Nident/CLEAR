from __future__ import annotations
from typing import Any, Dict, List, Optional
import json

from utils.llm import get_llm
from agent.llm_agent_utils import extract_json_obj

SYSTEM_PROMPT_JUDGE = """
You are a Judge Agent that evaluates vulnerability causal chains by strictly relying on explicit code-level evidence.
Your role is to EVALUATE EACH claim (by claim_id) and decide whether it remains valid after considering counterclaims.

Return JSON only in the following format:
{
  "judgments": [
    {
      "claim_id": "<claim_id from input>",
      "cwe_id": "<CWE-XXX or N/A>",
      "precondition": "<precondition>",
      "precondition_status": "SUPPORTED | POTENTIALLY_SUPPORTED | CONTRADICTED",
      "precondition_reasoning": "<reasoning for the precondition status>",
      "root_cause": "<root_cause>",
      "root_cause_status": "PRESENT | POTENTIALLY_PRESENT | INVALIDATED",
      "root_cause_reasoning": "<reasoning for the root cause status>"
    }
  ]
}
""".strip()

SYSTEM_PROMPT_JUDGE_CLASSIFICATION = """
You are a Judge Agent that evaluates vulnerability arguments by strictly relying on explicit code-level evidence.
Your role is to EVALUATE EACH claims and decide whether it remains valid after considering counterclaims.

Return JSON only in the following format:
{
  "judgments": [
    {
      "claim_id": "<claim_id from input>",
      "cwe_id": "<CWE-XXX>",
      "preconditions": "<precondition>",
      "status": "SUPPORTED | POTENTIALLY_SUPPORTED | CONTRADICTED",
      "precondition_reasoning": "<reasoning for the precondition status>",
      "root_cause": "<root_cause>",
      "root_cause_status": "PRESENT | POTENTIALLY_PRESENT | INVALIDATED",
      "root_cause_reasoning": "<reasoning for the root cause status>"
    }
  ]
}
""".strip()

SYSTEM_PROMPT_JUDGE_EVIDENCE_ONLY = """
You are a Judge Agent that evaluates whether code is vulnerable by strictly relying on explicit code-level evidence and the provided knowledge-graph evidences.
Your role is to assess the most plausible vulnerability causal chain grounded in the code and the given evidences (Entrypoint, Vulnerability_Precondition, Vulnerability_Root_Cause, Fix_Intent).

Return JSON only in the following format:
{
  "judgments": [
    {
      "claim_id": "E1",
      "cwe_id": "<CWE-XXX or N/A>",
      "precondition": "<vulnerability precondition>",
      "precondition_status": "SUPPORTED | POTENTIALLY_SUPPORTED | CONTRADICTED",
      "precondition_reasoning": "<reasoning for the precondition status>",
      "root_cause": "<vulnerability root cause>",
      "root_cause_status": "PRESENT | POTENTIALLY_PRESENT | INVALIDATED",
      "root_cause_reasoning": "<reasoning for the root cause status>"
    }
  ]
}
""".strip()


def generate_judgment(
    code: str,
    claims: List[Dict[str, Any]],
    counterclaims: List[Dict[str, Any]],
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    task_mode: str = "detection",
) -> Dict[str, Any]:
    llm = get_llm(model_name, config)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
    except Exception as e:
        raise ImportError(
            "langchain_core is required for LLM agent messages. "
            "If you see pydantic 'Discriminator' import errors, install pydantic>=2 and compatible langchain_core."
        ) from e

    payload = {
        "code": code,
        "claims": claims,
        "counterclaims": counterclaims,
    }

    use_classification = task_mode == "classification"
    system_prompt = SYSTEM_PROMPT_JUDGE_CLASSIFICATION if use_classification else SYSTEM_PROMPT_JUDGE
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        obj = extract_json_obj(raw)
    except Exception:
        return {"judgments": []}

    judgments: List[Dict[str, Any]] = []
    if isinstance(obj.get("judgments"), list):
        judgments = [x for x in obj.get("judgments") if isinstance(x, dict)]
    elif isinstance(obj.get("judgment"), dict):
        judgments = [obj["judgment"]]
    out: Dict[str, Any] = {"judgments": judgments}
    if use_classification:
        final_cwe = obj.get("final_cwe") if isinstance(obj.get("final_cwe"), str) and str(obj.get("final_cwe")).strip() else None
        if not final_cwe and judgments:
            for j in judgments:
                if isinstance(j, dict) and str(j.get("root_cause_status", "")).strip() in ("PRESENT", "POTENTIALLY_PRESENT"):
                    cid = j.get("cwe_id") or j.get("cwe")
                    if isinstance(cid, str) and cid.strip():
                        final_cwe = cid.strip()
                        break
        out["final_cwe"] = final_cwe if final_cwe else "N/A"
    return out

def generate_judgment_from_evidence(
    code: str,
    evidences: List[Dict[str, Any]],
    fix_intents: List[Dict[str, Any]],
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    llm = get_llm(model_name, config)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
    except Exception as e:
        raise ImportError(
            "langchain_core is required for LLM agent messages. "
            "If you see pydantic 'Discriminator' import errors, install pydantic>=2 and compatible langchain_core."
        ) from e

    payload = {
        "code": code,
        "evidences": evidences,
        "fix_intents": fix_intents,
    }
    messages = [
        SystemMessage(content=SYSTEM_PROMPT_JUDGE_EVIDENCE_ONLY),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        obj = extract_json_obj(raw)
    except Exception:
        return {"judgments": []}

    judgments: List[Dict[str, Any]] = []
    if isinstance(obj.get("judgments"), list):
        judgments = [x for x in obj.get("judgments") if isinstance(x, dict)]
    elif isinstance(obj.get("judgment"), dict):
        judgments = [obj["judgment"]]
    if judgments and not judgments[0].get("claim_id"):
        judgments[0]["claim_id"] = "E1"
    return {"judgments": judgments}

