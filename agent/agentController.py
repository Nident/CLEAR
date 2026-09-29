from __future__ import annotations

from typing import Any, Dict, Optional

from utils.kg_retriever import KGRetriever
from utils.llm import get_llm
from utils.dataloader.message_saver import (
    get_messages_dir_for_model_and_version,
    load_agent_messages,
    save_agent_message,
)
from tools.tools import execute_tool

from agent.collector_agent import collect_evidence
from agent.claim_agent import generate_claims
from agent.critic_agent import generate_counterclaims
from agent.judge_agent import generate_judgment, generate_judgment_from_evidence
from agent.decision import evaluate_judge_llm_output, normalize_judgments_from_llm_output
from agent import evidence_utils as ev
from agent.controller_helpers import persist_all_field_safe, run_with_retry

MAX_EMPTY_RETRIES = 3
EMPTY_RETRY_DELAY_SEC = 3


class AgentController:
    def __init__(
        self,
        kg_retriever: KGRetriever,
        model_name: str,
        storage_model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        self.kg = kg_retriever
        self.model = model_name
        self.storage_model = storage_model_name or model_name
        self.config = config
        self.debug = debug

    def _run_ablation_reuse(
        self,
        code: str,
        index: int,
        version: str,
        claims_count: int,
        task_mode: str,
        ablation: str,
        messages_dir: str,
        reuse_messages_dir: str,
        debug: bool,
    ) -> Dict[str, Any]:
        msgs = load_agent_messages(index=index, messages_dir=reuse_messages_dir)
        coll = msgs.get("collector") or {}
        if not coll and ablation != "no_evidence":
            return {"evidence": {}, "claims": [], "counterclaims": [], "judgment": {}, "judge_raw": {}}

        if ablation == "no_evidence":
            evidence = ev.ensure_evidence_struct({})
        else:
            evidence = ev.ensure_evidence_struct((coll.get("output") or {}).get("evidence"))
            if ablation == "no_precondition":
                evidence = ev.evidence_without_precondition(evidence)
            elif ablation == "no_root_cause":
                evidence = ev.evidence_without_root_cause(evidence)
            elif ablation == "no_entrypoint":
                evidence = ev.evidence_without_entrypoint(evidence)
        evidence_for_claim = ev.evidence_for_claim(evidence, task_mode)
        filtered = ev.filter_evidence(evidence_for_claim)
        evidences = ev.evidences_with_full_type(filtered.get("nodes", []))
        if ablation == "no_evidence" or ablation == "no_fix_intent":
            fix_intents = []
        else:
            orig_evidence = ev.ensure_evidence_struct((coll.get("output") or {}).get("evidence"))
            fix_intents = ev.get_all_fix_intents(orig_evidence)

        claims_out = run_with_retry(
            fn=lambda: generate_claims(
                code=code,
                evidences=evidences,
                model_name=self.model,
                config=self.config,
                debug=debug,
                claims_count=claims_count,
                task_mode=task_mode,
            ),
            is_ok=lambda out: bool((out or {}).get("claims")),
            retries=MAX_EMPTY_RETRIES,
            delay_sec=EMPTY_RETRY_DELAY_SEC,
        )
        claims = (claims_out or {}).get("claims", []) or []
        save_agent_message(
            index=index,
            agent_type="claim",
            input_data={"code": code, "evidences": evidences, "claims_count": claims_count},
            output_data=claims_out or {},
            messages_dir=messages_dir,
            run_id=None,
            parent_run_id=None,
            model_name=self.model,
            cleanup_legacy=True,
        )

        counterclaims_out = run_with_retry(
            fn=lambda: generate_counterclaims(
                code=code,
                claims=claims,
                fix_intents=fix_intents,
                model_name=self.model,
                config=self.config,
                debug=debug,
                task_mode=task_mode,
            ),
            is_ok=lambda out: (out or {}).get("counterclaims") is not None,
            retries=MAX_EMPTY_RETRIES,
            delay_sec=EMPTY_RETRY_DELAY_SEC,
        )
        counterclaims = (counterclaims_out or {}).get("counterclaims", []) or []
        save_agent_message(
            index=index,
            agent_type="critic",
            input_data={"code": code, "claims": claims, "fix_intents": fix_intents},
            output_data=counterclaims_out or {},
            messages_dir=messages_dir,
            run_id=None,
            parent_run_id=None,
            model_name=self.model,
            cleanup_legacy=True,
        )

        def _run_judge_once() -> Dict[str, Any]:
            judge_raw_local = generate_judgment(
                code=code,
                claims=claims,
                counterclaims=counterclaims,
                model_name=self.model,
                config=self.config,
                task_mode=task_mode,
            )
            judgments, overall = evaluate_judge_llm_output(judge_raw_local or {})
            judge_out_local = {"raw": judge_raw_local or {}, "judgments": judgments, "judgment": overall}
            if task_mode == "classification" and isinstance(judge_out_local.get("judgment"), dict):
                judge_out_local["judgment"] = dict(judge_out_local["judgment"])
                judge_out_local["judgment"]["final_cwe"] = (judge_raw_local or {}).get("final_cwe") or (overall or {}).get("cwe_id", "N/A")
            return judge_out_local

        judge_out = run_with_retry(
            fn=_run_judge_once,
            is_ok=lambda out: isinstance((out or {}).get("judgment"), dict) and "is_vulnerable" in (out or {}).get("judgment", {}),
            retries=MAX_EMPTY_RETRIES,
            delay_sec=EMPTY_RETRY_DELAY_SEC,
        )
        judge_raw = (judge_out or {}).get("raw", {})
        raw_judgments = normalize_judgments_from_llm_output(judge_raw or {})
        judge_message_output = {
            **(judge_out or {}),
            "judgments": [{k: v for k, v in j.items() if k not in ("is_vulnerable", "cwe")} for j in raw_judgments],
        }
        save_agent_message(
            index=index,
            agent_type="judge",
            input_data={"code": code, "claims": claims, "counterclaims": counterclaims},
            output_data=judge_message_output,
            messages_dir=messages_dir,
            run_id=None,
            parent_run_id=None,
            model_name=self.model,
            cleanup_legacy=True,
        )
        persist_all_field_safe(index=index, messages_dir=messages_dir, model_name=self.model)
        return {
            "evidence": evidence,
            "claims": claims,
            "counterclaims": counterclaims,
            "judgment": (judge_out or {}).get("judgment", {}),
            "judge_raw": judge_raw or {},
            "cwe": ((judge_out or {}).get("judgment") or {}).get("cwe"),
            "trace": [],
            "meta": {},
        }

    def _run_ablation_evidence_only(
        self,
        code: str,
        index: int,
        messages_dir: str,
        reuse_messages_dir: str,
        debug: bool,
    ) -> Dict[str, Any]:
        msgs = load_agent_messages(index=index, messages_dir=reuse_messages_dir)
        coll = msgs.get("collector") or {}
        if not coll:
            return {"evidence": {}, "claims": [], "counterclaims": [], "judgment": {}, "judge_raw": {}}

        evidence = ev.ensure_evidence_struct((coll.get("output") or {}).get("evidence"))
        evidence_for_claim = ev.evidence_for_claim(evidence, "detection")
        filtered = ev.filter_evidence(evidence_for_claim)
        evidences = ev.evidences_with_full_type(filtered.get("nodes", []))
        fix_intents = ev.get_all_fix_intents(evidence)

        def _run_judge_once() -> Dict[str, Any]:
            judge_raw_local = generate_judgment_from_evidence(
                code=code,
                evidences=evidences,
                fix_intents=fix_intents,
                model_name=self.model,
                config=self.config,
            )
            judgments, overall = evaluate_judge_llm_output(judge_raw_local or {})
            return {"raw": judge_raw_local or {}, "judgments": judgments, "judgment": overall}

        judge_out = run_with_retry(
            fn=_run_judge_once,
            is_ok=lambda out: isinstance((out or {}).get("judgment"), dict) and "is_vulnerable" in (out or {}).get("judgment", {}),
            retries=MAX_EMPTY_RETRIES,
            delay_sec=EMPTY_RETRY_DELAY_SEC,
        )
        judge_raw = (judge_out or {}).get("raw", {})
        raw_judgments = normalize_judgments_from_llm_output(judge_raw or {})
        judge_message_output = {
            **(judge_out or {}),
            "judgments": [{k: v for k, v in j.items() if k not in ("is_vulnerable", "cwe")} for j in raw_judgments],
        }
        save_agent_message(
            index=index,
            agent_type="judge",
            input_data={"code": code, "evidences": evidences, "fix_intents": fix_intents},
            output_data=judge_message_output,
            messages_dir=messages_dir,
            run_id=None,
            parent_run_id=None,
            model_name=self.model,
            cleanup_legacy=True,
        )
        persist_all_field_safe(index=index, messages_dir=messages_dir, model_name=self.model)
        return {
            "evidence": evidence,
            "claims": [],
            "counterclaims": [],
            "judgment": (judge_out or {}).get("judgment", {}),
            "judge_raw": judge_raw or {},
            "cwe": ((judge_out or {}).get("judgment") or {}).get("cwe"),
            "trace": [],
            "meta": {},
        }


    def run(
        self,
        code: str,
        index: int = 0,
        version: str = "default",
        claims_count: int = 3,
        task_mode: str = "detection",
        ablation: Optional[str] = None,
        reuse_messages_dir: Optional[str] = None,
        results_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        debug = self.debug
        messages_dir = get_messages_dir_for_model_and_version(
            self.storage_model or "default",
            version,
            task_mode=task_mode,
            ablation=ablation,
            results_base=results_base,
        )

        if ablation == "evidence_only" and reuse_messages_dir:
            return self._run_ablation_evidence_only(
                code=code,
                index=index,
                messages_dir=messages_dir,
                reuse_messages_dir=reuse_messages_dir,
                debug=debug,
            )

        if ablation and reuse_messages_dir:
            return self._run_ablation_reuse(
                code=code,
                index=index,
                version=version,
                claims_count=claims_count,
                task_mode=task_mode,
                ablation=ablation,
                messages_dir=messages_dir,
                reuse_messages_dir=reuse_messages_dir,
                debug=debug,
            )

        llm = get_llm(self.model, self.config)

        collection = run_with_retry(
            fn=lambda: collect_evidence(
                code=code,
                kg_retriever=self.kg,
                execute_tool=execute_tool,
                llm=llm,
                debug=debug,
                task_mode=task_mode,
            ),
            is_ok=lambda out: bool((out or {}).get("evidence", {}).get("nodes", [])),
            retries=MAX_EMPTY_RETRIES,
            delay_sec=EMPTY_RETRY_DELAY_SEC,
        )

        evidence = ev.ensure_evidence_struct((collection or {}).get("evidence"))
        save_agent_message(
            index=index,
            agent_type="collector",
            input_data={"code": code},
            output_data=collection or {},
            messages_dir=messages_dir,
            run_id=None,
            parent_run_id=None,
            model_name=self.model,
            cleanup_legacy=True,
        )

        evidence_for_claim = ev.evidence_for_claim(evidence, task_mode)
        filtered = ev.filter_evidence(evidence_for_claim)
        evidences = ev.evidences_with_full_type(filtered.get("nodes", []))
        claims_out = run_with_retry(
            fn=lambda: generate_claims(
                code=code,
                evidences=evidences,
                model_name=self.model,
                config=self.config,
                debug=debug,
                claims_count=claims_count,
                task_mode=task_mode,
            ),
            is_ok=lambda out: bool((out or {}).get("claims")),
            retries=MAX_EMPTY_RETRIES,
            delay_sec=EMPTY_RETRY_DELAY_SEC,
        )
        claims = (claims_out or {}).get("claims", []) or []
        save_agent_message(
            index=index,
            agent_type="claim",
            input_data={"code": code, "evidences": evidences, "claims_count": claims_count},
            output_data=claims_out or {},
            messages_dir=messages_dir,
            run_id=None,
            parent_run_id=None,
            model_name=self.model,
            cleanup_legacy=True,
        )

        fix_intents = ev.get_all_fix_intents(evidence)
        counterclaims_out = run_with_retry(
            fn=lambda: generate_counterclaims(
                code=code,
                claims=claims,
                fix_intents=fix_intents,
                model_name=self.model,
                config=self.config,
                debug=debug,
                task_mode=task_mode,
            ),
            is_ok=lambda out: (out or {}).get("counterclaims") is not None,
            retries=MAX_EMPTY_RETRIES,
            delay_sec=EMPTY_RETRY_DELAY_SEC,
        )
        counterclaims = (counterclaims_out or {}).get("counterclaims", []) or []
        save_agent_message(
            index=index,
            agent_type="critic",
            input_data={"code": code, "claims": claims, "fix_intents": fix_intents},
            output_data=counterclaims_out or {},
            messages_dir=messages_dir,
            run_id=None,
            parent_run_id=None,
            model_name=self.model,
            cleanup_legacy=True,
        )

        def _run_judge_once() -> Dict[str, Any]:
            judge_raw_local = generate_judgment(
                code=code,
                claims=claims,
                counterclaims=counterclaims,
                model_name=self.model,
                config=self.config,
                task_mode=task_mode,
            )
            judgments, overall = evaluate_judge_llm_output(judge_raw_local or {})
            judge_out_local = {"raw": judge_raw_local or {}, "judgments": judgments, "judgment": overall}
            if task_mode == "classification" and isinstance(judge_out_local.get("judgment"), dict):
                judge_out_local["judgment"]["final_cwe"] = (judge_raw_local or {}).get("final_cwe") or (overall or {}).get("cwe_id", "N/A")
            return judge_out_local

        judge_out = run_with_retry(
            fn=_run_judge_once,
            is_ok=lambda out: isinstance((out or {}).get("judgment"), dict) and "is_vulnerable" in (out or {}).get("judgment", {}),
            retries=MAX_EMPTY_RETRIES,
            delay_sec=EMPTY_RETRY_DELAY_SEC,
        )
        judge_raw = (judge_out or {}).get("raw", {})
        judgment = (judge_out or {}).get("judgment", {}) if judge_out else {}
        raw_judgments = normalize_judgments_from_llm_output(judge_raw or {})
        judge_message_output = {
            **(judge_out or {}),
            "judgments": [
                {k: v for k, v in j.items() if k not in ("is_vulnerable", "cwe")}
                for j in raw_judgments
            ],
        }
        save_agent_message(
            index=index,
            agent_type="judge",
            input_data={"code": code, "claims": claims, "counterclaims": counterclaims},
            output_data=judge_message_output,
            messages_dir=messages_dir,
            run_id=None,
            parent_run_id=None,
            model_name=self.model,
            cleanup_legacy=True,
        )

        persist_all_field_safe(index=index, messages_dir=messages_dir, model_name=self.model)

        return {
            "evidence": evidence,
            "claims": claims,
            "counterclaims": counterclaims,
            "judgment": judgment,
            "judge_raw": judge_raw or {},
            "cwe": (judgment or {}).get("cwe"),
            "trace": (collection or {}).get("trace", []),
            "meta": (collection or {}).get("meta", {}),
        }