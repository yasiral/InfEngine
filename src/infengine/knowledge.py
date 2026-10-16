"""Deterministic operator knowledge for ``infengine explain``.

Static, repository-derived metadata — NOT an LLM call. It describes what each
control does, its native engine mapping, transition class, verification
mechanism, conditions where it is inert, known limitations and the applicable
historical M12 evidence label. Runtime values (current/bounds/supported) are
merged in by the CLI from the live capability contract.
"""

from typing import Any, Dict, Optional

# Evidence is explicitly HISTORICAL and never a recommendation for the current
# deployment (M13 product principle).
EVIDENCE_DISCLAIMER = (
    "Historical experimental evidence only (M12/M12.1). It is NOT a "
    "recommendation for the current deployment; evaluate your own workload."
)

CONTROL_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "runtime_concurrency": {
        "summary": "Upper bound on concurrently running requests (admission gate).",
        "native": {
            "vllm": "EngineCore scheduler.max_num_running_reqs",
            "sglang": "pp_max_micro_batch_size (per DP rank)",
        },
        "semantic": "CONCURRENCY_UPPER_BOUND",
        "transition": "HOT",
        "controls": (
            "How many requests the scheduler admits to run at once. It is a cap, "
            "not a guarantee of exactly N concurrent requests."
        ),
        "operational_effects": [
            "Higher values can raise throughput and queueing pressure.",
            "Lower values can reduce per-request latency and KV pressure.",
        ],
        "inert_when": [
            "Offered load is below the bound (the bound is not the constraint).",
            "KV capacity, not concurrency, is the limiting factor.",
        ],
        "verification": "Engine readback of the native concurrency field equals the requested value.",
        "limitations": [
            "Bounded by the provisioned (boot-time) max_num_seqs / max_running_requests.",
            "SGLang scope is per DP rank; it is not a cluster-wide guarantee.",
        ],
        "engine_differences": (
            "vLLM bounds the scheduler running set; SGLang bounds pp_max_micro_batch_size "
            "and admission is further limited by available request-token pool slots."
        ),
        "evidence": {"available": True, "label": EVIDENCE_DISCLAIMER},
    },
    "runtime_max_scheduled_tokens": {
        "summary": "Strict per-step total scheduled-token budget (vLLM only).",
        "native": {"vllm": "EngineCore scheduler.max_num_scheduled_tokens"},
        "semantic": "STRICT_SCHEDULED_TOKEN_BUDGET",
        "transition": "HOT",
        "controls": "The total number of tokens the scheduler may schedule in one engine step.",
        "operational_effects": [
            "Lower values cap batch work per step; may change TTFT and throughput.",
            "Higher values may raise throughput but increase per-step latency.",
        ],
        "inert_when": ["The workload never approaches the per-step token budget."],
        "verification": "Engine readback of the live scheduler token budget equals the requested value.",
        "limitations": [
            "vLLM only; NOT the SGLang prefill-admission budget and never translated to it.",
            "Bounded by the provisioned boot-time token ceiling.",
        ],
        "engine_differences": (
            "No SGLang equivalent. A strict total scheduled-token budget against SGLang "
            "is rejected rather than silently weakened."
        ),
        "evidence": {
            "available": True,
            "label": (
                "Historical M12/M12.1 A100 evidence showed large, reproducible, "
                "workload-dependent latency/throughput/admission effects. " + EVIDENCE_DISCLAIMER
            ),
        },
    },
    "long_prefill_token_threshold": {
        "summary": "Max prefill tokens scheduled for a single long prompt per step (vLLM only).",
        "native": {"vllm": "EngineCore scheduler_config.long_prefill_token_threshold"},
        "semantic": "LONG_PREFILL_TOKEN_THRESHOLD",
        "transition": "HOT",
        "controls": "How a single long prompt's prefill is chunked within one scheduler step.",
        "operational_effects": [
            "Smaller values chunk long prefills more; can reduce head-of-line blocking.",
            "Larger values prefills long prompts in fewer steps.",
        ],
        "inert_when": ["Prompts are short relative to the threshold."],
        "verification": "Engine readback of the threshold equals the requested value.",
        "limitations": ["vLLM only."],
        "engine_differences": "No SGLang equivalent (SGLang uses runtime_max_prefill_tokens).",
        "evidence": {"available": False, "label": "No M12 headline evidence; validate on your workload."},
    },
    "runtime_kv_watermark": {
        "summary": "KV admission headroom in [0.0, 1.0) reserved for waiting/preempted requests (vLLM only).",
        "native": {"vllm": "EngineCore scheduler.kv_cache_manager.watermark_blocks"},
        "semantic": "KV_ADMISSION_WATERMARK",
        "transition": "HOT",
        "controls": (
            "Fraction of GPU KV blocks reserved as headroom before admitting new work. "
            "Physical KV capacity is unchanged."
        ),
        "operational_effects": [
            "Higher watermark reserves more headroom; may reduce preemptions.",
            "Too high can reduce effective capacity and throughput.",
        ],
        "inert_when": ["KV usage never approaches the watermark."],
        "verification": "Engine readback of the watermark equals the requested value (< 1.0).",
        "limitations": ["vLLM only; must be < 1.0."],
        "engine_differences": "No SGLang equivalent.",
        "evidence": {"available": True, "label": "Historical M7D2 policy evidence. " + EVIDENCE_DISCLAIMER},
    },
    "runtime_reserve_full_isl": {
        "summary": "Admit a request only when its full input sequence fits KV (vLLM only).",
        "native": {"vllm": "EngineCore scheduler.scheduler_reserve_full_isl"},
        "semantic": "RESERVE_FULL_ISL",
        "transition": "HOT",
        "controls": (
            "When True, a request is admitted only if its FULL input fits the KV cache; "
            "when False, only the first chunk must fit (chunked prefill)."
        ),
        "operational_effects": [
            "True can reject/queue long requests under KV pressure.",
            "False enables chunked prefill of long prompts.",
        ],
        "inert_when": ["All inputs are short enough to fit KV regardless."],
        "verification": "Engine readback of the boolean equals the requested value.",
        "limitations": ["vLLM only; affects future admissions only."],
        "engine_differences": "No SGLang equivalent.",
        "evidence": {"available": True, "label": "Historical M7D3 actuator evidence. " + EVIDENCE_DISCLAIMER},
    },
    "runtime_max_prefill_tokens": {
        "summary": "Aggregate prefill input-token admission budget (SGLang only).",
        "native": {"sglang": "Scheduler runtime_max_prefill_tokens (patched, scheduler-local)"},
        "semantic": "AGGREGATE_PREFILL_INPUT_ADMISSION_BUDGET_FIRST_REQUEST_UNBOUNDED",
        "transition": "HOT",
        "controls": (
            "The aggregate prefill input-token budget per scheduling pass. The first "
            "request of a pass is unbounded by it."
        ),
        "operational_effects": [
            "Lower values limit aggregate prefill work per pass.",
            "Can change TTFT and queue behavior under prefill-heavy load.",
        ],
        "inert_when": ["A single request dominates each scheduling pass."],
        "verification": "Scheduler readback (via /server_info internal_states) equals the requested value.",
        "limitations": [
            "SGLang only; NOT a strict per-request cap, NOT a total scheduler-token "
            "budget, NOT vLLM's runtime_max_scheduled_tokens.",
            "Bounded by the boot provisioned max_prefill_tokens.",
        ],
        "engine_differences": "No vLLM equivalent; never equated with runtime_max_scheduled_tokens.",
        "evidence": {"available": True, "label": "Historical M11/M12 SGLang evidence. " + EVIDENCE_DISCLAIMER},
    },
}


def explain(intent: str, capability: Optional[Dict[str, Any]] = None,
            current: Any = None) -> Dict[str, Any]:
    """Merge static knowledge with live capability/current state."""
    knowledge = CONTROL_KNOWLEDGE.get(intent)
    if knowledge is None:
        return {
            "intent": intent,
            "known": False,
            "detail": f"no operator knowledge recorded for {intent!r}",
        }
    cap = capability or {}
    out = {"intent": intent, "known": True, **knowledge}
    out["engine"] = cap.get("engine")
    out["internal_key"] = cap.get("internal_key")
    out["supported_on_selected_engine"] = cap.get("supported_on_selected_engine")
    out["scope"] = cap.get("scope")
    out["bounds"] = {"min": cap.get("min"), "max": cap.get("max"), "dtype": cap.get("dtype")}
    out["current"] = current
    if cap.get("transition"):
        out["transition"] = cap["transition"]
    return out
