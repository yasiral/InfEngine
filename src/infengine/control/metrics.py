from typing import Any, Dict


def collect_engine_metrics(engine) -> Dict[str, Any]:
    """Engine-internal signals the routing layer never sees.

    These are the inputs the closed-loop controller needs to tune scheduling,
    KV cache and admission on the running process.
    """
    sched = engine.scheduler
    pending_prefill = sum(
        1 for r in sched.requests.values() if r.status == "prefill" and not r.finished
    )
    pending_decode = sum(
        1 for r in sched.requests.values() if r.status == "decode" and not r.finished
    )
    capacity = engine.store.k_pool[0].shape[0] if engine.store.k_pool else 0
    used = engine.store.used_blocks
    return {
        "active_requests": len(sched.requests),
        "pending_prefill": pending_prefill,
        "pending_decode": pending_decode,
        "kv_blocks_used": used,
        "kv_blocks_capacity": capacity,
        "kv_utilization": round(used / capacity, 4) if capacity else 0.0,
        "max_batch": sched.max_batch,
        "chunk_size": sched.chunk_size,
        "kv_block_size": engine.store.block_size,
        "kv_num_blocks": capacity,
    }