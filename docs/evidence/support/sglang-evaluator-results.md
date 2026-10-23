# SGLang evaluator results

Selected fields from the historical HTML evaluation reports. Local T2000 functional acceptance, not A100 performance. Values and phase policy retained.

## Single-control report

```json
{
  "configuration": {
    "engine": "sglang",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "baseline": {
      "runtime_max_prefill_tokens": 16384
    },
    "candidate": {
      "runtime_max_prefill_tokens": 4096
    },
    "repetitions": 1,
    "run_order": [
      [
        "baseline",
        "candidate"
      ]
    ]
  },
  "restore": {
    "http": 200,
    "status": "CONVERGED",
    "verified": true,
    "observed": {
      "runtime_max_running_seqs": 64,
      "runtime_max_prefill_tokens": 16384,
      "engine_cache_flush": 0,
      "prefix_cache_reset": 0
    },
    "steps": [
      {
        "param": "runtime_max_running_seqs",
        "transition": "HOT",
        "desired": 64,
        "observed_before": 64,
        "observed_after": 64,
        "status": "NOOP",
        "reason": null
      },
      {
        "param": "runtime_max_prefill_tokens",
        "transition": "HOT",
        "desired": 16384,
        "observed_before": 4096,
        "observed_after": 16384,
        "status": "VERIFIED",
        "reason": null
      }
    ],
    "rollback": null,
    "revision": 5
  }
}
```

Unavailable in comparison: TTFT p95, E2E p50/p95, queue p95 and request throughput. Phase application converged and was verified. Outcome label is not performance equivalence.

## Multi-control report

```json
{
  "configuration": {
    "engine": "sglang",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "baseline": {
      "runtime_max_running_seqs": 16,
      "runtime_max_prefill_tokens": 16384
    },
    "candidate": {
      "runtime_max_running_seqs": 8,
      "runtime_max_prefill_tokens": 4096
    },
    "repetitions": 2,
    "run_order": [
      [
        "baseline",
        "candidate"
      ],
      [
        "candidate",
        "baseline"
      ]
    ]
  },
  "restore": {
    "http": 200,
    "status": "CONVERGED",
    "verified": true,
    "observed": {
      "runtime_max_running_seqs": 64,
      "runtime_max_prefill_tokens": 16384,
      "engine_cache_flush": 0,
      "prefix_cache_reset": 0
    },
    "steps": [
      {
        "param": "runtime_max_prefill_tokens",
        "transition": "HOT",
        "desired": 16384,
        "observed_before": 16384,
        "observed_after": 16384,
        "status": "NOOP",
        "reason": null
      },
      {
        "param": "runtime_max_running_seqs",
        "transition": "HOT",
        "desired": 64,
        "observed_before": 16,
        "observed_after": 64,
        "status": "VERIFIED",
        "reason": null
      }
    ],
    "rollback": null,
    "revision": 10
  }
}
```

Unavailable in comparison: TTFT p95, E2E p50/p95, queue p95 and request throughput. Phase application converged and was verified. Outcome label is not performance equivalence.
