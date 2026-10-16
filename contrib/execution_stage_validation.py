"""M9B live execution-stage validation harness (vLLM / SGLang).

Targeted, READ-ONLY validation for the pinned patched engines:

  1. Confirms the actual Prometheus exposition names/labels of the native
     execution-stage families (vLLM prefill/decode/inference/ITL/TPOT/tokens;
     SGLang per-stage/inter-token/tokens).
  2. Drives a small prefill-heavy and decode-heavy workload through the
     engine's native ``/v1/completions``.
  3. Polls twice so the windowed histograms produce quantiles, then prints the
     classified execution evidence and the observed sample counts.

It never mutates runtime policy, never proxies inference, and never induces
artificial pressure. Run it against a running engine:

    python contrib/execution_stage_validation.py --engine vllm \\
        --base-url http://localhost:8000 --tokenizer-model Qwen/Qwen2.5-0.5B-Instruct

If no engine is available, this harness is simply not runnable and live
validation remains DEFERRED (see M9B_EXECUTION_STAGE_TELEMETRY.md).
"""

import argparse
import json
import sys
import time
import urllib.request
from typing import Dict, List

from infengine.telemetry import (
    SglangTelemetryCollector,
    VllmTelemetryCollector,
    build_execution_evidence,
    validate_snapshot,
)

PREFILL_PROMPT = "Summarize the following passage in one sentence: " + ("word " * 400)
DECODE_PROMPT = "Count slowly from one to two hundred, one number per line."


def _post_completion(base_url: str, prompt: str, max_tokens: int, timeout: float) -> int:
    body = json.dumps(
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
        return resp.status


def _drive_workload(base_url: str, timeout: float) -> List[str]:
    notes = []
    for label, prompt, max_tokens in (
        ("prefill-heavy", PREFILL_PROMPT, 8),
        ("decode-heavy", DECODE_PROMPT, 200),
    ):
        try:
            status = _post_completion(base_url, prompt, max_tokens, timeout)
            notes.append(f"{label}: HTTP {status}")
        except Exception as exc:  # noqa: BLE001 - report, do not fail the audit
            notes.append(f"{label}: skipped ({exc})")
    return notes


def _collect(engine: str, base_url: str, polls: int, interval: float, workload):
    """Baseline poll -> workload -> poll (delta) -> workload -> poll ...

    The windowed histograms need a *delta* between scrapes, so traffic must be
    driven AFTER the baseline poll and BEFORE each subsequent poll.
    """
    if engine == "vllm":
        collector = VllmTelemetryCollector(base_url, timeout=10.0)
    else:
        collector = SglangTelemetryCollector(base_url, timeout=10.0)
    snapshots = [collector.poll_once()]
    notes = []
    for _ in range(max(1, polls - 1)):
        if workload is not None:
            notes.extend(workload())
        time.sleep(interval)
        snapshots.append(collector.poll_once())
    return collector, snapshots, notes


def main() -> int:
    ap = argparse.ArgumentParser("execution-stage-validation")
    ap.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    ap.add_argument("--base-url", required=True,
                    help="engine base URL (vLLM API server or SGLang server)")
    ap.add_argument("--polls", type=int, default=3)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--no-workload", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    workload = None
    if not args.no_workload:
        def workload():
            return _drive_workload(args.base_url, args.timeout)

    collector, snapshots, workload_notes = _collect(
        args.engine, args.base_url, args.polls, args.interval, workload
    )
    for snap in snapshots:
        validate_snapshot(snap)
    evidence = build_execution_evidence(snapshots[-1])

    measured = {
        key: {
            "classification": m.classification,
            "source": m.source,
            "samples": m.sample_count,
            "quantiles": m.quantiles,
        }
        for section in (evidence.request_stage, evidence.engine_level)
        for key, m in section.items()
    }

    report: Dict[str, object] = {
        "engine": args.engine,
        "engine_version": evidence.engine_version,
        "base_url": args.base_url,
        "workload": workload_notes,
        "window_s": evidence.window_s,
        "request_stage": {k: v for k, v in measured.items()},
        "engine_specific_keys": sorted(snapshots[-1].engine_specific),
        "quality": dict(snapshots[-1].quality),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    # Report (do not assert) the expected phase evidence; unavailability is a
    # legitimate, recorded outcome (one scrape / metrics disabled).
    for key in ("prefill_time", "decode_time", "inter_token_latency"):
        m = evidence.request_stage.get(key)
        if m is not None:
            print(f"[evidence] {key}: {m.classification} samples={m.sample_count}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
