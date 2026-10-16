"""Shared helpers for M5B telemetry tests.

Provides a minimal local HTTP server that serves canned Prometheus exposition
per path (used as a stand-in for the engines' native telemetry surfaces) and
builders for source-grounded vLLM / SGLang exposition text.
"""

import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Dict, Optional, Tuple, Union

VLLM_LABELS = 'engine="0",model_name="test-model"'
SGLANG_LABELS = 'dp_rank="0"'


def histogram_text(
    name: str,
    buckets,
    total_sum: float,
    total_count: int,
    labels: str = VLLM_LABELS,
    add_help: bool = True,
) -> str:
    """Prometheus text for a histogram family (cumulative buckets + sum + count)."""
    lines = []
    if add_help:
        lines.append(f"# HELP {name} test histogram")
        lines.append(f"# TYPE {name} histogram")
    for le, count in buckets:
        lines.append(f'{name}_bucket{{{labels},le="{le}"}} {count}')
    lines.append(f"{name}_sum{{{labels}}} {total_sum}")
    lines.append(f"{name}_count{{{labels}}} {total_count}")
    return "\n".join(lines)


def vllm_exposition(
    running: int = 2,
    waiting: int = 1,
    kv_usage: float = 0.42,
    preemptions: int = 3,
    prompt_tokens: int = 100,
    prompt_tokens_cached: int = 40,
    generation_tokens: int = 500,
    prefix_queries: int = 120,
    prefix_hits: int = 60,
    waiting_reasons: Optional[Dict[str, int]] = None,
    ttft=None,
    tpot=None,
    e2e=None,
    queue_time=None,
    prefill=None,
    decode=None,
    inference=None,
    request_tpot=None,
    request_prompt_tokens_hist=None,
    request_generation_tokens_hist=None,
    request_success: Optional[Dict[str, int]] = None,
    prompt_tokens_by_source: Optional[Dict[str, int]] = None,
    legacy_counters: bool = False,
    empty: bool = False,
) -> str:
    """Source-grounded vLLM v0.28.0 /metrics exposition (real metric names).

    Counters carry the Prometheus ``_total`` suffix as verified live on the
    patched v0.28.0 runtime. ``legacy_counters=True`` emits the unsuffixed forms
    (for the exact-name fallback test only).
    """
    if empty:
        return "# HELP none\n# TYPE none gauge\n"
    sfx = "" if legacy_counters else "_total"
    parts = [
        f'# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{{{VLLM_LABELS}}} {running}',
        f'# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting{{{VLLM_LABELS}}} {waiting}',
        f'# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc{{{VLLM_LABELS}}} {kv_usage}',
        f'# TYPE vllm:num_preemptions{sfx} counter\nvllm:num_preemptions{sfx}{{{VLLM_LABELS}}} {preemptions}',
        f'# TYPE vllm:prompt_tokens{sfx} counter\nvllm:prompt_tokens{sfx}{{{VLLM_LABELS}}} {prompt_tokens}',
        f'# TYPE vllm:prompt_tokens_cached{sfx} counter\nvllm:prompt_tokens_cached{sfx}{{{VLLM_LABELS}}} {prompt_tokens_cached}',
        f'# TYPE vllm:generation_tokens{sfx} counter\nvllm:generation_tokens{sfx}{{{VLLM_LABELS}}} {generation_tokens}',
        f'# TYPE vllm:prefix_cache_queries{sfx} counter\nvllm:prefix_cache_queries{sfx}{{{VLLM_LABELS}}} {prefix_queries}',
        f'# TYPE vllm:prefix_cache_hits{sfx} counter\nvllm:prefix_cache_hits{sfx}{{{VLLM_LABELS}}} {prefix_hits}',
        "# TYPE vllm:iteration_tokens_total histogram\n"
        + histogram_text("vllm:iteration_tokens_total", [(1.0, 4), ("+Inf", 8)], 16.0, 8, add_help=False),
    ]
    if waiting_reasons:
        for reason, count in waiting_reasons.items():
            parts.append(
                f'# TYPE vllm:num_requests_waiting_by_reason gauge\n'
                f'vllm:num_requests_waiting_by_reason{{{VLLM_LABELS},reason="{reason}"}} {count}'
            )
    if ttft is not None:
        parts.append(histogram_text("vllm:time_to_first_token_seconds", *ttft))
    if tpot is not None:
        parts.append(histogram_text("vllm:inter_token_latency_seconds", *tpot))
    if e2e is not None:
        parts.append(histogram_text("vllm:e2e_request_latency_seconds", *e2e))
    if queue_time is not None:
        parts.append(histogram_text("vllm:request_queue_time_seconds", *queue_time))
    if prefill is not None:
        parts.append(histogram_text("vllm:request_prefill_time_seconds", *prefill))
    if decode is not None:
        parts.append(histogram_text("vllm:request_decode_time_seconds", *decode))
    if inference is not None:
        parts.append(histogram_text("vllm:request_inference_time_seconds", *inference))
    if request_tpot is not None:
        parts.append(
            histogram_text("vllm:request_time_per_output_token_seconds", *request_tpot)
        )
    if request_prompt_tokens_hist is not None:
        parts.append(histogram_text("vllm:request_prompt_tokens", *request_prompt_tokens_hist))
    if request_generation_tokens_hist is not None:
        parts.append(
            histogram_text("vllm:request_generation_tokens", *request_generation_tokens_hist)
        )
    if request_success:
        # Live v0.28.0 exposition carries the Prometheus counter '_total' suffix.
        for reason, count in request_success.items():
            parts.append(
                f'# TYPE vllm:request_success_total counter\n'
                f'vllm:request_success_total{{{VLLM_LABELS},finished_reason="{reason}"}} {count}'
            )
    if prompt_tokens_by_source:
        for source, count in prompt_tokens_by_source.items():
            parts.append(
                f'# TYPE vllm:prompt_tokens_by_source_total counter\n'
                f'vllm:prompt_tokens_by_source_total{{{VLLM_LABELS},source="{source}"}} {count}'
            )
    return "\n".join(parts) + "\n"


def sglang_loads_exposition(
    running: int = 0,
    waiting: int = 0,
    token_usage: float = 0.0,
    gen_throughput: float = 0.0,
    cache_hit_rate: float = 0.0,
    max_total_num_tokens: int = 100000,
    max_running_requests: int = 8,
    utilization: float = 0.0,
    kv_cache_gb: float = 1.9,
    weight_gb: float = 2.1,
    num_used_tokens: int = 0,
    num_total_tokens: int = 0,
    num_active_tokens: int = 0,
    num_waiting_uncached_tokens: int = 0,
    total_prefill_uncached_tokens: int = 0,
    total_prefill_busy_us: int = 0,
    timestamp: Optional[float] = None,
    labels: str = SGLANG_LABELS,
) -> str:
    """Source-grounded SGLang /v1/loads?format=prometheus exposition.

    Mirrors ``_format_loads_prometheus`` (M9A source-verified): every flat
    ``LoadSnapshot`` scalar is rendered as a ``sglang_<key>`` gauge. The
    cumulative counters (``total_prefill_*``) are therefore available on this
    surface without a second endpoint.
    """
    import time as _time

    ts = timestamp if timestamp is not None else _time.time()
    families = {
        "sglang_timestamp": ts,
        "sglang_num_running_reqs": running,
        "sglang_num_waiting_reqs": waiting,
        "sglang_token_usage": token_usage,
        "sglang_gen_throughput": gen_throughput,
        "sglang_cache_hit_rate": cache_hit_rate,
        "sglang_max_total_num_tokens": max_total_num_tokens,
        "sglang_max_running_requests": max_running_requests,
        "sglang_utilization": utilization,
        "sglang_memory_kv_cache_gb": kv_cache_gb,
        "sglang_memory_weight_gb": weight_gb,
        "sglang_num_used_tokens": num_used_tokens,
        "sglang_num_total_tokens": num_total_tokens,
        "sglang_num_active_tokens": num_active_tokens,
        "sglang_num_waiting_uncached_tokens": num_waiting_uncached_tokens,
        "sglang_total_prefill_uncached_tokens": total_prefill_uncached_tokens,
        "sglang_total_prefill_busy_us": total_prefill_busy_us,
    }
    lines = []
    for name, value in families.items():
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{{{labels}}} {value}")
    return "\n".join(lines) + "\n"


def sglang_metrics_exposition(
    ttft=None,
    tpot=None,
    e2e=None,
    queue_time=None,
    per_stage: Optional[Dict[str, tuple]] = None,
    prompt_tokens_hist=None,
    uncached_prompt_tokens_hist=None,
    generation_tokens_hist=None,
    forward_execution_seconds_total: Optional[float] = None,
    fwd_occupancy: Optional[float] = None,
    decode_sum_seq_lens: Optional[float] = None,
    num_requests_total: Optional[int] = None,
    num_aborted_requests_total: Optional[int] = None,
    kv_used_tokens: Optional[int] = None,
    labels: str = SGLANG_LABELS,
) -> str:
    """Source-grounded SGLang v0.5.18 ``/metrics`` exposition (--enable-metrics)."""
    parts = []
    if ttft is not None:
        parts.append(histogram_text("sglang:time_to_first_token_seconds", *ttft,
                                    labels=f'{labels},is_streaming="false"'))
    if tpot is not None:
        parts.append(histogram_text("sglang:inter_token_latency_seconds", *tpot, labels=labels))
    if e2e is not None:
        parts.append(histogram_text("sglang:e2e_request_latency_seconds", *e2e,
                                    labels=f'{labels},is_streaming="false"'))
    if queue_time is not None:
        parts.append(histogram_text("sglang:queue_time_seconds", *queue_time, labels=labels))
    if per_stage:
        for stage, hist in per_stage.items():
            parts.append(histogram_text(
                "sglang:per_stage_req_latency_seconds", *hist,
                labels=f'{labels},stage="{stage}"',
            ))
    if prompt_tokens_hist is not None:
        parts.append(histogram_text("sglang:prompt_tokens_histogram", *prompt_tokens_hist, labels=labels))
    if uncached_prompt_tokens_hist is not None:
        parts.append(histogram_text(
            "sglang:uncached_prompt_tokens_histogram", *uncached_prompt_tokens_hist, labels=labels,
        ))
    if generation_tokens_hist is not None:
        parts.append(histogram_text(
            "sglang:generation_tokens_histogram", *generation_tokens_hist, labels=labels,
        ))
    scalars = {
        "sglang:forward_execution_seconds_total": forward_execution_seconds_total,
        "sglang:fwd_occupancy": fwd_occupancy,
        "sglang:decode_sum_seq_lens": decode_sum_seq_lens,
        "sglang:num_requests_total": num_requests_total,
        "sglang:num_aborted_requests_total": num_aborted_requests_total,
        "sglang:kv_used_tokens": kv_used_tokens,
    }
    for name, value in scalars.items():
        if value is not None:
            parts.append(f"# TYPE {name} gauge")
            parts.append(f'{name}{{{labels}}} {value}')
    return "\n".join(parts) + "\n"


# ------------------------------------------------------------------ M5C GPU
def gpu_csv(*rows) -> str:
    """``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` output. Each
    row is an 8-field tuple in the collector's field order: (util, clock, used,
    free, total, power, power_limit, temperature)."""
    return "\n".join(",".join(str(v) for v in row) for row in rows) + "\n"


def compute_apps_csv(*rows) -> str:
    """``nvidia-smi --query-compute-apps=pid,process_name,used_memory`` output.
    Each row is (pid, name, used_memory_mb); names containing commas are
    double-quoted exactly like nvidia-smi CSV output."""
    lines = []
    for row in rows:
        fields = []
        for v in row:
            text = str(v)
            if "," in text or '"' in text:
                text = '"' + text.replace('"', '""') + '"'
            fields.append(text)
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"


def pmon_text(rows) -> str:
    """``nvidia-smi pmon -c 1`` output. Each row is (gpu, pid, type, sm, mem,
    enc, dec, byte, read, write); ``sm`` is the per-process SM utilization %."""
    lines = [
        "# gpu        pid  type    sm   mem   enc   dec   byte   read   write",
        "# Idx          Pid Ctx   SM%   MEM%   ENC%   DEC%  MB/s   MB/s   MB/s",
    ]
    for row in rows:
        lines.append("   ".join(str(v) for v in row))
    return "\n".join(lines) + "\n"


class FakeGpuSource:
    """Injectable nvidia-smi runner for M5C tests: maps argv to canned outputs
    (returncode, stdout) so no GPU / subprocess is needed."""

    def __init__(self, gpu_out="", gpu_rc=0, apps_out="", apps_rc=0,
                 pmon_out="", pmon_rc=0):
        self.gpu_out, self.gpu_rc = gpu_out, gpu_rc
        self.apps_out, self.apps_rc = apps_out, apps_rc
        self.pmon_out, self.pmon_rc = pmon_out, pmon_rc

    def __call__(self, argv):
        joined = " ".join(argv)
        if "--query-gpu=" in joined:
            return self.gpu_rc, self.gpu_out
        if "--query-compute-apps=" in joined:
            return self.apps_rc, self.apps_out
        if "pmon" in joined:
            return self.pmon_rc, self.pmon_out
        return 1, ""


Route = Union[
    Tuple[int, str],
    Callable[[], Tuple[int, str]],
]


@contextmanager
def telemetry_server(routes: Dict[str, Route]):
    """Local HTTP server serving canned responses per request path."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            route = routes.get(self.path)
            if route is None:
                self.send_response(404)
                self.end_headers()
                return
            status, body = route() if callable(route) else route
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, *args):  # silence
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)