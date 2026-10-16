"""Throughput / latency benchmark (Ch7: production observability).

Loads a model once, then runs ``--num-requests`` generations with a pool of
``--concurrency`` worker threads. Because every thread drives the *same*
scheduler, requests are continuously batched (prefill + decode are shared
across the batch), which is exactly what the engine was built to do.

Usage:
    python -m infengine.bench --model Qwen/Qwen2.5-0.5B-Instruct --concurrency 4 --num-requests 16
"""

import argparse
import queue
import statistics
import threading
import time

import torch

from .backend import build_model_from_hf, build_tokenizer, resolve_device, resolve_dtype
from .engine import LLMEngine


def _worker(engine, prompt_ids, max_new_tokens, results, idx):
    t_start = time.perf_counter()
    ttft = None
    tokens = 0
    for _ in engine.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=0.0):
        if ttft is None:
            ttft = time.perf_counter() - t_start
        tokens += 1
    results[idx] = {"ttft": ttft, "latency": time.perf_counter() - t_start, "tokens": tokens}


def run_bench(model_id, concurrency, num_requests, max_new_tokens, prompt,
              bits=0, max_batch=4, chunk_size=256, block_size=64, num_blocks=512,
              device_name="auto", seed=None):
    if device_name == "auto":
        device = resolve_device()
    else:
        device = torch.device(device_name)
    dtype = resolve_dtype(device)
    print(f"loading {model_id} (bits={bits})...")
    model, cfg = build_model_from_hf(model_id, device=device, dtype=dtype, bits=bits)
    tokenizer = build_tokenizer(model_id)
    engine = LLMEngine(model, tokenizer, device, dtype,
                       max_batch=max_batch, chunk_size=chunk_size,
                       block_size=block_size, num_blocks=num_blocks)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"prompt: {len(prompt_ids)} tokens, concurrency={concurrency}, "
          f"requests={num_requests}, max_new_tokens={max_new_tokens}")

    work = queue.Queue()
    for i in range(num_requests):
        work.put(i)
    results = [None] * num_requests

    def run():
        while True:
            try:
                i = work.get_nowait()
            except queue.Empty:
                return
            _worker(engine, prompt_ids, max_new_tokens, results, i)
            work.task_done()

    threads = [threading.Thread(target=run) for _ in range(concurrency)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0

    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    lats = [r["latency"] for r in results]
    toks = sum(r["tokens"] for r in results)

    def pct(vals, p):
        return statistics.quantiles(sorted(vals), n=100)[p - 1] if vals else 0.0

    print("\n=== benchmark results ===")
    print(f"wall time:          {wall:.2f}s")
    print(f"requests/sec:       {num_requests / wall:.2f}")
    print(f"tokens/sec:         {toks / wall:.2f}")
    print(f"avg tokens/request: {toks / num_requests:.1f}")
    print(f"TTFT  avg/p50/p95:  {statistics.mean(ttfts) * 1000:.1f}/{pct(ttfts, 50) * 1000:.1f}/{pct(ttfts, 95) * 1000:.1f} ms")
    print(f"lat   avg/p50/p95:  {statistics.mean(lats) * 1000:.1f}/{pct(lats, 50) * 1000:.1f}/{pct(lats, 95) * 1000:.1f} ms")
    m = engine.metrics
    print("\n=== engine metrics ===")
    print(f"prefill steps:      {m['prefill_steps']}  ({m['prefill_time_s']:.2f}s)")
    print(f"decode steps:       {m['decode_steps']}  ({m['decode_time_s']:.2f}s)")
    print(f"max KV blocks used: {m['kv_blocks_used']} / {num_blocks}")
    return engine


def main():
    ap = argparse.ArgumentParser("infengine-bench")
    ap.add_argument("--model", dest="model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=512)
    ap.add_argument("--device", dest="device_name", type=str, default="auto")
    args = ap.parse_args()
    run_bench(**vars(args))


if __name__ == "__main__":
    main()