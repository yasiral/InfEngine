import pytest
import torch
import torch.nn.functional as F

from infengine.config import ModelConfig
from infengine.model import LlamaModel, precompute_freqs, apply_rope
from infengine.kv_store import PagedKVStore
from infengine.engine import LLMEngine


def reference_logits(model, input_ids, cos, sin):
    B, T = input_ids.shape
    pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
    c = cos[pos]
    s = sin[pos]
    x = model.embed(input_ids)
    for layer in model.layers:
        r = x
        n = layer.input_norm(x)
        h = model.cfg.head_dim
        nh = model.cfg.num_heads
        k_h = model.cfg.num_kv_heads
        q = layer.attn.q(n).view(B, T, nh, h).transpose(1, 2)
        k = layer.attn.k(n).view(B, T, k_h, h).transpose(1, 2)
        v = layer.attn.v(n).view(B, T, k_h, h).transpose(1, 2)
        q = apply_rope(q, c, s)
        k = apply_rope(k, c, s)
        if k_h != nh:
            rp = nh // k_h
            k = k.repeat_interleave(rp, dim=1)
            v = v.repeat_interleave(rp, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = r + layer.attn.o(attn.transpose(1, 2).contiguous().view(B, T, -1))
        r = x
        x = r + layer.mlp(layer.post_norm(x))
    x = model.norm(x)
    return model.lm_head(x)


def test_prefill_logits_match_reference():
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    device = "cpu"
    model = LlamaModel(cfg).to(dtype=torch.float32, device=device)
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, device)
    store = PagedKVStore(cfg, block_size=4, num_blocks=32, dtype=torch.float32, device=device)
    prompt_ids = [5, 8, 23, 12, 7]
    sid = 0
    store.init_seq(sid)
    store.reserve(sid, len(prompt_ids))
    input_t = torch.tensor([prompt_ids], device=device)
    pos = torch.arange(len(prompt_ids), device=device).unsqueeze(0)
    L = len(prompt_ids)
    attn_mask = (torch.arange(L, device=device)[None, None, :] <= pos[:, :, None]).bool()
    logits_engine = model.forward(
        input_t, store, [sid], pos, attn_mask, cos, sin,
        n_valid=torch.tensor([len(prompt_ids)], device=device),
    )
    logits_ref = reference_logits(model, input_t, cos, sin)
    assert torch.allclose(logits_engine, logits_ref, atol=1e-5), (
        f"prefill logit mismatch\nengine:{logits_engine}\nref:{logits_ref}"
    )

    tok = logits_engine[:, -1].argmax(-1).item()
    store.reserve(sid, 1)
    input_t = torch.tensor([[tok]], device=device)
    pos = torch.tensor([[len(prompt_ids)]], device=device)
    L = store.length(sid)
    attn_mask = (torch.arange(L, device=device)[None, None, :] <= pos[:, :, None]).bool()
    logits_engine = model.forward(
        input_t, store, [sid], pos, attn_mask, cos, sin,
        n_valid=torch.tensor([1], device=device),
    )
    full_ids = prompt_ids + [tok]
    logits_ref = reference_logits(model, torch.tensor([full_ids], device=device), cos, sin)
    assert torch.allclose(logits_engine, logits_ref[:, -1:], atol=1e-5), (
        f"decode logit mismatch\nengine:{logits_engine}\nref:{logits_ref[:, -1:]}"
    )


def test_engine_generates_deterministic():
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    device = "cpu"
    model = LlamaModel(cfg).to(dtype=torch.float32, device=device)
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, device)
    store = PagedKVStore(cfg, block_size=4, num_blocks=32, dtype=torch.float32, device=device)
    engine = LLMEngine(model, None, device, torch.float32,
                       cos=cos, sin=sin, block_size=4, num_blocks=32,
                       max_batch=1, chunk_size=256)
    prompt_ids = [5, 8, 23, 12, 7]
    out1 = list(engine.generate(prompt_ids, max_new_tokens=8, temperature=0.0))
    store2 = PagedKVStore(cfg, block_size=4, num_blocks=32, dtype=torch.float32, device=device)
    engine2 = LLMEngine(model, None, device, torch.float32,
                        cos=cos, sin=sin, block_size=4, num_blocks=32,
                        max_batch=1, chunk_size=256)
    out2 = list(engine2.generate(prompt_ids, max_new_tokens=8, temperature=0.0))
    assert out1 == out2, f"{out1} != {out2}"


def test_engine_prefix_reuse_matches_reference():
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    device = "cpu"
    model = LlamaModel(cfg).to(dtype=torch.float32, device=device)
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, device)

    prefix = list(range(16))  # 4 blocks of block_size 4
    suffix = [10, 11, 12, 13]
    full = prefix + suffix

    # reference: run the full prompt from scratch, no cache priming
    eng1 = LLMEngine(model, None, device, torch.float32, cos=cos, sin=sin,
                     block_size=4, num_blocks=64, max_batch=1, chunk_size=8)
    out1 = list(eng1.generate(full, max_new_tokens=8, temperature=0.0))

    # cached: prime with the prefix first, then run the full prompt
    eng2 = LLMEngine(model, None, device, torch.float32, cos=cos, sin=sin,
                     block_size=4, num_blocks=64, max_batch=1, chunk_size=8)
    list(eng2.generate(prefix, max_new_tokens=1, temperature=0.0))
    out2 = list(eng2.generate(full, max_new_tokens=8, temperature=0.0))

    assert out1 == out2, f"cached({out1}) != reference({out2})"
    assert eng2.store.prefix  # prefix cache retained blocks after request completion


def test_engine_metrics_accumulate():
    torch.manual_seed(123)  # deterministic model that generates > 1 token
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    device = "cpu"
    model = LlamaModel(cfg).to(dtype=torch.float32, device=device)
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, device)
    engine = LLMEngine(model, None, device, torch.float32, cos=cos, sin=sin,
                       block_size=4, num_blocks=32, max_batch=1, chunk_size=8)
    prompt_ids = [5, 8, 23, 12, 7, 3, 9, 10, 11, 2]
    n = list(engine.generate(prompt_ids, max_new_tokens=6, temperature=0.0))
    assert len(n) > 1  # model did not immediately emit EOS
    assert engine.metrics["requests"] == 1
    assert engine.metrics["tokens"] == len(n)
    assert engine.metrics["prefill_steps"] >= 1
    assert engine.metrics["decode_steps"] >= 1
    assert engine.metrics["total_time_s"] > 0
    assert engine.metrics["kv_blocks_used"] > 0


def test_engine_concurrent_generates_are_correct():
    import threading

    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    device = "cpu"
    model = LlamaModel(cfg).to(dtype=torch.float32, device=device)
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, device)
    engine = LLMEngine(model, None, device, torch.float32, cos=cos, sin=sin,
                       block_size=4, num_blocks=64, max_batch=2, chunk_size=8)
    prompt_a = [5, 8, 23, 12, 7, 3, 9, 10, 11, 2, 4]
    prompt_b = [9, 3, 10, 12, 7, 1, 5, 8, 2, 4, 6, 11]

    # serial references
    ref_a = list(engine.generate(prompt_a, max_new_tokens=6, temperature=0.0))
    ref_b = list(engine.generate(prompt_b, max_new_tokens=6, temperature=0.0))

    engine2 = LLMEngine(model, None, device, torch.float32, cos=cos, sin=sin,
                        block_size=4, num_blocks=64, max_batch=2, chunk_size=8)
    results = {}

    def run(name, prompt):
        results[name] = list(engine2.generate(prompt, max_new_tokens=6, temperature=0.0))

    t1 = threading.Thread(target=run, args=("a", prompt_a))
    t2 = threading.Thread(target=run, args=("b", prompt_b))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert results["a"] == ref_a, f"concurrent A {results['a']} != serial {ref_a}"
    assert results["b"] == ref_b, f"concurrent B {results['b']} != serial {ref_b}"


def test_fully_prefix_cached_prompt_no_crash():
    """A prompt entirely covered by the prefix cache must not crash prefill.

    Regression: adopt() returning shared == len(prompt) left remaining_prompt 0
    and _do_prefill built a T=0 tensor. The engine re-prefills the final token.
    """
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    device = "cpu"
    model = LlamaModel(cfg).to(dtype=torch.float32, device=device)
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, device)
    engine = LLMEngine(model, None, device, torch.float32, cos=cos, sin=sin,
                       block_size=4, num_blocks=64, max_batch=1, chunk_size=8)

    prompt = list(range(16))  # 4 full blocks
    out1 = list(engine.generate(prompt, max_new_tokens=4, temperature=0.0))
    assert len(out1) == 4
    # second identical request: fully prefix-cached -> previously crashed
    out2 = list(engine.generate(prompt, max_new_tokens=4, temperature=0.0))
    assert len(out2) == 4
    assert out1 == out2
    assert engine.metrics["requests"] == 2


def test_generate_failure_releases_resources():
    """A generation failure must not leak scheduler/KV state.

    Regression: generate() had no finally, so a KV pool exhaustion leaked the
    request and its blocks. Leaked requests permanently block DRAIN quiescence.
    """
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    device = "cpu"
    model = LlamaModel(cfg).to(dtype=torch.float32, device=device)
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, device)
    engine = LLMEngine(model, None, device, torch.float32, cos=cos, sin=sin,
                       block_size=4, num_blocks=8, max_batch=1, chunk_size=8)

    long_prompt = list(range(32))  # 40 tokens with output exceeds the 8-block pool
    with pytest.raises(RuntimeError, match="KV block pool exhausted"):
        list(engine.generate(long_prompt, max_new_tokens=8, temperature=0.0))

    assert engine.scheduler.requests == {}       # no leaked scheduler entry
    assert engine.store.seqs == {}                # no leaked live KV sequences
    # prefix cache may retain completed prompt blocks by design; the failed
    # request must not own any of them
    assert engine.store.used_blocks < 8

    # engine still usable after the failure (needs 2 of 8 blocks)
    out = list(engine.generate([5, 8, 23, 12], max_new_tokens=4, temperature=0.0))
    assert len(out) >= 1
    assert engine.metrics["requests"] == 2