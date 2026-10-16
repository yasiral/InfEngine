import torch

from infengine.config import ModelConfig
from infengine.model import LlamaModel, precompute_freqs
from infengine.engine import LLMEngine
from infengine.disagg import PrefillEngine, DecodeEngine


def test_disaggregated_prefill_decode_matches_generate():
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    model = LlamaModel(cfg).to(torch.float32)
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, "cpu")
    prompt = [5, 8, 23, 12, 7, 3, 9, 10, 11, 2, 4]

    eng = LLMEngine(model, None, "cpu", torch.float32, cos=cos, sin=sin,
                    block_size=4, num_blocks=64, max_batch=1, chunk_size=16)
    ref = list(eng.generate(prompt, max_new_tokens=12, temperature=0.0))

    prefill = PrefillEngine(model, None, "cpu", torch.float32, block_size=4, num_blocks=64)
    handle = prefill.prefill(prompt)
    assert handle.length == len(prompt)
    assert len(handle.k) == cfg.num_layers
    assert handle.k[0].shape[1] == cfg.num_kv_heads  # [L, kv_heads, head_dim]

    decode = DecodeEngine(model, None, "cpu", torch.float32, block_size=4, num_blocks=64)
    out = list(decode.decode(handle, max_new_tokens=12))
    assert out == ref, f"disaggregated({out}) != generate({ref})"


def test_handle_kv_roundtrip_preserves_values():
    cfg = ModelConfig(hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2,
                      vocab_size=64, intermediate_size=64, max_seq_len=32)
    model = LlamaModel(cfg).to(torch.float32)
    prefill = PrefillEngine(model, None, "cpu", torch.float32, block_size=4, num_blocks=32)
    handle = prefill.prefill([1, 2, 3, 4, 5])

    # rebuild a store from the handle and confirm the KV matches what was gathered
    from infengine.kv_store import PagedKVStore
    store = PagedKVStore(cfg, 4, 32, torch.float32, "cpu")
    sid = 0
    store.init_seq(sid)
    store.reserve(sid, handle.length)
    store.write(sid, 0, handle.k[0], handle.v[0], handle.length)
    k_g, v_g = store.gather(sid, 0)
    assert torch.allclose(k_g.transpose(0, 1), handle.k[0])
    assert torch.allclose(v_g.transpose(0, 1), handle.v[0])