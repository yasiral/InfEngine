import torch

from infengine.config import ModelConfig
from infengine.model import LlamaModel, precompute_freqs
from infengine.engine import LLMEngine
from infengine.kv_store import PagedKVStore


def _mk(cfg):
    return LlamaModel(cfg).to(dtype=torch.float32)


def test_speculative_matches_greedy():
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    target = _mk(cfg)
    draft_cfg = ModelConfig(hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2,
                            vocab_size=128, intermediate_size=64, max_seq_len=64)
    draft = _mk(draft_cfg)  # smaller/weaker draft -> many rejections, but still lossless
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, "cpu")

    eng = LLMEngine(target, None, "cpu", torch.float32, cos=cos, sin=sin,
                    block_size=4, num_blocks=128, max_batch=1, chunk_size=32)
    prompt = [5, 8, 23, 12, 7, 3, 9, 10, 11, 2, 4, 6, 1, 14, 15, 13, 7, 8]
    ref = list(eng.generate(prompt, max_new_tokens=20, temperature=0.0))

    eng2 = LLMEngine(target, None, "cpu", torch.float32, cos=cos, sin=sin,
                     block_size=4, num_blocks=128, max_batch=1, chunk_size=32)
    spec = list(eng2.generate_speculative(prompt, draft, None, spec_len=4, max_new_tokens=20))
    assert spec == ref, f"speculative({spec}) != greedy({ref})"


def test_speculative_perfect_draft_matches():
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    target = _mk(cfg)
    draft = _mk(cfg)
    draft.load_state_dict(target.state_dict())  # perfect draft -> everything accepted
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, "cpu")

    eng = LLMEngine(target, None, "cpu", torch.float32, cos=cos, sin=sin,
                    block_size=4, num_blocks=128, max_batch=1, chunk_size=32)
    prompt = [5, 8, 23, 12, 7, 3, 9, 10, 11, 2, 4]
    ref = list(eng.generate(prompt, max_new_tokens=16, temperature=0.0))

    eng2 = LLMEngine(target, None, "cpu", torch.float32, cos=cos, sin=sin,
                     block_size=4, num_blocks=128, max_batch=1, chunk_size=32)
    spec = list(eng2.generate_speculative(prompt, draft, None, spec_len=3, max_new_tokens=16))
    assert spec == ref


def test_rollback_trims_length():
    cfg = ModelConfig(hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2,
                      vocab_size=64, intermediate_size=64, max_seq_len=32)
    store = PagedKVStore(cfg, block_size=4, num_blocks=16, dtype=torch.float32, device="cpu")
    store.init_seq(0)
    store.reserve(0, 5)
    assert store.length(0) == 5
    store.rollback(0, 3)
    assert store.length(0) == 2
    store.rollback(0, 5)
    assert store.length(0) == 0
