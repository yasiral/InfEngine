import torch

from infengine.config import ModelConfig
from infengine.model import LlamaModel, precompute_freqs
from infengine.kv_store import PagedKVStore
from infengine.parallel import make_tensor_parallel
from infengine.engine import LLMEngine


def _cfg():
    return ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                       vocab_size=128, intermediate_size=64, max_seq_len=64)


def _forward_logits(model, ids):
    cfg = model.cfg
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, "cpu")
    store = PagedKVStore(cfg, block_size=4, num_blocks=32, dtype=torch.float32, device="cpu")
    sid = 0
    store.init_seq(sid)
    store.reserve(sid, len(ids))
    pos = torch.arange(len(ids)).unsqueeze(0)
    L = len(ids)
    mask = (torch.arange(L)[None, None, :] <= pos[:, :, None]).bool()
    return model.forward(torch.tensor([ids]), store, [sid], pos, mask, cos, sin,
                         n_valid=torch.tensor([len(ids)]))


def test_tensor_parallel_matches_full_model():
    torch.manual_seed(0)
    full = LlamaModel(_cfg()).to(torch.float32)
    sharded = LlamaModel(_cfg()).to(torch.float32)
    sharded.load_state_dict(full.state_dict())
    make_tensor_parallel(sharded, n_shards=2)

    ids = [5, 8, 23, 12, 7, 3, 9, 10, 11]
    lo = _forward_logits(full, ids)
    ls = _forward_logits(sharded, ids)
    assert lo.shape == ls.shape
    assert torch.allclose(lo, ls, atol=1e-4), (lo - ls).abs().max()


def test_tensor_parallel_generation_matches():
    torch.manual_seed(0)
    cfg = _cfg()
    full = LlamaModel(cfg).to(torch.float32)
    sharded = LlamaModel(cfg).to(torch.float32)
    sharded.load_state_dict(full.state_dict())
    make_tensor_parallel(sharded, n_shards=2)

    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, "cpu")
    e1 = LLMEngine(full, None, "cpu", torch.float32, cos=cos, sin=sin,
                   block_size=4, num_blocks=64, max_batch=1, chunk_size=16)
    e2 = LLMEngine(sharded, None, "cpu", torch.float32, cos=cos, sin=sin,
                   block_size=4, num_blocks=64, max_batch=1, chunk_size=16)
    prompt = [5, 8, 23, 12, 7, 3, 9, 10, 11, 2, 4, 6, 1, 14]
    assert list(e1.generate(prompt, max_new_tokens=10, temperature=0.0)) == \
           list(e2.generate(prompt, max_new_tokens=10, temperature=0.0))