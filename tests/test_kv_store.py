import torch
import pytest

from infengine.config import ModelConfig
from infengine.kv_store import PagedKVStore


@pytest.fixture
def cfg():
    return ModelConfig(
        hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
        vocab_size=64, intermediate_size=64, max_seq_len=32,
    )


@pytest.fixture
def store(cfg):
    return PagedKVStore(cfg, block_size=8, num_blocks=16, dtype=torch.float32, device="cpu")


def test_alloc_and_free(store):
    store.init_seq(0)
    store.init_seq(1)
    assert store.length(0) == 0
    assert store.length(1) == 0
    store.release(0)
    store.release(1)
    assert store._count == 0


def test_write_gather_roundtrip(store):
    sid = 42
    store.init_seq(sid)
    cfg = store.cfg
    k = torch.randn(1, cfg.num_kv_heads, cfg.head_dim)
    v = torch.randn_like(k)
    store.reserve(sid, 1)
    store.write(sid, 0, k, v, n=1)
    assert store.length(sid) == 1
    k_g, v_g = store.gather(sid, 0)
    assert torch.allclose(k[0], k_g[:, 0])
    assert torch.allclose(v[0], v_g[:, 0])


def test_multi_tokens_multilayer(store):
    sid = 7
    store.init_seq(sid)
    cfg = store.cfg
    store.reserve(sid, 4)
    for layer in range(2):
        k = torch.randn(4, cfg.num_kv_heads, cfg.head_dim)
        v = torch.randn_like(k)
        store.write(sid, layer, k, v, n=4)
        assert store.length(sid) == 4
        k_g, v_g = store.gather(sid, layer)
        assert k_g.shape[1] == 4
        assert torch.allclose(k, k_g.transpose(0, 1))


def test_gather_batch_pads(store):
    for sid in (0, 1):
        store.init_seq(sid)
    for sid, n in ((0, 2), (1, 4)):
        k = torch.randn(n, store.cfg.num_kv_heads, store.cfg.head_dim)
        v = torch.randn_like(k)
        store.reserve(sid, n)
        store.write(sid, 0, k, v, n=n)
    k, v, lens = store.gather_batch([0, 1], 0)
    assert k.shape[1] == store.cfg.num_kv_heads
    assert k.shape[2] == 4  # max_len
    assert lens == [2, 4]


def test_prefix_reuse_shared_blocks(store):
    bs = store.block_size  # 8
    tokens = list(range(24))  # 3 full blocks
    store.init_seq(0)
    store.reserve(0, len(tokens))
    for layer in range(store.cfg.num_layers):
        k = torch.randn(len(tokens), store.cfg.num_kv_heads, store.cfg.head_dim)
        v = torch.randn_like(k)
        store.write(0, layer, k, v, n=len(tokens))
    store.register_prefix(0, tokens)

    tokens2 = tokens + [99, 100, 101, 102, 103]  # shares 3 full blocks
    shared = store.adopt(1, tokens2)
    assert shared == 24
    # adopted seq shares physical blocks with the original
    assert store.seqs[0]["blocks"][:3] == store.seqs[1]["blocks"][:3]

    # releasing the original must not free the adopted blocks (cache + seq hold refs)
    store.release(0)
    assert store.seqs[1]["blocks"][:3] == [0, 1, 2]

    # adopted blocks still gather the original values
    k_g, _ = store.gather(1, 0)
    assert k_g.shape[1] == 24


def test_prefix_partial_and_dedup(store):
    bs = store.block_size
    tokens = list(range(bs * 2))  # 2 full blocks
    store.init_seq(0)
    store.reserve(0, len(tokens))
    for layer in range(store.cfg.num_layers):
        k = torch.randn(len(tokens), store.cfg.num_kv_heads, store.cfg.head_dim)
        v = torch.randn_like(k)
        store.write(0, layer, k, v, n=len(tokens))
    store.register_prefix(0, tokens)
    # new seq sharing only the first full block
    tokens2 = list(range(bs)) + [200] * (bs + 3)
    shared = store.adopt(2, tokens2)
    assert shared == bs
    assert len(store.seqs[2]["blocks"]) == 1