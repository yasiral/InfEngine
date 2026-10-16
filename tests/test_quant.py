import torch
import torch.nn as nn

from infengine.config import ModelConfig
from infengine.model import LlamaModel, precompute_freqs
from infengine.quant import quantize_per_channel, quantize_model, WQLinear


def test_per_channel_roundtrip_error():
    w = torch.randn(64, 128)
    qw, scale = quantize_per_channel(w, bits=8)
    assert qw.dtype == torch.int8
    assert qw.shape == w.shape
    assert scale.shape == (64, 1)
    # relative reconstruction error should be small (per-channel INT8)
    w_rec = qw.to(torch.float32) * scale
    rel = (w - w_rec).abs().mean() / w.abs().mean()
    assert rel < 0.05


def test_quantize_model_replaces_linears():
    cfg = ModelConfig(hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2,
                      vocab_size=64, intermediate_size=64, max_seq_len=32)
    model = LlamaModel(cfg).to(dtype=torch.float32)
    quantize_model(model, bits=8)
    # every nn.Linear under the hood is now a WQLinear
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linears == []
    wqls = [m for m in model.modules() if isinstance(m, WQLinear)]
    # q,k,v,o in 1 layer + gate,up,down in 1 layer + lm_head = 8
    assert len(wqls) == 8
    for m in wqls:
        assert m.qweight.dtype == torch.int8


def test_quantized_forward_close_to_fp():
    cfg = ModelConfig(hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2,
                      vocab_size=64, intermediate_size=64, max_seq_len=32)
    torch.manual_seed(0)
    fp = LlamaModel(cfg).to(dtype=torch.float32)
    q = LlamaModel(cfg).to(dtype=torch.float32)
    q.load_state_dict(fp.state_dict())
    quantize_model(q, bits=8)

    ids = torch.randint(0, 64, (1, 8))
    cos, sin = precompute_freqs(cfg.head_dim, 32, 10000.0, torch.float32, "cpu")
    pos = torch.arange(8).unsqueeze(0)
    mask = (torch.arange(8)[None, None, :] <= pos[:, :, None]).bool()
    from infengine.kv_store import PagedKVStore
    from infengine.backend import resolve_device
    store = PagedKVStore(cfg, block_size=8, num_blocks=16, dtype=torch.float32, device="cpu")
    sid = 0
    store.init_seq(sid)
    store.reserve(sid, 8)

    lo = fp.forward(ids, store, [sid], pos, mask, cos, sin, n_valid=torch.tensor([8]))
    store2 = PagedKVStore(cfg, block_size=8, num_blocks=16, dtype=torch.float32, device="cpu")
    store2.init_seq(0)
    store2.reserve(0, 8)
    lq = q.forward(ids, store2, [0], pos, mask, cos, sin, n_valid=torch.tensor([8]))

    assert lo.shape == lq.shape
    # quantized logits close to fp (same argmax on a small model)
    assert (lo.argmax(-1) == lq.argmax(-1)).float().mean() >= 0.8
    # and the argmax of the final position matches exactly for determinism
    assert lo[0, -1].argmax() == lq[0, -1].argmax()
