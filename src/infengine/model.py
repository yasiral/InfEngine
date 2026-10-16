import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def precompute_freqs(head_dim: int, max_len: int, base: float, dtype: torch.dtype, device: torch.device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos[..., :half].unsqueeze(1)
    s = sin[..., :half].unsqueeze(1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        n, h, k = cfg.num_heads, cfg.head_dim, cfg.num_kv_heads
        self.q = nn.Linear(cfg.hidden_size, n * h, bias=cfg.attention_bias)
        self.k = nn.Linear(cfg.hidden_size, k * h, bias=cfg.attention_bias)
        self.v = nn.Linear(cfg.hidden_size, k * h, bias=cfg.attention_bias)
        self.o = nn.Linear(n * h, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                store, seq_ids, n_valid, attn_mask):
        B, T, _ = x.shape
        h, k = self.cfg.head_dim, self.cfg.num_kv_heads
        nh = self.cfg.num_heads

        q = self.q(x).view(B, T, nh, h).transpose(1, 2)
        kv = self.k(x).view(B, T, k, h).transpose(1, 2)
        v = self.v(x).view(B, T, k, h).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        kv = apply_rope(kv, cos, sin)

        store.write_batch(seq_ids, self.layer_id, kv, v, n_valid)

        kv_full, v_full, _ = store.gather_batch(seq_ids, self.layer_id)

        if k != nh:
            r = nh // k
            kv_full = kv_full.repeat_interleave(r, dim=1)
            v_full = v_full.repeat_interleave(r, dim=1)

        attn = F.scaled_dot_product_attention(q, kv_full, v_full, attn_mask=attn_mask.unsqueeze(1))
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o(attn)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_size
        i = cfg.intermediate_size
        self.gate = nn.Linear(d, i, bias=False)
        self.up = nn.Linear(d, i, bias=False)
        self.down = nn.Linear(i, d, bias=False)

    def forward(self, x: torch.Tensor):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg, layer_id)
        self.post_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, store, seq_ids, n_valid, attn_mask):
        r = self.attn(self.input_norm(x), cos, sin, store, seq_ids, n_valid, attn_mask)
        x = x + r
        return x + self.mlp(self.post_norm(x))


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, store, seq_ids, positions: torch.Tensor,
                attn_mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, n_valid):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        c = cos[positions]
        s = sin[positions]
        for layer in self.layers:
            x = layer(x, c, s, store, seq_ids, n_valid, attn_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits