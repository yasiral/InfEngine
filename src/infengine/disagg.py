"""Disaggregation (Ch5): split prefill and decode into separate engines.

Production systems put prefill and decode on different GPU pools so one pool
isn't blocked doing long prompts while the other idles. The contract between
them is the KV cache: prefill produces it, decode consumes it. Here a
``PrefillEngine`` returns a ``KVHandle`` (the gathered KV tensors + last
logits), and a ``DecodeEngine`` rebuilds its own KV store from that handle and
continues generation — decoupled from how/where the prefill ran.
"""

from dataclasses import dataclass, field

import torch

from .config import ModelConfig
from .kv_store import PagedKVStore
from .engine import _single_step
from .model import precompute_freqs


@dataclass
class KVHandle:
    cfg: ModelConfig
    k: list = field(default_factory=list)   # per layer [L, kv_heads, head_dim]
    v: list = field(default_factory=list)
    length: int = 0
    last_logits: torch.Tensor = None        # (1,1,vocab) predicting the next token
    cos: torch.Tensor = None
    sin: torch.Tensor = None
    pad_id: int = 0
    eos_id: int = 0
    dtype: torch.dtype = None
    device: torch.device = None


class PrefillEngine:
    """Runs the prompt through the model and hands off the resulting KV."""

    def __init__(self, model, tokenizer, device, dtype, block_size=64, num_blocks=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.cfg = model.cfg
        self.pad_id = tokenizer.pad_token_id if tokenizer else 0
        self.eos_id = tokenizer.eos_token_id if tokenizer else 0
        self.cos, self.sin = precompute_freqs(self.cfg.head_dim, self.cfg.max_seq_len,
                                              self.cfg.rope_theta, dtype, device)
        self.block_size = block_size
        self.num_blocks = num_blocks

    def prefill(self, prompt) -> KVHandle:
        if isinstance(prompt, str):
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            ids = list(prompt)
        T = len(ids)
        sid = id(ids)
        store = PagedKVStore(self.cfg, self.block_size, self.num_blocks, self.dtype, self.device)
        store.init_seq(sid)
        store.reserve(sid, T)

        input_ids = torch.full((1, T), self.pad_id, device=self.device)
        input_ids[0] = torch.tensor(ids, device=self.device)
        positions = torch.arange(T, device=self.device).unsqueeze(0)
        t = torch.arange(T, device=self.device)[None, None, :]
        mask = (t <= positions[:, :, None]) & (t < T)
        logits = self.model.forward(input_ids, store, [sid], positions, mask,
                                    self.cos, self.sin, n_valid=torch.tensor([T], device=self.device))

        k = [store.gather(sid, layer)[0].transpose(0, 1) for layer in range(self.cfg.num_layers)]
        v = [store.gather(sid, layer)[1].transpose(0, 1) for layer in range(self.cfg.num_layers)]
        store.release(sid)

        return KVHandle(
            cfg=self.cfg, k=k, v=v, length=T,
            last_logits=logits[:, -1:, :], cos=self.cos, sin=self.sin,
            pad_id=self.pad_id, eos_id=self.eos_id, dtype=self.dtype, device=self.device,
        )


class DecodeEngine:
    """Rebuilds a KV store from a KVHandle and continues decoding."""

    def __init__(self, model, tokenizer, device, dtype, block_size=64, num_blocks=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.num_blocks = num_blocks

    def decode(self, handle: KVHandle, max_new_tokens: int):
        cfg = handle.cfg
        store = PagedKVStore(cfg, self.block_size, self.num_blocks, handle.dtype, handle.device)
        sid = id(handle.k)
        store.init_seq(sid)
        store.reserve(sid, handle.length)
        for layer in range(cfg.num_layers):
            store.write(sid, layer, handle.k[layer], handle.v[layer], handle.length)

        last_logits = handle.last_logits
        tok = last_logits[0, 0].argmax().item()
        gen = 1
        yield tok
        try:
            while gen < max_new_tokens and tok != handle.eos_id:
                pos = handle.length + gen - 1
                last_logits = _single_step(self.model, store, sid, tok, pos,
                                           handle.cos, handle.sin, handle.device, handle.pad_id)
                tok = last_logits[0, 0].argmax().item()
                gen += 1
                yield tok
        finally:
            store.release(sid)