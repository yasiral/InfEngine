from collections import deque
import torch

from .config import ModelConfig


class PagedKVStore:
    """Block-pooled KV cache with prefix (KV-reuse) caching.

    Blocks are reference counted so a block can be shared between the prefix
    cache and multiple live sequences. A full block of a sequence is registered
    in ``self.prefix`` keyed by its token tuple; future requests whose prompt
    shares that exact token prefix adopt the same physical blocks and only
    prefill the unseen suffix.
    """

    def __init__(self, cfg: ModelConfig, block_size: int, num_blocks: int, dtype: torch.dtype, device: torch.device):
        self.cfg = cfg
        self.block_size = block_size
        self.device = device
        self.free_blocks = deque(range(num_blocks))
        self.seqs: dict[int, dict] = {}
        self.block_refs: dict[int, int] = {}
        hd = cfg.head_dim
        kv = cfg.num_kv_heads
        self.k_pool = [
            torch.zeros(num_blocks, kv, block_size, hd, dtype=dtype, device=device)
            for _ in range(cfg.num_layers)
        ]
        self.v_pool = [
            torch.zeros(num_blocks, kv, block_size, hd, dtype=dtype, device=device)
            for _ in range(cfg.num_layers)
        ]
        self.prefix: dict[tuple, int] = {}
        self._registered: dict[int, int] = {}
        self._count = 0

    def _alloc(self) -> int:
        if not self.free_blocks:
            raise RuntimeError(f"KV block pool exhausted")
        b = self.free_blocks.popleft()
        self.block_refs[b] = 1
        return b

    def _free(self, b: int):
        self.block_refs[b] -= 1
        if self.block_refs[b] <= 0:
            self.free_blocks.append(b)
            del self.block_refs[b]

    def init_seq(self, sid: int, blocks=None, length: int = 0):
        blocks = list(blocks) if blocks is not None else []
        self.seqs[sid] = {"blocks": blocks, "len": length}
        self._registered[sid] = length // self.block_size
        for b in blocks:
            self.block_refs[b] = self.block_refs.get(b, 0) + 1
        self._count += 1

    def adopt(self, sid: int, tokens) -> int:
        """Start a sequence reusing the longest cached prefix of ``tokens``.

        Returns the number of leading tokens covered by reused blocks.
        """
        shared, blocks = self.lookup_prefix(tokens)
        if shared:
            self.init_seq(sid, blocks=blocks, length=shared)
        else:
            self.init_seq(sid)
        return shared

    def lookup_prefix(self, tokens):
        bs = self.block_size
        full = len(tokens) // bs
        blocks = []
        for j in range(full):
            blk = tuple(tokens[j * bs:(j + 1) * bs])
            bid = self.prefix.get(blk)
            if bid is None:
                break
            blocks.append(bid)
        return len(blocks) * bs, blocks

    def register_prefix(self, sid: int, tokens):
        """Register the sequence's completed full blocks into the prefix cache."""
        st = self.seqs.get(sid)
        if st is None:
            return
        bs = self.block_size
        total_full = min(len(tokens) // bs, st["len"] // bs)
        start = self._registered.get(sid, 0)
        for j in range(start, total_full):
            blk = tuple(tokens[j * bs:(j + 1) * bs])
            if blk not in self.prefix:
                bid = st["blocks"][j]
                self.prefix[blk] = bid
                self.block_refs[bid] = self.block_refs.get(bid, 0) + 1
        self._registered[sid] = total_full

    def reserve(self, sid: int, n: int):
        st = self.seqs[sid]
        self._ensure_capacity(sid, st["len"] + n)
        st["len"] += n

    def reserve_batch(self, seq_ids, n_valid):
        for sid, n in zip(seq_ids, n_valid):
            self.reserve(sid, n)

    def release(self, sid: int):
        st = self.seqs.pop(sid, None)
        self._registered.pop(sid, None)
        if st is None:
            return
        for b in st["blocks"]:
            self._free(b)
        self._count -= 1

    def _ensure_capacity(self, sid: int, need: int):
        st = self.seqs[sid]
        while need > len(st["blocks"]) * self.block_size:
            st["blocks"].append(self._alloc())

    def length(self, sid: int):
        return self.seqs[sid]["len"]

    def rollback(self, sid: int, n: int):
        """Trim ``n`` most-recently-written tokens (speculative rejections)."""
        st = self.seqs[sid]
        st["len"] -= n
        if st["len"] < 0:
            st["len"] = 0

    def write(self, sid: int, layer: int, k: torch.Tensor, v: torch.Tensor, n: int):
        st = self.seqs[sid]
        offset = st["len"] - n
        need = offset + n
        blocks = torch.tensor(st["blocks"], device=self.device)
        g = torch.arange(offset, need, device=self.device)
        b = blocks[g // self.block_size]
        o = g % self.block_size
        self.k_pool[layer][b, :, o, :] = k[:n]
        self.v_pool[layer][b, :, o, :] = v[:n]

    def write_batch(self, seq_ids, layer: int, k: torch.Tensor, v: torch.Tensor, n_valid):
        for i, sid in enumerate(seq_ids):
            kw = k[i].permute(1, 0, 2)
            vw = v[i].permute(1, 0, 2)
            self.write(sid, layer, kw, vw, n_valid[i])

    def gather(self, sid: int, layer: int):
        st = self.seqs.get(sid)
        if st is None or st["len"] == 0:
            return None, None
        blocks = torch.tensor(st["blocks"], device=self.device)
        nb = len(st["blocks"])
        kv = self.cfg.num_kv_heads
        k = self.k_pool[layer][blocks].permute(1, 0, 2, 3).reshape(kv, nb * self.block_size, -1)[:, :st["len"]]
        v = self.v_pool[layer][blocks].permute(1, 0, 2, 3).reshape(kv, nb * self.block_size, -1)[:, :st["len"]]
        return k, v

    def gather_batch(self, seq_ids, layer: int):
        ks, vs, lens = [], [], []
        ml = max(self.seqs[s]["len"] for s in seq_ids)
        for s in seq_ids:
            k, v = self.gather(s, layer)
            L = k.shape[1]
            if L < ml:
                pad = torch.zeros(k.shape[0], ml - L, k.shape[2], dtype=k.dtype, device=k.device)
                k = torch.cat((k, pad), 1)
                v = torch.cat((v, pad), 1)
            ks.append(k)
            vs.append(v)
            lens.append(L)
        return torch.stack(ks), torch.stack(vs), lens

    @property
    def used_blocks(self):
        return (self.k_pool[0].shape[0] if self.k_pool else 0) - len(self.free_blocks)
