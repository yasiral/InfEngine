import threading
import time

import torch

from .model import precompute_freqs
from .kv_store import PagedKVStore
from .scheduler import Scheduler, Request
from .sampling import sample
from .config import ModelConfig


class AdmissionError(RuntimeError):
    """The engine is not admitting new requests (e.g. a DRAIN is active)."""


def _single_step(model, store, sid, token, position, cos, sin, device, pad_id):
    """Run one decode step: write ``token`` at ``position``, return logits
    predicting the next token. Shared by the target and draft models."""
    input_ids = torch.full((1, 1), pad_id, device=device)
    input_ids[0, 0] = token
    positions = torch.tensor([[position]], device=device)
    store.reserve(sid, 1)
    L = store.length(sid)
    t = torch.arange(L, device=device)[None, None, :]
    lens = torch.tensor([L], device=device).view(1, 1, 1)
    mask = (t <= positions[:, :, None]) & (t < lens)
    return model.forward(input_ids, store, [sid], positions, mask, cos, sin,
                         n_valid=torch.tensor([1], device=device))


class LLMEngine:
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        dtype: torch.dtype,
        cos=None,
        sin=None,
        block_size: int = 64,
        num_blocks: int = 512,
        max_batch: int = 4,
        chunk_size: int = 256,
        eos_id: int = 0,
        pad_id: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.cfg: ModelConfig = model.cfg
        self.pad_id = pad_id
        self.eos_id = eos_id
        if tokenizer:
            self.pad_id = tokenizer.pad_token_id
            self.eos_id = tokenizer.eos_token_id

        if cos is None or sin is None:
            cos, sin = precompute_freqs(self.cfg.head_dim, self.cfg.max_seq_len,
                                        self.cfg.rope_theta, dtype, device)
        self.cos = cos
        self.sin = sin

        self.store = PagedKVStore(self.cfg, block_size, num_blocks, dtype, device)
        self.scheduler = Scheduler(max_batch=max_batch, chunk_size=chunk_size)
        # Guards each scheduler iteration. torch releases the GIL during compute,
        # so concurrent generate() callers could otherwise interleave mid-forward.
        # With the lock, iterations serialize but each still batches ALL pending
        # requests; tokens are demultiplexed through per-request queues.
        self._lock = threading.Lock()
        # Authoritative admission gate. Closed by DRAIN so no new request can
        # enter scheduler/KV state; checked under _lock atomically with enqueue.
        self._admitting = True
        self.metrics = {
            "requests": 0,
            "tokens": 0,
            "prefill_steps": 0,
            "decode_steps": 0,
            "prefill_time_s": 0.0,
            "decode_time_s": 0.0,
            "total_time_s": 0.0,
            "kv_blocks_used": 0,
        }

    @property
    def admitting(self) -> bool:
        return self._admitting

    def _begin_request(self, ids, *, max_new_tokens, temperature, top_p):
        """Atomically: check admission, adopt KV state, and enqueue.

        Runs under ``_lock`` so it is atomic with respect to DRAIN's quiescence
        checks: a request is either fully admitted before DRAIN observes an
        empty scheduler, or rejected because admission is already closed. Never
        leaves a request adopted into a KV store that DRAIN has discarded.
        """
        with self._lock:
            if not self._admitting:
                raise AdmissionError("engine not admitting; drain in progress")
            sid = id(ids)
            shared = self.store.adopt(sid, ids)
            if shared == len(ids) and len(ids) > 0:
                # Fully prefix-cached prompt: keep the final token to re-prefill
                # so we have prediction logits. Otherwise _do_prefill sees a
                # zero-length chunk and builds a T=0 tensor.
                self.store.rollback(sid, 1)
                shared = len(ids) - 1
            req = Request(
                seq_id=sid,
                prompt_ids=ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                prompt_consumed=shared,
            )
            self.scheduler.add(req)
            return req

    def _end_request(self, sid):
        """Release a sequence's KV blocks and drop its scheduler entry."""
        with self._lock:
            self.store.release(sid)
            self.scheduler.drop(sid)

    def reconfigure_kv(self, block_size=None, num_blocks=None):
        """Rebuild the KV pools with new sizing (DRAIN-class transition).

        Refuses while requests are in flight so a live pool is never mutated
        underneath running sequences. Returns False if the engine is busy.
        Unspecified dimensions keep their current value.
        """
        with self._lock:
            if self.scheduler.requests:
                return False
            bs = block_size if block_size is not None else self.store.block_size
            nb = num_blocks if num_blocks is not None else (
                self.store.k_pool[0].shape[0] if self.store.k_pool else 0
            )
            self.store = PagedKVStore(self.cfg, bs, nb, self.dtype, self.device)
            return True

    def generate(self, prompt, max_new_tokens=128, temperature=0.0, top_p=1.0):
        if isinstance(prompt, str):
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            ids = list(prompt)
        if not ids:
            raise ValueError("empty prompt not supported")

        req = self._begin_request(ids, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_p=top_p)
        sid = req.seq_id

        t0 = time.perf_counter()
        try:
            while not req.finished or req.token_queue:
                with self._lock:
                    pgroup = self.scheduler.prefill_group()
                    if pgroup:
                        p0 = time.perf_counter()
                        self._do_prefill(pgroup)
                        self.metrics["prefill_time_s"] += time.perf_counter() - p0
                        self.metrics["prefill_steps"] += 1
                        # queue the first token each request produces when its
                        # prompt is consumed (prefill samples it; decode yields the rest)
                        for r in pgroup:
                            if r.num_gen == 1:
                                r.token_queue.append(r.output_ids[0])

                    dgroup = self.scheduler.decode_group()
                    if dgroup:
                        d0 = time.perf_counter()
                        yielded = self._do_decode(dgroup)
                        self.metrics["decode_time_s"] += time.perf_counter() - d0
                        self.metrics["decode_steps"] += 1
                        for r, t in zip(dgroup, yielded):
                            r.token_queue.append(t)

                    for r in self.scheduler.requests.values():
                        if not r.finished:
                            self.store.register_prefix(r.seq_id, r.prompt_ids + r.output_ids)

                    done = [r for r in self.scheduler.requests.values() if r.finished]
                    for r in done:
                        self.store.release(r.seq_id)
                        self.scheduler.drop(r.seq_id)

                while req.token_queue:
                    yield req.token_queue.popleft()
                    self.metrics["tokens"] += 1
        finally:
            # Reliable cleanup on success, failure, or abandoned generator,
            # synchronized with the engine state DRAIN/KV reconfiguration
            # inspect. release/drop are idempotent after normal completion.
            self._end_request(sid)
            self.metrics["requests"] += 1
            self.metrics["total_time_s"] += time.perf_counter() - t0
            with self._lock:
                self.metrics["kv_blocks_used"] = max(self.metrics["kv_blocks_used"], self.store.used_blocks)

    def generate_speculative(self, prompt, draft_model, draft_tokenizer, spec_len=4, max_new_tokens=128):
        """Speculative decoding (Ch5): a small draft model proposes ``spec_len``
        tokens, the target verifies them in one parallel forward pass, and we keep
        the accepted prefix. Greedy (temperature 0) verification is exact."""
        device = self.device
        dtype = self.dtype

        if isinstance(prompt, str):
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            ids = list(prompt)
        T0 = len(ids)

        # --- prefill target, remembering the last prompt token's logits ---
        with self._lock:
            if not self._admitting:
                raise AdmissionError("engine not admitting; drain in progress")
            sid = id(ids)
            shared = self.store.adopt(sid, ids)
        off = shared if shared < T0 else 0  # fully cached -> re-prefill for logits
        chunk = ids[off:]
        T = len(chunk)
        input_ids = torch.full((1, T), self.pad_id, device=device)
        input_ids[0] = torch.tensor(chunk, device=device)
        positions = (off + torch.arange(T, device=device)).unsqueeze(0)
        self.store.reserve_batch([sid], [T])
        L = self.store.length(sid)
        t = torch.arange(L, device=device)[None, None, :]
        mask = (t <= positions[:, :, None]) & (t < L)
        logits = self.model.forward(input_ids, self.store, [sid], positions, mask, self.cos, self.sin,
                                    n_valid=torch.tensor([T], device=device))
        self.store.register_prefix(sid, ids)
        last_logits = logits[:, -1:, :]  # predicts the token at position T0

        # --- draft model (fresh context each iteration for simplicity) ---
        d_cos, d_sin = precompute_freqs(draft_model.cfg.head_dim, draft_model.cfg.max_seq_len,
                                        draft_model.cfg.rope_theta, dtype, device)

        gen = 0
        out = []
        t0 = time.perf_counter()
        try:
            while gen < max_new_tokens:
                cur_pos = T0 + gen
                k = min(spec_len, max_new_tokens - gen)

                # 1) draft k tokens with the small model: prefill it over the full
                # accepted context, then walk forward greedily.
                d0 = time.perf_counter()
                ctx = ids + out
                d_sid = id(ctx)
                draft_store = PagedKVStore(draft_model.cfg, self.store.block_size, 256, dtype, device)
                draft_store.init_seq(d_sid)
                d_T = len(ctx)
                d_ids = torch.full((1, d_T), self.pad_id, device=device)
                d_ids[0] = torch.tensor(ctx, device=device)
                d_pos = torch.arange(d_T, device=device).unsqueeze(0)
                draft_store.reserve_batch([d_sid], [d_T])
                dL = draft_store.length(d_sid)
                dt = torch.arange(dL, device=device)[None, None, :]
                d_mask = (dt <= d_pos[:, :, None]) & (dt < dL)
                d_logits = draft_model.forward(d_ids, draft_store, [d_sid], d_pos, d_mask, d_cos, d_sin,
                                               n_valid=torch.tensor([d_T], device=device))
                d_last_logits = d_logits[:, -1:, :]
                drafts = [d_last_logits[0, 0].argmax().item()]
                for j in range(1, k):
                    dl = _single_step(draft_model, draft_store, d_sid, drafts[j - 1],
                                      cur_pos + (j - 1), d_cos, d_sin, device, self.pad_id)
                    drafts.append(sample(dl, 0.0, 1.0).item())
                draft_store.release(d_sid)
                self.metrics["prefill_time_s"] += time.perf_counter() - d0

                # 2) verify all k drafts in ONE parallel target forward
                cand = torch.full((1, k), self.pad_id, device=device)
                cand[0] = torch.tensor(drafts, device=device)
                cpos = (cur_pos + torch.arange(k, device=device)).unsqueeze(0)
                self.store.reserve_batch([sid], [k])
                L = self.store.length(sid)
                t = torch.arange(L, device=device)[None, None, :]
                mask = (t <= cpos[:, :, None]) & (t < L)
                vlogits = self.model.forward(cand, self.store, [sid], cpos, mask, self.cos, self.sin,
                                             n_valid=torch.tensor([k], device=device))

                # 3) accept greedily up to first disagreement
                m = 0
                rejected = False
                while m < k:
                    pred = last_logits[0, 0].argmax().item()
                    if pred == drafts[m]:
                        out.append(drafts[m]); gen += 1
                        yield drafts[m]
                        self.metrics["tokens"] += 1
                        m += 1
                        last_logits = vlogits[:, m - 1:m, :]  # predicts cur_pos + m
                        if drafts[m - 1] == self.eos_id or gen >= max_new_tokens:
                            break
                    else:
                        out.append(pred); gen += 1
                        yield pred
                        self.metrics["tokens"] += 1
                        rejected = True
                        break

                if rejected:
                    # keep a_0..a_{m-1}, drop the rest, write the target token at cur_pos+m
                    self.store.rollback(sid, k - m)
                    pred = out[-1]
                    last_logits = _single_step(self.model, self.store, sid, pred, cur_pos + m,
                                               self.cos, self.sin, device, self.pad_id)
                    if pred == self.eos_id or gen >= max_new_tokens:
                        break
                elif m == k:
                    # all drafts accepted -> bonus token from the target
                    if gen < max_new_tokens and out[-1] != self.eos_id:
                        bonus = last_logits[0, 0].argmax().item()
                        out.append(bonus); gen += 1
                        yield bonus
                        self.metrics["tokens"] += 1
                        last_logits = _single_step(self.model, self.store, sid, bonus, cur_pos + k,
                                                   self.cos, self.sin, device, self.pad_id)
                        if bonus == self.eos_id or gen >= max_new_tokens:
                            break
                else:
                    # stopped early (eos / token cap) before using all drafts
                    self.store.rollback(sid, k - m)
        finally:
            with self._lock:
                self.store.release(sid)
            self.metrics["requests"] += 1
            self.metrics["total_time_s"] += time.perf_counter() - t0
            with self._lock:
                self.metrics["kv_blocks_used"] = max(self.metrics["kv_blocks_used"], self.store.used_blocks)

    def _do_prefill(self, group):
        B = len(group)
        chunk_lens = [min(r.remaining_prompt, self.scheduler.chunk_size) for r in group]
        T = max(chunk_lens)
        offsets = [r.prompt_consumed for r in group]

        input_ids = torch.full((B, T), self.pad_id, device=self.device)
        positions = torch.zeros(B, T, dtype=torch.long, device=self.device)

        for i, r in enumerate(group):
            chunk = r.prompt_ids[offsets[i] : offsets[i] + chunk_lens[i]]
            for j, tok in enumerate(chunk):
                input_ids[i, j] = tok
                positions[i, j] = offsets[i] + j

        for i in range(B):
            if chunk_lens[i] < T:
                for j in range(chunk_lens[i], T):
                    positions[i, j] = 0

        self.store.reserve_batch([r.seq_id for r in group], chunk_lens)

        Lmax = max(offsets[i] + chunk_lens[i] for i in range(B))
        device = self.device
        qpos = positions[:, :, None]
        t = torch.arange(Lmax, device=device)[None, None, :]
        mask = (t <= qpos) & (t < torch.tensor([offsets[i] + chunk_lens[i] for i in range(B)], device=device).view(B, 1, 1))

        for i in range(B):
            if chunk_lens[i] < T:
                mask[i, chunk_lens[i]:, :] = False

        logits = self.model.forward(
            input_ids, self.store, [r.seq_id for r in group],
            positions, mask, self.cos, self.sin,
            n_valid=torch.tensor(chunk_lens, device=device),
        )

        for i, r in enumerate(group):
            r.prompt_consumed += chunk_lens[i]
            if r.prompt_consumed >= r.prompt_len:
                r.status = "decode"
                tok = sample(logits[i, chunk_lens[i] - 1].unsqueeze(0), r.temperature, r.top_p).item()
                r.output_ids.append(tok)
                r.num_gen += 1
                if tok == self.eos_id or r.num_gen >= r.max_new_tokens:
                    r.status = "done"

    def _do_decode(self, group):
        B = len(group)
        input_ids = torch.tensor([[r.output_ids[-1] for r in group]], device=self.device).T  # [B,1]
        positions = torch.tensor([r.prompt_len + r.num_gen - 1 for r in group], device=self.device).view(B, 1)

        Lmax = max(self.store.length(r.seq_id) for r in group)
        self.store.reserve_batch([r.seq_id for r in group], [1] * B)
        Lmax = max(self.store.length(r.seq_id) for r in group)

        device = self.device
        t = torch.arange(Lmax, device=device)[None, None, :]
        lens = torch.tensor([self.store.length(r.seq_id) for r in group], device=device).view(B, 1, 1)
        mask = (t <= positions[:, :, None]) & (t < lens)

        logits = self.model.forward(
            input_ids, self.store, [r.seq_id for r in group],
            positions, mask, self.cos, self.sin,
            n_valid=torch.tensor([1] * B, device=device),
        )

        toks = sample(logits.squeeze(1), group[0].temperature, group[0].top_p).squeeze(-1)
        out = []
        for i, r in enumerate(group):
            tok = toks[i].item()
            r.output_ids.append(tok)
            r.num_gen += 1
            out.append(tok)
            if tok == self.eos_id or r.num_gen >= r.max_new_tokens:
                r.status = "done"
        return out