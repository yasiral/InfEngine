from collections import deque
from dataclasses import dataclass, field


@dataclass
class Request:
    seq_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    prompt_consumed: int = 0
    output_ids: list[int] = field(default_factory=list)
    status: str = "prefill"
    num_gen: int = 0
    token_queue: deque = field(default_factory=deque)

    @property
    def prompt_len(self):
        return len(self.prompt_ids)

    @property
    def remaining_prompt(self):
        return self.prompt_len - self.prompt_consumed

    @property
    def finished(self):
        return self.status == "done"


class Scheduler:
    def __init__(self, max_batch: int = 4, chunk_size: int = 256):
        self.max_batch = max_batch
        self.chunk_size = chunk_size
        self.requests: dict[int, Request] = {}

    def add(self, req: Request):
        self.requests[req.seq_id] = req

    def drop(self, sid: int):
        self.requests.pop(sid, None)

    def _pending(self, status: str):
        return [r for r in self.requests.values() if r.status == status and not r.finished]

    def prefill_group(self):
        return self._pending("prefill")[: self.max_batch]

    def decode_group(self):
        return self._pending("decode")[: self.max_batch]

    def has_work(self):
        return any(not r.finished for r in self.requests.values())