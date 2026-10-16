"""Model parallelism (Ch5): tensor parallelism by sharding the linear layers.

The trick that makes this drop-in: the rest of the model calls each layer the
same way (``q(x)``, ``o(attn)``, ...), so we just swap each ``nn.Linear`` for a
sharded version that computes all shards and combines them. In production the
shards live on separate GPUs and the combine step is an AllGather (column
parallel) or AllReduce (row parallel); here we demonstrate the exact same math,
testable on a single device.

Two split styles, both must reproduce ``y = x @ W.T + b`` exactly:

- Column parallel: split W along its *output* rows, each shard computes part of
  y, then concatenate. Used for q, k, v, gate, up (and lm_head).
- Row parallel: split W along its *input* columns, split x, each shard computes
  a partial y, then sum (the AllReduce). Used for o and down.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _shards(weight: torch.Tensor, n_shards: int, dim: int, devices):
    size = weight.shape[dim]
    assert size % n_shards == 0, f"dim {dim} ({size}) not divisible by {n_shards} shards"
    step = size // n_shards
    idx = torch.arange(size)
    return [
        weight.index_select(dim, idx[i * step:(i + 1) * step]).detach().clone().to(devices[i])
        for i in range(n_shards)
    ]


class ColumnParallelLinear(nn.Module):
    """Split across output rows; forward concatenates each shard's output."""

    def __init__(self, weight: torch.Tensor, bias, n_shards: int, devices):
        super().__init__()
        self.n_shards = n_shards
        self.devices = list(devices)
        self.w_shards = nn.ParameterList([nn.Parameter(w) for w in _shards(weight, n_shards, 0, devices)])
        self.b_shards = None
        if bias is not None:
            self.b_shards = [bias[i * step:(i + 1) * step].detach().clone().to(devices[i])
                             for i, step in enumerate([weight.shape[0] // n_shards] * n_shards)]

    def forward(self, x: torch.Tensor):
        outs = []
        for i in range(self.n_shards):
            xi = x.to(self.devices[i])
            b = self.b_shards[i] if self.b_shards is not None else None
            outs.append(F.linear(xi, self.w_shards[i], b))
        return torch.cat(outs, dim=-1).to(self.devices[0])


class RowParallelLinear(nn.Module):
    """Split across input columns; forward sums each shard's partial output."""

    def __init__(self, weight: torch.Tensor, bias, n_shards: int, devices):
        super().__init__()
        self.n_shards = n_shards
        self.devices = list(devices)
        self.w_shards = nn.ParameterList([nn.Parameter(w) for w in _shards(weight, n_shards, 1, devices)])
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x: torch.Tensor):
        parts = torch.split(x, x.shape[-1] // self.n_shards, dim=-1)
        acc = None
        for i in range(self.n_shards):
            part = parts[i].to(self.devices[i])
            y = F.linear(part, self.w_shards[i], None)
            acc = y if acc is None else acc + y
        acc = acc.to(self.devices[0])
        return acc if self.bias is None else acc + self.bias.to(acc.device)


# attribute names that are column-parallel (split output) vs row-parallel (split input)
_COLUMN = {"q", "k", "v", "gate", "up", "lm_head"}
_ROW = {"o", "down"}


def make_tensor_parallel(model: nn.Module, n_shards: int, devices=None) -> nn.Module:
    """Replace every nn.Linear with a sharded parallel linear (in place)."""
    if devices is None:
        devices = [None] * n_shards
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            style = "row" if name in _ROW else "column"
            sharded = (
                RowParallelLinear(child.weight.data, child.bias, n_shards, devices)
                if style == "row" else
                ColumnParallelLinear(child.weight.data, child.bias, n_shards, devices)
            )
            setattr(model, name, sharded)
        else:
            make_tensor_parallel(child, n_shards, devices)
    return model