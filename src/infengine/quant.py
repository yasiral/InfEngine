"""Weight-only quantization (Ch5: quantization).

Quantization shrinks a model by storing its weights in fewer bits than float32.
We use symmetric, per-channel, round-to-nearest (RTN) INT8:

  - symmetric: we store the magnitude only (zero-point == 0), so
      q = round(w / scale)          and       w ~= q * scale
  - per-channel: each output row of a weight matrix gets its own ``scale``,
    because different rows can have very different magnitudes.
  - weight-only: we quantize the *weights* only; the KV cache and activations
    stay in full precision. This is the safe first step.

INT8 (8 bits) holds values in [-128, 127]. The range [-127, 127] is split into
127 steps, so ``scale`` is "how much one integer step is worth" for that row.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_per_channel(weight: torch.Tensor, bits: int = 8):
    """Quantize a [out, in] weight matrix to symmetric per-channel INT*.

    Returns (qweight int, scale float). w ~= qweight * scale.
    """
    qmax = 2 ** (bits - 1) - 1  # 127 for int8
    # one scale per output row = largest abs value in that row / half-range
    amax = weight.abs().amax(dim=1, keepdim=True)
    scale = amax / qmax
    scale = scale.clamp(min=1e-12)  # avoid div-by-zero on all-zero rows
    qweight = torch.clamp(torch.round(weight / scale), -qmax - 1, qmax)
    qweight = qweight.to(torch.int8)
    return qweight, scale


class WQLinear(nn.Module):
    """Drop-in replacement for nn.Linear that stores quantized weights.

    Keeps the int8 weights + per-row scales in memory, then dequantizes on the
    fly in forward(). This is the naive (unfused) version: it trades compute for
    memory. A production kernel would dequantize inside the matmul instead.
    """

    def __init__(self, out_features: int, in_features: int, bits: int = 8, bias: bool = True, dtype=torch.float32):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.bits = bits
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.empty(out_features, 1, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 8):
        w = linear.weight.detach()
        qw, s = quantize_per_channel(w, bits)
        m = cls(w.shape[0], w.shape[1], bits, bias=linear.bias is not None, dtype=s.dtype)
        m.qweight.copy_(qw)
        m.scale.copy_(s)
        if linear.bias is not None:
            m.bias.data.copy_(linear.bias.detach())
        return m

    def forward(self, x: torch.Tensor):
        # dequantize [out, in] then do a normal linear layer
        w = self.qweight.to(x.dtype) * self.scale
        return F.linear(x, w, self.bias)


def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
    """Replace every nn.Linear in ``model`` with a quantized WQLinear."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, WQLinear.from_linear(child, bits))
        else:
            quantize_model(child, bits)
    return model
