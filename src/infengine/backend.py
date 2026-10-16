import torch

from .config import ModelConfig
from .model import LlamaModel
from .quant import quantize_model


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(device: torch.device):
    if device.type == "cuda":
        return torch.float16
    if torch.xpu.is_available():
        return torch.bfloat16
    return torch.float32


def build_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def build_model_from_hf(model_id: str, device=None, dtype=None, bits: int = 0):
    from transformers import AutoConfig, AutoModelForCausalLM

    if device is None:
        device = resolve_device()
    if dtype is None:
        dtype = resolve_dtype(device)

    hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    hf = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    )
    hf_sd = hf.state_dict()
    del hf

    attention_bias = bool(
        getattr(hf_cfg, "attention_bias", False)
        or f"model.layers.0.self_attn.q_proj.bias" in hf_sd
    )

    cfg = ModelConfig(
        hidden_size=hf_cfg.hidden_size,
        num_layers=hf_cfg.num_hidden_layers,
        num_heads=hf_cfg.num_attention_heads,
        num_kv_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
        vocab_size=hf_cfg.vocab_size,
        intermediate_size=hf_cfg.intermediate_size,
        max_seq_len=min(getattr(hf_cfg, "max_position_embeddings", 4096), 4096),
        rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
        rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-5),
        attention_bias=attention_bias,
    )

    model = LlamaModel(cfg).to(dtype=dtype, device="cpu")

    key_map = {
        "model.embed_tokens.weight": "embed.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for i in range(cfg.num_layers):
        key_map[f"model.layers.{i}.input_layernorm.weight"] = f"layers.{i}.input_norm.weight"
        key_map[f"model.layers.{i}.post_attention_layernorm.weight"] = f"layers.{i}.post_norm.weight"
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            key_map[f"model.layers.{i}.self_attn.{n}.weight"] = f"layers.{i}.attn.{n.replace('_proj', '')}.weight"
        if cfg.attention_bias:
            for n in ("q_proj", "k_proj", "v_proj"):
                key_map[f"model.layers.{i}.self_attn.{n}.bias"] = f"layers.{i}.attn.{n.replace('_proj', '')}.bias"
        for n in ("gate_proj", "up_proj", "down_proj"):
            key_map[f"model.layers.{i}.mlp.{n}.weight"] = f"layers.{i}.mlp.{n.replace('_proj', '')}.weight"

    sd = {}
    for hf_key, w in hf_sd.items():
        mapped = key_map.get(hf_key)
        if mapped is None:
            continue
        sd[mapped] = w.to(dtype=dtype)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")

    if bits and bits > 0:
        model = quantize_model(model, bits)
        # qweight buffers are int8 and must NOT be dtype-cast; move device only
        return model.to(device=device), cfg
    return model.to(device=device, dtype=dtype), cfg