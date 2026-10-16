from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    intermediate_size: int
    max_seq_len: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads