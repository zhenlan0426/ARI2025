# Deprecated 2D rotary embedding implementation from commit bf25c22
import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import dynamic_rope_update
from transformers.models.qwen3.modeling_qwen3 import rotate_half, ROPE_INIT_FUNCTIONS


def apply_rotary_pos_emb(q, k, cos_expanded, sin_expanded, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    q_embed = (q * cos_expanded) + (rotate_half(q) * sin_expanded)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3RotaryEmbedding2d(nn.Module):
    def __init__(self, config: Qwen3Config, max_head_freq=100, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        mask = torch.arange(len(inv_freq)) % 2 == 0
        freq_x = inv_freq.clone()
        freq_y = inv_freq.clone()
        freq_x[~mask] = 0.0
        freq_y[mask] = 0.0
        self.register_buffer("freq_x", freq_x)
        self.register_buffer("freq_y", freq_y)
        self.head_freq = torch.linspace(1, max_head_freq, 8)

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        rows, cols = position_ids
        rows = rows[None, :, None].float()
        cols = cols[None, :, None].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freq_x = self.freq_x.float()[None, None, :].to(x.device)
            freq_y = self.freq_y.float()[None, None, :].to(x.device)
            angles = rows * freq_x + cols * freq_y
            emb = torch.cat((angles, angles), dim=-1)[:, None]
            emb = self.head_freq[None, :, None, None].to(x.device) * emb
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            cos_expanded = cos.repeat_interleave(4, dim=1)
            sin_expanded = sin.repeat_interleave(4, dim=1)
        return cos_expanded.to(dtype=x.dtype), sin_expanded.to(dtype=x.dtype), cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
