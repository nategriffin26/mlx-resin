"""Routed and shared experts; attention and recurrent execution use the dense family."""

import mlx.core as mx
import mlx.nn as nn

from resin_models_mlx.qwen3_5.model import (
    MLP,
    Attention,
    DecoderLayer,
    GatedDeltaNet,
    Qwen35ForCausalLM,
    RMSNorm,
    TextModel,
)


def route(logits, count):
    """Normalize only the selected experts; tied experts follow MLX argpartition."""
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    indices = mx.argpartition(probabilities, kth=-count, axis=-1)[..., -count:]
    scores = mx.take_along_axis(probabilities, indices, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    return indices, scores.astype(logits.dtype)


class ExpertLinear(nn.Module):
    def __init__(self, inputs, outputs, experts):
        super().__init__()
        self.weight = mx.zeros((experts, outputs, inputs))

    def __call__(self, hidden, indices, sorted_indices=False):
        return mx.gather_mm(
            hidden,
            self.weight.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )


class Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, width, count = (
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
        )
        self.gate_proj = ExpertLinear(hidden, width, count)
        self.up_proj = ExpertLinear(hidden, width, count)
        self.down_proj = ExpertLinear(width, hidden, count)

    def __call__(self, hidden, indices):
        shape = indices.shape
        expanded = hidden[..., None, None, :]
        # Group long prefills by expert to avoid repeatedly loading the same weights.
        ordered = indices.size >= 64
        if ordered:
            order = mx.argsort(indices.reshape(-1))
            inverse = mx.argsort(order)
            expanded = expanded.reshape(-1, 1, hidden.shape[-1])[order // shape[-1]]
            indices = indices.reshape(-1)[order]
        gate = self.gate_proj(expanded, indices, ordered)
        up = self.up_proj(expanded, indices, ordered)
        output = self.down_proj(nn.silu(gate) * up, indices, ordered)
        if ordered:
            output = output[inverse].reshape(*shape, 1, hidden.shape[-1])
        return output.squeeze(-2)


class SparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = Experts(config)
        self.shared_expert = MLP(config)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def __call__(self, hidden, addition=None):
        indices, scores = route(self.gate(hidden), self.top_k)
        routed = (self.switch_mlp(hidden, indices) * scores[..., None]).sum(axis=-2)
        shared = self.shared_expert(hidden, addition)
        return routed + mx.sigmoid(self.shared_expert_gate(hidden)) * shared


class MoeDecoderLayer(DecoderLayer):
    def __init__(self, config, kind, layer):
        nn.Module.__init__(self)
        self.is_linear = kind == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(config, layer)
        else:
            self.self_attn = Attention(config, layer)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SparseMLP(config)


class MoeTextModel(TextModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        counts = {"linear_attention": 0, "full_attention": 0}
        for kind in config.layer_types:
            self.layers.append(MoeDecoderLayer(config, kind, counts[kind]))
            counts[kind] += 1
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Qwen35MoeForCausalLM(Qwen35ForCausalLM):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.model = MoeTextModel(config)
        self.tied = config.tie_word_embeddings
        if not self.tied:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
