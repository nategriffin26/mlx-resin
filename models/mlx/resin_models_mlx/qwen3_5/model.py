# Portions are adapted from Hugging Face Transformers (Apache-2.0).

"""Dense Qwen3.5 text layers with Engine-owned hybrid state."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from resin.backend.mlx import paged_attention
from resin_models_mlx.qwen3_5.delta import gated_delta_update


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.zeros(dim)
        self.eps = eps

    def __call__(self, hidden):
        # Add in FP32: rounding 1 + weight in BF16 changes Qwen's checkpoint semantics.
        return mx.fast.rms_norm(
            hidden.astype(mx.float32), 1 + self.weight.astype(mx.float32), self.eps
        ).astype(hidden.dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.layer = layer
        self.in_proj_qkv = nn.Linear(
            config.hidden_size, config.linear_conv_dim, bias=False
        )
        self.in_proj_z = nn.Linear(
            config.hidden_size, config.linear_value_dim, bias=False
        )
        self.in_proj_b = nn.Linear(
            config.hidden_size, config.linear_num_value_heads, bias=False
        )
        self.in_proj_a = nn.Linear(
            config.hidden_size, config.linear_num_value_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            config.linear_conv_dim,
            config.linear_conv_dim,
            config.linear_conv_kernel_dim,
            groups=config.linear_conv_dim,
            bias=False,
        )
        self.dt_bias = mx.ones(config.linear_num_value_heads)
        self.A_log = mx.zeros(config.linear_num_value_heads)
        self.norm = nn.RMSNorm(config.linear_value_head_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(
            config.linear_value_dim, config.hidden_size, bias=False
        )

    def __call__(self, hidden, cache):
        config = self.config
        batch, length, _ = hidden.shape
        keep = config.linear_conv_kernel_dim - 1
        if cache is None:
            convolution = mx.zeros((batch, keep, config.linear_conv_dim), hidden.dtype)
            recurrent = mx.zeros(
                (
                    batch,
                    config.linear_num_value_heads,
                    config.linear_value_head_dim,
                    config.linear_key_head_dim,
                )
            )
        else:
            convolution, recurrent = cache.state(self.layer)
        convolution = mx.concatenate(
            (convolution.astype(hidden.dtype), self.in_proj_qkv(hidden)), axis=1
        )
        mixed = nn.silu(self.conv1d(convolution))
        query, key, value = mx.split(
            mixed, [config.linear_key_dim, 2 * config.linear_key_dim], axis=-1
        )
        query = query.reshape(
            batch, length, config.linear_num_key_heads, config.linear_key_head_dim
        )
        key = key.reshape(query.shape)
        value = value.reshape(
            batch, length, config.linear_num_value_heads, config.linear_value_head_dim
        )
        # The recurrence expects normalized Q/K and linear-space decay.
        scale = config.linear_key_head_dim**-0.5
        query = (
            mx.fast.rms_norm(query.astype(mx.float32), None, 1e-6 * scale**2) * scale**2
        )
        key = mx.fast.rms_norm(key.astype(mx.float32), None, 1e-6 * scale**2) * scale
        decay = mx.exp(
            -mx.exp(self.A_log.astype(mx.float32))
            * nn.softplus(
                self.in_proj_a(hidden).astype(mx.float32)
                + self.dt_bias.astype(mx.float32)
            )
        )
        beta = mx.sigmoid(self.in_proj_b(hidden)).astype(mx.float32)
        # FP32 recurrence keeps the same accumulation and checkpoint dtype as HF.
        output, recurrent = gated_delta_update(
            query, key, value.astype(mx.float32), decay, beta, recurrent
        )
        if cache is not None:
            cache.store(self.layer, convolution[:, -keep:], recurrent)
        output = output.astype(hidden.dtype)
        gate = self.in_proj_z(hidden).reshape(output.shape)
        normalized = self.norm(output)
        output = (
            normalized.astype(mx.float32) * nn.silu(gate.astype(mx.float32))
        ).astype(hidden.dtype)
        return self.out_proj(output.reshape(batch, length, config.linear_value_dim))


class Attention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.layer = layer
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim * 2,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def __call__(self, hidden, positions, cache):
        config = self.config
        batch, length, _ = hidden.shape
        query, gate = mx.split(
            self.q_proj(hidden).reshape(
                batch, length, config.num_attention_heads, 2 * config.head_dim
            ),
            2,
            axis=-1,
        )
        query = self.q_norm(query)
        key = self.k_norm(
            self.k_proj(hidden).reshape(
                batch, length, config.num_key_value_heads, config.head_dim
            )
        )
        value = self.v_proj(hidden).reshape(key.shape)
        dim = config.rotary_dim
        frequencies = config.rope_theta ** (
            -mx.arange(0, dim, 2, dtype=mx.float32) / dim
        )
        angles = positions.astype(mx.float32)[..., None] * frequencies
        angles = mx.concatenate((angles, angles), axis=-1)[:, :, None]
        cos, sin = (
            mx.cos(angles).astype(hidden.dtype),
            mx.sin(angles).astype(hidden.dtype),
        )

        def rotate(tensor):
            rotary, passthrough = tensor[..., :dim], tensor[..., dim:]
            first, second = mx.split(rotary, 2, axis=-1)
            return mx.concatenate(
                (
                    rotary * cos + mx.concatenate((-second, first), axis=-1) * sin,
                    passthrough,
                ),
                axis=-1,
            )

        query, key = rotate(query), rotate(key)
        if cache is not None:
            cache.append(self.layer, key, value)
        if cache is not None and length == 1:
            output = paged_attention(
                query,
                key,
                value,
                cache.arena,
                0,
                self.layer,
                cache.history_slabs,
                cache.history_slots,
                cache.contexts,
                scale=config.head_dim**-0.5,
            )
        else:
            previous = cache.history(self.layer) if cache is not None else None
            mask = "causal"
            if previous is not None:
                past = previous.shape[1]
                key = mx.concatenate((previous[:, :, 0], key), axis=1)
                value = mx.concatenate((previous[:, :, 1], value), axis=1)
                valid = mx.arange(past)[None] < cache.contexts[:, None]
                mask = mx.concatenate(
                    (
                        mx.broadcast_to(valid[:, None], (batch, length, past)),
                        mx.broadcast_to(
                            mx.arange(length)[None] <= mx.arange(length)[:, None],
                            (batch, length, length),
                        ),
                    ),
                    axis=-1,
                )[:, None]
            output = mx.fast.scaled_dot_product_attention(
                query.transpose(0, 2, 1, 3),
                key.transpose(0, 2, 1, 3),
                value.transpose(0, 2, 1, 3),
                scale=config.head_dim**-0.5,
                mask=mask,
            ).transpose(0, 2, 1, 3)
        return self.o_proj((output * mx.sigmoid(gate)).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, hidden, addition):
        intermediate = nn.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        if addition is not None:
            intermediate = intermediate + addition[:, None].astype(intermediate.dtype)
        return self.down_proj(intermediate)


class DecoderLayer(nn.Module):
    def __init__(self, config, kind, layer):
        super().__init__()
        self.is_linear = kind == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(config, layer)
        else:
            self.self_attn = Attention(config, layer)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, hidden, positions, cache, addition, mlp_inputs=None):
        """``addition`` is ``(activation, residual)``, each ``[batch, width]`` or None.

        ``activation`` adds to ``silu(gate) * up``; ``residual`` adds to the MLP output.
        A list passed as ``mlp_inputs`` receives the normalized MLP input.
        """
        normalized = self.input_layernorm(hidden)
        hidden = hidden + (
            self.linear_attn(normalized, cache)
            if self.is_linear
            else self.self_attn(normalized, positions, cache)
        )
        normalized = self.post_attention_layernorm(hidden)
        if mlp_inputs is not None:
            mlp_inputs.append(normalized)
        activation, residual = addition if addition is not None else (None, None)
        output = self.mlp(normalized, activation)
        if residual is not None:
            output = output + residual[:, None].astype(output.dtype)
        return hidden + output


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        counts = {"linear_attention": 0, "full_attention": 0}
        for kind in config.layer_types:
            self.layers.append(DecoderLayer(config, kind, counts[kind]))
            counts[kind] += 1
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, tokens, positions, cache, additions, residual_norms=None):
        hidden = self.embed_tokens(tokens)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, positions, cache, additions.get(index))
            if residual_norms is not None:
                residual_norms.append(
                    mx.linalg.norm(hidden[:, 1:].astype(mx.float32), axis=-1)
                )
        return self.norm(hidden)


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TextModel(config)
        self.tied = config.tie_word_embeddings
        if not self.tied:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, tokens, positions, cache, additions=None, residual_norms=None):
        return self.model(tokens, positions, cache, additions or {}, residual_norms)

    def logits(self, hidden):
        return (
            self.model.embed_tokens.as_linear(hidden)
            if self.tied
            else self.lm_head(hidden)
        )
