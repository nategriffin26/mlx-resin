# Portions are adapted from Hugging Face Transformers (Apache-2.0).

"""LFM2 modules with Engine-owned history and recurrent state."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.layer = layer
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.hidden_size // self.heads
        self.theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.dim, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, self.kv_heads * self.dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.kv_heads * self.dim, bias=False
        )
        self.out_proj = nn.Linear(self.heads * self.dim, config.hidden_size, bias=False)
        self.q_layernorm = nn.RMSNorm(self.dim, config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.dim, config.norm_eps)

    def __call__(self, hidden, positions, cache):
        batch, length, _ = hidden.shape
        queries = self.q_layernorm(
            self.q_proj(hidden).reshape(batch, length, self.heads, self.dim)
        )
        keys = self.k_layernorm(
            self.k_proj(hidden).reshape(batch, length, self.kv_heads, self.dim)
        )
        values = self.v_proj(hidden).reshape(batch, length, self.kv_heads, self.dim)
        frequencies = self.theta ** (
            -mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim
        )
        angles = positions.astype(mx.float32)[..., None] * frequencies
        angles = mx.concatenate((angles, angles), axis=-1)[:, :, None, :]
        cos, sin = (
            mx.cos(angles).astype(hidden.dtype),
            mx.sin(angles).astype(hidden.dtype),
        )

        def rotate(value):
            first, second = mx.split(value, 2, axis=-1)
            return value * cos + mx.concatenate((-second, first), axis=-1) * sin

        queries, keys = rotate(queries), rotate(keys)
        cache.append(self.layer, keys, values)
        previous = cache.history(self.layer)
        if previous is not None:
            past_length = previous.shape[1]
            keys = mx.concatenate((previous[:, :, 0], keys), axis=1)
            values = mx.concatenate((previous[:, :, 1], values), axis=1)
            previous_valid = (
                mx.arange(past_length)[None, :] < mx.array(cache.contexts)[:, None]
            )
            previous_mask = mx.broadcast_to(
                previous_valid[:, None, :], (batch, length, past_length)
            )
            current_mask = mx.broadcast_to(
                mx.arange(length)[None, :] <= mx.arange(length)[:, None],
                (batch, length, length),
            )
            mask = mx.concatenate((previous_mask, current_mask), axis=-1)[:, None]
        else:
            mask = "causal"
        output = mx.fast.scaled_dot_product_attention(
            queries.transpose(0, 2, 1, 3),
            keys.transpose(0, 2, 1, 3),
            values.transpose(0, 2, 1, 3),
            scale=self.dim**-0.5,
            mask=mask,
        )
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class ShortConv(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.layer = layer
        self.state_length = config.conv_cache_length - 1
        self.in_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.conv_bias
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.conv_bias
        )
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_cache_length,
            groups=config.hidden_size,
            bias=config.conv_bias,
        )

    def __call__(self, hidden, cache):
        gate_b, gate_c, value = mx.split(self.in_proj(hidden), 3, axis=-1)
        convolution_input = mx.concatenate(
            (cache.state(self.layer), gate_b * value), axis=1
        )
        cache.store_states(self.layer, convolution_input, self.state_length)
        return self.out_proj(gate_c * self.conv(convolution_input))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        size = config.intermediate_size
        if config.auto_adjust_ff_dim:
            size = int(config.ffn_dim_multiplier * int(2 * size / 3))
            size = config.multiple_of * (
                (size + config.multiple_of - 1) // config.multiple_of
            )
        self.w1 = nn.Linear(config.hidden_size, size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, size, bias=False)
        self.w2 = nn.Linear(size, config.hidden_size, bias=False)

    def __call__(self, hidden):
        return self.w2(nn.silu(self.w1(hidden)) * self.w3(hidden))


class DecoderLayer(nn.Module):
    def __init__(self, config, kind, layer):
        super().__init__()
        self.is_attention = kind == "full_attention"
        if self.is_attention:
            self.self_attn = Attention(config, layer)
        else:
            self.conv = ShortConv(config, layer)
        self.operator_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.feed_forward = MLP(config)

    def __call__(self, hidden, positions, cache):
        normalized = self.operator_norm(hidden)
        hidden = hidden + (
            self.self_attn(normalized, positions, cache)
            if self.is_attention
            else self.conv(normalized, cache)
        )
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class Lfm2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        counts = {"full_attention": 0, "conv": 0}
        self.layers = []
        for kind in config.layer_types:
            self.layers.append(DecoderLayer(config, kind, counts[kind]))
            counts[kind] += 1
        self.embedding_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

    def __call__(self, tokens, positions, cache):
        hidden = self.embed_tokens(tokens)
        for layer in self.layers:
            hidden = layer(hidden, positions, cache)
        return self.embedding_norm(hidden)


class Lfm2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Lfm2Model(config)

    def __call__(self, tokens, positions, cache):
        return self.model(tokens, positions, cache)

    def logits(self, hidden):
        return self.model.embed_tokens.as_linear(hidden)
