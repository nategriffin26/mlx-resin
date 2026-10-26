"""Validated LFM2 checkpoint configuration, independent of the torch runner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Lfm2Config:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    norm_eps: float
    conv_bias: bool
    conv_cache_length: int
    intermediate_size: int
    multiple_of: int
    auto_adjust_ff_dim: bool
    ffn_dim_multiplier: float
    rope_theta: float
    layer_types: tuple[str, ...]

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.intermediate_size,
                self.multiple_of,
            )
            <= 0
        ):
            raise ValueError("LFM2 dimensions must be positive")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per hidden layer")
        if set(self.layer_types) - {"conv", "full_attention"}:
            raise ValueError("unsupported LFM2 layer type")
        if (
            self.hidden_size % self.num_attention_heads
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError("LFM2 attention dimensions must divide evenly")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("the attention head dimension must be even for RoPE")
        if self.conv_cache_length < 2:
            raise ValueError(
                "LFM2 recurrent cache requires a convolution width of at least two"
            )

    @classmethod
    def from_dict(cls, config):
        if config.get("model_type") != "lfm2":
            raise ValueError(
                f"expected an LFM2 checkpoint, got {config.get('model_type')!r}"
            )
        if config.get("quantization") or config.get("quantization_config"):
            raise ValueError("the MLX runner requires an unquantized LFM2 checkpoint")
        rope = config.get("rope_parameters") or {}
        return cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config.get(
                "num_key_value_heads", config["num_attention_heads"]
            ),
            norm_eps=config.get("norm_eps", config.get("block_norm_eps", 1e-5)),
            conv_bias=config["conv_bias"],
            conv_cache_length=config["conv_L_cache"],
            intermediate_size=config.get("block_ff_dim", config["intermediate_size"]),
            multiple_of=config.get("block_multiple_of", 256),
            auto_adjust_ff_dim=config.get("block_auto_adjust_ff_dim", False),
            ffn_dim_multiplier=config.get("block_ffn_dim_multiplier", 1.0),
            rope_theta=rope.get("rope_theta", config.get("rope_theta", 1_000_000.0)),
            layer_types=tuple(config["layer_types"]),
        )
