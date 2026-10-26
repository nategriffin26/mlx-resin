"""Backend-neutral Qwen3.5 MoE text geometry, including Qwen3.6 checkpoints."""

from dataclasses import asdict, dataclass
from typing import Any, Self

from resin.models.qwen3_5.config import Qwen35Config


@dataclass(frozen=True, slots=True, kw_only=True)
class Qwen35MoeConfig(Qwen35Config):
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        if "text_config" in values:
            if values.get("model_type") != "qwen3_5_moe":
                raise ValueError("expected a qwen3_5_moe checkpoint")
            text = values["text_config"]
        else:
            text = values
        if not isinstance(text, dict) or text.get("model_type") != "qwen3_5_moe_text":
            raise ValueError("checkpoint is missing qwen3_5_moe_text config")
        # Attention and state geometry are identical to the dense family. The shared
        # expert defines the activation width of the inherited steering interface.
        dense = Qwen35Config.from_hugging_face(
            {
                **text,
                "model_type": "qwen3_5_text",
                "intermediate_size": text["shared_expert_intermediate_size"],
            }
        )
        config = cls(
            **asdict(dense),
            num_experts=int(text["num_experts"]),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(
                text["shared_expert_intermediate_size"]
            ),
        )
        if config.moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be positive")
        if not 1 <= config.num_experts_per_tok <= config.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")
        if text.get("norm_topk_prob", True) is not True:
            raise ValueError("Qwen3.5 MoE requires normalized top-k routing")
        return config


__all__ = ["Qwen35MoeConfig"]
