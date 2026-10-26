"""MLX execution primitives; imported only by Apple Silicon runners."""

from resin.backend.mlx.attention import paged_attention
from resin.backend.mlx.sampling import (
    MlxSampler,
    PromptScores,
    format_output,
    prompt_scores,
)
from resin.backend.mlx.slab import DevicePageGroup, SlabArena, stage_page_groups

__all__ = [
    "DevicePageGroup",
    "MlxSampler",
    "PromptScores",
    "SlabArena",
    "format_output",
    "paged_attention",
    "prompt_scores",
    "stage_page_groups",
]
