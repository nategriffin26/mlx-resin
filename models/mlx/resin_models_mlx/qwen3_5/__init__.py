"""Dense Qwen3.5 text inference and calibrated neuron steering on MLX."""

from resin_models_mlx.qwen3_5.mlx import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Qwen35MlxRunner,
)
from resin_models_mlx.qwen3_5.steering import Qwen35Steering, SteeringCalibration

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "Qwen35MlxRunner",
    "Qwen35Steering",
    "SteeringCalibration",
]
