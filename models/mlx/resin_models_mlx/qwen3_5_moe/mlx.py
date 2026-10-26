"""Qwen3.6 MoE text inference on the dense family's Engine-owned hybrid cache."""

import math
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from resin import RunnerConfig
from resin.backend.loading import read_config, resolve_model
from resin.backend.mlx import MlxSampler, SlabArena
from resin.models.qwen3_5_moe import Qwen35MoeConfig
from resin_models_mlx.qwen3_5.cache import CacheGeometry
from resin_models_mlx.qwen3_5.mlx import Qwen35MlxRunner, _download_model
from resin_models_mlx.qwen3_5.model import RMSNorm
from resin_models_mlx.qwen3_5_moe.model import ExpertLinear, Qwen35MoeForCausalLM
from resin_models_mlx.qwen3_5_moe.quantization import (
    QuantizedEmbedding,
    QuantizedExpertLinear,
    QuantizedLinear,
)

DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
DEFAULT_REVISION = "38740b847e4cb78f352aba30aa41c76e08e6eb46"


def tensor_name(name):
    name = name.replace("model.language_model.", "model.", 1)
    if name.startswith("language_model."):
        name = name[len("language_model.") :]
    return name


def load_model(path, *, dtype):
    raw = read_config(path)
    config = Qwen35MoeConfig.from_hugging_face(raw)
    model = Qwen35MoeForCausalLM(config)
    quantization = raw.get("quantization") or raw.get("quantization_config")
    weights = {}
    hf_convolution_layout = False
    for shard in sorted(path.glob("model*.safetensors")):
        for name, value in mx.load(str(shard)).items():
            if name.startswith(("model.visual.", "visual.", "vision_tower.")):
                continue
            name = tensor_name(name)
            if name.startswith(("mtp.", "model.mtp.")):
                continue
            if name in weights:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            if name.endswith(".conv1d.weight") and value.shape[-1] != 1:
                hf_convolution_layout = True
                value = value.transpose(0, 2, 1)
            if name.endswith(".experts.gate_up_proj"):
                gate, up = mx.split(value, 2, axis=-2)
                prefix = name.removesuffix("experts.gate_up_proj")
                weights[prefix + "switch_mlp.gate_proj.weight"] = gate.astype(dtype)
                weights[prefix + "switch_mlp.up_proj.weight"] = up.astype(dtype)
                continue
            name = name.replace(".experts.down_proj", ".switch_mlp.down_proj.weight")
            if value.dtype != mx.uint32:
                value = value.astype(
                    mx.float32 if name.endswith((".A_log", ".dt_bias")) else dtype
                )
            weights[name] = value
    replacements = []
    for name, module in model.named_modules():
        if not hf_convolution_layout and isinstance(module, RMSNorm):
            # MLX conversion changes both convolution layout and the norm offset.
            # Quantization metadata alone does not prove that conversion happened.
            weights[name + ".weight"] = weights[name + ".weight"].astype(mx.float32) - 1
        if name + ".scales" not in weights:
            continue
        if not quantization:
            raise ValueError("packed checkpoint tensors require quantization metadata")
        settings = {
            key: quantization.get(key, default)
            for key, default in (("group_size", 64), ("bits", 4), ("mode", "affine"))
        }
        settings.update(
            quantization.get("language_model." + name, quantization.get(name, {}))
        )
        projection = (
            QuantizedExpertLinear
            if isinstance(module, ExpertLinear)
            else QuantizedEmbedding
            if isinstance(module, nn.Embedding)
            else QuantizedLinear
        )
        replacements.append(
            (
                name,
                projection(
                    tuple(
                        weights[name + suffix]
                        for suffix in (".weight", ".scales", ".biases")
                    ),
                    **settings,
                ),
            )
        )
    from mlx.utils import tree_unflatten

    model.update_modules(tree_unflatten(replacements))
    if config.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, raw, config


class Qwen35MoeMlxRunner(Qwen35MlxRunner):
    def __init__(
        self,
        model=DEFAULT_MODEL,
        *,
        revision=DEFAULT_REVISION,
        dtype=mx.bfloat16,
        cache_slab_capacity=32,
        block_tokens=128,
        prefill_chunk_size=4096,
    ):
        if dtype not in (mx.float32, mx.float16, mx.bfloat16):
            raise ValueError("Qwen3.5 MoE dtype must be float32, float16, or bfloat16")
        if (
            block_tokens <= 0
            or prefill_chunk_size <= 0
            or prefill_chunk_size % block_tokens
        ):
            raise ValueError(
                "prefill_chunk_size must be a positive multiple of block_tokens"
            )
        self.model_path = resolve_model(model, revision, _download_model)
        self._model, self.raw_config, self.config = load_model(
            self.model_path, dtype=dtype
        )
        self.model_identity = f"{self.model_path.resolve()}:{dtype}"
        self.eos_token_id, self.vocab_size = (
            self.config.eos_token_id,
            self.config.vocab_size,
        )
        self.geometry = CacheGeometry(
            self.config,
            dtype=dtype,
            block_tokens=block_tokens,
            prefill_chunk_size=prefill_chunk_size,
        )
        layout = self.geometry.layout(cache_slab_capacity)
        self.runner_config = RunnerConfig(
            cache_layout=layout,
            continuation_prefill=True,
            prefill_chunk_size=prefill_chunk_size,
        )
        self.arena = SlabArena(
            layout, self.geometry.shapes, group_dtypes=(dtype, mx.float32)
        )
        self._sampler = MlxSampler()

    def steer(self, token_ids, neurons, calibration, directions=()):
        """Steer shared-expert neurons or layer residual directions, not routed experts."""
        steering = super().steer(token_ids, (), calibration, directions)
        combined = {}
        for layer, neuron, dose in neurons:
            if (
                not isinstance(layer, int)
                or not 0 <= layer < self.config.num_hidden_layers
            ):
                raise ValueError("steering layer is outside the model")
            if (
                not isinstance(neuron, int)
                or not 0 <= neuron < self.config.intermediate_size
            ):
                raise ValueError("steering neuron is outside the shared expert")
            if not math.isfinite(dose):
                raise ValueError("steering dose must be finite")
            combined[layer, neuron] = combined.get((layer, neuron), 0.0) + float(dose)
        neurons = tuple(
            (layer, neuron, dose)
            for (layer, neuron), dose in sorted(combined.items())
            if dose
        )
        additions = []
        for layer, neuron, dose in neurons:
            projection = self._model.model.layers[layer].mlp.shared_expert.down_proj
            column = (
                projection.column(neuron)
                if isinstance(projection, QuantizedLinear)
                else projection.weight[:, neuron]
            )
            norm = float(mx.linalg.norm(column.astype(mx.float32)).item())
            if not math.isfinite(norm) or norm == 0:
                raise ValueError(
                    "steering requires a nonzero finite down-projection column"
                )
            additions.append(
                (layer, neuron, dose * calibration.residual_medians[layer] / norm)
            )
        return replace(steering, neurons=neurons, additions=tuple(additions))
