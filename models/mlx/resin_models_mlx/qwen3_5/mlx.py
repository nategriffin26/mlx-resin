"""Dense Qwen3.5 text execution and per-request neuron steering on MLX."""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from resin import RunnerConfig
from resin.backend.loading import MODEL_FILES, read_config, resolve_model
from resin.backend.mlx import (
    MlxSampler,
    PromptScores,
    SlabArena,
    format_output,
    prompt_scores,
    stage_page_groups,
)
from resin.backend.plans import PagedCachePlan
from resin.backend.topology import ParallelismCapabilities
from resin.models.qwen3_5.config import Qwen35Config
from resin.protocols import ManagedRunner
from resin_models_mlx.qwen3_5.cache import CacheGeometry, DecodeState, DeviceCache
from resin_models_mlx.qwen3_5.model import Qwen35ForCausalLM
from resin_models_mlx.qwen3_5.steering import Qwen35Steering, SteeringCalibration

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _download_model(repo_id, revision):
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=list(MODEL_FILES)
        )
    )


def load_model(path, *, dtype):
    raw = read_config(path)
    config = Qwen35Config.from_hugging_face(raw)
    if raw.get("quantization_config"):
        raise ValueError("Qwen3.5 MLX runner requires unquantized text weights")
    model = Qwen35ForCausalLM(config)
    weights = {}
    for shard in sorted(path.glob("model*.safetensors")):
        for name, value in mx.load(str(shard)).items():
            # Vision and bundled MTP tensors are not part of text-only execution.
            if name.startswith(("model.visual.", "visual.", "mtp.", "model.mtp.")):
                continue
            name = name.replace("model.language_model.", "model.", 1)
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            if name.startswith("model.mtp."):
                continue
            if name in weights:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            if name.endswith(".conv1d.weight"):
                value = value.transpose(0, 2, 1)
            tensor_dtype = (
                mx.float32 if name.endswith((".A_log", ".dt_bias")) else dtype
            )
            weights[name] = value.astype(tensor_dtype)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, raw, config


class Qwen35MlxRunner(ManagedRunner):
    parallelism_capabilities = ParallelismCapabilities()

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
            raise ValueError("Qwen3.5 dtype must be float32, float16, or bfloat16")
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        if prefill_chunk_size <= 0 or prefill_chunk_size % block_tokens:
            raise ValueError(
                "prefill_chunk_size must be a positive multiple of block_tokens"
            )
        self.model_path = resolve_model(model, revision, _download_model)
        self._model, self.raw_config, self.config = load_model(
            self.model_path, dtype=dtype
        )
        self.model_identity = f"{self.model_path.resolve()}:{dtype}"
        self.eos_token_id = self.config.eos_token_id
        self.vocab_size = self.config.vocab_size
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
        self._decode_state = DecodeState(self.arena)

    def bind_resources(self, binding):
        self.arena.bind_resources(binding)

    def release(self, sequence_ids):
        # The Engine calls this before completion, cancellation, retraction, and close.
        self._decode_state.flush()

    def calibrate(self, token_ids):
        """Median residual norms after each layer, excluding the first prompt token."""
        tokens = tuple(token_ids)
        if len(tokens) < 2 or any(
            token < 0 or token >= self.vocab_size for token in tokens
        ):
            raise ValueError("calibration requires at least two in-vocabulary tokens")
        norms = []
        hidden = self._model(
            mx.array(tokens)[None],
            mx.arange(len(tokens))[None],
            None,
            residual_norms=norms,
        )
        mx.eval(hidden, norms)
        return SteeringCalibration(
            self.model_identity,
            len(tokens),
            tuple(float(np.median(np.array(row))) for row in norms),
        )

    def steer(self, token_ids, neurons, calibration, directions=()):
        """Prepare additive values once; layer/neuron indices are zero-based.

        ``directions`` holds ``(layer, vector, dose)``: the MLP output of ``layer`` gains
        ``dose`` residual medians along ``vector``, the same push size a neuron dose
        gives along its down-projection column.
        """
        if (
            calibration.model != self.model_identity
            or len(calibration.residual_medians) != self.config.num_hidden_layers
        ):
            raise ValueError(
                "steering calibration belongs to a different model or dtype"
            )
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
                raise ValueError("steering neuron is outside the MLP")
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
            column = (
                self._model.model.layers[layer]
                .mlp.down_proj.weight[:, neuron]
                .astype(mx.float32)
            )
            norm = float(mx.linalg.norm(column).item())
            if not math.isfinite(norm) or norm == 0:
                raise ValueError(
                    "steering requires a nonzero finite down-projection column"
                )
            additions.append(
                (layer, neuron, dose * calibration.residual_medians[layer] / norm)
            )
        residual_additions = []
        for layer, vector, dose in directions:
            if (
                not isinstance(layer, int)
                or not 0 <= layer < self.config.num_hidden_layers
            ):
                raise ValueError("steering layer is outside the model")
            vector = np.asarray(vector, np.float32)
            norm = float(np.linalg.norm(vector))
            if (
                vector.shape != (self.config.hidden_size,)
                or not math.isfinite(norm)
                or norm == 0
            ):
                raise ValueError(
                    "steering directions need a nonzero finite hidden-size vector"
                )
            if not math.isfinite(dose):
                raise ValueError("steering dose must be finite")
            if dose:
                scale = dose * calibration.residual_medians[layer] / norm
                residual_additions.append(
                    (layer, (vector * scale).astype("<f4").tobytes())
                )
        tokens = tuple(token_ids)
        if not tokens or any(token < 0 or token >= self.vocab_size for token in tokens):
            raise ValueError("steering requires a nonempty in-vocabulary prompt")
        return Qwen35Steering(
            tokens, neurons, calibration, tuple(additions), tuple(residual_additions)
        )

    def _additions(self, media, rows):
        """Per-layer ``(activation, residual)`` additions for the batch rows."""
        activations, residuals = {}, {}
        for batch_row, row in enumerate(rows):
            steering = media[row]
            if steering is None:
                continue
            for layer, neuron, value in steering.additions:
                if layer not in activations:
                    activations[layer] = np.zeros(
                        (len(rows), self.config.intermediate_size), np.float32
                    )
                activations[layer][batch_row, neuron] += value
            for layer, vector in steering.residual_additions:
                if layer not in residuals:
                    residuals[layer] = np.zeros(
                        (len(rows), self.config.hidden_size), np.float32
                    )
                residuals[layer][batch_row] += np.frombuffer(vector, "<f4")
        return {
            layer: tuple(
                mx.array(values[layer]) if layer in values else None
                for values in (activations, residuals)
            )
            for layer in activations.keys() | residuals.keys()
        }

    def forward(self, plan):
        if self.arena.binding.pool_id is None:
            raise RuntimeError("bind MLX cache resources before execution")
        cache_plan = PagedCachePlan.from_plan(plan, self.arena.layout)
        if np.any(plan.token_ids >= self.vocab_size) or np.any(
            plan.prompt_target_token_ids >= self.vocab_size
        ):
            raise ValueError("token id is outside the model vocabulary")
        for media in plan.media:
            if media is not None and (
                not isinstance(media, Qwen35Steering)
                or media.calibration.model != self.model_identity
            ):
                raise ValueError(
                    "Qwen3.5 text runner accepts only its own calibrated steering media"
                )
        groups = stage_page_groups(cache_plan)
        resident_key = self._decode_state.prepare(cache_plan)
        try:
            has_prefill = np.any(cache_plan.batch.query_lengths > 1)
            if has_prefill:
                # Decode-sized free buffers cannot satisfy large prefill allocations.
                mx.clear_cache()
            self.arena.initialize_fresh(cache_plan)
            logits_rows, prompt_rows = [], []
            batches = (
                [list(range(cache_plan.rows))]
                if np.all(
                    cache_plan.batch.query_lengths == cache_plan.batch.query_lengths[0]
                )
                else [[row] for row in range(cache_plan.rows)]
            )
            for rows in batches:
                cache = DeviceCache.from_plan(
                    self.arena, self.geometry, cache_plan, groups, rows
                )
                if resident_key is not None:
                    cache.resident = self._decode_state.layers
                    cache.updated = []
                if len(rows) > 1:
                    tokens, positions = (
                        mx.array(plan.token_ids.reshape(len(rows), -1)),
                        mx.array(plan.positions.reshape(len(rows), -1)),
                    )
                    token_start = 0
                else:
                    token_start, token_end = map(
                        int, plan.token_indptr[rows[0] : rows[0] + 2]
                    )
                    tokens = mx.array(plan.token_ids[token_start:token_end][None])
                    positions = mx.array(plan.positions[token_start:token_end][None])
                additions = self._additions(plan.media, rows)
                hidden = self._model(tokens, positions, cache, additions)
                logits_rows.append(self._model.logits(hidden[:, -1]))
                if plan.needs_prompt_logprobs:
                    prompt_rows.append(
                        prompt_scores(
                            plan,
                            hidden.reshape(-1, hidden.shape[-1]),
                            self._model.logits,
                            token_start=token_start,
                        )
                    )
            logits = mx.concatenate(logits_rows)
            sampled = self._sampler(logits, plan)
            # Formatting can fail too: no new state is visible until it succeeds.
            output = format_output(
                logits, sampled, plan, prompt_logprobs=PromptScores.concat(prompt_rows)
            )
            boundary = resident_key is not None and np.any(
                (cache_plan.context_lengths + 1) % self.geometry.block_tokens == 0
            )
            if boundary:
                self._decode_state.queue(
                    cache.updated, cache.destination_slabs, cache.destination_slots
                )
            self.arena.commit(
                logits, sampled, cache.updated if resident_key is not None else []
            )
            if resident_key is not None:
                if boundary:
                    self._decode_state.key = self._decode_state.layers = None
                else:
                    self._decode_state.publish(
                        resident_key, cache_plan.context_lengths, cache
                    )
            if has_prefill:
                # Prefill scratch can exceed live cache size and evict decode buffers.
                mx.clear_cache()
            return output
        except Exception as error:
            self.arena.abort()
            error._resin_post_accept = True
            raise
