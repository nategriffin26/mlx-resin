"""Tokenizer-free LFM2 execution on MLX with Engine-owned cache pages."""

from __future__ import annotations

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
from resin.protocols import ManagedRunner
from resin_models_mlx.lfm2.cache import CacheGeometry, DeviceCache
from resin_models_mlx.lfm2.config import Lfm2Config
from resin_models_mlx.lfm2.model import Lfm2ForCausalLM

DEFAULT_MODEL = "LiquidAI/LFM2.5-230M"
DEFAULT_REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"


def _download_model(repo_id, revision):
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=list(MODEL_FILES)
        )
    )


def load_model(path, *, dtype):
    config = read_config(path)
    args = Lfm2Config.from_dict(config)
    model = Lfm2ForCausalLM(args)
    weights = {}
    for shard in sorted(path.glob("model*.safetensors")):
        for name, value in mx.load(str(shard)).items():
            if name in weights:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            if name.endswith(".conv.conv.weight"):
                value = value.transpose(0, 2, 1)
            weights[name] = value.astype(dtype)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, config, args


class Lfm2MlxRunner(ManagedRunner):
    """LFM2 managed runner with paged decode and checkpoint-aligned prefill."""

    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        model=DEFAULT_MODEL,
        *,
        revision=DEFAULT_REVISION,
        dtype=mx.bfloat16,
        cache_slab_capacity=64,
        block_tokens=128,
    ):
        if dtype not in (mx.float32, mx.float16, mx.bfloat16):
            raise ValueError("LFM2 dtype must be float32, float16, or bfloat16")
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        self.model_path = resolve_model(model, revision, _download_model)
        self._model, self.config, args = load_model(self.model_path, dtype=dtype)
        self.eos_token_id = self.config.get("eos_token_id")
        self.vocab_size = args.vocab_size
        self.geometry = CacheGeometry(args, block_tokens=block_tokens)
        layout = self.geometry.layout(cache_slab_capacity)
        self.runner_config = RunnerConfig(
            cache_layout=layout,
            continuation_prefill=True,
            prefill_chunk_size=block_tokens,
        )
        self.arena = SlabArena(
            layout, self.geometry.shapes, group_dtypes=(dtype, dtype)
        )
        self._sampler = MlxSampler()
        self.forward_calls = 0

    def bind_resources(self, binding):
        self.arena.bind_resources(binding)

    def forward(self, plan):
        if self.arena.binding.pool_id is None:
            raise RuntimeError("bind MLX cache resources before execution")
        cache_plan = PagedCachePlan.from_plan(plan, self.arena.layout)
        if np.any(plan.token_ids >= self.vocab_size):
            raise ValueError("token id is outside the model vocabulary")
        if np.any(plan.prompt_target_token_ids >= self.vocab_size):
            raise ValueError("prompt target token id is outside the model vocabulary")
        groups = stage_page_groups(cache_plan)
        try:
            self.arena.initialize_fresh(cache_plan)
            self.forward_calls += 1
            logits_rows, prompt_rows = [], []
            batches = (
                [list(range(cache_plan.rows))]
                if np.all(cache_plan.batch.query_lengths == 1)
                else [[row] for row in range(cache_plan.rows)]
            )
            for rows in batches:
                cache = DeviceCache.from_plan(self.arena, cache_plan, groups, rows)
                if len(rows) > 1:
                    tokens = mx.array(plan.token_ids[:, None])
                    positions = mx.array(plan.positions[:, None])
                    token_start = 0
                else:
                    token_start, token_end = map(
                        int, plan.token_indptr[rows[0] : rows[0] + 2]
                    )
                    tokens = mx.array(plan.token_ids[token_start:token_end][None])
                    positions = mx.array(plan.positions[token_start:token_end][None])
                hidden = self._model(tokens, positions, cache)
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
            self.arena.commit(logits, sampled)
            return format_output(
                logits, sampled, plan, prompt_logprobs=PromptScores.concat(prompt_rows)
            )
        except Exception as error:
            # Lazy GPU execution and recurrent writes cannot be rolled back safely.
            error._resin_post_accept = True
            raise
