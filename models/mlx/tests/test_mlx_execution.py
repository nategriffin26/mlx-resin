"""Attack batched prefill and recurrent checkpoints across large chunks."""

import os
import sys

import numpy as np
import pytest

from resin import Engine, SamplingParams

if sys.platform == "darwin":
    os.environ["MLX_ENABLE_TF32"] = "0"
    mx = pytest.importorskip("mlx.core")
    from resin_models_mlx.lfm2 import Lfm2MlxRunner
    from resin_models_mlx.qwen3_5 import Qwen35MlxRunner
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


def make_runner(model, checkpoint, **kwargs):
    if model == "lfm2":
        return Lfm2MlxRunner(dtype=mx.float32, cache_slab_capacity=128, **kwargs)
    return Qwen35MlxRunner(
        checkpoint[0], dtype=mx.float32, cache_slab_capacity=128, **kwargs
    )


def generate(engine, prompts, count):
    requests = [
        engine.submit(
            prompt,
            sampling_params=SamplingParams(temperature=0, max_output_tokens=count),
        )
        for prompt in prompts
    ]
    while engine.has_work():
        engine.step()
    return [request.take_result() for request in requests], requests


@pytest.mark.parametrize(
    "model,chunk,hit", [("lfm2", 2048, 4096), ("qwen35", 4096, 8192)]
)
def test_large_prefill_chunks_keep_their_starting_recurrent_checkpoint(
    model, chunk, hit, qwen35_checkpoint
):
    tokens = [1 + index * 17 % 250 for index in range(hit + 9)]
    small = make_runner(model, qwen35_checkpoint, prefill_chunk_size=128)
    with Engine(small, prefix_cache=False) as engine:
        expected, _ = generate(engine, [tokens], 4)
    large = make_runner(model, qwen35_checkpoint)
    assert large.arena.layout.groups[0].block_tokens == 128
    assert large.arena.layout.groups[1].block_tokens == chunk
    with Engine(large, prefix_cache=True) as engine:
        actual, _ = generate(engine, [tokens], 4)
        repeated, requests = generate(engine, [tokens], 4)
    assert actual == repeated == expected
    assert requests[0].cached_tokens == hit


@pytest.mark.parametrize("model", ["lfm2", "qwen35"])
def test_equal_length_prefill_batches_rows_without_mixing_tokens(
    model, qwen35_checkpoint, monkeypatch
):
    prompts = [
        [1 + (index * 17 + offset) % 250 for index in range(257)] for offset in (0, 3)
    ]
    runner = make_runner(model, qwen35_checkpoint)
    sampled = []
    sampler = runner._sampler

    def record_logits(logits, plan):
        sampled.append(np.array(logits))
        return sampler(logits, plan)

    runner._sampler = record_logits
    shapes = []
    model_type = type(runner._model)
    forward = model_type.__call__

    def record(self, tokens, *args, **kwargs):
        shapes.append(tokens.shape)
        return forward(self, tokens, *args, **kwargs)

    monkeypatch.setattr(model_type, "__call__", record)
    with Engine(runner, prefix_cache=False, enable_overlap=False) as engine:
        expected = [generate(engine, [prompt], 4)[0][0] for prompt in prompts]
        expected_logits = np.stack(sampled).reshape(2, 4, -1).transpose(1, 0, 2)
        sampled.clear()
        shapes.clear()
        actual, _ = generate(engine, prompts, 4)
    assert actual == expected
    assert shapes[0] == (2, 257)
    np.testing.assert_allclose(np.stack(sampled), expected_logits, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("model", ["lfm2", "qwen35"])
def test_allocator_scratch_is_released_after_prefill_not_each_decode(
    model, qwen35_checkpoint, monkeypatch
):
    runner = make_runner(model, qwen35_checkpoint)
    releases = []
    clear_cache = mx.clear_cache

    def release():
        clear_cache()
        releases.append(True)

    monkeypatch.setattr(mx, "clear_cache", release)
    with Engine(runner) as engine:
        generate(engine, [[1, 2, 3, 4]], 4)
    # The resource API call is the boundary: retain decode allocations for reuse.
    assert releases == [True]
