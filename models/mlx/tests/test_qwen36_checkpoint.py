"""Pinned 35B 4-bit parity; the reference's BF16/FP32 noise sets logit bounds."""

import gc
import os
import sys

import numpy as np
import pytest
from tokenizers import Tokenizer

from resin import Engine, SamplingParams, SchedulerConfig
from resin.backend.loading import resolve_model
from resin.recipe import Message, Role, Text, get_native_recipe

if sys.platform == "darwin":
    os.environ["MLX_ENABLE_TF32"] = "0"
    mx = pytest.importorskip("mlx.core")
    from mlx.utils import tree_map
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    from resin_models_mlx.qwen3_5.mlx import _download_model
    from resin_models_mlx.qwen3_5_moe import (
        DEFAULT_MODEL,
        DEFAULT_REVISION,
        Qwen35MoeMlxRunner,
    )
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)

pytestmark = pytest.mark.checkpoint


def reference_logits(model, tokens, count=4, continuation=None):
    cache = make_prompt_cache(model)
    inputs, output, rows = mx.array(tokens)[None], [], []
    for step in range(count):
        logits = model(inputs, cache=cache)[:, -1]
        mx.eval(logits)
        row = np.array(logits.astype(mx.float32))[0]
        token = int(row.argmax()) if continuation is None else continuation[step]
        rows.append(row)
        output.append(token)
        inputs = mx.array([[token]])
    return output, np.stack(rows)


def test_pinned_quantized_checkpoint_matches_mlx_lm_greedy_and_precision_noise():
    path = resolve_model(DEFAULT_MODEL, DEFAULT_REVISION, _download_model)
    tokenizer = Tokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    recipe = get_native_recipe("qwen3_5", tokenizer, enable_thinking=False)
    prompts = [
        recipe.render([Message(Role.USER, [Text(text)])]).token_ids
        for text in (
            "What is the capital of France?",
            "Calculate 17 times 23. Give only the number.",
            "Write a Python function that returns the square of an integer.",
        )
    ]
    model, _ = load_model(path)
    mx.eval(model.parameters())
    references = [reference_logits(model, prompt) for prompt in prompts]
    # Keep packed uint32 weights unchanged: this measures activation/scale arithmetic
    # noise on exactly the same quantized model, not the error of quantization itself.
    model.update(
        tree_map(
            lambda value: (
                value.astype(mx.float32)
                if mx.issubdtype(value.dtype, mx.floating)
                else value
            ),
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    precise = [
        reference_logits(model, prompt, continuation=expected[0])[1]
        for prompt, expected in zip(prompts, references, strict=True)
    ]
    del model
    gc.collect()
    mx.clear_cache()
    runner = Qwen35MoeMlxRunner(path, cache_slab_capacity=8)
    observed = {}
    sampler = runner._sampler

    def record(logits, plan):
        rows = np.array(logits.astype(mx.float32))
        for sequence, valid, row in zip(
            plan.sequence_ids, plan.sampling_mask, rows, strict=True
        ):
            if valid:
                observed.setdefault(int(sequence), []).append(row)
        return sampler(logits, plan)

    runner._sampler = record
    with Engine(runner) as engine:
        requests = [
            engine.submit(
                prompt,
                sampling_params=SamplingParams(temperature=0, max_output_tokens=4),
            )
            for prompt in prompts
        ]
        while engine.has_work():
            engine.step()
        actual = [request.take_result() for request in requests]
    assert actual == [tokens for tokens, _ in references]
    for rows, (_, expected), fp32 in zip(
        observed.values(), references, precise, strict=True
    ):
        noise = expected - fp32
        error = np.asarray(rows) - expected
        noise_rms = float(np.sqrt(np.mean(noise**2)))
        noise_max = float(np.max(np.abs(noise)))
        error_rms = float(np.sqrt(np.mean(error**2)))
        error_max = float(np.max(np.abs(error)))
        print(
            {
                "reference_noise_rms": noise_rms,
                "reference_noise_max": noise_max,
                "resin_error_rms": error_rms,
                "resin_error_max": error_max,
            }
        )
        assert np.isfinite(error).all()
        # Two independently rounded evaluations can differ by the sum of their
        # errors; the floor is the existing FP32 tiny-model oracle's tolerance.
        assert error_rms <= max(3e-4, 2 * noise_rms)
        assert error_max <= max(3e-4, 2 * noise_max)


def test_real_checkpoint_prefix_hit_matches_fresh_chunked_prefill():
    runner = Qwen35MoeMlxRunner(cache_slab_capacity=8)
    tokenizer = Tokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    text = "Alice lives in Paris. Bob lives in Rome. The train leaves at noon. "
    tokens = tokenizer.encode(text * 400).ids[:4097]
    assert len(tokens) == 4097
    observed = []
    sampler = runner._sampler

    def record(logits, plan):
        if plan.sampling_mask.any():
            observed.extend(np.array(logits.astype(mx.float32))[plan.sampling_mask])
        return sampler(logits, plan)

    runner._sampler = record
    with Engine(
        runner,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(prefill_chunk_size=512),
    ) as engine:
        requests, outputs, logits = [], [], []
        for _ in range(2):
            observed.clear()
            request = engine.submit(
                tokens,
                sampling_params=SamplingParams(temperature=0, max_output_tokens=4),
            )
            while engine.has_work():
                engine.step()
            requests.append(request)
            outputs.append(request.take_result())
            logits.append(np.stack(observed))
        assert requests[0].cached_tokens == 0
        assert requests[1].cached_tokens == 4096
    assert outputs[0] == outputs[1]
    np.testing.assert_allclose(logits[0], logits[1], atol=3e-4, rtol=3e-4)
