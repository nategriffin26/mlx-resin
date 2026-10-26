"""Catch cache addressing bugs with the real checkpoint and native Engine."""

import os
import sys
from functools import lru_cache

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from resin import Engine, SamplingParams, SchedulerConfig
from resin_models_torch.lfm2 import Lfm2TorchRunner

if sys.platform == "darwin":
    # M5 defaults float32 GEMM to TF32; the oracle requires full mantissas.
    os.environ["MLX_ENABLE_TF32"] = "0"
    mx = pytest.importorskip("mlx.core")
    from resin_models_mlx.lfm2 import DEFAULT_MODEL, DEFAULT_REVISION, Lfm2MlxRunner
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


@pytest.fixture(scope="module")
def oracle():
    torch.set_num_threads(1)
    model = Lfm2TorchRunner(device="cpu", dtype=torch.float32)
    model.sampled_logits = []
    sampler = model._sampler

    def record(logits, plan):
        if plan.context_lengths.min() == 0:
            model.sampled_logits.clear()
        rows = np.flatnonzero(plan.sampling_mask)
        if len(rows):
            model.sampled_logits.append(logits[rows].detach().numpy().copy())
        return sampler(logits, plan)

    model._sampler = record
    return model


@pytest.fixture(scope="module")
def model_path(oracle):
    return oracle.model_path


@lru_cache
def prompt(length, offset=0):
    tokenizer = Tokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    text = (
        "Alice lives in Paris. Bob lives in Rome. Carol lives in Berlin. "
        "The train leaves at noon and arrives at six. "
    ) * 100
    tokens = tokenizer.encode(text).ids
    return tokens[offset : offset + length]


def generate(engine, prompts, *, count=6, scores=False):
    requests = [
        engine.submit(
            tokens,
            sampling_params=SamplingParams(
                temperature=0,
                max_output_tokens=count,
                logprobs=scores,
                top_logprobs=3 if scores else 0,
                prompt_logprobs=scores,
                prompt_top_logprobs=2 if scores else 0,
            ),
        )
        for tokens in prompts
    ]
    events = []
    while engine.has_work():
        events.extend(engine.step())
    return [request.take_result() for request in requests], requests, events


class RecordingRunner(Lfm2MlxRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plans = []
        self.sampled_logits = []
        sampler = self._sampler

        def record(logits, plan):
            rows = np.flatnonzero(plan.sampling_mask)
            if len(rows):
                self.sampled_logits.append(np.array(logits[mx.array(rows)]))
            return sampler(logits, plan)

        self._sampler = record

    def forward(self, plan):
        self.plans.append(plan)
        return super().forward(plan)


def runner(model_path, **kwargs):
    return RecordingRunner(
        model_path, dtype=mx.float32, cache_slab_capacity=16, **kwargs
    )


def test_single_request_matches_torch_greedy_tokens(model_path, oracle):
    tokens = prompt(23)
    with Engine(oracle) as engine:
        expected, _, _ = generate(engine, [tokens])
    with Engine(runner(model_path)) as engine:
        actual, _, _ = generate(engine, [tokens])
    assert actual == expected


def test_batched_decode_does_not_mix_unequal_history_lengths(model_path, oracle):
    tokens = [prompt(7), prompt(35, 3), prompt(133, 8)]
    with Engine(oracle) as engine:
        expected, _, _ = generate(engine, tokens)
    model = runner(model_path)
    with Engine(model) as engine:
        actual, _, _ = generate(engine, tokens)
    assert actual == expected
    assert any(
        plan.num_sequences == 3 and len(plan.token_ids) == 3 for plan in model.plans
    )


@pytest.mark.parametrize("chunk", [63, 128])
def test_chunked_prefill_preserves_convolution_checkpoint(model_path, oracle, chunk):
    tokens = prompt(263)
    with Engine(oracle) as engine:
        expected, _, _ = generate(engine, [tokens])
    model = runner(model_path)
    with Engine(
        model, scheduler_config=SchedulerConfig(prefill_chunk_size=chunk)
    ) as engine:
        actual, _, _ = generate(engine, [tokens])
    assert actual == expected
    np.testing.assert_allclose(
        np.concatenate(model.sampled_logits),
        np.concatenate(oracle.sampled_logits),
        atol=3e-4,
        rtol=3e-4,
    )


@pytest.mark.parametrize(
    "length,hit", [(127, 0), (128, 0), (129, 128), (255, 128), (256, 128), (257, 256)]
)
def test_prefix_hit_restores_state_on_both_sides_of_page_boundary(
    model_path, oracle, length, hit
):
    tokens = prompt(length)
    with Engine(oracle) as engine:
        reference, _, _ = generate(engine, [tokens])
    fresh, cached = runner(model_path), runner(model_path)
    with Engine(fresh) as engine:
        expected, _, _ = generate(engine, [tokens])
    with Engine(cached, prefix_cache=True) as engine:
        generate(engine, [prompt(263)])
        cached.sampled_logits.clear()
        actual, requests, _ = generate(engine, [tokens])
        assert requests[0].cached_tokens == hit
    assert actual == expected == reference
    np.testing.assert_allclose(
        np.concatenate(cached.sampled_logits),
        np.concatenate(oracle.sampled_logits),
        atol=3e-4,
        rtol=3e-4,
    )
    np.testing.assert_allclose(
        np.concatenate(cached.sampled_logits),
        np.concatenate(fresh.sampled_logits),
        atol=3e-4,
        rtol=3e-4,
    )


def test_nan_filled_fresh_pages_and_null_state_are_initialized(model_path, oracle):
    tokens = [prompt(3), prompt(129, 5)]
    with Engine(oracle) as engine:
        expected, _, _ = generate(engine, tokens)
    model = runner(model_path)
    model.arena.storage[:] = float("nan")
    mx.eval(model.arena.storage)
    with Engine(model) as engine:
        actual, _, _ = generate(engine, tokens)
    assert actual == expected
    assert bool(mx.all(mx.isfinite(model.arena.storage[: model.arena.slab_elements])))
    for plan in model.plans:
        for index, group in enumerate(plan.cache_groups):
            for slab, slot in zip(
                group.block_slab_ids, group.block_slotnums, strict=True
            ):
                start = (
                    int(slab) * model.arena.slab_elements
                    + int(slot) * model.arena.block_elements[index]
                )
                block = model.arena.storage[
                    start : start + model.arena.block_elements[index]
                ]
                assert bool(mx.all(mx.isfinite(block)))


def test_eviction_and_release_do_not_reuse_stale_state(model_path, oracle):
    tokens = [prompt(257, offset) for offset in range(6)]
    with Engine(oracle) as engine:
        expected = [generate(engine, [item])[0] for item in tokens]
    model = Lfm2MlxRunner(model_path, dtype=mx.float32, cache_slab_capacity=6)
    with Engine(model, prefix_cache=True) as engine:
        actual = [generate(engine, [item])[0] for item in tokens]
        repeated, requests, _ = generate(engine, [tokens[0]])
        assert requests[0].cached_tokens == 0
        assert engine.prefix_cache_stats[3] > 0
    assert actual == expected
    assert repeated == expected[0]


def test_prompt_and_output_logprobs_use_raw_model_distribution(model_path, oracle):
    tokens = prompt(17)
    with Engine(oracle) as engine:
        expected, reference_requests, reference_events = generate(
            engine, [tokens], scores=True
        )
    with Engine(runner(model_path)) as engine:
        actual, requests, events = generate(engine, [tokens], scores=True)
    assert actual == expected
    reference = reference_requests[0].prompt_logprobs
    scores = requests[0].prompt_logprobs
    assert scores[0] is reference[0] is None
    np.testing.assert_allclose(
        [score.logprob for score in scores[1:]],
        [score.logprob for score in reference[1:]],
        atol=3e-4,
        rtol=3e-4,
    )
    np.testing.assert_allclose(
        [event.logprobs.logprob for event in events],
        [event.logprobs.logprob for event in reference_events],
        atol=3e-4,
        rtol=3e-4,
    )
