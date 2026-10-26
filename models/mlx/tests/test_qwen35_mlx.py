"""Attack hybrid checkpoints and steering isolation through the real native Engine."""

import os
import sys

import numpy as np
import pytest
import torch

from resin import Engine, SamplingParams, SchedulerConfig

if sys.platform == "darwin":
    os.environ["MLX_ENABLE_TF32"] = "0"
    mx = pytest.importorskip("mlx.core")
    from resin_models_mlx.qwen3_5 import Qwen35MlxRunner, SteeringCalibration
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


def prompt(length, offset=0):
    return [1 + (index * 17 + offset) % 250 for index in range(length)]


@torch.inference_mode()
def reference_generate(model, tokens, count=4, additions=()):
    hooks = []
    for layer, neuron, value in additions:

        def add(module, inputs, neuron=neuron, value=value):
            hidden = inputs[0].clone()
            hidden[..., neuron] += value
            return (hidden,)

        hooks.append(
            model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(add)
        )
    try:
        inputs = torch.tensor([tokens])
        cache, output, logits = None, [], []
        for _ in range(count):
            result = model(
                inputs, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
            cache = result.past_key_values
            row = result.logits[:, -1]
            token = row.argmax(-1)
            logits.append(row.numpy().copy())
            output.append(token.item())
            inputs = token[:, None]
        return output, np.concatenate(logits)
    finally:
        for hook in hooks:
            hook.remove()


class RecordingRunner(Qwen35MlxRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plans = []
        self.logits = []
        sampler = self._sampler

        def record(logits, plan):
            self.logits.append(
                (
                    plan.sequence_ids.copy(),
                    plan.sampling_mask.copy(),
                    np.array(logits.astype(mx.float32)),
                )
            )
            return sampler(logits, plan)

        self._sampler = record

    def forward(self, plan):
        self.plans.append(plan)
        return super().forward(plan)


def runner(qwen35_checkpoint, capacity=16):
    return RecordingRunner(
        qwen35_checkpoint[0], dtype=mx.float32, cache_slab_capacity=capacity
    )


def generate(engine, inputs, count=4):
    requests = [
        engine.submit(
            tokens if not hasattr(tokens, "cache_key") else None,
            media=tokens if hasattr(tokens, "cache_key") else None,
            sampling_params=SamplingParams(temperature=0, max_output_tokens=count),
        )
        for tokens in inputs
    ]
    while engine.has_work():
        engine.step()
    return [request.take_result() for request in requests], requests


def sampled(model):
    return np.concatenate(
        [logits[mask] for _, mask, logits in model.logits if mask.any()]
    )


def test_single_greedy_decode_matches_transformers(qwen35_checkpoint):
    tokens = prompt(17)
    expected, logits = reference_generate(qwen35_checkpoint[1], tokens)
    model = runner(qwen35_checkpoint)
    with Engine(model) as engine:
        actual, _ = generate(engine, [tokens])
    assert actual == [expected]
    np.testing.assert_allclose(sampled(model), logits, atol=3e-4, rtol=3e-4)


def test_batched_decode_keeps_unequal_histories_separate(qwen35_checkpoint):
    inputs = [prompt(7), prompt(35, 1), prompt(133, 2)]
    references = [reference_generate(qwen35_checkpoint[1], tokens) for tokens in inputs]
    expected = [tokens for tokens, _ in references]
    model = runner(qwen35_checkpoint)
    with Engine(model) as engine:
        actual, _ = generate(engine, inputs)
    assert actual == expected
    observed = {}
    for ids, mask, logits in model.logits:
        for sequence, valid, row in zip(ids, mask, logits, strict=True):
            if valid:
                observed.setdefault(int(sequence), []).append(row)
    for rows, (_, logits) in zip(observed.values(), references, strict=True):
        np.testing.assert_allclose(rows, logits, atol=3e-4, rtol=3e-4)
    assert any(
        plan.num_sequences == 3 and len(plan.token_ids) == 3 for plan in model.plans
    )


@pytest.mark.parametrize("chunk", [63, 128])
def test_chunked_prefill_retains_deltanet_and_convolution_state(
    qwen35_checkpoint, chunk
):
    tokens = prompt(263)
    expected, logits = reference_generate(qwen35_checkpoint[1], tokens)
    model = runner(qwen35_checkpoint)
    with Engine(
        model, scheduler_config=SchedulerConfig(prefill_chunk_size=chunk)
    ) as engine:
        actual, _ = generate(engine, [tokens])
    assert actual == [expected]
    np.testing.assert_allclose(sampled(model), logits, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize(
    "length,hit", [(127, 0), (128, 0), (129, 128), (255, 128), (256, 128), (257, 256)]
)
def test_prefix_hit_restores_boundary_state_not_final_partial_state(
    qwen35_checkpoint, length, hit
):
    tokens = prompt(length)
    reference, reference_logits = reference_generate(qwen35_checkpoint[1], tokens)
    fresh, cached = runner(qwen35_checkpoint), runner(qwen35_checkpoint)
    with Engine(fresh) as engine:
        expected, _ = generate(engine, [tokens])
    # The Engine cuts the requested 63-token chunks at reusable 128-token boundaries.
    with Engine(
        cached,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(prefill_chunk_size=63),
    ) as engine:
        generate(engine, [prompt(263)])
        cached.logits.clear()
        actual, requests = generate(engine, [tokens])
        assert requests[0].cached_tokens == hit
    assert actual == expected == [reference]
    np.testing.assert_allclose(sampled(cached), sampled(fresh), atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(sampled(cached), reference_logits, atol=3e-4, rtol=3e-4)


def test_nan_filled_fresh_pages_and_null_state_are_cleared(qwen35_checkpoint):
    inputs = [prompt(3), prompt(129, 5)]
    expected = [
        reference_generate(qwen35_checkpoint[1], tokens)[0] for tokens in inputs
    ]
    model = runner(qwen35_checkpoint)
    model.arena.storage = mx.full(
        (model.arena.storage.shape[0], model.arena.slab_bytes // 4),
        float("nan"),
        mx.float32,
    ).view(mx.uint8)
    mx.eval(model.arena.storage)
    with Engine(model) as engine:
        actual, _ = generate(engine, inputs)
    assert actual == expected
    assert np.isfinite(sampled(model)).all()
    assert bool(mx.all(model.arena.storage[0] == 0))


def test_evicted_pages_do_not_restore_previous_requests_state(qwen35_checkpoint):
    inputs = [prompt(257, offset) for offset in range(6)]
    references = [reference_generate(qwen35_checkpoint[1], tokens) for tokens in inputs]
    expected = [tokens for tokens, _ in references]
    model = runner(qwen35_checkpoint, capacity=6)
    with Engine(model, prefix_cache=True) as engine:
        actual = []
        for tokens, (_, logits) in zip(inputs, references, strict=True):
            model.logits.clear()
            actual.append(generate(engine, [tokens])[0][0])
            np.testing.assert_allclose(sampled(model), logits, atol=3e-4, rtol=3e-4)
        repeated, requests = generate(engine, [inputs[0]])
        assert requests[0].cached_tokens == 0
        assert engine.prefix_cache_stats[3] > 0
    assert actual == expected
    assert repeated == [expected[0]]


def test_steered_batch_matches_hf_addition_without_changing_unsteered_row(
    qwen35_checkpoint,
):
    tokens = prompt(17)
    model = runner(qwen35_checkpoint)
    calibration = model.calibrate(prompt(33))
    steering = model.steer(tokens, [(1, 19, 1), (2, 71, -1)], calibration)
    expected_plain, plain_logits = reference_generate(qwen35_checkpoint[1], tokens)
    expected_steered, steered_logits = reference_generate(
        qwen35_checkpoint[1], tokens, additions=steering.additions
    )
    with Engine(model) as engine:
        actual, _ = generate(engine, [tokens, steering])
    assert actual == [expected_plain, expected_steered]
    observed = sampled(model).reshape(4, 2, -1)
    np.testing.assert_allclose(observed[:, 0], plain_logits, atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(observed[:, 1], steered_logits, atol=3e-4, rtol=3e-4)
    assert not np.allclose(observed[:, 0], observed[:, 1])


def test_steering_prefixes_share_only_identical_configuration(qwen35_checkpoint):
    model = runner(qwen35_checkpoint)
    calibration = model.calibrate(prompt(33))
    tokens = prompt(257)
    first = model.steer(tokens, [(1, 19, 1)], calibration)
    other = model.steer(tokens, [(1, 19, -1)], calibration)
    with Engine(model, prefix_cache=True) as engine:
        generate(engine, [first])
        _, different = generate(engine, [other])
        _, plain = generate(engine, [tokens])
        _, identical = generate(engine, [first])
    assert different[0].cached_tokens == 0
    assert plain[0].cached_tokens == 0
    assert identical[0].cached_tokens == 256


def test_generated_steered_prefix_continuation_retains_salted_root(qwen35_checkpoint):
    model = runner(qwen35_checkpoint)
    calibration = model.calibrate(prompt(33))
    tokens = prompt(125)
    settings = [(1, 19, 1)]
    with Engine(model, prefix_cache=True) as engine:
        generated, _ = generate(
            engine, [model.steer(tokens, settings, calibration)], count=6
        )
        continuation = tokens + generated[0] + prompt(3, 7)
        model.logits.clear()
        actual, requests = generate(
            engine, [model.steer(continuation, settings, calibration)]
        )
        assert requests[0].cached_tokens == 128
    fresh = runner(qwen35_checkpoint)
    with Engine(fresh) as engine:
        expected, _ = generate(
            engine, [fresh.steer(continuation, settings, calibration)]
        )
    assert actual == expected
    np.testing.assert_allclose(sampled(model), sampled(fresh), atol=3e-4, rtol=3e-4)


def test_calibration_measures_layer_outputs_excluding_first_token(
    qwen35_checkpoint, tmp_path
):
    model = runner(qwen35_checkpoint)
    tokens = prompt(33)
    calibration = model.calibrate(tokens)
    # HF's final hidden state is normalized, so capture the last residual separately.
    residuals = []
    hooks = [
        layer.register_forward_hook(
            lambda module, inputs, output: residuals.append(output.detach())
        )
        for layer in qwen35_checkpoint[1].model.layers
    ]
    try:
        with torch.inference_mode():
            qwen35_checkpoint[1](torch.tensor([tokens]))
    finally:
        for hook in hooks:
            hook.remove()
    expected = [np.median(row[:, 1:].float().norm(dim=-1).numpy()) for row in residuals]
    np.testing.assert_allclose(
        calibration.residual_medians, expected, atol=3e-5, rtol=3e-5
    )
    assert calibration.token_count == 33
    steering = model.steer(tokens, [(1, 19, 2.0)], calibration)
    column_norm = (
        qwen35_checkpoint[1]
        .model.layers[1]
        .mlp.down_proj.weight[:, 19]
        .detach()
        .float()
        .norm()
        .item()
    )
    assert steering.additions[0][:2] == (1, 19)
    assert steering.additions[0][2] == pytest.approx(
        2 * expected[1] / column_norm, rel=3e-5
    )
    path = tmp_path / "calibration.json"
    calibration.save(path)
    assert SteeringCalibration.load(path) == calibration
