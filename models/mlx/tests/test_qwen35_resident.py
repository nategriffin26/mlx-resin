"""Attack write-back visibility, cohort identity, and failed-forward isolation."""

import os
import sys

import numpy as np
import pytest

from resin import BatchPhase, Engine, SamplingParams

if sys.platform == "darwin":
    os.environ["MLX_ENABLE_TF32"] = "0"
    mx = pytest.importorskip("mlx.core")
    from test_qwen35_moe import checkpoint as moe_checkpoint  # noqa: F401

    from resin_models_mlx.qwen3_5 import Qwen35MlxRunner
    from resin_models_mlx.qwen3_5_moe import Qwen35MoeMlxRunner
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


def prompt(length, offset=0):
    return [1 + (index * 17 + offset) % 250 for index in range(length)]


@pytest.fixture(params=["dense", "moe"])
def model_factory(request, qwen35_checkpoint, moe_checkpoint):
    cls, path = (
        (Qwen35MlxRunner, qwen35_checkpoint[0])
        if request.param == "dense"
        else (Qwen35MoeMlxRunner, moe_checkpoint[0])
    )

    def create(flush=False):
        model = cls(path, dtype=mx.float32, cache_slab_capacity=128)
        model.observed = []
        sampler = model._sampler

        def record(logits, plan):
            model.observed.append(np.array(logits))
            model.last_plan = plan
            return sampler(logits, plan)

        model._sampler = record
        if flush:
            forward = model.forward

            def flushed(plan):
                result = forward(plan)
                model._decode_state.flush()
                return result

            model.forward = flushed
        return model

    return create


def submit(engine, tokens, count=10, media=None):
    return engine.submit(
        tokens if media is None else None,
        media=media,
        sampling_params=SamplingParams(temperature=0, max_output_tokens=count),
    )


def compare(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left.argmax(-1), right.argmax(-1))
        np.testing.assert_allclose(left, right, atol=3e-4, rtol=3e-4)


def test_resident_cohort_join_finish_cancel_and_shutdown(model_factory):
    def execute(flush):
        model = model_factory(flush)
        with Engine(model, prefix_cache=False) as engine:
            requests = [
                submit(engine, prompt(17, i), n) for i, n in enumerate((6, 14, 14))
            ]
            engine.step()
            engine.step()
            assert flush or model._decode_state.layers is not None
            requests.append(submit(engine, prompt(19, 5), 8))
            engine.step()
            engine.step()
            assert engine.cancel(requests[2])
            assert model._decode_state.layers is None
            while not requests[0].done:
                engine.step()
            while engine.has_work():
                engine.step()
            tokens = [request.take_result() for request in requests]
            live = submit(engine, prompt(21, 7), 20)
            engine.step()
            engine.step()
            assert not live.done
        assert model._decode_state.layers is None
        assert not model.arena._writes
        return tokens, model.observed

    actual, observed = execute(False)
    expected, reference = execute(True)
    assert actual == expected
    compare(observed, reference)


@pytest.mark.parametrize("length", [125, 4093])
def test_resident_crosses_page_and_checkpoint_with_mixed_steering(
    model_factory, length
):
    def execute(flush):
        model = model_factory(flush)
        calibration = model.calibrate(prompt(33))
        steering = model.steer(prompt(length, 1), [(1, 19, 1)], calibration)
        with Engine(model, prefix_cache=True) as engine:
            requests = [
                submit(engine, prompt(length), 8),
                submit(engine, None, 8, steering),
            ]
            while engine.has_work():
                engine.step()
                plan = model.last_plan
                if plan.phase == BatchPhase.DECODE and np.any(
                    (plan.context_lengths + 1) % 128 == 0
                ):
                    assert model._decode_state.layers is None
            result = [request.take_result() for request in requests]
        return result, model.observed

    actual, observed = execute(False)
    expected, reference = execute(True)
    assert actual == expected
    compare(observed, reference)
    assert any(not np.allclose(rows[0], rows[1]) for rows in observed if len(rows) == 2)


def test_resident_published_prefix_is_visible_to_later_cohort(model_factory):
    model = model_factory()
    with Engine(model, prefix_cache=True) as engine:
        first = submit(engine, prompt(4093), 10)
        for _ in range(4):
            engine.step()
        assert model._decode_state.layers is None
        engine.step()
        engine.step()
        # The checkpoint is published while its generating request is still alive.
        assert not first.done
        generated = np.concatenate(
            [rows.argmax(-1) for rows in model.observed]
        ).tolist()
        continuation = prompt(4093) + generated[:3] + prompt(3, 9)
        second = submit(engine, continuation, 5)
        model.observed.clear()
        while engine.has_work():
            engine.step()
        assert second.cached_tokens == 4096
        actual = second.take_result()
    fresh = model_factory(True)
    with Engine(fresh, prefix_cache=False) as engine:
        second = submit(engine, continuation, 5)
        while engine.has_work():
            engine.step()
        expected = second.take_result()
    assert actual == expected


@pytest.mark.parametrize("length,steps", [(125, 2), (128, 1)])
def test_failed_forward_preserves_arena_and_resident_state(
    model_factory, length, steps
):
    model = model_factory()
    engine = Engine(model, prefix_cache=False)
    submit(engine, prompt(length), 10)
    for _ in range(steps):
        engine.step()
    # Dirty padding makes a failed fresh-page clear observable, not a zero-to-zero write.
    if length == 128:
        model.arena.storage = mx.ones_like(model.arena.storage)
    mx.eval(model.arena._storage)
    before = np.array(model.arena._storage)
    layers = model._decode_state.layers
    values = (
        None
        if layers is None
        else [tuple(np.array(x) for x in pair) for pair in layers]
    )

    def fail(logits, plan):
        mx.eval(logits)
        raise RuntimeError("injected sampler failure after model execution")

    model._sampler = fail
    with pytest.raises(RuntimeError, match="injected sampler failure"):
        engine.step()
    np.testing.assert_array_equal(np.array(model.arena._storage), before)
    assert model._decode_state.layers is layers
    if values is not None:
        for current, previous in zip(layers, values, strict=True):
            for left, right in zip(current, previous, strict=True):
                np.testing.assert_array_equal(np.array(left), right)
    assert not model.arena._writes


@pytest.mark.parametrize("access", ["storage", "state_layer"])
def test_arena_read_barrier_materializes_resident_state(model_factory, access):
    model = model_factory()
    with Engine(model, prefix_cache=False) as engine:
        submit(engine, prompt(17), 8)
        engine.step()
        engine.step()
        state = model._decode_state
        convolution, recurrent = state.layers[0]
        expected = np.concatenate(
            (np.array(convolution).reshape(1, -1), np.array(recurrent).reshape(1, -1)),
            axis=-1,
        )
        if access == "storage":
            mx.eval(model.arena.storage)
            assert state.layers is None
        actual = model.arena.read_layer(1, 0, state.slabs, state.slots)
        np.testing.assert_array_equal(np.array(actual), expected)
        assert state.layers is None
