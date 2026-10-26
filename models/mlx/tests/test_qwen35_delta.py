"""Qwen's recurrence must preserve complete forgetting, including zero decay."""

import os
import sys

import numpy as np
import pytest

if sys.platform == "darwin":
    os.environ["MLX_ENABLE_TF32"] = "0"
    mx = pytest.importorskip("mlx.core")
    from resin_models_mlx.qwen3_5.delta import gated_delta_update
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


def test_zero_decay_does_not_resurrect_previous_state():
    query = mx.ones((1, 128, 16, 128))
    value = mx.zeros((1, 128, 32, 128))
    state = mx.full((1, 32, 128, 128), 10000.0)
    output, final = gated_delta_update(
        query, query, value, mx.zeros((1, 128, 32)), mx.zeros((1, 128, 32)), state
    )
    np.testing.assert_array_equal(np.array(output), 0)
    np.testing.assert_array_equal(np.array(final), 0)


@pytest.mark.parametrize("width", [31, 32, 33])
def test_grouped_recurrence_keeps_key_tail_and_value_heads(width):
    query = mx.full((1, 2, 1, width), width**-0.5)
    value = mx.broadcast_to(mx.array([[[[2.0], [4.0]], [[3.0], [6.0]]]]), (1, 2, 2, 5))
    output, final = gated_delta_update(
        query,
        query,
        value,
        mx.full((1, 2, 2), 0.25),
        mx.full((1, 2, 2), 0.5),
        mx.zeros((1, 2, 5, width)),
    )
    expected = np.broadcast_to(
        np.array([[[[1.0], [2.0]], [[1.625], [3.25]]]]), (1, 2, 2, 5)
    )
    np.testing.assert_allclose(np.array(output), expected, atol=1e-6, rtol=1e-6)
    expected_state = np.broadcast_to(
        np.array([[[[1.625]], [[3.25]]]]) / width**0.5, (1, 2, 5, width)
    )
    np.testing.assert_allclose(np.array(final), expected_state, atol=1e-6, rtol=1e-6)


def test_fp32_recurrence_does_not_use_relaxed_matrix_precision():
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_recurrent_gated_delta_rule,
    )

    random = np.random.default_rng(41)
    query = random.standard_normal((1, 17, 16, 128), dtype=np.float32) * 0.1
    key = random.standard_normal(query.shape, dtype=np.float32) * 0.1
    value = random.standard_normal((1, 17, 32, 128), dtype=np.float32)
    log_decay = random.uniform(-0.7, 0, (1, 17, 32)).astype(np.float32)
    beta = random.uniform(0, 1, log_decay.shape).astype(np.float32)
    reference, reference_state = torch_recurrent_gated_delta_rule(
        torch.from_numpy(query).repeat_interleave(2, dim=2),
        torch.from_numpy(key).repeat_interleave(2, dim=2),
        torch.from_numpy(value),
        torch.from_numpy(log_decay),
        torch.from_numpy(beta),
        None,
        True,
    )
    output, state = gated_delta_update(
        mx.array(query * (128**-0.5)),
        mx.array(key),
        mx.array(value),
        mx.array(np.exp(log_decay)),
        mx.array(beta),
        mx.zeros((1, 32, 128, 128)),
    )
    np.testing.assert_allclose(
        np.array(output), reference.numpy(), atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(
        np.array(state), reference_state.transpose(-1, -2).numpy(), atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("width", [31, 32, 33])
def test_paged_recurrence_reads_permuted_state_offsets_without_padding(width):
    stride = 2 * 5 * width + 17
    offsets = mx.array([3 * stride + 17, stride + 17], mx.uint64)
    arena = mx.full((5, stride), float("nan"))
    arena[3, 17:] = 4.0
    arena[1, 17:] = 8.0
    query = mx.full((2, 2, 1, width), 1 / width)
    output, final = gated_delta_update(
        query,
        mx.zeros_like(query),
        mx.zeros((2, 2, 2, 5)),
        mx.full((2, 2, 2), 0.25),
        mx.zeros((2, 2, 2)),
        arena.view(mx.uint8),
        state_offsets=offsets,
    )
    expected = np.broadcast_to(
        np.array([[[[1.0]], [[0.25]]], [[[2.0]], [[0.5]]]]), (2, 2, 2, 5)
    )
    expected_state = np.broadcast_to(
        np.array([[[[0.25]]], [[[0.5]]]]), (2, 2, 5, width)
    )
    np.testing.assert_allclose(np.array(output), expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.array(final), expected_state, atol=1e-6, rtol=1e-6)
