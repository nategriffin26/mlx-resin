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
