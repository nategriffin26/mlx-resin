"""Check direct page attention against independent dense arithmetic."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from resin import CacheGroupLayout, CacheLayout

if sys.platform == "darwin":
    mx = pytest.importorskip("mlx.core")
    from resin.backend.mlx import SlabArena
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


@pytest.mark.parametrize(
    "length",
    [0, 1, 7, 8, 9, 31, 32, 33, 127, 128, 129, 255, 256, 257, 4095, 4096, 32768],
)
def test_each_history_token_is_visited_once_without_reading_nan_padding(length):
    from resin.backend.mlx import paged_attention

    lengths = [length, max(0, length - 17)]
    width = max(1, math.ceil(length / 128))
    arena = SlabArena(
        CacheLayout(
            parent_capacity=2 * width,
            groups=(CacheGroupLayout(128, blocks_per_slab=2),),
        ),
        ((1, 128, 2, 1, 32),),
        group_dtypes=(mx.float32,),
    )
    arena.storage = mx.full(arena.storage.size // 4, float("nan")).view(mx.uint8)
    slabs = (
        np.arange(1, 2 * width + 1, dtype=np.uint32).reshape(2, width)[:, ::-1].copy()
    )
    slots = np.ones_like(slabs)
    for row, count in enumerate(lengths):
        if not count:
            continue
        token = np.arange(count, dtype=np.uint32)
        values = np.zeros((count, 2, 1, 32), np.float32)
        values[:, 1, 0] = (token[:, None] % 29) - 14.0
        arena.write_layer(
            0,
            0,
            mx.array(slabs[row, token // 128]),
            mx.array(slots[row, token // 128]),
            mx.array(values),
            offsets=mx.array(token % 128),
        )
    arena.commit()
    output = paged_attention(
        mx.zeros((2, 1, 2, 32)),
        mx.zeros((2, 1, 1, 32)),
        mx.full((2, 1, 1, 32), 3.0),
        arena,
        0,
        0,
        mx.array(slabs),
        mx.array(slots),
        mx.array(lengths, mx.uint32),
        scale=32**-0.5,
    )
    actual = np.array(output)
    for row, count in enumerate(lengths):
        expected = (sum(index % 29 - 14 for index in range(count)) + 3) / (count + 1)
        np.testing.assert_allclose(actual[row], expected, atol=2e-5, rtol=2e-5)


def test_paged_attention_preserves_softmax_and_grouped_query_heads():
    from resin.backend.mlx import paged_attention

    rng = np.random.default_rng(42)
    keys = rng.normal(size=(129, 2, 64)).astype(np.float32)
    values = rng.normal(size=(129, 2, 64)).astype(np.float32)
    queries = rng.normal(size=(1, 1, 4, 64)).astype(np.float32)
    arena = SlabArena(
        CacheLayout(parent_capacity=2, groups=(CacheGroupLayout(128),)),
        ((1, 128, 2, 2, 64),),
        group_dtypes=(mx.float32,),
    )
    arena.write_layer(
        0,
        0,
        mx.array([1], mx.uint32),
        mx.array([0], mx.uint32),
        mx.array(np.stack((keys[:128], values[:128]), axis=1)[None]),
    )
    arena.commit()
    actual = paged_attention(
        mx.array(queries),
        mx.array(keys[128:][None]),
        mx.array(values[128:][None]),
        arena,
        0,
        0,
        mx.array([[1]], mx.uint32),
        mx.array([[0]], mx.uint32),
        mx.array([128], mx.uint32),
        scale=0.125,
    )
    expected = []
    for head in range(4):
        scores = keys[:, head // 2] @ queries[0, 0, head] * 0.125
        probabilities = np.exp(scores - scores.max())
        expected.append(probabilities @ values[:, head // 2] / probabilities.sum())
    np.testing.assert_allclose(np.array(actual)[0, 0], expected, atol=2e-6, rtol=2e-5)


def test_novel_shapes_do_not_compile_new_metal_page_kernels():
    # A fresh process prevents earlier tests from prewarming the very misses
    # this regression must detect. The probe counts the native compiler entry.
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/benchmark_mlx_kernels.py")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
