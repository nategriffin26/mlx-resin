"""Attack wide physical addresses and mixed-precision cache ownership."""

import sys

import numpy as np
import pytest

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout

if sys.platform == "darwin":
    mx = pytest.importorskip("mlx.core")
    from resin.backend.mlx import SlabArena
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


def test_physical_offsets_do_not_wrap_at_uint32_element_boundary():
    arena = SlabArena(
        CacheLayout(parent_capacity=2, groups=(CacheGroupLayout(128),)), ((1, 2),)
    )
    # Exercise real address arithmetic without allocating an 8 GiB test fixture.
    arena.slab_bytes = 2**32
    actual = arena._base(0, 0, mx.array([1, 2], mx.uint32), mx.array([0, 1], mx.uint32))
    assert actual.tolist() == [2**31, 2**32 + 2]
    arena.slab_bytes = 2**33 - 4
    actual = arena._base(
        0, 0, mx.array([1, 1, 1], mx.uint32), mx.array([0, 1, 2], mx.uint32)
    )
    assert actual.tolist() == [2**32 - 2, 2**32, 2**32 + 2]


def test_mixed_group_dtypes_preserve_fp32_state_and_bfloat16_history():
    layout = CacheLayout(
        parent_capacity=2,
        groups=(
            CacheGroupLayout(2),
            CacheGroupLayout(2, blocks_per_slab=2, family=CacheGroupFamily.STATE),
        ),
    )
    arena = SlabArena(
        layout, ((1, 2, 4), (1, 2)), group_dtypes=(mx.bfloat16, mx.float32)
    )
    history = mx.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], mx.bfloat16)
    state = mx.array([[1.0000001192092896, -12345.6787109375]], mx.float32)
    slab = mx.array([1], mx.uint32)
    zero, one = mx.array([0], mx.uint32), mx.array([1], mx.uint32)
    arena.write_layer(0, 0, slab, zero, history)
    arena.commit()
    assert arena.read_layer(0, 0, slab, zero).tolist() == history.tolist()
    # The Engine may recycle the same physical slab for a different group.
    arena.write_layer(1, 0, slab, one, state)
    arena.commit()
    result = arena.read_layer(1, 0, slab, one)
    assert result.dtype == mx.float32
    np.testing.assert_array_equal(np.array(result), np.array(state))
    # A single commit also accepts writes of different dtypes on distinct slabs.
    arena.write_layer(0, 0, mx.array([2], mx.uint32), zero, history)
    arena.write_layer(1, 0, slab, zero, state)
    arena.commit()
    np.testing.assert_array_equal(
        np.array(arena.read_layer(1, 0, slab, zero)), np.array(state)
    )
    assert arena.storage.nbytes == (layout.parent_capacity + 1) * arena.slab_bytes


@pytest.mark.parametrize("capacity", [2**21 - 2, 2**21 - 1, 2**21])
def test_byte_capacity_crosses_signed_dimension_limit_without_flattening(
    monkeypatch, capacity
):
    # MLX validates lazy array dimensions before allocation. Suppress only
    # evaluation so this boundary test never reserves a multi-GiB buffer.
    monkeypatch.setattr(mx, "eval", lambda *arrays: None)
    arena = SlabArena(
        CacheLayout(parent_capacity=capacity, groups=(CacheGroupLayout(1),)),
        ((1, 512),),
    )
    assert arena.storage.shape == (capacity + 1, 1024)
    assert arena.storage.nbytes == (capacity + 1) * 1024
