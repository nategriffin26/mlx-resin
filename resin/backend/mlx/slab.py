"""Mixed-precision MLX views over Engine-addressed fungible byte slabs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import numpy as np

from resin import CacheGroupFamily, CacheLayout
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding


def _physical_base(slabs, slots, slab_stride, block_stride, layer_offset):
    return (
        slabs.astype(mx.uint64) * slab_stride
        + slots.astype(mx.uint64) * block_stride
        + layer_offset
    )


@partial(mx.compile, shapeless=True)
def _physical_indices(
    slabs,
    slots,
    offsets,
    elements,
    slab_stride,
    block_stride,
    layer_offset,
    token_stride,
):
    return (
        _physical_base(slabs, slots, slab_stride, block_stride, layer_offset)
        + offsets.astype(mx.uint64) * token_stride
        + elements
    )


_read_pages = mx.fast.metal_kernel(
    name="resin_read_pages",
    input_names=["storage", "addresses", "geometry"],
    output_names=["pages"],
    source="""
        size_t i = thread_position_in_grid.x;
        size_t width = geometry[3];
        size_t page = i / width;
        if (page >= geometry[4]) return;
        size_t base = size_t(addresses[page]) * geometry[0]
                    + size_t(addresses[geometry[4] + page]) * geometry[1] + geometry[2];
        const device T* typed = reinterpret_cast<const device T*>(storage);
        pages[i] = typed[base + i % width];
    """,
)


@dataclass(frozen=True, slots=True)
class DevicePageGroup:
    slab_ids: mx.array
    slotnums: mx.array
    first_blocks: mx.array
    counts: mx.array


def stage_page_groups(plan: PagedCachePlan) -> tuple[DevicePageGroup, ...]:
    """Pad validated CSR tables with the reserved null slab, not live pages."""
    groups = []
    for source in plan.groups:
        counts = np.diff(source.block_indptr)
        width = int(counts.max())
        slabs = np.zeros((plan.rows, width), dtype=np.uint32)
        slots = np.zeros_like(slabs)
        for row in range(plan.rows):
            slabs[row, : counts[row]] = source.block_slabs(row)
            slots[row, : counts[row]] = source.block_slots(row)
        groups.append(
            DevicePageGroup(
                mx.array(slabs),
                mx.array(slots),
                mx.array(source.block_table_start_indices),
                mx.array(counts),
            )
        )
    return tuple(groups)


class SlabArena:
    """One byte allocation with a model-declared shape and dtype per group.

    Writes are deferred until all reads and new values have been evaluated. This
    lets MLX donate the backing buffer instead of copying it for every layer.
    No tensor view or request identity is retained between forwards.
    """

    def __init__(self, layout: CacheLayout, group_shapes, *, group_dtypes=None):
        self.layout = layout
        self.group_shapes = tuple(tuple(shape) for shape in group_shapes)
        if len(self.group_shapes) != len(layout.groups):
            raise ValueError("one tensor shape is required per cache group")
        if any(
            not shape or any(size <= 0 for size in shape) for shape in self.group_shapes
        ):
            raise ValueError("cache shapes must have positive layer and element counts")
        self.group_dtypes = (
            tuple(group_dtypes)
            if group_dtypes is not None
            else (mx.bfloat16,) * len(self.group_shapes)
        )
        if len(self.group_dtypes) != len(self.group_shapes):
            raise ValueError("one dtype is required per cache group")
        self.item_sizes = tuple(dtype.size for dtype in self.group_dtypes)
        self.block_elements = tuple(math.prod(shape) for shape in self.group_shapes)
        self.block_bytes = tuple(
            count * size
            for count, size in zip(self.block_elements, self.item_sizes, strict=True)
        )
        alignment = math.lcm(*self.item_sizes)
        largest = max(
            size * group.blocks_per_slab
            for size, group in zip(self.block_bytes, layout.groups, strict=True)
        )
        self.slab_bytes = (largest + alignment - 1) // alignment * alignment
        # Scatter contiguous blocks, not an address for every recurrent-state word.
        self.write_unit_bytes = math.gcd(
            *(
                math.prod(
                    shape[1:] if group.family == CacheGroupFamily.STATE else shape[2:]
                )
                * size
                for shape, size, group in zip(
                    self.group_shapes, self.item_sizes, layout.groups, strict=True
                )
            )
        )
        self.write_size = math.gcd(4, self.write_unit_bytes)
        self.write_unit_words = self.write_unit_bytes // self.write_size
        self.write_dtype = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}[self.write_size]
        self._write_ranges = tuple(
            tuple(
                mx.arange(
                    math.prod(shape[start:]) * size // self.write_unit_bytes,
                    dtype=mx.uint64,
                )
                for start in (1, 2)
            )
            for shape, size in zip(self.group_shapes, self.item_sizes, strict=True)
        )
        mx.eval(self._write_ranges)
        self.storage = mx.zeros((layout.parent_capacity + 1, self.slab_bytes), mx.uint8)
        mx.eval(self.storage)
        self.binding = CachePoolBinding(layout, resource="MLX slab arena")
        self._writes: list[tuple[mx.array, mx.array]] = []

    def bind_resources(self, binding) -> None:
        self.binding.bind_resources(binding)

    def _base(self, group, layer, slabs, slots):
        shape = self.group_shapes[group]
        if not 0 <= layer < shape[0]:
            raise ValueError("cache layer lies outside its cache group")
        return _physical_base(
            slabs,
            slots,
            self.slab_bytes // self.item_sizes[group],
            self.block_elements[group],
            layer * math.prod(shape[1:]),
        )

    def read_layer(self, group, layer, slabs, slots):
        """Gather complete layer pages at validated physical addresses."""
        shape = self.group_shapes[group][1:]
        if not 0 <= layer < self.group_shapes[group][0]:
            raise ValueError("cache layer lies outside its cache group")
        width = math.prod(shape)
        return _read_pages(
            inputs=[
                self.storage,
                mx.concatenate(
                    (slabs.reshape(-1), slots.reshape(-1), mx.zeros(32, mx.uint32))
                ),
                mx.array(
                    [
                        self.slab_bytes // self.item_sizes[group],
                        self.block_elements[group],
                        layer * width,
                        width,
                        slabs.size,
                    ],
                    mx.uint64,
                ),
            ],
            template=[("T", self.group_dtypes[group])],
            grid=(slabs.size * width, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(*slabs.shape, *shape)],
            output_dtypes=[self.group_dtypes[group]],
        )[0]

    def write_layer(self, group, layer, slabs, slots, values, *, offsets=None):
        """Queue layer-page writes, or token writes into a history layer page."""
        shape = self.group_shapes[group][1:]
        if not 0 <= layer < self.group_shapes[group][0]:
            raise ValueError("cache layer lies outside its cache group")
        token_write = offsets is not None
        if token_write:
            if self.layout.groups[group].family == CacheGroupFamily.STATE:
                raise ValueError("token offsets require a history cache group")
            shape = shape[1:]
        else:
            offsets = mx.zeros_like(slabs)
        expected = (*slabs.shape, *shape)
        if values.shape != expected:
            raise ValueError(
                f"cache write shape {values.shape} does not match {expected}"
            )
        size = self.item_sizes[group]
        indices = _physical_indices(
            slabs[..., None],
            slots[..., None],
            offsets[..., None],
            self._write_ranges[group][int(token_write)],
            self.slab_bytes // self.write_unit_bytes,
            self.block_bytes[group] // self.write_unit_bytes,
            layer
            * math.prod(self.group_shapes[group][1:])
            * size
            // self.write_unit_bytes,
            math.prod(shape) * size // self.write_unit_bytes,
        )
        self._writes.append(
            (
                indices.reshape(-1),
                values.astype(self.group_dtypes[group])
                .reshape(-1)
                .view(self.write_dtype)
                .reshape(-1, self.write_unit_words),
            )
        )

    def initialize_fresh(self, plan: PagedCachePlan) -> None:
        """Clear blocks starting here, including recycled padding and state."""
        indices = []
        if np.any(plan.context_lengths == 0):
            indices.append(
                mx.arange(self.slab_bytes // self.write_unit_bytes, dtype=mx.uint64)
            )
        for index, group in enumerate(plan.groups):
            fresh = group.append_block_offsets == 0
            addresses = np.unique(
                np.stack(
                    (
                        group.append_slab_ids[fresh],
                        group.append_slotnums[fresh],
                    ),
                    axis=1,
                ),
                axis=0,
            )
            if not len(addresses):
                continue
            base = (
                addresses[:, 0].astype(np.uint64) * self.slab_bytes
                + addresses[:, 1].astype(np.uint64) * self.block_bytes[index]
            )
            indices.append(
                (
                    mx.array(base // self.write_unit_bytes)[:, None]
                    + mx.arange(
                        self.block_bytes[index] // self.write_unit_bytes,
                        dtype=mx.uint64,
                    )
                ).reshape(-1)
            )
        if indices:
            units = self.slab_bytes // self.write_unit_bytes
            shape = self.storage.shape
            self.storage = self.storage.view(self.write_dtype).reshape(
                shape[0], units, self.write_unit_words
            )
            addresses = mx.concatenate(indices)
            self.storage[addresses // units, addresses % units] = 0
            self.storage = self.storage.reshape(shape[0], -1).view(mx.uint8)
            mx.eval(self.storage)

    def commit(self, *outputs) -> None:
        """Evaluate outputs and reads together, then synchronously commit leased writes."""
        if not self._writes:
            mx.eval(*outputs)
            return
        indices = mx.concatenate([index for index, _ in self._writes])
        values = mx.concatenate([value for _, value in self._writes])
        mx.eval(indices, values, *outputs)
        self._writes.clear()
        units = self.slab_bytes // self.write_unit_bytes
        shape = self.storage.shape
        self.storage = self.storage.view(self.write_dtype).reshape(
            shape[0], units, self.write_unit_words
        )
        self.storage[indices // units, indices % units] = values
        self.storage = self.storage.reshape(shape[0], -1).view(mx.uint8)
        mx.eval(self.storage)
