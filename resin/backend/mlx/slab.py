"""Typed MLX views over a single Engine-addressed fungible slab allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from resin import CacheLayout
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding


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
    """One allocation, with model-declared [layer, ...] shapes for each group.

    Writes are deferred until all reads and new values have been evaluated. This
    lets MLX donate the backing buffer instead of copying it for every layer.
    No tensor view or request identity is retained between forwards.
    """

    def __init__(self, layout: CacheLayout, group_shapes, *, dtype=mx.bfloat16):
        self.layout = layout
        self.group_shapes = tuple(tuple(shape) for shape in group_shapes)
        if len(self.group_shapes) != len(layout.groups):
            raise ValueError("one tensor shape is required per cache group")
        if any(
            not shape or any(size <= 0 for size in shape) for shape in self.group_shapes
        ):
            raise ValueError("cache shapes must have positive layer and element counts")
        self.block_elements = tuple(math.prod(shape) for shape in self.group_shapes)
        self.slab_elements = max(
            elements * group.blocks_per_slab
            for elements, group in zip(self.block_elements, layout.groups, strict=True)
        )
        self.storage = mx.zeros(
            (layout.parent_capacity + 1) * self.slab_elements, dtype
        )
        mx.eval(self.storage)
        self.binding = CachePoolBinding(layout, resource="MLX slab arena")
        self._writes: list[tuple[mx.array, mx.array]] = []

    def bind_resources(self, binding) -> None:
        self.binding.bind_resources(binding)

    def _base(self, group, layer, slabs, slots):
        shape = self.group_shapes[group]
        if not 0 <= layer < shape[0]:
            raise ValueError("cache layer lies outside its cache group")
        return (
            slabs * self.slab_elements
            + slots * self.block_elements[group]
            + layer * math.prod(shape[1:])
        )

    def read_layer(self, group, layer, slabs, slots):
        """Gather complete layer pages at validated physical addresses."""
        shape = self.group_shapes[group][1:]
        base = self._base(group, layer, slabs, slots)
        indices = base[..., None] + mx.arange(math.prod(shape), dtype=mx.uint32)
        return self.storage[indices].reshape(*base.shape, *shape)

    def write_layer(self, group, layer, slabs, slots, values, *, offsets=None):
        """Queue layer-page writes, or token writes into a history layer page."""
        shape = self.group_shapes[group][1:]
        base = self._base(group, layer, slabs, slots)
        if offsets is not None:
            shape = shape[1:]
            base = base + offsets * math.prod(shape)
        expected = (*base.shape, *shape)
        if values.shape != expected:
            raise ValueError(
                f"cache write shape {values.shape} does not match {expected}"
            )
        indices = base[..., None] + mx.arange(math.prod(shape), dtype=mx.uint32)
        self._writes.append((indices.reshape(-1), values.reshape(-1)))

    def initialize_fresh(self, plan: PagedCachePlan) -> None:
        """Clear blocks starting here, including recycled padding and state."""
        indices = []
        if np.any(plan.context_lengths == 0):
            indices.append(mx.arange(self.slab_elements, dtype=mx.uint32))
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
                addresses[:, 0] * self.slab_elements
                + addresses[:, 1] * self.block_elements[index]
            )
            indices.append(
                (
                    mx.array(base)[:, None]
                    + mx.arange(
                        self.block_elements[index],
                        dtype=mx.uint32,
                    )
                ).reshape(-1)
            )
        if indices:
            self.storage[mx.concatenate(indices)] = 0
            mx.eval(self.storage)

    def commit(self) -> None:
        """Finish reads before mutation, then synchronously commit leased writes."""
        if not self._writes:
            return
        indices = mx.concatenate([index for index, _ in self._writes])
        values = mx.concatenate([value for _, value in self._writes])
        mx.eval(indices, values)
        self._writes.clear()
        self.storage[indices] = values
        mx.eval(self.storage)
