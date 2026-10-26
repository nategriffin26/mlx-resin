"""Engine-owned KV pages and FP32 convolution/DeltaNet checkpoints."""

from __future__ import annotations

import math
from weakref import proxy

import mlx.core as mx
import numpy as np

from resin import BatchPhase, CacheGroupFamily, CacheGroupLayout, CacheLayout


class CacheGeometry:
    def __init__(self, config, *, dtype, block_tokens=128, prefill_chunk_size=4096):
        self.block_tokens = block_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.conv_shape = (config.linear_conv_kernel_dim - 1, config.linear_conv_dim)
        # MLX's recurrence stores value-major state, unlike Transformers' key-major state.
        self.recurrent_shape = (
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_key_head_dim,
        )
        self.conv_elements = math.prod(self.conv_shape)
        self.shapes = (
            (
                config.full_layer_count,
                block_tokens,
                2,
                config.num_key_value_heads,
                config.head_dim,
            ),
            (
                config.linear_layer_count,
                self.conv_elements + math.prod(self.recurrent_shape),
            ),
        )
        sizes = (math.prod(self.shapes[0]) * dtype.size, math.prod(self.shapes[1]) * 4)
        self.packings = tuple(max(sizes) // size for size in sizes)

    def layout(self, capacity):
        return CacheLayout(
            parent_capacity=capacity,
            groups=(
                CacheGroupLayout(self.block_tokens, blocks_per_slab=self.packings[0]),
                CacheGroupLayout(
                    self.prefill_chunk_size,
                    blocks_per_slab=self.packings[1],
                    family=CacheGroupFamily.STATE,
                ),
            ),
        )


class DecodeState:
    """One write-back cohort, identified only by Engine-owned physical blocks."""

    group = 1

    def __init__(self, arena):
        # The arena owns the read barrier; do not retain a cycle of GPU allocations.
        self.arena = proxy(arena)
        arena.write_back = self
        self.key = None
        self.layers = None

    def prepare(self, plan):
        decode = plan.phase == BatchPhase.DECODE and np.all(
            plan.batch.query_lengths == 1
        )
        key = (
            tuple(
                column.tobytes()
                for group in plan.groups
                for column in (group.append_slab_ids, group.append_slotnums)
            )
            if decode
            else None
        )
        if key != self.key or (
            self.key is not None
            and not np.array_equal(plan.context_lengths, self.contexts)
        ):
            self.flush()
        return key

    def publish(self, key, contexts, cache):
        self.key = key
        self.contexts = contexts + 1
        self.slabs = cache.destination_slabs
        self.slots = cache.destination_slots
        self.layers = cache.updated

    def queue(self, layers, slabs, slots):
        for layer, (convolution, recurrent) in enumerate(layers):
            batch = convolution.shape[0]
            packed = mx.concatenate(
                (
                    convolution.reshape(batch, -1).astype(mx.float32),
                    recurrent.reshape(batch, -1),
                ),
                axis=-1,
            )
            self.arena.write_layer(1, layer, slabs, slots, packed)

    def flush(self):
        """Persist before a different plan, publication, or physical-block retirement."""
        if self.layers is not None:
            self.queue(self.layers, self.slabs, self.slots)
            self.arena.commit()
            self.key = self.layers = None


class DeviceCache:
    """Ephemeral views; the Engine aligns reusable checkpoints to block boundaries."""

    def __init__(self, arena, geometry, tensors, *, history_length=0):
        self.arena = arena
        self.geometry = geometry
        (
            self.history_slabs,
            self.history_slots,
            self.contexts,
            self.append_slabs,
            self.append_slots,
            self.append_offsets,
            self.state_slabs,
            self.state_slots,
            self.destination_slabs,
            self.destination_slots,
        ) = tensors
        self.history_length = history_length
        self.resident = None
        self.updated = None

    @classmethod
    def from_plan(cls, arena, geometry, plan, groups, rows):
        contexts = plan.context_lengths[rows]
        lengths = plan.batch.query_lengths[rows]
        block = geometry.prefill_chunk_size
        previous, destinations = [], []
        for row, context, length in zip(rows, contexts, lengths, strict=True):
            context, length = int(context), int(length)
            previous.append(
                plan.groups[1].address(row, (context - 1) // block)
                if context
                else (0, 0)
            )
            destinations.append(
                plan.groups[1].address(row, (context + length - 1) // block)
            )
        state_slabs, state_slots = np.asarray(previous, np.uint32).T
        destination_slabs, destination_slots = np.asarray(destinations, np.uint32).T
        indices = np.concatenate(
            [
                np.arange(plan.token_indptr[row], plan.token_indptr[row + 1])
                for row in rows
            ]
        )
        history = plan.groups[0]
        device_rows = mx.array(rows)
        tensors = (
            groups[0].slab_ids[device_rows],
            groups[0].slotnums[device_rows],
            mx.array(contexts),
            mx.array(history.append_slab_ids[indices]),
            mx.array(history.append_slotnums[indices]),
            mx.array(history.append_block_offsets[indices]),
            mx.array(state_slabs),
            mx.array(state_slots),
            mx.array(destination_slabs),
            mx.array(destination_slots),
        )
        return cls(
            arena,
            geometry,
            tensors,
            history_length=int(contexts.max()),
        )

    def history(self, layer):
        if not self.history_length:
            return None
        columns = math.ceil(self.history_length / self.geometry.block_tokens)
        pages = self.arena.read_layer(
            0, layer, self.history_slabs[:, :columns], self.history_slots[:, :columns]
        ).reshape(self.contexts.size, -1, *self.geometry.shapes[0][-3:])
        pages = pages[:, : self.history_length]
        valid = mx.arange(self.history_length)[None] < self.contexts[:, None]
        return mx.where(valid[:, :, None, None, None], pages, 0)

    def append(self, layer, keys, values):
        self.arena.write_layer(
            0,
            layer,
            self.append_slabs,
            self.append_slots,
            mx.stack((keys, values), axis=2).reshape(-1, 2, *keys.shape[-2:]),
            offsets=self.append_offsets,
        )

    def state(self, layer):
        if self.resident is not None:
            convolution, recurrent = self.resident[layer]
            return convolution, recurrent, None
        convolution = self.arena.read_layer(
            1,
            layer,
            self.state_slabs,
            self.state_slots,
            prefix_elements=self.geometry.conv_elements,
        )
        offsets = self.arena._base(1, layer, self.state_slabs, self.state_slots)
        # Padding keeps metadata in Metal's device address space at every batch size.
        offsets = mx.concatenate(
            (offsets + self.geometry.conv_elements, mx.zeros(32, mx.uint64))
        )
        return (
            convolution.reshape(-1, *self.geometry.conv_shape),
            self.arena.storage,
            offsets,
        )

    def store(self, layer, convolution, recurrent):
        if self.updated is not None:
            self.updated.append((convolution, recurrent))
            return
        batch = convolution.shape[0]
        packed = mx.concatenate(
            (
                convolution.reshape(batch, -1).astype(mx.float32),
                recurrent.reshape(batch, -1),
            ),
            axis=-1,
        )
        self.arena.write_layer(
            1,
            layer,
            self.destination_slabs,
            self.destination_slots,
            packed,
        )
