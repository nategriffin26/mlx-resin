"""Engine-owned KV pages and FP32 convolution/DeltaNet checkpoints."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout


class CacheGeometry:
    def __init__(self, config, *, dtype, block_tokens=128):
        self.block_tokens = block_tokens
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
                    self.block_tokens,
                    blocks_per_slab=self.packings[1],
                    family=CacheGroupFamily.STATE,
                ),
            ),
        )


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

    @classmethod
    def from_plan(cls, arena, geometry, plan, groups, rows):
        contexts = plan.context_lengths[rows]
        lengths = plan.batch.query_lengths[rows]
        block = geometry.block_tokens
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
        packed = self.arena.read_layer(1, layer, self.state_slabs, self.state_slots)
        split = self.geometry.conv_elements
        return (
            packed[:, :split].reshape(-1, *self.geometry.conv_shape),
            packed[:, split:].reshape(-1, *self.geometry.recurrent_shape),
        )

    def store(self, layer, convolution, recurrent):
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
