"""LFM2 history pages and short-convolution checkpoints in one MLX arena."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout
from resin.backend.mlx import SlabArena


class CacheGeometry:
    def __init__(self, config, *, block_tokens=128):
        self.block_tokens = block_tokens
        heads = config.num_key_value_heads
        dim = config.hidden_size // config.num_attention_heads
        self.shapes = (
            (config.layer_types.count("full_attention"), block_tokens, 2, heads, dim),
            (
                config.layer_types.count("conv"),
                config.conv_cache_length - 1,
                config.hidden_size,
            ),
        )
        elements = tuple(math.prod(shape) for shape in self.shapes)
        largest = max(elements)
        self.packings = tuple(largest // size for size in elements)

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
    """Ephemeral views for a prefill row or a batch of one-token decode rows."""

    def __init__(self, arena: SlabArena, tensors, *, history_length=0):
        self.arena = arena
        self.tensors = tensors
        (
            self.history_slabs,
            self.history_slots,
            self.device_contexts,
            self.append_slabs,
            self.append_slots,
            self.append_offsets,
            self.state_slabs,
            self.state_slots,
            self.destination_slabs,
            self.destination_slots,
        ) = tensors
        self.history_length = history_length
        self.block_tokens = arena.layout.groups[0].block_tokens

    @classmethod
    def from_plan(cls, arena, plan, groups, rows):
        contexts = plan.context_lengths[rows]
        lengths = plan.batch.query_lengths[rows]
        block_tokens = arena.layout.groups[0].block_tokens
        previous, destinations = [], []
        for row, context, length in zip(rows, contexts, lengths, strict=True):
            previous.append(
                plan.groups[1].address(row, (int(context) - 1) // block_tokens)
                if context
                else (0, 0)
            )
            destinations.append(
                plan.groups[1].address(row, (int(context + length) - 1) // block_tokens)
            )
        state_slabs, state_slots = np.asarray(previous, np.uint32).T
        destination_slabs, destination_slots = np.asarray(destinations, np.uint32).T
        token_indices = np.concatenate(
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
            mx.array(history.append_slab_ids[token_indices]),
            mx.array(history.append_slotnums[token_indices]),
            mx.array(history.append_block_offsets[token_indices]),
            mx.array(state_slabs),
            mx.array(state_slots),
            mx.array(destination_slabs),
            mx.array(destination_slots),
        )
        return cls(arena, tensors, history_length=int(contexts.max()))

    def history(self, layer):
        length = self.history_length
        if not length:
            return None
        columns = math.ceil(length / self.block_tokens)
        pages = self.arena.read_layer(
            0, layer, self.history_slabs[:, :columns], self.history_slots[:, :columns]
        )
        pages = pages.reshape(self.device_contexts.size, -1, *pages.shape[-3:])[
            :, :length
        ]
        valid = mx.arange(length)[None, :] < self.device_contexts[:, None]
        # A mask inside SDPA is too late if a padded value already contains NaN.
        return mx.where(valid[:, :, None, None, None], pages, 0)

    def append(self, layer, keys, values):
        values = mx.stack((keys, values), axis=2).reshape(-1, 2, *keys.shape[-2:])
        self.arena.write_layer(
            0,
            layer,
            self.append_slabs,
            self.append_slots,
            values,
            offsets=self.append_offsets,
        )

    def state(self, layer):
        return self.arena.read_layer(1, layer, self.state_slabs, self.state_slots)

    def store_states(self, layer, convolution_input, state_length):
        self.arena.write_layer(
            1,
            layer,
            self.destination_slabs,
            self.destination_slots,
            convolution_input[:, -state_length:],
        )
