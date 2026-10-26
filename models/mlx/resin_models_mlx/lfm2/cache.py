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

    def __init__(self, arena: SlabArena, plan, groups, rows):
        self.arena, self.plan, self.groups, self.rows = arena, plan, groups, rows
        self.contexts = plan.context_lengths[rows]
        self.lengths = plan.batch.query_lengths[rows]
        self.block_tokens = arena.layout.groups[0].block_tokens
        token_indices = np.concatenate(
            [
                np.arange(plan.token_indptr[row], plan.token_indptr[row + 1])
                for row in rows
            ]
        )
        history = plan.groups[0]
        self.append_slabs = mx.array(history.append_slab_ids[token_indices])
        self.append_slots = mx.array(history.append_slotnums[token_indices])
        self.append_offsets = mx.array(history.append_block_offsets[token_indices])

    def history(self, layer):
        length = int(self.contexts.max())
        if not length:
            return None
        columns = math.ceil(length / self.block_tokens)
        group = self.groups[0]
        rows = mx.array(self.rows)
        pages = self.arena.read_layer(
            0, layer, group.slab_ids[rows, :columns], group.slotnums[rows, :columns]
        )
        pages = pages.reshape(len(self.rows), -1, *pages.shape[-3:])[:, :length]
        valid = mx.arange(length)[None, :] < mx.array(self.contexts)[:, None]
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
        addresses = [
            self.plan.groups[1].address(row, (int(context) - 1) // self.block_tokens)
            if context
            else (0, 0)
            for row, context in zip(self.rows, self.contexts, strict=True)
        ]
        slabs, slots = np.asarray(addresses, np.uint32).T
        return self.arena.read_layer(1, layer, mx.array(slabs), mx.array(slots))

    def store_states(self, layer, convolution_input, state_length):
        slabs, slots, states = [], [], []
        for batch_row, row in enumerate(self.rows):
            context, length = (
                int(self.contexts[batch_row]),
                int(self.lengths[batch_row]),
            )
            ends = list(
                range(
                    self.block_tokens - context % self.block_tokens,
                    length + 1,
                    self.block_tokens,
                )
            )
            if not ends or ends[-1] != length:
                ends.append(length)
            for end in ends:
                slab, slot = self.plan.groups[1].address(
                    row, (context + end - 1) // self.block_tokens
                )
                slabs.append(slab)
                slots.append(slot)
                states.append(convolution_input[batch_row, end : end + state_length])
        self.arena.write_layer(
            1,
            layer,
            mx.array(slabs, mx.uint32),
            mx.array(slots, mx.uint32),
            mx.stack(states),
        )
