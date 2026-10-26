"""Split-history decode attention without materialized K/V or element indices."""

from __future__ import annotations

import mlx.core as mx

from resin.backend.mlx.slab import SlabArena

_partials = mx.fast.metal_kernel(
    name="resin_attention_partials",
    input_names=[
        "queries",
        "keys",
        "values",
        "storage",
        "metadata",
        "geometry",
        "scale",
    ],
    output_names=["partials"],
    source="""
        uint lane = thread_position_in_threadgroup.x;
        uint head = thread_position_in_grid.y;
        uint partition = thread_position_in_grid.z % geometry[5];
        uint row = thread_position_in_grid.z / geometry[5];
        uint kv_head = head / (queries_shape[2] / keys_shape[2]);
        size_t table_size = size_t(queries_shape[0]) * geometry[6];
        uint context = metadata[2 * table_size + row];
        uint begin = partition * 32;
        uint end = min(begin + 32, context + 1);
        float q[D / 32], acc[D / 32];
        for (uint d = 0; d < D / 32; ++d) {
            q[d] = float(queries[(row * queries_shape[2] + head) * D + d * 32 + lane]);
            acc[d] = 0;
        }
        float maximum = -INFINITY, denominator = 0;
        const device T* history = reinterpret_cast<const device T*>(storage);
        for (uint token = begin; token < end; ++token) {
            const device T* k;
            const device T* v;
            if (token < context) {
                uint page = token / geometry[3];
                size_t table = size_t(row) * geometry[6] + page;
                size_t base = size_t(metadata[table]) * geometry[0]
                            + size_t(metadata[table_size + table]) * geometry[1] + geometry[2]
                            + size_t(token % geometry[3]) * 2 * keys_shape[2] * D;
                k = history + base + kv_head * D;
                v = k + keys_shape[2] * D;
            } else {
                k = keys + (row * keys_shape[2] + kv_head) * D;
                v = values + (row * keys_shape[2] + kv_head) * D;
            }
            // Bounds precede both loads: masked NaNs never enter arithmetic.
            float dot = 0;
            for (uint d = 0; d < D / 32; ++d)
                dot += q[d] * float(k[d * 32 + lane]);
            float score = simd_sum(dot) * scale[0];
            float next_maximum = max(maximum, score);
            float correction = exp(maximum - next_maximum);
            float probability = exp(score - next_maximum);
            denominator = denominator * correction + probability;
            for (uint d = 0; d < D / 32; ++d)
                acc[d] = acc[d] * correction + probability * float(v[d * 32 + lane]);
            maximum = next_maximum;
        }
        size_t output = ((size_t(row) * queries_shape[2] + head) * geometry[5] + partition) * (D + 2);
        for (uint d = 0; d < D / 32; ++d)
            partials[output + d * 32 + lane] = acc[d];
        if (lane == 0) {
            partials[output + D] = maximum;
            partials[output + D + 1] = denominator;
        }
    """,
)

_combine = mx.fast.metal_kernel(
    name="resin_attention_combine",
    input_names=["partials"],
    output_names=["output"],
    source="""
        uint lane = thread_position_in_threadgroup.x;
        uint head = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;
        uint count = partials_shape[2];
        size_t base = (size_t(row) * partials_shape[1] + head) * count * (D + 2);
        float maximum = -INFINITY;
        for (uint p = 0; p < count; ++p)
            maximum = max(maximum, partials[base + p * (D + 2) + D]);
        float acc[D / 32], denominator = 0;
        for (uint d = 0; d < D / 32; ++d) acc[d] = 0;
        for (uint p = 0; p < count; ++p) {
            size_t offset = base + p * (D + 2);
            float weight = exp(partials[offset + D] - maximum);
            denominator += weight * partials[offset + D + 1];
            for (uint d = 0; d < D / 32; ++d)
                acc[d] += weight * partials[offset + d * 32 + lane];
        }
        size_t destination = (size_t(row) * partials_shape[1] + head) * D;
        for (uint d = 0; d < D / 32; ++d)
            output[destination + d * 32 + lane] = T(acc[d] / denominator);
    """,
)


def paged_attention(
    queries,
    keys,
    values,
    arena: SlabArena,
    group,
    layer,
    slabs,
    slots,
    contexts,
    *,
    scale,
):
    """Attend one query per row to history plus this forward's uncommitted K/V.

    Q/K/V use [batch, 1, heads, dim]; cache pages use [layer, token, 2, kv_heads, dim].
    Tables start at logical block zero. Head dimensions are multiples of 32.
    Lengths, batch sizes and table widths do not specialize the kernels.
    """
    batch, length, heads, dim = queries.shape
    if length != 1 or dim % 32:
        raise ValueError(
            "paged decode attention needs one query and a head dimension divisible by 32"
        )
    if queries.dtype != arena.group_dtypes[group]:
        raise ValueError("attention queries and history must have the same dtype")
    partitions = (slabs.shape[1] * arena.layout.groups[group].block_tokens + 32) // 32
    shape = arena.group_shapes[group]
    geometry = mx.array(
        [
            arena.slab_bytes // arena.item_sizes[group],
            arena.block_elements[group],
            layer * shape[1] * 2 * shape[3] * dim,
            shape[1],
            shape[3],
            partitions,
            slabs.shape[1],
        ],
        mx.uint64,
    )
    # Avoid MLX's small-host-array constant-address-space specialization.
    metadata = mx.concatenate(
        (slabs.reshape(-1), slots.reshape(-1), contexts, mx.zeros(32, mx.uint32))
    )
    partials = _partials(
        inputs=[
            queries,
            keys,
            values,
            arena.storage,
            metadata,
            geometry,
            mx.array([scale]),
        ],
        template=[("T", queries.dtype), ("D", dim)],
        grid=(32, heads, batch * partitions),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, heads, partitions, dim + 2)],
        output_dtypes=[mx.float32],
    )[0]
    return _combine(
        inputs=[partials],
        template=[("T", queries.dtype), ("D", dim)],
        grid=(32, heads, batch),
        threadgroup=(32, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
