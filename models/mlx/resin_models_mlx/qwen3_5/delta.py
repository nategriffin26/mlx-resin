"""FP32 gated DeltaNet recurrence without decay clamping or relaxed matmul."""

import mlx.core as mx

_recurrence = mx.fast.metal_kernel(
    name="resin_qwen35_delta",
    input_names=["query", "key", "value", "decay", "beta", "initial"],
    output_names=["output", "final"],
    source="""
        uint lane = thread_position_in_threadgroup.x;
        uint column = thread_position_in_grid.y;
        uint head = thread_position_in_grid.z % value_shape[2];
        uint row = thread_position_in_grid.z / value_shape[2];
        uint key_head = head / (value_shape[2] / key_shape[2]);
        uint tokens = query_shape[1];
        size_t state_base = ((size_t(row) * value_shape[2] + head)
                            * value_shape[3] + column) * D;
        float state[(D + 31) / 32];
        for (uint index = 0; index < (D + 31) / 32; ++index)
            state[index] = index * 32 + lane < D
                ? initial[state_base + index * 32 + lane] : 0;
        for (uint token = 0; token < tokens; ++token) {
            size_t gate_offset = (size_t(row) * tokens + token) * value_shape[2] + head;
            size_t key_offset = ((size_t(row) * tokens + token) * key_shape[2]
                                 + key_head) * D;
            size_t value_offset = gate_offset * value_shape[3] + column;
            float memory = 0;
            float keys[(D + 31) / 32];
            for (uint index = 0; index < (D + 31) / 32; ++index) {
                keys[index] = index * 32 + lane < D
                    ? key[key_offset + index * 32 + lane] : 0;
                state[index] *= decay[gate_offset];
                memory += state[index] * keys[index];
            }
            memory = simd_sum(memory);
            float update = (value[value_offset] - memory) * beta[gate_offset];
            float result = 0;
            for (uint index = 0; index < (D + 31) / 32; ++index) {
                state[index] += keys[index] * update;
                if (index * 32 + lane < D)
                    result += state[index] * query[key_offset + index * 32 + lane];
            }
            result = simd_sum(result);
            if (lane == 0) output[value_offset] = result;
        }
        for (uint index = 0; index < (D + 31) / 32; ++index)
            if (index * 32 + lane < D)
                final[state_base + index * 32 + lane] = state[index];
    """,
)


def gated_delta_update(query, key, value, decay, beta, state):
    """Normalized queries already include the key-dimension inverse-square-root scale."""
    return _recurrence(
        inputs=[query, key, value, decay, beta, state],
        template=[("D", query.shape[-1])],
        grid=(32, value.shape[-1], value.shape[0] * value.shape[2]),
        threadgroup=(32, 4, 1),
        output_shapes=[value.shape, state.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
