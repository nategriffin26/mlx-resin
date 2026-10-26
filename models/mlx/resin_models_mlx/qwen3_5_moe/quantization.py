"""Affine checkpoint tensors stay packed through linear and routed projections."""

import mlx.core as mx
import mlx.nn as nn


class QuantizedLinear(nn.Module):
    def __init__(self, weights, *, group_size=64, bits=4, mode="affine"):
        super().__init__()
        if mode != "affine":
            raise ValueError("Qwen3.5 MoE supports MLX affine quantization")
        self.group_size, self.bits, self.mode = group_size, bits, mode
        self.weight, self.scales, self.biases = weights

    def __call__(self, hidden):
        return mx.quantized_matmul(
            hidden,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def column(self, index):
        # Steering is prepared once, outside inference. Decode never unpacks weights.
        return mx.dequantize(
            self.weight,
            self.scales,
            self.biases,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )[:, index]


class QuantizedExpertLinear(QuantizedLinear):
    def __call__(self, hidden, indices, sorted_indices=False):
        return mx.gather_qmm(
            hidden,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


class QuantizedEmbedding(QuantizedLinear):
    def __call__(self, tokens):
        return mx.dequantize(
            self.weight[tokens],
            self.scales[tokens],
            self.biases[tokens],
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def as_linear(self, hidden):
        return super().__call__(hidden)
