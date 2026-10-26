"""Regression coverage for RMSNorm aliasing the float32 residual stream."""

import pytest
import torch

from resin_models_torch.lfm2.model import RMSNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_does_not_mutate_residual(dtype):
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=dtype)
    original = hidden.clone()
    norm = RMSNorm(4, eps=1e-5).to(dtype)
    output = norm(hidden)
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)
    expected = (original.float() * torch.rsqrt(torch.tensor(7.5 + 1e-5))).to(dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
