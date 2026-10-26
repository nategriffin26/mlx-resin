# Qwen3.5 on MLX

Dense, text-only Qwen3.5 with Engine-owned full-attention pages and FP32 Gated
DeltaNet checkpoints. The default is `Qwen/Qwen3.5-4B` pinned to
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, in BF16. The loader accepts tied
embeddings and an absent BOS token; these are real properties of the 4B model.
The shared config records those properties; CUDA still requires untied embeddings.
Vision and bundled MTP tensors are ignored. MoE, quantized weights, and speculative
decoding are not supported by this runner.

```sh
uv run python scripts/qwen35_mlx_chat.py --prompt 'What is the capital of France?'
uv run python scripts/qwen35_mlx_chat.py --steering-demo --prompt 'Describe yourself.'
```

The chat script uses Resin's `qwen3_5` recipe with thinking disabled. The steering
demo compares greedy baseline and steered requests in one Engine, using zero-based
L14 N8949 +1 and L13 N7956 −1. It calibrates on 256 tokens, excludes token zero
from each layer's residual median, and saves `local/qwen35-calibration.json`.
Delete that file to recalibrate; calibration from another model path or dtype is
rejected. The corpus and token count affect the dose: this reproduces the
published intervention, not a claim of identical text to a different calibration.

```python
from resin import Engine, SamplingParams
from resin_models_mlx.qwen3_5 import Qwen35MlxRunner

runner = Qwen35MlxRunner()
calibration = runner.calibrate(calibration_token_ids)
media = runner.steer(prompt_token_ids, [(14, 8949, 1), (13, 7956, -1)], calibration)
with Engine(runner, prefix_cache=True) as engine:
    request = engine.submit(media=media, sampling_params=SamplingParams(temperature=0))
    while engine.has_work():
        engine.step()
    output_token_ids = request.take_result()
```

See [the steering contract](../../../../docs/steering.md) for dose units and
prefix isolation. Prefill chunks are at most one 128-token checkpoint interval.
With prefix caching enabled, the Engine cuts chunks at reusable boundaries; each
forward stores its final recurrent state. No unused mid-chunk checkpoint is
computed. Decode reads history pages plus current K/V through the Metal
paged-attention kernel; writes persist only after the complete forward.

## Validation and measurement

```sh
uv run pytest models/mlx/tests/test_qwen35_mlx.py models/mlx/tests/test_qwen35_delta.py models/mlx/tests/test_qwen35_checkpoint.py -q
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen35_mlx.py --backend resin --batch-size 1 --prompt-tokens 2048
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen35_mlx.py --backend mlx-lm --batch-size 1 --prompt-tokens 2048
```

The regression suite uses an independent Transformers model and the real Engine.
Small-model FP32 logits use `atol=rtol=3e-4`. Pinned 4B FP32 logits use
`atol=3e-3, rtol=1e-3` against HF's recurrent rule. BF16 requires per-row RMS
error ≤ 0.5 and maximum absolute error ≤ 3.5. Both require exact greedy tokens.
Benchmarks disable prefix hits and EOS, warm once, then report three-run medians
and ranges. Prefill includes the first generated token; decode excludes it.
Batch throughput is aggregate tokens per second, not per-sequence speed.

## MLX recurrence precision

MLX 0.32.4's GPU `mx.fast.gated_delta_update` produced a 0.00636 maximum FP32
logit error on the pinned model's 17-token probe, despite `MLX_ENABLE_TF32=0`.
The exact SIMD recurrence reduced that error to 0.000042. Its FP32 state updates
use the reference delta rule directly, without a relaxed matrix product.

The FP32 bound comes from measured reference noise. HF's chunked versus recurrent
rule differs by 0.00206 maximum and 0.00049 RMS at 128 tokens. Equivalent L2
normalization orderings move the observed test violation from 0.00102 to 0.00175.
Forced-input layer errors are usually `1e-6–1e-5`, with sensitive later layers up
to `2.6e-4`. The oracle uses HF's recurrent rule; the `3e-3` absolute / `1e-3`
relative policy admits those rounding effects without weakening greedy equality.
Pytest disables TF32 before importing MLX.

BF16 has a separate measured rounding bound. At 127/128 tokens, HF's own BF16
versus FP32 errors are 0.68/3.08 maximum and 0.13/0.46 RMS; MLX's exact BF16
path measures 0.84/1.78 maximum and 0.11/0.31 RMS. The BF16 policy admits that
reference noise, while the FP32 and primitive-level checks retain strict bounds.

The installed MLX headers identify two upstream review points:
`gated_delta_nax_ops.h:260–261` hardcodes relaxed matrix precision, and
`gated_delta_update_nax.h:124–128` floors decay at `1e-6` before taking its log.
The measured precision failure is reproducible with identical recurrence inputs;
the header's decay floor is a separate semantic concern, not the measured cause.
Installed mlx-lm 0.32.2 uses its own SIMD recurrence, not this native primitive.

`benchmark_qwen35_mlx.py --delta-only` compares native and exact GDN calls.
`--delta-kernel native --backend resin` is an explicit diagnostic of the native
primitive, not the validated runner. `benchmark_qwen35_precision.py --backend
hf-fp32` saves the reference; `hf-bf16`, `mlx-native`, and `mlx-exact` compare
BF16 execution against it on identical 17-, 129-, and 257-token prompts.

## M5 Pro measurements

MLX 0.32.4, mlx-lm 0.32.2, unquantized BF16, 48 GB unified memory. Values are
three-run medians with observed ranges. Decode measures 64 tokens after the first
output; the b1/b8 decode prompts have 128 tokens. No prefix hits or stop tokens.

| Measurement | Engine tok/s | mlx-lm tok/s |
| --- | ---: | ---: |
| Decode b1 | 20.85 (20.76–20.86) | 25.22 (24.95–25.53) |
| Decode b8, aggregate | 89.73 (88.21–90.01) | 174.64 (158.15–179.43) |
| Prefill b1, 2048 tokens | 1086.72 (1074.76–1142.06) | 2134.49 (2087.75–2151.67) |

```sh
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen35_mlx.py --backend resin --batch-size 1 --prompt-tokens 128 --decode-tokens 64
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen35_mlx.py --backend mlx-lm --batch-size 1 --prompt-tokens 128 --decode-tokens 64
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen35_mlx.py --backend resin --batch-size 8 --prompt-tokens 128 --decode-tokens 64
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen35_mlx.py --backend mlx-lm --batch-size 8 --prompt-tokens 128 --decode-tokens 64
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen35_mlx.py --backend resin --batch-size 1 --prompt-tokens 2048 --decode-tokens 64
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen35_mlx.py --backend mlx-lm --batch-size 1 --prompt-tokens 2048 --decode-tokens 64
```

The exact recurrence costs about 6% end-to-end prefill latency versus the native
primitive in the paired 2k/16-decode-token diagnostic (1.886 s versus 1.782 s).
The isolated 2k GDN call costs 7.68 ms versus 2.04 ms, but the Engine uses 128-token
chunks, not a single 2k recurrent call.

The current byte arena has MLX's signed-32-bit single-dimension limit. Qwen4B's
default 32 slabs fit; requesting 64 slabs fails allocation at 3.42 GB. Larger
arenas need a backend storage-layout change, not a runner-private allocator.
