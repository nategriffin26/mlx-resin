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
`atol=rtol=1e-3`; BF16 execution allows `atol=0.3, rtol=0.01` against the HF
FP32 oracle. All require exact greedy token equality on the tested inputs.
Benchmarks disable prefix hits and EOS, warm once, then report three-run medians
and ranges. Prefill includes the first generated token; decode excludes it.
Batch throughput is aggregate tokens per second, not per-sequence speed.

## MLX recurrence precision

MLX 0.32.4's GPU `mx.fast.gated_delta_update` produced a 0.00636 maximum FP32
logit error on the pinned model's 17-token probe, despite `MLX_ENABLE_TF32=0`.
The exact SIMD recurrence reduced that error to 0.000042. Its FP32 state updates
use the reference delta rule directly, without a relaxed matrix product.

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
