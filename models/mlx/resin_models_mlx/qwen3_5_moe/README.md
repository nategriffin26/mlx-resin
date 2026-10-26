# Qwen3.6 MoE on MLX

`Qwen35MoeMlxRunner` runs the text model in
`mlx-community/Qwen3.6-35B-A3B-4bit`, pinned to
`38740b847e4cb78f352aba30aa41c76e08e6eb46`. The architecture remains
`qwen3_5_moe`: 40 hybrid-attention layers, 256 experts, eight routed experts per
token and one sigmoid-gated shared expert. Vision and bundled MTP tensors are
ignored; speculative decoding is not implemented.

```sh
uv run python scripts/qwen36_mlx_chat.py --prompt 'What is the capital of France?'
```

The checkpoint uses MLX affine quantization, group size 64: four-bit projections
and experts, eight-bit routing and shared-expert gates. Packed weights are never
expanded during projection: ordinary layers call `mx.quantized_matmul`, routed
layers call `mx.gather_qmm`. Embedding lookup dequantizes only selected rows.
Long prefills group routed tokens by expert, then restore their original order.
The loader honors per-layer quantization settings, splits HF's fused expert
weights when loading unquantized fixtures, and reverses the converted checkpoint's
RMSNorm offset before using the shared dense-family normalization.

Convolution tensor layout identifies that conversion; a quantization block alone
does not imply the norm offset was changed.

## State and steering

Attention, Gated DeltaNet, cache geometry, forward scheduling and steering media
come from the dense Qwen3.5 implementation, not a second copy. The Engine owns
128-token KV pages and FP32 convolution/recurrent checkpoints. The recurrent
checkpoint interval equals the runner's prefill chunk size (4096 by default).
Prefix reuse restores both groups at that interval; there is no runner-private
request cache.

`runner.calibrate` and `runner.steer` use the dense family's dose and prefix
isolation contract. Neuron indices address the **shared expert**; routed experts
have no single input-independent down-projection column. The shared expert's
sigmoid gate still applies to neuron additions. Residual-direction steering applies
after the combined expert output. Quantized steering columns are unpacked once
while preparing the request, never during decode.

## Oracle contract

```sh
uv run pytest models/mlx/tests/test_qwen35_moe.py -q
uv run pytest models/mlx/tests/test_qwen36_checkpoint.py -m checkpoint -q -s
```

The tiny random-weight oracle is Transformers FP32 on CPU, with exact greedy
tokens and the dense family's `atol=rtol=3e-4` logit bound. It covers unequal
histories, chunk boundaries, prefix hits immediately around checkpoint boundaries,
NaN-filled pages, eviction/reuse, normalized routing, shared-expert gating and
request-local steering. A mixed four/eight-bit checkpoint is independently
dequantized into the HF oracle to check loading and packed execution together.

Top-k ties do not have portable expert identities: on the installed libraries,
eight equal scores choose `[0, 1]` in Torch and `[6, 7]` in MLX. This runner keeps
MLX/mlx-lm's `argpartition` choice. Tied experts must be distinct and carry equal,
normalized weights; cross-backend tied-ID equality is not promised. Non-tied
routing and full-model FP32 parity retain the HF oracle.

The real-checkpoint test loads the same packed weights in mlx-lm and mlx-resin,
sequentially so two 35B models need not coexist. It checks four greedy tokens on
three chat prompts. Its noise reference is mlx-lm evaluated twice on the same
forced token sequence: checkpoint BF16 arithmetic versus FP32 floating parameters
and activations, leaving packed weights unchanged. RMS and maximum logit error
must be no larger than twice that independently measured noise, with a `3e-4`
floor. Greedy equality is a separate requirement, never relaxed by this bound.

## Benchmark commands

```sh
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen36_mlx.py --matrix --runs 3 --decode-tokens 16 --output local/qwen36-benchmarks.jsonl
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen36_mlx.py --backend resin --batch-size 32 --prompt-tokens 4096 --decode-tokens 16 --runs 3
```

The matrix alternates fresh processes in A/B, B/A, A/B order. Each process loads
one backend, warms once and measures a fresh-cache trial. Loading is excluded;
preprocessing through the first generated token is included in prefill. Decode
excludes that first token. Prefix hits and EOS stopping are disabled, and the
complete cohort must prefill before decode. Reported throughput is aggregate
tokens per second; `peak_bytes` is MLX's peak active allocation, not process RSS
or the machine's total memory. Raw JSONL records checkpoint revision, device,
cache bytes, generated token counts and exact subprocess commands.
