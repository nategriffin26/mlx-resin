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
The default `max_batch_tokens=4096` bounds total tokens per step independently of
that checkpoint interval. Without it, eight 4096-token rows reached 43.31 GB peak
allocation and swapped heavily; a 32-row warmup was killed. Bounding the step
retains 4096-token prefix reuse instead of changing checkpoint granularity.

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
The real model also checks a 4096-token prefix hit against fresh, 512-token-chunk
prefill on a 4097-token prompt.

On the pinned checkpoint, all 12 greedy tokens matched. Measured logit errors:

| Prompt | mlx-lm BF16/FP32 RMS / max | mlx-resin versus mlx-lm RMS / max |
| --- | ---: | ---: |
| Capital of France | 0.13994 / 1.28884 | 0.12018 / 0.75000 |
| 17 × 23 | 0.21445 / 5.89944 | 0.11452 / 0.84375 |
| Python square function | 0.18074 / 1.42432 | 0.14872 / 1.23438 |

The 4096-token prefix test also passed. The chat command returned
`The capital of France is **Paris**.` with a 32-token output limit.

## Benchmark commands

```sh
uv run python scripts/benchmark_qwen36_mlx.py --matrix --decode-batches 1 8 --runs 3 --decode-tokens 128 --output local/resident-state/moe-matrix.jsonl
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen36_mlx.py --matrix --runs 3 --decode-tokens 128 --output local/qwen36-benchmarks-final.jsonl
```

The resident-state matrix measures 128-token prompts and 128 decoded tokens at
batches 1 and 8. Omitting `--decode-batches` also measures batch 32 and
single-request 2048/4096-token prefills. Fresh processes alternate in
A/B, B/A, A/B order. Each process loads one backend, warms once and measures a
fresh-cache trial. Loading is excluded; prefill includes the first generated token.
Decode excludes that first token. Prefix hits and EOS stopping are disabled, and the
complete cohort must prefill before decode. Reported throughput is aggregate
tokens per second; `peak_bytes` is MLX's peak active allocation, not process RSS
or the machine's total memory. Raw JSONL records checkpoint revision, device,
cache bytes, generated token counts and exact subprocess commands.
Each measured trial records `vm_stat` pageout, swap-in and swap-out counters before
and after execution. New swap-outs mark the trial `accepted: false`; rejected
trials do not contribute to reported throughput. Existing swap usage and swap-ins
are reported but do not, by themselves, reject a cell.

## M5 Pro measurements

Measured 2026-10-26: MLX 0.32.4, mlx-lm 0.32.2, 48 GB unified memory, pinned
four-bit weights with BF16 activations. Values are aggregate tokens/second,
median (range). The resident-state decode matrix has three interleaved trials
per backend. The default 4096-token step budget and checkpoint interval are
unchanged. Prefill rows below retain the earlier matrix's accepted trials.

| Measurement | mlx-resin tok/s | mlx-lm tok/s | Accepted trials, Resin / mlx-lm |
| --- | ---: | ---: | ---: |
| Decode b1, 128-token prompt | Not verified: unstable under load | Not verified: unstable under load | 3 / 3 |
| Decode b8, 128-token prompt | 255.48 (222.03–261.79) | 326.69 (267.74–328.20) | 3 / 3 |
| Decode b32, prior matrix | Rejected: swapping | Not requalified | 0 / 2 |
| Prefill b1, 2048 tokens, prior matrix | 1842.49 (1812.52–1854.43) | 2054.78 (1976.76–2334.36) | 3 / 3 |
| Prefill b1, 4096 tokens, prior matrix | 1790.21 (1788.06–1792.35) | 2374.13 (1981.00–2375.68) | 2 / 3 |

All 12 current measured trials had zero new swap-outs and identical complete
greedy-output hashes across backends. Batch 8 reaches 78% of mlx-lm, versus
53% in the earlier matrix (165.49 / 313.69 tok/s). Those matrices were not
interleaved with each other, so this is not an isolated causal speedup.

Batch 1 is not qualified despite zero new swap-outs in its measured trials:
repeats varied widely, and later comparisons stalled during warmup. A sampled
warmup used 39.2% CPU while system counters rose by 25,089 swap-in and 38,208
swap-out pages. It was thrashing, not idle. No rejected or incomplete trial is
published as model throughput.

The raw current and historical records, commands, hashes, and process sample
are in [`artifacts/benchmarks/2026-10-26-mlx-resident/`](../../../../artifacts/benchmarks/2026-10-26-mlx-resident/).
Local originals are `local/resident-state/moe-matrix.jsonl` and
`local/qwen36-benchmarks-final.jsonl`. Measurements use `880b139`; the subsequent
`90d1aa7` error-guard and initialization-memory fixes are correctness-tested,
not separately throughput-qualified under this Mac's increased load.

## Memory capacity

With the default 4096-token step budget, 32 requests with 4096-token contexts
completed prefill and one decode step (two output tokens per request). Peak active
MLX allocation was **30.14 GB during prefill and 32.20 GB during decode**, including
weights and the 7.38 GB arena; resident allocation was 26.89 GB. GB is decimal.

This is a capacity result, **not swap-free operation or a speed result**. The Mac's
other live workloads occupied roughly 15–18 GB, and the capacity run swapped.
The unbounded warmup failure and the bounded completion are retained separately
from the speed matrix; neither becomes an accepted timing by changing the threshold.

```sh
MLX_ENABLE_TF32=0 uv run python scripts/benchmark_qwen36_mlx.py --backend resin --batch-size 32 --prompt-tokens 4096 --capacity-only
```

`--capacity-only` skips warmup and runs one cold-cache prefill plus one decode
step. It records peak allocation and paging, not a warmed throughput claim.
A later b32 invocation exceeded 360 seconds under pressure. The completed run
demonstrates allocation capacity, not reliable b32 latency on this loaded Mac.
