# Scripts and deployment modules

Run commands from the repository root. The shared entry points are
`scripts/bench.py` for Engine performance, `scripts/check_server.py` for endpoint
qualification, and `deploy/modal_serve.py` for Modal serving. Model-specific
configuration belongs in `deploy/models.py`.

## Scripts

| File | Purpose |
| --- | --- |
| [bench.py](../scripts/bench.py) | Construct a registered runner locally or on Modal; sweep prefill contexts and decode batches through `resin.bench`; save JSONL, plots and Perfetto traces. Also download or publish checkpoint inventories. |
| [plot_bench.py](../scripts/plot_bench.py) | Plot prefill latency versus input throughput/GPU and decode interactivity versus output throughput/GPU. Actual committed output is the default; acceptance projections are explicit. |
| [merge_bench_traces.py](../scripts/merge_bench_traces.py) | Merge per-rank prefill/decode traces, align timestamp bases and isolate process/flow IDs for Perfetto. |
| [check_server.py](../scripts/check_server.py) | Qualify an existing HTTP endpoint: JSON/SSE, logprobs, concurrency, prefix hits, context/TTFT, optional media/reasoning, MMLU subset and stress/recovery. Writes a pass/fail/skip report. |
| [stress_http.py](../scripts/stress_http.py) | Standalone bounded HTTP load/protocol/disconnect/recovery checks, plus the reusable client/parser/reporting machinery used by `check_server.py`. |
| [bench_startup.py](../scripts/bench_startup.py) | Measure cold `resin` and Torch backend imports in separate fresh processes. This measures Python startup, not model loading or inference. |
| [lfm2_chat.py](../scripts/lfm2_chat.py) | Minimal CPU example: render a fixed LFM2 conversation with the native recipe and stream one Engine completion. |
| [lfm2_mlx_chat.py](../scripts/lfm2_mlx_chat.py) | Apple Silicon example: the same native LFM2 recipe and prompt, streamed through an Engine-managed MLX runner. |
| [benchmark_lfm2_mlx.py](../scripts/benchmark_lfm2_mlx.py) | Compare synchronized Engine and mlx-lm batch-API throughput with the pinned checkpoint and identical dtype/input tokens. |
| [qwen35_mlx_chat.py](../scripts/qwen35_mlx_chat.py) | [Dense Qwen3.5 MLX chat](../models/mlx/resin_models_mlx/qwen3_5/README.md), with a calibrated baseline-versus-steered demo through the existing [media contract](steering.md). |
| [qwen35_neuron_sweep.py](../scripts/qwen35_neuron_sweep.py) | Qwen3.5 [neuron sweeps](neuron-sweeps.md): activation statistics, steered `main_rel` rollouts with random-direction controls, steer lens, direct-path connections and MLP-input captures, in the neuron explorer's file formats. |
| [benchmark_qwen35_mlx.py](../scripts/benchmark_qwen35_mlx.py) | Compare pinned Qwen3.5-4B BF16 Engine and mlx-lm throughput with fresh caches, synchronized phase timings, and three-run medians. |
| [benchmark_qwen35_precision.py](../scripts/benchmark_qwen35_precision.py) | Measure HF's own BF16 rounding noise beside native and exact MLX recurrence on the same pinned inputs and saved HF FP32 logits. |
| [qwen36_mlx_chat.py](../scripts/qwen36_mlx_chat.py) | [Qwen3.6-35B-A3B MLX chat](../models/mlx/resin_models_mlx/qwen3_5_moe/README.md), using pinned affine 4-bit weights and the native Qwen recipe. |
| [benchmark_qwen36_mlx.py](../scripts/benchmark_qwen36_mlx.py) | Interleave Qwen3.6 Engine and mlx-lm trials on the same quantized checkpoint; record throughput, peak allocation, cache capacity and exact commands as JSONL. |
| [benchmark_mlx_arena.py](../scripts/benchmark_mlx_arena.py) | Measure MLX slice writes with donation versus a retained old view that forces copying. |
| [lfm2_serve.py](../scripts/lfm2_serve.py) | Minimal CPU example: serve the Torch LFM2 runner through the native HTTP server on localhost. |
| [mlx_serve.py](../scripts/mlx_serve.py) | Apple Silicon: serve `qwen3_5` (Qwen3.5-4B) or `qwen3_5_moe` (Qwen3.6-35B-A3B) through the native OpenAI and Anthropic HTTP server on localhost, with the `qwen3_5` recipe and prefix caching. Qualify it with `check_server.py --prefix-cache-repeats 1100`: recurrent state checkpoints every 4096 tokens, so the default probe prompt (about 1,000 tokens) can never hit. |
| [rust_build_and_types.py](../scripts/rust_build_and_types.py) | Development build: sync type-library sources, build the Rust extension with generated stubs, normalize stubs and format them. |
| [sync_type_libraries.py](../scripts/sync_type_libraries.py) | Make locked Linux dependency sources available to `ty` on the development host without importing those libraries or running their installation hooks. |
| [normalize_stubs.py](../scripts/normalize_stubs.py) | Add property types, protocol bases and imports missing from generated native bindings. Called by the build script. |
| [cloc.py](../scripts/cloc.py) | Count production Rust/Python code using Tokei, excluding tests and grouping the report by package. |

## Deployment modules

| File | Purpose |
| --- | --- |
| [models.py](../deploy/models.py) | Shared checkpoint revisions/inventories, GPU/topology defaults, runner and recipe factories, and model-specific runtime preparation. |
| [modal_serve.py](../deploy/modal_serve.py) | One parameterized serving app selected by `RESIN_MODEL`; checkpoint publication, worker startup/warmup and native HTTP serving. Each model selection has its own endpoint. |
| [modal_bench.py](../deploy/modal_bench.py) | Thin remote adapter for `scripts/bench.py`, including CPU checkpoint actions and GPU sweep execution. |
| [modal_image.py](../deploy/modal_image.py) | Build the shared CUDA image and its pinned native/kernel dependencies. |
| [checkpoint_cache.py](../deploy/checkpoint_cache.py) | Validate checkpoint file inventories and atomically publish/require completion markers. |
| [jit_cache.py](../deploy/jit_cache.py) | Configure persistent compiler namespaces and content-validated FlashInfer artifact loading. |
| [__init__.py](../deploy/__init__.py) | Package marker for shared deployment imports. |

The artifact checker, one-off kernel/parallelism probes, Modal validation wrappers
and per-model deployment files have been retired. Reusable numerical tests remain
in `tests/` and `models/cuda/tests/`; performance measurement lives in `resin.bench`,
with command-line orchestration in `scripts/`. Historical reports retain the
revisions and scope of their original measurements.

See [benchmarking and endpoint checks](benchmarks.md) for command examples and
[Modal serving](../deploy/README.md) for checkpoint and deployment commands.
