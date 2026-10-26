"""Compare pinned Qwen3.5-4B BF16 through the Engine and mlx-lm BatchGenerator."""

import argparse
import json
import statistics
import time

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.utils import load_model
from tokenizers import Tokenizer

from resin import Engine, SamplingParams
from resin.backend.loading import read_config, resolve_model
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.qwen3_5 import DEFAULT_MODEL, DEFAULT_REVISION, Qwen35MlxRunner
from resin_models_mlx.qwen3_5.cache import CacheGeometry
from resin_models_mlx.qwen3_5.mlx import _download_model


def delta_trials(runs):
    from resin_models_mlx.qwen3_5.delta import gated_delta_update

    mx.random.seed(41)
    for batch, tokens in ((1, 1), (8, 1), (1, 2048)):
        query = mx.random.normal((batch, tokens, 16, 128)) / 128
        key = mx.random.normal(query.shape) / 128**0.5
        value = mx.random.normal((batch, tokens, 32, 128))
        decay = mx.full((batch, tokens, 32), 0.95)
        beta = mx.full(decay.shape, 0.5)
        initial = mx.zeros((batch, 32, 128, 128))
        mx.eval(query, key, value, decay, beta, initial)
        for name, kernel in (
            ("native", mx.fast.gated_delta_update),
            ("exact", gated_delta_update),
        ):
            output, state = kernel(query, key, value, decay, beta, initial)
            mx.eval(output, state)
            samples = []
            for _ in range(runs):
                start = time.perf_counter()
                for _ in range(10):
                    output, state = kernel(query, key, value, decay, beta, state)
                    mx.eval(output, state)
                samples.append((time.perf_counter() - start) / 10)
            print(
                json.dumps(
                    {
                        "kernel": name,
                        "batch_size": batch,
                        "tokens": tokens,
                        "median_ms": statistics.median(samples) * 1000,
                        "min_ms": min(samples) * 1000,
                        "max_ms": max(samples) * 1000,
                        "method": "warmup; 3 trials of 10 synchronized state-dependent calls; FP32 inputs",
                    }
                ),
                flush=True,
            )


def resin_trial(engine, prompts, count):
    requests = [
        engine.submit(
            prompt,
            sampling_params=SamplingParams(temperature=0, max_output_tokens=count + 1),
        )
        for prompt in prompts
    ]
    mx.synchronize()
    start = time.perf_counter()
    first = []
    while len(first) < len(prompts):
        first.extend(engine.step())
    mx.synchronize()
    prefill = time.perf_counter() - start
    assert len(first) == len(prompts)
    assert {event.request_id for event in first} == {
        request.request_id for request in requests
    }, "the complete cohort must prefill before any request decodes"
    start = time.perf_counter()
    completed = 0
    while completed < count:
        events = engine.step()
        if events:
            assert len(events) == len(prompts)
            completed += 1
    mx.synchronize()
    decode = time.perf_counter() - start
    tokens = [request.take_result() for request in requests]
    assert all(len(row) == count + 1 for row in tokens)
    assert all(row == tokens[0] for row in tokens), "identical greedy rows diverged"
    assert not engine.has_work()
    return prefill, decode, tokens


def mlx_lm_trial(model, prompts, count):
    generator = BatchGenerator(
        model,
        max_tokens=count + 1,
        stop_tokens=[],
        prefill_batch_size=len(prompts),
        completion_batch_size=len(prompts),
    )
    uids = generator.insert(prompts)
    tokens = {uid: [] for uid in uids}
    mx.synchronize()
    start = time.perf_counter()
    first = generator.next_generated()
    mx.synchronize()
    prefill = time.perf_counter() - start
    assert len(first) == len(prompts)
    for row in first:
        tokens[row.uid].append(row.token)
    start = time.perf_counter()
    for _ in range(count):
        rows = generator.next_generated()
        assert len(rows) == len(prompts)
        for row in rows:
            tokens[row.uid].append(row.token)
    mx.synchronize()
    decode = time.perf_counter() - start
    assert not generator.next_generated()
    generator.close()
    tokens = [tokens[uid] for uid in uids]
    assert all(len(row) == count + 1 for row in tokens)
    assert all(row == tokens[0] for row in tokens), "identical greedy rows diverged"
    return prefill, decode, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("resin", "mlx-lm"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--delta-only", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--delta-kernel",
        choices=("exact", "native"),
        default="exact",
        help="native is a numerical diagnostic, not the validated runner",
    )
    args = parser.parse_args()
    if args.delta_only:
        delta_trials(args.runs)
        return
    if args.backend is None:
        parser.error("--backend is required unless --delta-only is selected")
    if args.delta_kernel == "native":
        if args.backend != "resin":
            parser.error("--delta-kernel native requires --backend resin")
        import resin_models_mlx.qwen3_5.model as qwen_model

        qwen_model.gated_delta_update = mx.fast.gated_delta_update
    tokenizer = Tokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    text = "Alice lives in Paris. Bob lives in Rome. The train leaves at noon. "
    source = tokenizer.encode(text * (args.prompt_tokens // 8 + 1)).ids
    prompts = [source[: args.prompt_tokens] for _ in range(args.batch_size)]
    assert all(len(prompt) == args.prompt_tokens for prompt in prompts)
    path = resolve_model(DEFAULT_MODEL, DEFAULT_REVISION, _download_model)
    if args.backend == "resin":
        geometry = CacheGeometry(
            Qwen35Config.from_hugging_face(read_config(path)),
            dtype=mx.bfloat16,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        capacity = geometry.layout(1).estimated_parent_count(
            [args.prompt_tokens + args.decode_tokens + 1] * args.batch_size
        )
        runner = Qwen35MlxRunner(
            path,
            cache_slab_capacity=max(32, capacity + 4),
            prefill_chunk_size=args.prefill_chunk_size,
        )
        engine = Engine(
            runner, prefix_cache=False, max_running_requests=args.batch_size
        )

        def execute():
            return resin_trial(engine, prompts, args.decode_tokens)
    else:
        model, _ = load_model(path)
        mx.eval(model.parameters())

        def execute():
            return mlx_lm_trial(model, prompts, args.decode_tokens)

    print(
        json.dumps(
            {
                "backend": args.backend,
                "model": DEFAULT_MODEL,
                "revision": DEFAULT_REVISION,
                "dtype": "bfloat16",
                "delta_kernel": args.delta_kernel
                if args.backend == "resin"
                else "mlx-lm",
                "device": mx.device_info(),
                "batch_size": args.batch_size,
                "prompt_tokens": args.prompt_tokens,
                "prefill_chunk_size": args.prefill_chunk_size
                if args.backend == "resin"
                else 2048,
                "measured_decode_tokens": args.decode_tokens,
                "overlap": False,
                "method": "one warmup; three fresh-cache trials; stop disabled; prefill includes first token; decode excludes first token",
            }
        ),
        flush=True,
    )
    execute()
    results = []
    for index in range(args.runs):
        prefill, decode, tokens = execute()
        result = {
            "run": index + 1,
            "prefill_seconds": prefill,
            "decode_seconds": decode,
            "prefill_tps": args.batch_size * args.prompt_tokens / prefill,
            "decode_tps": args.batch_size * args.decode_tokens / decode,
            "generated_tokens": sum(map(len, tokens)),
            "first_tokens": tokens[0][:8],
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    print(
        json.dumps(
            {
                key: {
                    "median": statistics.median(row[key] for row in results),
                    "min": min(row[key] for row in results),
                    "max": max(row[key] for row in results),
                }
                for key in ("prefill_tps", "decode_tps")
            }
        ),
        flush=True,
    )
    if args.backend == "resin":
        engine.close()


if __name__ == "__main__":
    main()
