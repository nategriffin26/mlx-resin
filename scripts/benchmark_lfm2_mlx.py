"""Compare Engine and mlx-lm batch API on identical unquantized LFM2 inputs."""

import argparse
import cProfile
import json
import pstats
import statistics
import time
from collections import defaultdict

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.generate import BatchGenerator
from mlx_lm.utils import load_model
from tokenizers import Tokenizer

from resin import Engine, SamplingParams
from resin_models_mlx.lfm2 import DEFAULT_MODEL, DEFAULT_REVISION, Lfm2MlxRunner


def resin_trial(engine, prompts, count):
    requests = [
        engine.submit(
            prompt,
            sampling_params=SamplingParams(
                temperature=0,
                max_output_tokens=count + 1,
            ),
        )
        for prompt in prompts
    ]
    mx.synchronize()
    start = time.perf_counter()
    first = []
    while not first:
        first = engine.step()
    mx.synchronize()
    prefill_seconds = time.perf_counter() - start
    assert len(first) == len(prompts)
    start = time.perf_counter()
    for _ in range(count):
        assert len(engine.step()) == len(prompts)
    mx.synchronize()
    decode_seconds = time.perf_counter() - start
    tokens = [request.take_result() for request in requests]
    assert all(len(row) == count + 1 for row in tokens)
    assert not engine.has_work()
    return prefill_seconds, decode_seconds, tokens


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
    prefill_seconds = time.perf_counter() - start
    assert len(first) == len(prompts)
    for response in first:
        tokens[response.uid].append(response.token)
    start = time.perf_counter()
    for _ in range(count):
        responses = generator.next_generated()
        assert len(responses) == len(prompts)
        for response in responses:
            tokens[response.uid].append(response.token)
    mx.synchronize()
    decode_seconds = time.perf_counter() - start
    assert not generator.next_generated()
    generator.close()
    tokens = [tokens[uid] for uid in uids]
    assert all(len(row) == count + 1 for row in tokens)
    return prefill_seconds, decode_seconds, tokens


def profile_operations(execute):
    """Synchronize operation boundaries; these diagnostic timings are not throughput."""
    from resin.backend.mlx import SlabArena
    from resin_models_mlx.lfm2.cache import DeviceCache
    from resin_models_mlx.lfm2.model import MLP, Attention, ShortConv

    samples = defaultdict(list)
    originals = []

    def instrument(cls, name):
        original = getattr(cls, name)
        originals.append((cls, name, original))
        label = f"{cls.__name__}.{name}"

        def measured(self, *args, **kwargs):
            mx.synchronize()
            start = time.perf_counter()
            result = original(self, *args, **kwargs)
            if isinstance(result, mx.array):
                mx.eval(result)
            mx.synchronize()
            samples[label].append(time.perf_counter() - start)
            return result

        setattr(cls, name, measured)

    for cls, names in (
        (Attention, ("__call__",)),
        (ShortConv, ("__call__",)),
        (MLP, ("__call__",)),
        (DeviceCache, ("history", "state", "append", "store_states")),
        (SlabArena, ("initialize_fresh", "commit")),
    ):
        for name in names:
            instrument(cls, name)
    try:
        execute()
    finally:
        for cls, name, original in originals:
            setattr(cls, name, original)
    print(
        json.dumps(
            {
                "synchronized_operations": {
                    name: {
                        "calls": len(values),
                        "total_ms": sum(values) * 1000,
                        "median_us": statistics.median(values) * 1e6,
                    }
                    for name, values in samples.items()
                },
                "note": "Parent operation timings include child operations; synchronization changes scheduling.",
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("resin", "mlx-lm"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-operations", action="store_true")
    args = parser.parse_args()
    dtype = getattr(mx, args.dtype)
    capacity = (
        args.batch_size * ((args.prompt_tokens + args.decode_tokens) // 128 + 3) + 4
    )
    runner = Lfm2MlxRunner(dtype=dtype, cache_slab_capacity=max(64, capacity))
    model_path = runner.model_path
    tokenizer = Tokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    text = "Alice lives in Paris. Bob lives in Rome. The train leaves at noon. "
    source = tokenizer.encode(text * (args.prompt_tokens // 8 + 1)).ids
    prompts = [source[: args.prompt_tokens] for _ in range(args.batch_size)]
    assert all(len(prompt) == args.prompt_tokens for prompt in prompts)
    if args.backend == "resin":
        engine = Engine(runner, prefix_cache=False)

        def execute():
            return resin_trial(engine, prompts, args.decode_tokens)

        parameters = runner._model.parameters()
    else:
        del runner
        model, _ = load_model(model_path)
        model.set_dtype(dtype)
        mx.eval(model.parameters())

        def execute():
            return mlx_lm_trial(model, prompts, args.decode_tokens)

        parameters = model.parameters()
    assert all(value.dtype == dtype for _, value in tree_flatten(parameters))
    print(
        json.dumps(
            {
                "backend": args.backend,
                "model": DEFAULT_MODEL,
                "revision": DEFAULT_REVISION,
                "dtype": args.dtype,
                "device": mx.device_info(),
                "batch_size": args.batch_size,
                "prompt_tokens": args.prompt_tokens,
                "prefill_chunk_size": 128 if args.backend == "resin" else 2048,
                "measured_decode_tokens": args.decode_tokens,
                "method": "one warmup; fresh cache each run; prefill includes first token; decode excludes first token; stop disabled",
            }
        ),
        flush=True,
    )
    execute()
    results = []
    for run in range(args.runs):
        prefill, decode, tokens = execute()
        result = {
            "run": run + 1,
            "prefill_seconds": prefill,
            "prefill_tps": args.batch_size * args.prompt_tokens / prefill,
            "decode_seconds": decode,
            "decode_tps": args.batch_size * args.decode_tokens / decode,
            "generated_tokens": sum(map(len, tokens)),
            "first_tokens": tokens[0][:8],
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    print(
        json.dumps(
            {
                "median_prefill_tps": statistics.median(
                    row["prefill_tps"] for row in results
                ),
                "median_decode_tps": statistics.median(
                    row["decode_tps"] for row in results
                ),
            }
        ),
        flush=True,
    )
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(execute)
        pstats.Stats(profiler).sort_stats("tottime").print_stats(20)
    if args.profile_operations:
        if args.backend != "resin":
            parser.error("--profile-operations requires --backend resin")
        profile_operations(execute)
    if args.backend == "resin":
        engine.close()


if __name__ == "__main__":
    main()
