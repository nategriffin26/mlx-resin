"""Interleaved Qwen3.6 affine 4-bit benchmarks and allocated-memory measurements."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model
from tokenizers import Tokenizer

from resin import Engine
from resin.backend.loading import read_config, resolve_model
from resin.models.qwen3_5_moe import Qwen35MoeConfig
from resin_models_mlx.qwen3_5.cache import CacheGeometry
from resin_models_mlx.qwen3_5.mlx import _download_model
from resin_models_mlx.qwen3_5_moe import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Qwen35MoeMlxRunner,
)
from scripts.benchmark_qwen35_mlx import mlx_lm_trial, resin_trial


def paging_counters():
    output = subprocess.check_output(["vm_stat"], text=True)
    return {
        key: int(value.strip().rstrip("."))
        for line in output.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
        if key in ("Pageouts", "Swapins", "Swapouts")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("resin", "mlx-lm"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.capacity_only:
        if args.matrix:
            parser.error("--capacity-only requires a single --backend")
        args.runs, args.decode_tokens = 1, 1
    if args.matrix:
        if args.output is None:
            parser.error("--matrix requires --output for the raw JSONL measurements")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as stream:
            for batch, length in ((1, 128), (8, 128), (32, 128), (1, 2048), (1, 4096)):
                for trial in range(args.runs):
                    order = (
                        ("resin", "mlx-lm") if trial % 2 == 0 else ("mlx-lm", "resin")
                    )
                    for backend in order:
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--backend",
                            backend,
                            "--batch-size",
                            str(batch),
                            "--prompt-tokens",
                            str(length),
                            "--runs",
                            "1",
                            "--decode-tokens",
                            str(args.decode_tokens),
                        ]
                        print(
                            json.dumps({"command": command, "trial": trial + 1}),
                            flush=True,
                        )
                        result = subprocess.run(
                            command, text=True, stdout=subprocess.PIPE, check=True
                        )
                        for line in result.stdout.splitlines():
                            row = json.loads(line)
                            row["trial"] = trial + 1
                            row["command"] = command
                            stream.write(json.dumps(row) + "\n")
                            stream.flush()
                            print(json.dumps(row), flush=True)
        return
    if args.backend is None:
        parser.error("select --backend or --matrix")
    tokenizer = Tokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    text = "Alice lives in Paris. Bob lives in Rome. The train leaves at noon. "
    source = tokenizer.encode(text * (args.prompt_tokens // 8 + 1)).ids
    prompts = [source[: args.prompt_tokens] for _ in range(args.batch_size)]
    assert all(len(prompt) == args.prompt_tokens for prompt in prompts)
    path = resolve_model(DEFAULT_MODEL, DEFAULT_REVISION, _download_model)
    mx.reset_peak_memory()
    if args.backend == "resin":
        geometry = CacheGeometry(
            Qwen35MoeConfig.from_hugging_face(read_config(path)), dtype=mx.bfloat16
        )
        capacity = geometry.layout(1).estimated_parent_count(
            [args.prompt_tokens + args.decode_tokens + 1] * args.batch_size
        )
        runner = Qwen35MoeMlxRunner(path, cache_slab_capacity=capacity + 4)
        engine = Engine(
            runner, prefix_cache=False, max_running_requests=args.batch_size
        )

        def execute():
            return resin_trial(engine, prompts, args.decode_tokens)

        cache_bytes = runner.arena.storage.nbytes
    else:
        model, _ = load_model(path)
        mx.eval(model.parameters())

        def execute():
            return mlx_lm_trial(model, prompts, args.decode_tokens)

        cache_bytes = None
    metadata = {
        "backend": args.backend,
        "model": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "quantization": "affine 4-bit, 8-bit routers, group_size=64",
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "cache_bytes": cache_bytes,
        "max_batch_tokens": runner.runner_config.max_batch_tokens
        if args.backend == "resin"
        else None,
        "prefill_chunk_size": 4096 if args.backend == "resin" else 2048,
        "device": mx.device_info(),
        "method": "fresh-cache warmup then measured trials; stop disabled; prefill includes first token; decode excludes it",
        "capacity_only": args.capacity_only,
    }
    if args.capacity_only:
        metadata["method"] = (
            "one cold-cache capacity trial; one decode step after prefill; not a speed benchmark"
        )
    else:
        execute()
    for trial in range(args.runs):
        mx.reset_peak_memory()
        paging_before = paging_counters()
        prefill, decode, tokens, prefill_peak, decode_memory = execute()
        paging_after = paging_counters()
        paging_delta = {
            key: paging_after[key] - value for key, value in paging_before.items()
        }
        print(
            json.dumps(
                {
                    **metadata,
                    "trial": trial + 1,
                    "prefill_seconds": prefill,
                    "decode_seconds": decode,
                    "prefill_tps": args.batch_size * args.prompt_tokens / prefill,
                    "decode_tps": args.batch_size * args.decode_tokens / decode,
                    "peak_bytes": max(prefill_peak, mx.get_peak_memory()),
                    "prefill_peak_bytes": prefill_peak,
                    "decode_peak_bytes": mx.get_peak_memory(),
                    "decode_memory_before": decode_memory,
                    "active_bytes": mx.get_active_memory(),
                    "generated_tokens": sum(map(len, tokens)),
                    "first_tokens": tokens[0][:8],
                    "paging_before": paging_before,
                    "paging_after": paging_after,
                    "paging_delta": paging_delta,
                    "accepted": paging_delta["Swapouts"] == 0,
                }
            ),
            flush=True,
        )
    if args.backend == "resin":
        engine.close()


if __name__ == "__main__":
    main()
