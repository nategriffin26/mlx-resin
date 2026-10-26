"""Interleave Resin/mlx-lm trials and preserve commands, raw output, and medians."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=("lfm2", "qwen35"), default=("lfm2", "qwen35")
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("resin", "mlx-lm"),
        default=("resin", "mlx-lm"),
    )
    parser.add_argument(
        "--high-concurrency",
        action="store_true",
        help="Qwen3.5 decode at batches 1/8/32/64 and batch-one 2K/4K prefill",
    )
    parser.add_argument("--include-batch128", action="store_true")
    args = parser.parse_args()
    if args.high_concurrency and list(args.models) != ["qwen35"]:
        parser.error("--high-concurrency requires --models qwen35")
    if args.include_batch128 and not args.high_concurrency:
        parser.error("--include-batch128 requires --high-concurrency")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for model in args.models:
        cases = [
            (1, 128, 128),
            (8, 128, 128),
            (1, 2048, 0),
            (1, 4096, 128),
            (8, 4096, 128),
        ]
        if model == "qwen35":
            cases.extend(((32, 128, 128), (64, 128, 128)))
        if args.high_concurrency:
            batches = (1, 8, 32, 64, 128) if args.include_batch128 else (1, 8, 32, 64)
            cases = [(batch, 128, 128) for batch in batches]
            cases.extend(((1, 2048, 0), (1, 4096, 0)))
        for batch, prompt, decode in cases:
            records = {backend: [] for backend in args.backends}
            expected_hashes = None
            for trial in range(1, 4):
                for backend in args.backends:
                    command = [
                        sys.executable,
                        f"scripts/benchmark_{model}_mlx.py",
                        "--backend",
                        backend,
                        "--batch-size",
                        str(batch),
                        "--prompt-tokens",
                        str(prompt),
                        "--decode-tokens",
                        str(decode),
                        "--runs",
                        "1",
                    ]
                    if model == "lfm2":
                        command.extend(("--dtype", "bfloat16"))
                    path = (
                        args.output_dir
                        / f"{model}-{backend}-b{batch}-p{prompt}-r{trial}.jsonl"
                    )
                    print(
                        json.dumps({"command": command, "output": str(path)}),
                        flush=True,
                    )
                    with path.open("w") as output:
                        subprocess.run(command, stdout=output, check=True)
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    measured = [row for row in rows if "run" in row]
                    assert len(measured) == 1
                    if model == "qwen35":
                        hashes = measured[0]["token_hashes"]
                        assert len(hashes) == batch
                        if expected_hashes is None:
                            expected_hashes = hashes
                        assert hashes == expected_hashes, (
                            f"greedy token mismatch: {model} {backend} "
                            f"batch={batch} prompt={prompt} trial={trial}; see {path}"
                        )
                    records[backend].append(measured[0])
                    print(json.dumps(measured[0]), flush=True)
            for backend, trials in records.items():
                accepted = [
                    row
                    for row in trials
                    if not row.get("paging_delta", {}).get("Swapouts", 0)
                ]
                summary = {
                    "model": model,
                    "backend": backend,
                    "batch_size": batch,
                    "prompt_tokens": prompt,
                    "decode_tokens": decode,
                    "all_greedy_rows_validated": model == "qwen35",
                    "accepted_trials": len(accepted),
                    "rejected_swapping_trials": len(trials) - len(accepted),
                    **{
                        key: {
                            "median": statistics.median(row[key] for row in accepted),
                            "min": min(row[key] for row in accepted),
                            "max": max(row[key] for row in accepted),
                        }
                        for key in ("prefill_tps", "decode_tps")
                        if accepted
                    },
                }
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
            (args.output_dir / "summary.json").write_text(
                json.dumps(summaries, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
