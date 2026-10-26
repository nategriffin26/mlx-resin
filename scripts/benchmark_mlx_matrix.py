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
    args = parser.parse_args()
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
        for batch, prompt, decode in cases:
            records = {backend: [] for backend in args.backends}
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
                    records[backend].append(measured[0])
                    print(json.dumps(measured[0]), flush=True)
            for backend, trials in records.items():
                summary = {
                    "model": model,
                    "backend": backend,
                    "batch_size": batch,
                    "prompt_tokens": prompt,
                    "decode_tokens": decode,
                    **{
                        key: {
                            "median": statistics.median(row[key] for row in trials),
                            "min": min(row[key] for row in trials),
                            "max": max(row[key] for row in trials),
                        }
                        for key in ("prefill_tps", "decode_tps")
                    },
                }
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
            (args.output_dir / "summary.json").write_text(
                json.dumps(summaries, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
