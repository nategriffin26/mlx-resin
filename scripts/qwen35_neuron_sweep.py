"""Neuron sweeps on Qwen3.5-4B through mlx-resin, written in the neuron explorer's file formats.

    uv run python scripts/qwen35_neuron_sweep.py stats --layers 22 --tokens 2000000 --out stats/q35-4b
    uv run python scripts/qwen35_neuron_sweep.py sweep --stats stats/q35-4b_L22.npz --neurons all --run q35-4b-L22
    uv run python scripts/qwen35_neuron_sweep.py lens --layers 22 --calibration stats/q35-4b.calibration.json \
        --out stats/steer_lens --index explorer/cache/steer_index.npz
    uv run python scripts/qwen35_neuron_sweep.py connections --out explorer/cache/connections.npz
    uv run python scripts/qwen35_neuron_sweep.py capture runs/q35-4b-L22.jsonl

Method and formats: docs/neuron-sweeps.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from resin import Engine
from resin.recipe import Message, RenderOptions, Role, Text, get_native_recipe
from resin_models_mlx.qwen3_5 import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Qwen35MlxRunner,
    SteeringCalibration,
)
from resin_models_mlx.qwen3_5.neurons import (
    LENS_CHAT_PROMPTS,
    LENS_TEXT_PROMPTS,
    SWEEP_PROMPTS,
    ActivationSurvey,
    DirectConnections,
    SteeringSweep,
    capture_mlp_inputs,
    select_neurons,
    steer_lens,
    update_steer_index,
)

# Commands without Engine requests still construct the runner; one slab keeps its
# cache arena negligible.
UNSERVED_SLABS = 1


def chat_ids(recipe, text):
    return recipe.render([Message(Role.USER, [Text(text)])]).token_ids.tolist()


def corpus_windows(recipe, tokenizer, seq_len, batch, seed):
    """FineWeb-Edu documents alternating with UltraChat conversations, EOS-separated."""
    from datasets import load_dataset

    web = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    ).shuffle(seed=seed)
    chat = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True
    ).shuffle(seed=seed)
    roles = {"user": Role.USER, "assistant": Role.ASSISTANT, "system": Role.SYSTEM}
    options = RenderOptions(add_generation_prompt=False)
    eos = tokenizer.token_to_id("<|im_end|>")
    buffer = []
    for document, conversation in zip(web, chat):
        buffer += tokenizer.encode(document["text"], add_special_tokens=False).ids
        buffer.append(eos)
        messages = [
            Message(roles[message["role"]], [Text(message["content"])])
            for message in conversation["messages"]
        ]
        buffer += recipe.render(messages, options=options).token_ids.tolist()
        buffer.append(eos)
        while len(buffer) >= seq_len * batch:
            chunk, buffer = buffer[: seq_len * batch], buffer[seq_len * batch :]
            yield np.array(chunk, np.int64).reshape(batch, seq_len)


def run_stats(args, recipe, tokenizer):
    layers = [int(layer) for layer in args.layers.split(",")]
    runner = Qwen35MlxRunner(
        args.model, revision=args.revision, cache_slab_capacity=UNSERVED_SLABS
    )
    survey = ActivationSurvey(runner, layers)
    steps = -(-args.tokens // (args.seq_len * args.batch))
    start = time.perf_counter()
    windows = corpus_windows(recipe, tokenizer, args.seq_len, args.batch, args.seed)
    for step, window in enumerate(itertools.islice(windows, steps)):
        survey.update(window)
        if step % 10 == 0 or step == steps - 1:
            rate = survey.token_count / (time.perf_counter() - start)
            print(
                f"{step + 1}/{steps} windows, {survey.token_count:,} tokens, {rate:,.0f} tok/s",
                flush=True,
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    calibration = survey.calibration()
    calibration.save(f"{args.out}.calibration.json")
    for layer in layers:
        path = f"{args.out}_L{layer}.npz"
        values = survey.layer_statistics(layer, args.model)
        np.savez(path, **values)
        print(
            f"L{layer}: residual norm p10/p50/p90 = {values['resid_norm_q10_50_90'].round(1)}; "
            f"abs_max median {np.median(values['abs_max']):.3f}; "
            f"(-,-) holds the positive max for {(values['nn_max'] > values['pp_max']).mean():.1%} -> {path}"
        )


def run_sweep(args, recipe, tokenizer):
    statistics = np.load(args.stats)
    layer = int(statistics["layer"])
    calibration_path = (
        args.calibration or args.stats.rsplit("_L", 1)[0] + ".calibration.json"
    )
    calibration = SteeringCalibration.load(calibration_path)
    runner = Qwen35MlxRunner(
        args.model, revision=args.revision, cache_slab_capacity=args.cache_slabs
    )
    prompt_ids = {name: chat_ids(recipe, text) for name, text in SWEEP_PROMPTS.items()}
    neurons = select_neurons(args.neurons, runner.config.intermediate_size, args.seed)
    sweep = SteeringSweep(
        runner,
        calibration,
        layer,
        neurons,
        prompt_ids,
        random_controls=args.random_controls,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_layers = sorted(
        {layer + int(offset) for offset in args.capture_layers.split(",") if offset}
        & set(range(runner.config.num_hidden_layers))
    )
    meta = sweep.metadata(
        SWEEP_PROMPTS,
        model=args.model,
        revision=args.revision,
        stats=os.path.abspath(args.stats),
        calibration=os.path.abspath(calibration_path),
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        capture_layers=capture_layers,
    )
    (out_dir / f"{args.run}.meta.json").write_text(json.dumps(meta, indent=1))
    path = out_dir / f"{args.run}.jsonl"

    def key(job):
        return job["kind"], job["neuron"], job["prompt"], job["dose"]

    done = set()
    if path.exists():
        with path.open() as file:
            done = {key(json.loads(line)) for line in file}
    jobs = [job for job in sweep.jobs() if key(job) not in done]
    print(
        f"L{layer}: {len(neurons)} neurons + {args.random_controls} random controls, "
        f"{len(jobs)} rollouts to run ({len(done)} already done)",
        flush=True,
    )
    if not jobs:
        return
    start = time.perf_counter()
    generated = 0
    with (
        Engine(runner, max_running_requests=args.max_running) as engine,
        path.open("a") as file,
    ):
        records = sweep.run(
            engine,
            jobs,
            decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            stop_token_ids=recipe.stop_token_ids,
            queue=2 * args.max_running,
        )
        for count, record in enumerate(records, 1):
            file.write(json.dumps(record) + "\n")
            generated += record["n_gen"]
            if count % args.report_every == 0 or count == len(jobs):
                file.flush()
                elapsed = time.perf_counter() - start
                print(
                    f"{count}/{len(jobs)} rollouts, {generated / elapsed:,.1f} gen tok/s, "
                    f"{count / elapsed:.2f} rollouts/s",
                    flush=True,
                )


def run_lens(args, recipe, tokenizer):
    lo, _, hi = args.layers.partition("-")
    layers = list(range(int(lo), int(hi or lo) + 1))
    runner = Qwen35MlxRunner(
        args.model, revision=args.revision, cache_slab_capacity=UNSERVED_SLABS
    )
    calibration = SteeringCalibration.load(args.calibration)
    contexts = [chat_ids(recipe, text) for text in LENS_CHAT_PROMPTS] + [
        tokenizer.encode(text, add_special_tokens=False).ids
        for text in LENS_TEXT_PROMPTS
    ]
    neurons = (
        None
        if args.neurons is None
        else select_neurons(args.neurons, runner.config.intermediate_size, args.seed)
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    computed = {}
    for layer in layers:
        start = time.perf_counter()
        values = steer_lens(
            runner,
            contexts,
            layer,
            calibration,
            dose=args.dose,
            k=args.k,
            neurons=neurons,
            batch=args.neurons_per_batch,
        )
        computed[layer] = values
        np.savez(
            f"{args.out}_L{layer}.npz",
            **values,
            dose=args.dose,
            contexts=np.array(LENS_CHAT_PROMPTS + LENS_TEXT_PROMPTS),
        )
        print(
            f"L{layer}: {len(values['up_kl'])} neurons in {time.perf_counter() - start:.0f}s, "
            f"KL+ median {np.median(values['up_kl']):.3f}",
            flush=True,
        )
    if args.index:
        update_steer_index(
            args.index,
            computed,
            neurons=neurons,
            shape=(runner.config.num_hidden_layers, runner.config.intermediate_size),
            dose=args.dose,
            k=args.k,
        )
        print(
            f"steer index rows replaced for layers {sorted(computed)} -> {args.index}"
        )


def run_connections(args, recipe, tokenizer):
    runner = Qwen35MlxRunner(
        args.model, revision=args.revision, cache_slab_capacity=UNSERVED_SLABS
    )
    connections = DirectConnections(runner, k=args.k)
    start = time.perf_counter()
    for target in range(1, runner.config.num_hidden_layers):
        connections.add_target_layer(target)
        print(f"L{target} done ({time.perf_counter() - start:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **connections.arrays())
    print(f"-> {args.out} {os.path.getsize(args.out) / 1e6:.0f} MB")


def run_capture(args, recipe, tokenizer):
    runner = Qwen35MlxRunner(
        args.model, revision=args.revision, cache_slab_capacity=UNSERVED_SLABS
    )
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        meta = json.loads(Path(run.replace(".jsonl", ".meta.json")).read_text())
        name = Path(run).name.removesuffix(".jsonl")
        out = Path(args.out_dir) / f"{name}.mlp_in.npy"
        if out.exists():
            print(f"{name}: exists, skipping", flush=True)
            continue
        with open(run) as file:
            records = [json.loads(line) for line in file]
        calibration = SteeringCalibration.load(meta["calibration"])
        sweep = SteeringSweep(
            runner,
            calibration,
            meta["layer"],
            [],
            meta["prompt_ids"],
            random_controls=meta["random_controls"],
            seed=meta["seed"],
        )
        for record in records:
            if record["kind"] == "random" and record["ref_neuron"] != int(
                sweep.control_references[record["neuron"]]
            ):
                raise SystemExit(f"{run}: control directions do not match its seed")
        capture_layers = meta["capture_layers"]
        hidden = runner.config.hidden_size
        temporary = str(out) + ".tmp.npy"
        values = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=(len(records), len(capture_layers), hidden),
        )
        start = time.perf_counter()
        by_prompt = {}
        for row, record in enumerate(records):
            by_prompt.setdefault(record["prompt"], []).append(row)
        for prompt, rows in by_prompt.items():
            for first in range(0, len(rows), args.batch):
                chunk = rows[first : first + args.batch]
                pushes = np.stack([sweep.residual_push(records[row]) for row in chunk])
                values[chunk] = capture_mlp_inputs(
                    runner,
                    meta["layer"],
                    pushes,
                    meta["prompt_ids"][prompt],
                    capture_layers,
                ).astype(np.float16)
        values.flush()
        del values
        os.replace(temporary, out)
        out.with_suffix(".json").write_text(
            json.dumps(
                {
                    "run": name,
                    "capture_layers": capture_layers,
                    "hook": "mlp_in",
                    "position": "last_prompt",
                    "rows": len(records),
                }
            )
        )
        print(
            f"{name}: {len(records)} rows, layers {capture_layers}, "
            f"{time.perf_counter() - start:.0f}s",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    commands = parser.add_subparsers(dest="command", required=True)

    stats = commands.add_parser("stats", help="per-neuron activation extremes")
    stats.add_argument("--layers", default="22", help="comma-separated layers")
    stats.add_argument("--tokens", type=int, default=2_000_000)
    stats.add_argument("--seq-len", type=int, default=1024)
    stats.add_argument("--batch", type=int, default=16)
    stats.add_argument("--seed", type=int, default=0)
    stats.add_argument("--out", default="stats/q35-4b")
    stats.set_defaults(execute=run_stats)

    sweep = commands.add_parser("sweep", help="main_rel steered rollouts for one layer")
    sweep.add_argument("--stats", required=True, help="stats npz of the layer to steer")
    sweep.add_argument(
        "--calibration", help="defaults to the stats run's .calibration.json"
    )
    sweep.add_argument("--neurons", default="all", help="all, random:K, N, A-B, lists")
    sweep.add_argument("--random-controls", type=int, default=256)
    sweep.add_argument("--run", required=True)
    sweep.add_argument("--out-dir", default="runs")
    sweep.add_argument("--max-tokens", type=int, default=256)
    sweep.add_argument("--top-k", type=int, default=5)
    sweep.add_argument("--capture-layers", default="+0,+1,+4,+8")
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--max-running", type=int, default=16)
    sweep.add_argument("--cache-slabs", type=int, default=32)
    sweep.add_argument("--report-every", type=int, default=50)
    sweep.set_defaults(execute=run_sweep)

    lens = commands.add_parser("lens", help="next-token effect of ±dose pushes")
    lens.add_argument("--layers", default="0-31", help="A or A-B")
    lens.add_argument("--calibration", required=True)
    lens.add_argument("--dose", type=float, default=0.8)
    lens.add_argument("--k", type=int, default=12)
    lens.add_argument("--neurons", help="subset; default every neuron")
    lens.add_argument("--seed", type=int, default=0)
    lens.add_argument("--neurons-per-batch", type=int, default=64)
    lens.add_argument("--out", default="stats/steer_lens")
    lens.add_argument("--index", help="explorer steer_index.npz to update")
    lens.set_defaults(execute=run_lens)

    connections = commands.add_parser("connections", help="direct-path links")
    connections.add_argument("--k", type=int, default=24)
    connections.add_argument("--out", required=True)
    connections.set_defaults(execute=run_connections)

    capture = commands.add_parser("capture", help="MLP inputs at the last prompt token")
    capture.add_argument("runs", nargs="+")
    capture.add_argument("--batch", type=int, default=64)
    capture.add_argument("--out-dir", default="captures")
    capture.set_defaults(execute=run_capture)

    args = parser.parse_args()
    tokenizer = Tokenizer.from_pretrained(args.model, revision=args.revision)
    recipe = get_native_recipe("qwen3_5", tokenizer, enable_thinking=False)
    args.execute(args, recipe, tokenizer)


if __name__ == "__main__":
    main()
