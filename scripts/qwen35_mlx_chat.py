"""Chat with dense Qwen3.5, or compare baseline and calibrated neuron steering."""

import argparse
from pathlib import Path
from textwrap import wrap

from tokenizers import Tokenizer

from resin import Engine, SamplingParams
from resin.recipe import Message, Role, Text, get_native_recipe
from resin_models_mlx.qwen3_5 import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Qwen35MlxRunner,
    SteeringCalibration,
)

CALIBRATION_TEXT = (
    "Alice lives in Paris and studies the history of science. Bob takes a train to Rome. "
    "A library contains novels, mathematics textbooks, maps, and letters. "
    "Explain a difficult idea clearly, compare the evidence, and describe what remains uncertain. "
    "A farmer grows apples, a musician practices piano, and an engineer designs a bridge. "
) * 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--steering-demo", action="store_true")
    parser.add_argument(
        "--calibration", type=Path, default=Path("local/qwen35-calibration.json")
    )
    parser.add_argument("--calibration-tokens", type=int, default=256)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    recipe = get_native_recipe("qwen3_5", tokenizer, enable_thinking=False)
    prompt = recipe.render([Message(Role.USER, [Text(args.prompt)])])
    runner = Qwen35MlxRunner()
    inputs = [prompt.token_ids]
    if args.steering_demo:
        if args.calibration.exists():
            calibration = SteeringCalibration.load(args.calibration)
        else:
            tokens = tokenizer.encode(CALIBRATION_TEXT).ids[: args.calibration_tokens]
            if len(tokens) != args.calibration_tokens:
                parser.error("calibration text is shorter than --calibration-tokens")
            calibration = runner.calibrate(tokens)
            args.calibration.parent.mkdir(parents=True, exist_ok=True)
            calibration.save(args.calibration)
        inputs.append(
            runner.steer(
                prompt.token_ids, [(14, 8949, 1.0), (13, 7956, -1.0)], calibration
            )
        )
        print(
            f"Calibration: {calibration.token_count} tokens; medians exclude token 0; {args.calibration}"
        )
        print("Steering: L14 N8949 +1; L13 N7956 -1 (zero-based)")
    with Engine(runner) as engine:
        requests = [
            engine.submit(
                value if not args.steering_demo or index == 0 else None,
                media=value if index else None,
                sampling_params=SamplingParams(
                    temperature=0,
                    max_output_tokens=args.max_tokens,
                    stop_token_ids=recipe.stop_token_ids,
                ),
            )
            for index, value in enumerate(inputs)
        ]
        while engine.has_work():
            engine.step()
        replies = [tokenizer.decode(request.take_result()) for request in requests]
    if len(replies) == 1:
        print(replies[0])
    else:
        columns = [wrap(reply, width=60) for reply in replies]
        print(f"{'BASELINE':60} | STEERED")
        for index in range(max(map(len, columns))):
            left, right = [
                column[index] if index < len(column) else "" for column in columns
            ]
            print(f"{left:60} | {right}")


if __name__ == "__main__":
    main()
