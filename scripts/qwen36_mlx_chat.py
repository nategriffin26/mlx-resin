"""Chat with the pinned Qwen3.6-35B-A3B affine 4-bit checkpoint."""

import argparse

from tokenizers import Tokenizer

from resin import Engine, SamplingParams
from resin.recipe import Message, Role, Text, get_native_recipe
from resin_models_mlx.qwen3_5_moe import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    Qwen35MoeMlxRunner,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    recipe = get_native_recipe("qwen3_5", tokenizer, enable_thinking=False)
    prompt = recipe.render([Message(Role.USER, [Text(args.prompt)])])
    runner = Qwen35MoeMlxRunner()
    with Engine(runner) as engine:
        request = engine.submit(
            prompt.token_ids,
            sampling_params=SamplingParams(
                temperature=0,
                max_output_tokens=args.max_tokens,
                stop_token_ids=recipe.stop_token_ids,
            ),
        )
        while engine.has_work():
            engine.step()
        print(tokenizer.decode(request.take_result()))


if __name__ == "__main__":
    main()
