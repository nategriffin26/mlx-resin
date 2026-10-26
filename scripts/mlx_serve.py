"""Serve a Qwen MLX runner through Resin's OpenAI and Anthropic HTTP APIs."""

import argparse

from tokenizers import Tokenizer

from resin import Engine
from resin.recipe import get_native_recipe
from resin.server import HttpServer


def load_runner(name):
    if name == "qwen3_5":
        from resin_models_mlx.qwen3_5 import (
            DEFAULT_MODEL,
            DEFAULT_REVISION,
            Qwen35MlxRunner,
        )

        return DEFAULT_MODEL, DEFAULT_REVISION, Qwen35MlxRunner
    from resin_models_mlx.qwen3_5_moe import (
        DEFAULT_MODEL,
        DEFAULT_REVISION,
        Qwen35MoeMlxRunner,
    )

    return DEFAULT_MODEL, DEFAULT_REVISION, Qwen35MoeMlxRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["qwen3_5", "qwen3_5_moe"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model, revision, runner_class = load_runner(args.model)
    tokenizer = Tokenizer.from_pretrained(model, revision=revision)
    recipe = get_native_recipe("qwen3_5", tokenizer)
    engine = Engine(runner_class(), prefix_cache=True)
    print(f"Serving {model} at http://{args.host}:{args.port}", flush=True)
    HttpServer(engine, recipe=recipe, model=model).serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
