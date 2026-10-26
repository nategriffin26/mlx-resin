"""Chat with the LFM2 MLX runner using Resin's native conversation recipe."""

from tokenizers import Tokenizer

import resin
from resin import Engine, SamplingParams
from resin.recipe import get_native_recipe
from resin_models_mlx.lfm2 import DEFAULT_MODEL, DEFAULT_REVISION, Lfm2MlxRunner


def main() -> None:
    tokenizer = Tokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    recipe = get_native_recipe("lfm2", tokenizer)
    conversation = [
        resin.recipe.Message(
            resin.recipe.Role.SYSTEM,
            [resin.recipe.Text("You are a helpful assistant trained by Liquid AI.")],
        ),
        resin.recipe.Message(
            resin.recipe.Role.USER,
            [resin.recipe.Text("What is the capital of France?")],
        ),
    ]
    prompt = recipe.render(conversation)
    parser = recipe.create_parser(prompt)
    runner = Lfm2MlxRunner()
    with Engine(runner) as engine:
        request = engine.submit(
            prompt.token_ids,
            sampling_params=SamplingParams(
                max_output_tokens=256,
                stop_token_ids=recipe.stop_token_ids,
            ),
        )
        while engine.has_work():
            for token in engine.step():
                events = parser.feed([token.token_id])
                if token.finished:
                    reason = token.finish_reason
                    if reason not in ("stop", "length"):
                        raise RuntimeError(f"unexpected completion reason: {reason!r}")
                    events.extend(parser.finish(reason))
                for event in events:
                    if isinstance(event, resin.recipe.TextDelta):
                        print(event.delta, end="", flush=True)
        request.take_result()
    print()


if __name__ == "__main__":
    main()
