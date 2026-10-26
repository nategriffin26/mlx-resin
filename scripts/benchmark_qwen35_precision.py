"""Compare Qwen3.5 BF16 execution noise against an independently saved HF FP32 oracle."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

os.environ["MLX_ENABLE_TF32"] = "0"

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("hf-fp32", "hf-bf16", "mlx-native", "mlx-exact"),
        required=True,
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("local/qwen35-precision.npz")
    )
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[17, 129, 257])
    args = parser.parse_args()
    path = Path(
        snapshot_download(
            MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors"]
        )
    )
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    source = tokenizer.encode(
        "Alice lives in Paris. Bob lives in Rome. Carol lives in Berlin. " * 80
    ).ids
    prompts = [source[:length] for length in args.prompt_lengths]
    logits = []
    if args.backend.startswith("hf"):
        import torch
        from safetensors import safe_open
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextRotaryEmbedding,
        )

        dtype = torch.float32 if args.backend == "hf-fp32" else torch.bfloat16
        torch.set_num_threads(4)
        config = Qwen3_5TextConfig.from_dict(
            json.loads((path / "config.json").read_text())["text_config"]
        )
        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(config)
        weights = {}
        for shard in sorted(path.glob("model*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for name in tensors.keys():
                    if name.startswith("model.language_model."):
                        weights[name.replace("model.language_model.", "model.", 1)] = (
                            tensors.get_tensor(name).to(dtype)
                        )
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        model.load_state_dict(weights, strict=True, assign=True)
        model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        model.eval()
        del weights
        with torch.inference_mode():
            for prompt in prompts:
                logits.append(
                    model(torch.tensor([prompt]), use_cache=False, logits_to_keep=1)
                    .logits[0, -1]
                    .float()
                    .numpy()
                )
    else:
        import mlx.core as mx

        import resin_models_mlx.qwen3_5.model as qwen_model
        from resin import Engine, SamplingParams
        from resin_models_mlx.qwen3_5 import Qwen35MlxRunner

        if args.backend == "mlx-native":
            qwen_model.gated_delta_update = mx.fast.gated_delta_update
        runner = Qwen35MlxRunner(path, cache_slab_capacity=16)
        sampler = runner._sampler

        def record(rows, plan):
            if plan.sampling_mask[0]:
                logits.append(np.array(rows[0].astype(mx.float32)))
            return sampler(rows, plan)

        runner._sampler = record
        with Engine(runner) as engine:
            for prompt in prompts:
                request = engine.submit(
                    prompt,
                    sampling_params=SamplingParams(temperature=0, max_output_tokens=1),
                )
                while engine.has_work():
                    engine.step()
                request.take_result()
    logits = np.stack(logits)
    if args.backend == "hf-fp32":
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.reference,
            logits=logits,
            revision=REVISION,
            lengths=args.prompt_lengths,
        )
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "reference": str(args.reference),
                    "greedy_tokens": logits.argmax(-1).tolist(),
                }
            )
        )
    else:
        reference = np.load(args.reference)
        if str(reference["revision"]) != REVISION:
            raise ValueError("precision reference uses another checkpoint revision")
        if not np.array_equal(reference["lengths"], args.prompt_lengths):
            raise ValueError("precision reference uses different prompt lengths")
        difference = logits - reference["logits"]
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "lengths": args.prompt_lengths,
                    "max_absolute_error": np.max(np.abs(difference), axis=-1).tolist(),
                    "rms_error": np.sqrt(np.mean(difference**2, axis=-1)).tolist(),
                    "greedy_tokens": logits.argmax(-1).tolist(),
                    "hf_fp32_greedy_tokens": reference["logits"].argmax(-1).tolist(),
                }
            )
        )


if __name__ == "__main__":
    main()
