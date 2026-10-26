"""Pinned 4B parity against HF's recurrent FP32 formulation.

HF's chunked and recurrent rules differ by 2.06e-3 maximum and 4.9e-4 RMS at
128 tokens. Equivalent L2 normalization orderings move the observed maximum
violation from 1.02e-3 to 1.75e-3. Forced-input layer errors are usually 1e-6–1e-5
(up to 2.6e-4 in sensitive later layers). Use atol=3e-3, rtol=1e-3 against HF's
recurrent rule, with exact greedy equality, rather than tuning reduction order.

At 127/128 tokens, HF BF16 versus FP32 has max errors 0.68/3.08 and RMS
0.13/0.46; MLX BF16 has max 0.84/1.78 and RMS 0.11/0.31. BF16 therefore
requires exact greedy tokens, per-row RMS <= 0.5 and max absolute error <= 3.5.
"""

import gc
import json
import os
import sys

import numpy as np
import pytest
import torch

from resin import Engine, SchedulerConfig

if sys.platform == "darwin":
    os.environ["MLX_ENABLE_TF32"] = "0"
    mx = pytest.importorskip("mlx.core")
    from test_qwen35_mlx import RecordingRunner, generate, reference_generate, sampled

    from resin_models_mlx.qwen3_5 import DEFAULT_MODEL, DEFAULT_REVISION
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


pytestmark = pytest.mark.checkpoint


def test_pinned_4b_greedy_batch_chunking_and_prefix_match_hf_fp32():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding,
        torch_recurrent_gated_delta_rule,
    )

    path = snapshot_download(
        DEFAULT_MODEL,
        revision=DEFAULT_REVISION,
        allow_patterns=["*.json", "*.safetensors"],
    )
    from pathlib import Path

    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    config = Qwen3_5TextConfig.from_dict(raw["text_config"])
    torch.set_num_threads(4)
    with torch.device("meta"):
        reference = Qwen3_5ForCausalLM(config)
    weights = {}
    for shard in sorted(path.glob("model*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name.startswith("model.language_model."):
                    weights[name.replace("model.language_model.", "model.", 1)] = (
                        tensors.get_tensor(name).float()
                    )
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    reference.load_state_dict(weights, strict=True, assign=True)
    # Nonpersistent RoPE buffers are absent from state_dict and must leave meta too.
    reference.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    reference.eval()
    del weights

    def recurrent(query, key, value, **kwargs):
        return torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            kwargs["g"],
            kwargs["beta"],
            kwargs.get("initial_state"),
            kwargs.get("output_final_state"),
            kwargs.get("use_qk_l2norm_in_kernel", False),
        )

    for layer in reference.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.chunk_gated_delta_rule = recurrent
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    source = tokenizer.encode(
        "Alice lives in Paris. Bob lives in Rome. Carol lives in Berlin. " * 80
    ).ids
    inputs = [source[:17], source[:127], source[:128], source[:129], source[:257]]
    expected = [reference_generate(reference, tokens, count=3) for tokens in inputs]
    # Cache/MLX and HF weights need not occupy unified memory simultaneously.
    del reference
    gc.collect()
    model = RecordingRunner(path, dtype=mx.float32, cache_slab_capacity=16)
    with Engine(
        model, scheduler_config=SchedulerConfig(prefill_chunk_size=63)
    ) as engine:
        actual, _ = generate(engine, inputs, count=3)
    assert actual == [tokens for tokens, _ in expected]
    by_sequence = {}
    for ids, mask, logits in model.logits:
        for sequence, valid, row in zip(ids, mask, logits, strict=True):
            if valid:
                by_sequence.setdefault(int(sequence), []).append(row)
    for observed, (_, logits) in zip(by_sequence.values(), expected, strict=True):
        np.testing.assert_allclose(observed, logits, atol=3e-3, rtol=1e-3)
    del engine, model
    gc.collect()
    mx.clear_cache()
    cached = RecordingRunner(path, dtype=mx.float32, cache_slab_capacity=16)
    with Engine(cached, prefix_cache=True) as engine:
        generate(engine, [source[:263]], count=3)
        for tokens, (reference_tokens, reference_logits), hit in zip(
            inputs[1:], expected[1:], [0, 0, 128, 256], strict=True
        ):
            cached.logits.clear()
            actual, requests = generate(engine, [tokens], count=3)
            assert requests[0].cached_tokens == hit
            assert actual == [reference_tokens]
            np.testing.assert_allclose(
                sampled(cached), reference_logits, atol=3e-3, rtol=1e-3
            )
    del engine, cached
    gc.collect()
    mx.clear_cache()
    bf16 = RecordingRunner(path, dtype=mx.bfloat16, cache_slab_capacity=16)
    with Engine(bf16) as engine:
        for tokens, (reference_tokens, reference_logits) in zip(
            inputs, expected, strict=True
        ):
            bf16.logits.clear()
            actual, _ = generate(engine, [tokens], count=3)
            assert actual == [reference_tokens]
            difference = sampled(bf16).astype(np.float64) - reference_logits
            assert np.max(np.abs(difference)) <= 3.5
            assert np.all(np.sqrt(np.mean(difference**2, axis=-1)) <= 0.5)
