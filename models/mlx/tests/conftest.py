"""A tiny hybrid Qwen3.5 checkpoint shared by the MLX runner and neuron-sweep tests."""

import json

import pytest
import torch


@pytest.fixture(scope="session")
def qwen35_checkpoint(tmp_path_factory):
    """Return the checkpoint directory and its FP32 Transformers reference model."""
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.set_num_threads(2)
    torch.manual_seed(41)
    config = Qwen3_5TextConfig(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=True,
        eos_token_id=256,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [3, 3, 2],
        },
        dtype="bfloat16",
    )
    reference = Qwen3_5ForCausalLM(config).float().eval()
    path = tmp_path_factory.mktemp("qwen35-checkpoint")
    values = config.to_dict()
    values["dtype"] = "bfloat16"
    (path / "config.json").write_text(json.dumps(values))
    weights = {
        name: value.contiguous()
        for name, value in reference.state_dict().items()
        if name != "lm_head.weight"
    }
    save_file(weights, path / "model.safetensors")
    return path, reference
