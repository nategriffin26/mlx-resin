"""MoE semantics and Engine state against a tiny FP32 Transformers CPU oracle."""

import json
import os
import sys

import numpy as np
import pytest
import torch

from resin import Engine, SamplingParams, SchedulerConfig
from resin.models.qwen3_5_moe import Qwen35MoeConfig

if sys.platform == "darwin":
    os.environ["MLX_ENABLE_TF32"] = "0"
    mx = pytest.importorskip("mlx.core")
    from resin_models_mlx.qwen3_5_moe import Qwen35MoeMlxRunner
    from resin_models_mlx.qwen3_5_moe.model import route
    from resin_models_mlx.qwen3_5_moe.quantization import (
        QuantizedEmbedding,
        QuantizedExpertLinear,
        QuantizedLinear,
    )
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from safetensors.torch import save_file
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.set_num_threads(2)
    torch.manual_seed(57)
    config = Qwen3_5MoeTextConfig(
        vocab_size=257,
        hidden_size=64,
        num_hidden_layers=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=8,
        num_experts_per_tok=2,
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
    reference = Qwen3_5MoeForCausalLM(config).float().eval()
    path = tmp_path_factory.mktemp("qwen35-moe")
    values = config.to_dict()
    values["dtype"] = "bfloat16"
    (path / "config.json").write_text(json.dumps(values))
    save_file(
        {
            name: value.contiguous()
            for name, value in reference.state_dict().items()
            if name != "lm_head.weight"
        },
        path / "model.safetensors",
    )
    return path, reference


def prompt(length, offset=0):
    return [1 + (index * 17 + offset) % 250 for index in range(length)]


@torch.inference_mode()
def reference_generate(model, tokens, count=4):
    inputs = torch.tensor([tokens])
    cache, output, logits = None, [], []
    for _ in range(count):
        result = model(inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        row = result.logits[:, -1]
        token = row.argmax(-1)
        logits.append(row.numpy().copy())
        output.append(token.item())
        inputs = token[:, None]
    return output, np.concatenate(logits)


class RecordingRunner(Qwen35MoeMlxRunner):
    def __init__(self, path, **kwargs):
        super().__init__(
            path,
            dtype=mx.float32,
            prefill_chunk_size=128,
            cache_slab_capacity=kwargs.pop("capacity", 16),
            **kwargs,
        )
        self.logits, self.plans = {}, []
        sampler = self._sampler

        def record(logits, plan):
            rows = np.array(logits.astype(mx.float32))
            for sequence, valid, row in zip(
                plan.sequence_ids, plan.sampling_mask, rows, strict=True
            ):
                if valid:
                    self.logits.setdefault(int(sequence), []).append(row)
            return sampler(logits, plan)

        self._sampler = record

    def forward(self, plan):
        self.plans.append(plan)
        return super().forward(plan)


def generate(engine, inputs, count=4):
    requests = [
        engine.submit(
            tokens,
            sampling_params=SamplingParams(temperature=0, max_output_tokens=count),
        )
        for tokens in inputs
    ]
    while engine.has_work():
        engine.step()
    return [request.take_result() for request in requests], requests


def assert_oracle(runner, references, actual):
    assert actual == [tokens for tokens, _ in references]
    for rows, (_, logits) in zip(runner.logits.values(), references, strict=True):
        np.testing.assert_allclose(rows, logits, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize(
    "inputs", [[prompt(17)], [prompt(7), prompt(35, 1), prompt(133, 2)]]
)
def test_routed_decode_preserves_unequal_histories(checkpoint, inputs):
    references = [reference_generate(checkpoint[1], tokens) for tokens in inputs]
    runner = RecordingRunner(checkpoint[0])
    with Engine(runner) as engine:
        actual, _ = generate(engine, inputs)
    assert_oracle(runner, references, actual)


@pytest.mark.parametrize("chunk", [63, 128])
def test_chunked_moe_prefill_restores_recurrent_state(checkpoint, chunk):
    tokens = prompt(263)
    reference = reference_generate(checkpoint[1], tokens)
    runner = RecordingRunner(checkpoint[0])
    with Engine(
        runner, scheduler_config=SchedulerConfig(prefill_chunk_size=chunk)
    ) as engine:
        actual, _ = generate(engine, [tokens])
    assert_oracle(runner, [reference], actual)


@pytest.mark.parametrize(
    "length,hit", [(127, 0), (128, 0), (129, 128), (255, 128), (256, 128), (257, 256)]
)
def test_prefix_hit_restores_checkpoint_not_partial_state(checkpoint, length, hit):
    tokens = prompt(length)
    reference = reference_generate(checkpoint[1], tokens)
    fresh, cached = RecordingRunner(checkpoint[0]), RecordingRunner(checkpoint[0])
    with Engine(fresh) as engine:
        expected, _ = generate(engine, [tokens])
    with Engine(
        cached,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(prefill_chunk_size=63),
    ) as engine:
        generate(engine, [prompt(263)])
        cached.logits.clear()
        actual, requests = generate(engine, [tokens])
        assert requests[0].cached_tokens == hit
    assert actual == expected
    assert_oracle(cached, [reference], actual)
    np.testing.assert_allclose(
        next(iter(cached.logits.values())),
        next(iter(fresh.logits.values())),
        atol=3e-4,
        rtol=3e-4,
    )


def test_nan_pages_and_reused_pages_do_not_leak_state(checkpoint):
    runner = RecordingRunner(checkpoint[0], capacity=6)
    runner.arena.storage = mx.full(
        (runner.arena.storage.shape[0], runner.arena.slab_bytes // 4),
        float("nan"),
        mx.float32,
    ).view(mx.uint8)
    mx.eval(runner.arena.storage)
    inputs = [prompt(257, offset) for offset in range(6)]
    with Engine(runner, prefix_cache=True) as engine:
        for tokens in inputs:
            runner.logits.clear()
            actual, _ = generate(engine, [tokens])
            assert_oracle(runner, [reference_generate(checkpoint[1], tokens)], actual)
        runner.logits.clear()
        actual, requests = generate(engine, [inputs[0]])
        assert requests[0].cached_tokens == 0
        assert engine.prefix_cache_stats[3] > 0
        assert_oracle(runner, [reference_generate(checkpoint[1], inputs[0])], actual)
    assert bool(mx.all(runner.arena.storage[0] == 0))


def test_router_renormalizes_selected_experts_and_keeps_ties_equal():
    logits = mx.array([[0.0, 1.0, 2.0, 3.0], [2.0, 2.0, 2.0, 2.0]])
    indices, scores = route(logits, 2)
    indices, scores = np.array(indices), np.array(scores)
    assert set(indices[0]) == {2, 3}
    ordered = scores[0][np.argsort(indices[0])]
    np.testing.assert_allclose(ordered, [0.26894142, 0.73105858], atol=1e-7)
    assert len(set(indices[1])) == 2
    np.testing.assert_array_equal(scores[1], [0.5, 0.5])


@pytest.mark.parametrize("bits", [4, 8])
def test_packed_projections_match_their_dequantized_weights(bits):
    mx.random.seed(19)
    weight = mx.random.normal((4, 64, 64))
    packed = mx.quantize(weight, group_size=64, bits=bits)
    dequantized = mx.dequantize(*packed, group_size=64, bits=bits)
    hidden = mx.random.normal((3, 1, 64))
    indices = mx.array([3, 0, 2], mx.uint32)
    expert = QuantizedExpertLinear(packed, bits=bits)
    expected = mx.gather_mm(hidden, dequantized.swapaxes(-1, -2), rhs_indices=indices)
    np.testing.assert_allclose(
        np.array(expert(hidden, indices)), np.array(expected), atol=3e-5, rtol=3e-5
    )
    packed = tuple(value[0] for value in packed)
    linear, embedding = (
        QuantizedLinear(packed, bits=bits),
        QuantizedEmbedding(packed, bits=bits),
    )
    np.testing.assert_allclose(
        np.array(linear(hidden)),
        np.array(hidden @ dequantized[0].T),
        atol=3e-5,
        rtol=3e-5,
    )
    np.testing.assert_array_equal(
        np.array(embedding(indices)), np.array(dequantized[0][indices])
    )


def test_config_rejects_invalid_routing_without_weakening_dense_parser(checkpoint):
    raw = json.loads((checkpoint[0] / "config.json").read_text())
    config = Qwen35MoeConfig.from_hugging_face(raw)
    assert (
        config.num_experts,
        config.num_experts_per_tok,
        config.intermediate_size,
    ) == (8, 2, 64)
    for change in (
        {"num_experts_per_tok": 0},
        {"num_experts_per_tok": 9},
        {"moe_intermediate_size": 0},
        {"norm_topk_prob": False},
    ):
        with pytest.raises(ValueError):
            Qwen35MoeConfig.from_hugging_face({**raw, **change})
    from resin.models.qwen3_5.config import Qwen35Config

    with pytest.raises(ValueError):
        Qwen35Config.from_hugging_face(raw)


@pytest.mark.parametrize("converted_layout", [False, True])
def test_mixed_bit_checkpoint_loader_preserves_router_bits_and_norm_offsets(
    checkpoint, tmp_path, converted_layout
):
    import copy

    import mlx.nn as mx_nn
    from mlx.utils import tree_flatten

    from resin_models_mlx.qwen3_5.model import RMSNorm
    from resin_models_mlx.qwen3_5_moe.model import ExpertLinear

    plain = RecordingRunner(checkpoint[0])
    model = plain._model
    weights = dict(tree_flatten(model.parameters()))
    reference = copy.deepcopy(checkpoint[1])
    state = reference.state_dict()
    quantization = {"group_size": 64, "bits": 4, "mode": "affine"}
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm) and converted_layout:
            weights[name + ".weight"] = weights[name + ".weight"] + 1
        if name.endswith(".conv1d") and not converted_layout:
            weights[name + ".weight"] = weights[name + ".weight"].transpose(0, 2, 1)
        if not isinstance(module, (mx_nn.Linear, mx_nn.Embedding, ExpertLinear)):
            continue
        bits = 8 if name.endswith(("mlp.gate", "shared_expert_gate")) else 4
        if bits == 8:
            quantization["language_model." + name] = {"group_size": 64, "bits": 8}
        packed = mx.quantize(module.weight, group_size=64, bits=bits)
        for suffix, value in zip(
            (".weight", ".scales", ".biases"), packed, strict=True
        ):
            weights[name + suffix] = value
        dequantized = torch.from_numpy(
            np.array(mx.dequantize(*packed, group_size=64, bits=bits))
        )
        if ".switch_mlp." in name:
            prefix, projection = name.split(".switch_mlp.")
            if projection == "down_proj":
                state[prefix + ".experts.down_proj"] = dequantized
            else:
                part = 0 if projection == "gate_proj" else 1
                state[prefix + ".experts.gate_up_proj"][
                    :, part * 64 : (part + 1) * 64
                ] = dequantized
        else:
            state[name + ".weight"] = dequantized
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    reference.load_state_dict(state)
    raw = json.loads((checkpoint[0] / "config.json").read_text())
    raw["quantization"] = quantization
    (tmp_path / "config.json").write_text(json.dumps(raw))
    weights = {"language_model." + name: value for name, value in weights.items()}
    weights["language_model.model.mtp.unused.weight"] = mx.ones((2, 2))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    runner = RecordingRunner(tmp_path)
    assert runner._model.model.layers[0].mlp.gate.bits == 8
    assert runner._model.model.layers[0].mlp.switch_mlp.up_proj.bits == 4
    inputs = [prompt(17), prompt(133, 1)]
    with Engine(runner) as engine:
        actual, _ = generate(engine, inputs)
    assert_oracle(
        runner, [reference_generate(reference, tokens) for tokens in inputs], actual
    )


def test_shared_expert_steering_is_gated_and_isolated_by_request(checkpoint):
    runner = RecordingRunner(checkpoint[0])
    tokens = prompt(17)
    calibration = runner.calibrate(prompt(33))
    steering = runner.steer(tokens, [(1, 19, 1)], calibration)
    layer, neuron, value = steering.additions[0]
    expected_plain = reference_generate(checkpoint[1], tokens)

    def add(module, inputs):
        hidden = inputs[0].clone()
        hidden[..., neuron] += value
        return (hidden,)

    hook = (
        checkpoint[1]
        .model.layers[layer]
        .mlp.shared_expert.down_proj.register_forward_pre_hook(add)
    )
    try:
        expected_steered = reference_generate(checkpoint[1], tokens)
    finally:
        hook.remove()
    with Engine(runner) as engine:
        requests = [
            engine.submit(
                tokens,
                sampling_params=SamplingParams(temperature=0, max_output_tokens=4),
            ),
            engine.submit(
                media=steering,
                sampling_params=SamplingParams(temperature=0, max_output_tokens=4),
            ),
        ]
        while engine.has_work():
            engine.step()
        actual = [request.take_result() for request in requests]
    assert_oracle(runner, [expected_plain, expected_steered], actual)
    assert not np.allclose(expected_plain[1], expected_steered[1])
