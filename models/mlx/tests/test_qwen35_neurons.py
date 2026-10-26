"""Attack neuron sweeps against Transformers references and the explorer's file semantics."""

import glob
import json
import os
import sys

import numpy as np
import pytest
import torch

from resin import Engine

if sys.platform == "darwin":
    mx = pytest.importorskip("mlx.core")
    from resin_models_mlx.qwen3_5 import Qwen35MlxRunner
    from resin_models_mlx.qwen3_5.neurons import (
        ActivationSurvey,
        DirectConnections,
        SteeringSweep,
        capture_mlp_inputs,
        select_neurons,
        steer_lens,
        update_steer_index,
    )
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)

TOLERANCE = {"atol": 3e-4, "rtol": 3e-4}
DESIGN = (("short", -0.7), ("short", 0.8), ("long", 3.0))


def prompt(length, offset=0):
    return [1 + (index * 29 + offset) % 250 for index in range(length)]


class BatchRecordingRunner(Qwen35MlxRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sizes = []

    def forward(self, plan):
        self.batch_sizes.append(plan.num_sequences)
        return super().forward(plan)


def runner(qwen35_checkpoint, capacity=16):
    return BatchRecordingRunner(
        qwen35_checkpoint[0], dtype=mx.float32, cache_slab_capacity=capacity
    )


def sweep_for(model, calibration):
    return SteeringSweep(
        model,
        calibration,
        1,
        [3, 50],
        {"short": prompt(9), "long": prompt(12, 3)},
        random_controls=2,
        seed=0,
        design=DESIGN,
    )


def run(sweep, engine, jobs):
    return list(
        sweep.run(
            engine,
            jobs,
            decode=lambda ids: " ".join(map(str, ids)),
            max_tokens=6,
            top_k=3,
            stop_token_ids=[256],
            queue=len(jobs),
        )
    )


def key(record):
    return record["kind"], record["neuron"], record["prompt"], record["dose"]


@pytest.fixture(scope="module")
def mixed(qwen35_checkpoint):
    model = runner(qwen35_checkpoint)
    calibration = model.calibrate(prompt(33))
    sweep = sweep_for(model, calibration)
    with Engine(model) as engine:
        records = run(sweep, engine, sweep.jobs())
    return model, calibration, sweep, records


def test_steered_rollouts_in_a_mixed_batch_match_each_rollout_run_alone(
    qwen35_checkpoint, mixed
):
    model, calibration, sweep, records = mixed
    assert max(model.batch_sizes) >= 8
    alone_model = runner(qwen35_checkpoint)
    alone_sweep = sweep_for(alone_model, calibration)
    with Engine(alone_model) as engine:
        alone = {key(job): run(alone_sweep, engine, [job])[0] for job in sweep.jobs()}
    assert max(alone_model.batch_sizes) == 1
    assert sorted(map(key, records)) == sorted(alone)
    for record in records:
        expected = alone[key(record)]
        assert record["token_ids"] == expected["token_ids"], key(record)
        assert record["top_ids"] == expected["top_ids"], key(record)
        np.testing.assert_allclose(record["logprob"], expected["logprob"], **TOLERANCE)
    baseline = {r["prompt"]: r["token_ids"] for r in records if r["kind"] == "baseline"}
    for kind in ("neuron", "random"):
        assert any(
            r["token_ids"] != baseline[r["prompt"]]
            for r in records
            if r["kind"] == kind and r["dose"] == 3.0
        ), kind


@torch.inference_mode()
def reference_greedy(model, tokens, layer, vector, count=6):
    def push(module, inputs, output):
        return output + torch.from_numpy(vector)

    hook = model.model.layers[layer].mlp.register_forward_hook(push)
    try:
        inputs, cache, output, scores = torch.tensor([tokens]), None, [], []
        for _ in range(count):
            result = model(inputs, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            row = torch.log_softmax(result.logits[0, -1], -1)
            token = int(row.argmax())
            output.append(token)
            scores.append(float(row[token]))
            if token == 256:
                break
            inputs = torch.tensor([[token]])
        return output, scores
    finally:
        hook.remove()


def test_random_control_adds_its_direction_to_the_mlp_output(qwen35_checkpoint, mixed):
    _, calibration, sweep, records = mixed
    median = calibration.residual_medians[1]
    for record in records:
        if record["kind"] != "random":
            continue
        direction = sweep.control_directions[record["neuron"]]
        vector = record["dose"] * median * direction / np.linalg.norm(direction)
        np.testing.assert_allclose(sweep.residual_push(record), vector, rtol=1e-5)
        expected, scores = reference_greedy(
            qwen35_checkpoint[1],
            sweep.prompt_ids[record["prompt"]],
            1,
            vector.astype(np.float32),
        )
        assert record["token_ids"] == expected
        np.testing.assert_allclose(record["logprob"], scores, atol=3e-4)
        assert record["rel_push"] == pytest.approx(abs(record["dose"]))
        assert record["ref_neuron"] == int(sweep.control_references[record["neuron"]])


def test_selected_random_subsets_and_controls_follow_the_seed():
    assert select_neurons("random:3", 96, 0) == sorted(
        np.random.default_rng(1).choice(96, 3, replace=False).tolist()
    )
    assert select_neurons("1,4-6", 96, 0) == [1, 4, 5, 6]
    with pytest.raises(ValueError):
        select_neurons("96", 96, 0)


@torch.inference_mode()
def reference_layer_activity(model, windows, layers):
    captured = {layer: {} for layer in layers}
    residuals = []
    hooks = []
    for index, layer in enumerate(model.model.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda module, inputs, output: residuals.append(output.numpy().copy())
            )
        )
        if index in layers:
            mlp, store = layer.mlp, captured[index]
            hooks += [
                mlp.gate_proj.register_forward_hook(
                    lambda m, i, o, store=store: store.__setitem__("gate", o.numpy())
                ),
                mlp.up_proj.register_forward_hook(
                    lambda m, i, o, store=store: store.__setitem__("up", o.numpy())
                ),
                mlp.down_proj.register_forward_pre_hook(
                    lambda m, i, store=store: store.__setitem__("act", i[0].numpy())
                ),
            ]
    try:
        model(torch.tensor(windows))
    finally:
        for hook in hooks:
            hook.remove()
    return captured, residuals


def test_activation_extremes_match_a_direct_per_token_reference(qwen35_checkpoint):
    model = runner(qwen35_checkpoint)
    windows = [
        np.array([prompt(33, 2 * row + step) for row in range(2)]) for step in (0, 1)
    ]
    survey = ActivationSurvey(model, [1, 3])
    for window in windows:
        survey.update(window)
    expected = {layer: {} for layer in (1, 3)}
    norms = [[] for _ in range(4)]
    for window in windows:
        captured, residuals = reference_layer_activity(
            qwen35_checkpoint[1], window, (1, 3)
        )
        for index, residual in enumerate(residuals):
            norms[index].append(np.linalg.norm(residual[:, 1:], axis=-1).ravel())
        for layer, values in captured.items():
            gate, up, activation = (
                values[name][:, 1:].reshape(-1, 96) for name in ("gate", "up", "act")
            )
            extremes = expected[layer]
            for name in ("pp_max", "nn_max", "pn_min", "np_min"):
                extremes.setdefault(name, np.zeros(96))
            for token in range(len(activation)):
                for neuron in range(96):
                    value = float(activation[token, neuron])
                    regime = (gate[token, neuron] > 0, up[token, neuron] > 0)
                    if regime == (True, True):
                        extremes["pp_max"][neuron] = max(
                            extremes["pp_max"][neuron], value
                        )
                    elif regime == (False, False):
                        extremes["nn_max"][neuron] = max(
                            extremes["nn_max"][neuron], value
                        )
                    elif regime == (True, False):
                        extremes["pn_min"][neuron] = min(
                            extremes["pn_min"][neuron], value
                        )
                    else:
                        extremes["np_min"][neuron] = min(
                            extremes["np_min"][neuron], value
                        )
    for layer in (1, 3):
        actual = survey.layer_statistics(layer, "tiny")
        for name, values in expected[layer].items():
            np.testing.assert_allclose(
                actual[name], values, atol=1e-5, rtol=1e-4, err_msg=name
            )
        np.testing.assert_allclose(
            actual["abs_max"],
            np.max(np.abs(np.stack(list(expected[layer].values()))), axis=0),
            atol=1e-5,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            actual["resid_norm_q10_50_90"],
            np.quantile(np.concatenate(norms[layer]), [0.1, 0.5, 0.9]),
            rtol=1e-5,
        )
        assert actual["tokens"] == 132
    np.testing.assert_allclose(
        survey.calibration().residual_medians,
        [np.median(np.concatenate(rows)) for rows in norms],
        rtol=1e-5,
    )


def test_steer_lens_matches_full_forward_pushes(qwen35_checkpoint, mixed):
    model, calibration, _, _ = mixed
    reference = qwen35_checkpoint[1]
    contexts = [prompt(7), prompt(11, 5)]
    neurons = [5, 40, 77]
    lens = steer_lens(
        model, contexts, 2, calibration, dose=0.8, k=4, neurons=neurons, batch=2
    )
    down = reference.model.layers[2].mlp.down_proj
    for row, neuron in enumerate(neurons):
        value = (
            0.8 * calibration.residual_medians[2] / down.weight[:, neuron].norm().item()
        )
        for sign, name in ((1, "up"), (-1, "down")):

            def push(module, inputs, neuron=neuron, value=sign * value):
                hidden = inputs[0].clone()
                hidden[..., neuron] += value
                return (hidden,)

            delta, divergence = 0, 0
            for tokens in contexts:
                with torch.inference_mode():
                    base = torch.log_softmax(
                        reference(torch.tensor([tokens])).logits[0, -1], -1
                    )
                    hook = down.register_forward_pre_hook(push)
                    try:
                        steered = torch.log_softmax(
                            reference(torch.tensor([tokens])).logits[0, -1], -1
                        )
                    finally:
                        hook.remove()
                delta = delta + (steered - base).numpy() / len(contexts)
                divergence += float((steered.exp() * (steered - base)).sum()) / len(
                    contexts
                )
            assert lens[f"{name}_tok"][row].tolist() == np.argsort(-delta)[:4].tolist()
            assert lens[f"{name}_kl"][row] == pytest.approx(
                divergence, rel=1e-3, abs=1e-6
            )


def test_direct_connections_match_explicit_pair_scores(qwen35_checkpoint):
    model = runner(qwen35_checkpoint)
    connections = DirectConnections(model, k=5)
    for target in range(1, 4):
        connections.add_target_layer(target)
    actual = connections.arrays()
    layers = qwen35_checkpoint[1].model.layers
    weights = [
        {
            name: getattr(layer.mlp, f"{name}_proj").weight.detach().numpy()
            for name in ("gate", "up", "down")
        }
        | {"norm": 1 + layer.post_attention_layernorm.weight.detach().numpy()}
        for layer in layers
    ]
    candidates = {
        (side, layer, n): []
        for side in ("down", "up")
        for layer in range(4)
        for n in range(96)
    }
    for target in range(1, 4):
        for source in range(target):
            vectors = weights[source]["down"] * weights[target]["norm"][:, None]
            product = (weights[target]["gate"] @ vectors) * (
                weights[target]["up"] @ vectors
            )
            magnitude = np.abs(product)
            for a in range(96):
                column = magnitude[:, a]
                z = (column - column.mean()) / (column.std(ddof=1) + 1e-8)
                for b in range(96):
                    candidates["down", source, a].append(
                        (z[b], target, b, np.sign(product[b, a]))
                    )
            for b in range(96):
                row = magnitude[b]
                z = (row - row.mean()) / (row.std(ddof=1) + 1e-8)
                for a in range(96):
                    candidates["up", target, b].append(
                        (z[a], source, a, np.sign(product[b, a]))
                    )
    for (side, layer, n), entries in candidates.items():
        best = sorted(entries, key=lambda entry: -entry[0])[:5]
        if not best:
            assert (actual[f"{side}_z"][layer, n] == -60000).all()
            assert (actual[f"{side}_sign"][layer, n] == 0).all()
            continue
        np.testing.assert_allclose(
            actual[f"{side}_z"][layer, n],
            [entry[0] for entry in best],
            rtol=2e-3,
            atol=2e-3,
        )
        assert actual[f"{side}_layer"][layer, n].tolist() == [
            entry[1] for entry in best
        ]
        assert actual[f"{side}_n"][layer, n].tolist() == [entry[2] for entry in best]
        assert actual[f"{side}_sign"][layer, n].tolist() == [entry[3] for entry in best]


# A reader with the explorer's loading semantics (explorer/server.py, explorer/runtime.py,
# sweep/rank.py in lyramakesmusic/neuron-explorer), written from its documented behavior.
ROW_FIELDS = {
    "layer": int,
    "kind": str,
    "neuron": int,
    "prompt": str,
    "dose": float,
    "add": float,
    "rel_push": float,
    "text": str,
    "finish": str,
    "n_gen": int,
    "token_ids": list,
    "logprob": list,
    "top_ids": list,
    "top_lps": list,
    "capture_id": str,
}


def explorer_layer_runs(runs):
    """layer -> sweep jsonl: main_rel first, then the largest; calibration runs excluded."""
    best = {}
    for meta_path in glob.glob(os.path.join(runs, "*.meta.json")):
        with open(meta_path) as file:
            meta = json.load(file)
        if meta.get("design_name", "main") == "calib":
            continue
        path = meta_path.replace(".meta.json", ".jsonl")
        if not os.path.exists(path):
            continue
        rank = (meta.get("design_name") == "main_rel", os.path.getsize(path))
        if rank > best.get(meta["layer"], ("", (False, 0)))[1]:
            best[meta["layer"]] = (path, rank)
    return {layer: path for layer, (path, _) in best.items()}


def explorer_rows(path):
    with open(path.replace(".jsonl", ".meta.json")) as file:
        meta = json.load(file)
    with open(path) as file:
        rows = [json.loads(line) for line in file]
    for row in rows:
        for name, kind in ROW_FIELDS.items():
            assert isinstance(row[name], kind) or (
                kind is float and isinstance(row[name], int)
            ), name
        assert row["ref_neuron"] is None or isinstance(row["ref_neuron"], int)
        assert row["finish"] in ("stop", "length")
        assert row["prompt"] in meta["prompts"]
        assert (
            row["n_gen"]
            == len(row["token_ids"])
            == len(row["logprob"])
            == len(row["top_ids"])
        )
    return meta, rows


def explorer_connections(path, layer, n, k):
    archive = np.load(path)
    index = {name: archive[name] for name in archive.files}
    assert k <= int(archive["k"])
    return {
        side: [
            {
                "layer": int(index[f"{side}_layer"][layer, n, i]),
                "n": int(index[f"{side}_n"][layer, n, i]),
                "z": round(float(index[f"{side}_z"][layer, n][i]), 2),
                "sign": int(index[f"{side}_sign"][layer, n, i]),
            }
            for i in range(k)
            if index[f"{side}_z"][layer, n][i] > -1e4
        ]
        for side in ("down", "up")
    }


def test_written_files_round_trip_through_the_explorer_reader(
    qwen35_checkpoint, mixed, tmp_path
):
    model, calibration, sweep, records = mixed
    runs = tmp_path / "runs"
    runs.mkdir()
    meta = sweep.metadata(
        {"short": "S", "long": "L"}, model="tiny", max_tokens=6, top_k=3
    )
    (runs / "tiny-L1.meta.json").write_text(json.dumps(meta, indent=1))
    with open(runs / "tiny-L1.jsonl", "w") as file:
        file.writelines(json.dumps(record) + "\n" for record in records)
    (runs / "tiny-L1-calib.meta.json").write_text(
        json.dumps(meta | {"design_name": "calib", "layer": 2})
    )
    assert explorer_layer_runs(runs) == {1: str(runs / "tiny-L1.jsonl")}
    read_meta, rows = explorer_rows(str(runs / "tiny-L1.jsonl"))
    assert rows == records
    assert read_meta["prompt_ids"] == {"short": prompt(9), "long": prompt(12, 3)}
    assert {(r["kind"], r["prompt"]) for r in rows if r["kind"] == "baseline"} == {
        ("baseline", "short"),
        ("baseline", "long"),
    }
    for row in rows:
        if row["kind"] != "baseline":
            assert row["rel_push"] == pytest.approx(abs(row["dose"]))

    survey = ActivationSurvey(model, [1])
    survey.update(np.array([prompt(17)]))
    statistics = survey.layer_statistics(1, "tiny")
    np.savez(tmp_path / "q35-4b_L1.npz", **statistics)
    stored = np.load(tmp_path / "q35-4b_L1.npz")
    assert int(stored["layer"]) == 1 and int(stored["tokens"]) == 17
    assert stored["w_down"].dtype == np.float16 and stored["w_down"].shape == (96, 64)
    np.testing.assert_allclose(
        stored["w_down"][7],
        qwen35_checkpoint[1]
        .model.layers[1]
        .mlp.down_proj.weight[:, 7]
        .detach()
        .numpy(),
        rtol=1e-3,
        atol=1e-4,
    )
    assert (
        float(stored["resid_norm_q10_50_90"][1])
        == statistics["resid_norm_q10_50_90"][1]
    )
    for name in ("pp_max", "nn_max", "pn_min", "np_min", "abs_max"):
        assert stored[name].shape == (96,)
        np.testing.assert_array_equal(stored[name], statistics[name])

    connections = DirectConnections(model, k=5)
    for target in range(1, 4):
        connections.add_target_layer(target)
    arrays = connections.arrays()
    np.savez_compressed(tmp_path / "connections.npz", **arrays)
    for name, dtype in (
        ("z", np.float16),
        ("layer", np.int8),
        ("n", np.int16),
        ("sign", np.int8),
    ):
        assert arrays[f"down_{name}"].dtype == dtype and arrays[
            f"down_{name}"
        ].shape == (4, 96, 5)
    links = explorer_connections(tmp_path / "connections.npz", 1, 7, 5)
    assert [link["n"] for link in links["down"]] == arrays["down_n"][1, 7].tolist()
    assert all(link["layer"] > 1 for link in links["down"])
    assert all(link["layer"] < 1 for link in links["up"]) and len(links["up"]) == 5
    assert explorer_connections(tmp_path / "connections.npz", 0, 7, 5)["up"] == []

    index_path = tmp_path / "steer_index.npz"
    full = {
        layer: steer_lens(model, [prompt(7)], layer, calibration, k=4, batch=48)
        for layer in range(4)
    }
    update_steer_index(index_path, full, neurons=None, shape=(4, 96), dose=0.8, k=4)
    partial = steer_lens(model, [prompt(9)], 2, calibration, k=4, neurons=[3, 8])
    update_steer_index(
        index_path, {2: partial}, neurons=[3, 8], shape=(4, 96), dose=0.8, k=4
    )
    index = dict(np.load(index_path))
    assert index["up"].shape == (4, 96, 4) and index["up_kl"].shape == (4, 96)
    np.testing.assert_array_equal(index["up"][2, [3, 8]], partial["up_tok"])
    np.testing.assert_array_equal(index["down_kl"][2, [3, 8]], partial["down_kl"])
    np.testing.assert_array_equal(index["up"][2, 4], full[2]["up_tok"][4])
    np.testing.assert_array_equal(index["down"][1], full[1]["down_tok"])
    with pytest.raises(ValueError):
        update_steer_index(
            tmp_path / "new.npz",
            {2: partial},
            neurons=[3, 8],
            shape=(4, 96),
            dose=0.8,
            k=4,
        )
    with pytest.raises(ValueError):
        update_steer_index(
            index_path, {2: partial}, neurons=[3, 8], shape=(4, 96), dose=0.5, k=4
        )


def test_captured_mlp_inputs_include_the_rollout_push(qwen35_checkpoint, mixed):
    model, _, sweep, records = mixed
    chosen = [r for r in records if r["prompt"] == "long"][:4]
    pushes = np.stack([sweep.residual_push(r) for r in chosen])
    captured = capture_mlp_inputs(model, 1, pushes, sweep.prompt_ids["long"], [1, 2])
    reference = qwen35_checkpoint[1]
    for row, push in enumerate(pushes):
        inputs = {}
        hooks = [
            reference.model.layers[1].mlp.register_forward_hook(
                lambda module, args, output, push=push: output + torch.from_numpy(push)
            )
        ] + [
            reference.model.layers[layer].mlp.register_forward_pre_hook(
                lambda module, args, layer=layer: inputs.__setitem__(
                    layer, args[0][0, -1].numpy().copy()
                )
            )
            for layer in (1, 2)
        ]
        try:
            with torch.inference_mode():
                reference(torch.tensor([sweep.prompt_ids["long"]]))
        finally:
            for hook in hooks:
                hook.remove()
        np.testing.assert_allclose(captured[row, 0], inputs[1], **TOLERANCE)
        np.testing.assert_allclose(captured[row, 1], inputs[2], **TOLERANCE)
