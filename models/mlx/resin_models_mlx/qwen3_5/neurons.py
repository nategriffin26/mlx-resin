"""MLP-neuron sweeps on dense Qwen3.5: activation ranges, steered rollouts, steer lens and connections.

The method and file formats follow lyra's neuron explorer
(https://github.com/lyramakesmusic/neuron-explorer), reimplemented on this runner so the
explorer can read sweeps computed on Apple Silicon. ``docs/neuron-sweeps.md`` describes
each computation and file. A neuron's activation is ``silu(gate) * up``, the input to
``down_proj``; layer and neuron indices are zero-based.
"""

from __future__ import annotations

import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from resin import SamplingParams
from resin_models_mlx.qwen3_5.steering import SteeringCalibration

SWEEP_PROMPTS = {
    "who": "Who are you?",
    "snake": "Write a snake game in Python.",
    "orange": "Describe the color orange.",
    "free": "Write whatever you like.",
}
# Doses are fractions of the median residual norm leaving the steered layer: the
# coherent window is about 0.5-0.8, and +3 reads the direction out as wreckage tokens.
MAIN_REL_DESIGN = tuple(
    (prompt, dose) for prompt in ("who", "snake", "orange") for dose in (-0.7, 0.5, 0.8)
) + (("free", 3.0),)
LENS_CHAT_PROMPTS = ("Who are you?", "Write whatever you like.")
LENS_TEXT_PROMPTS = (
    "I think the most important thing to remember is",
    "Today we are going to talk about",
)


def select_neurons(spec, count, seed):
    """Parse ``all``, ``random:K``, ``N``, ``A-B`` or comma lists into sorted indices.

    ``random:K`` draws from ``default_rng(seed + 1)``, the same subset the source
    sweep selects for the same seed.
    """
    if spec == "all":
        return list(range(count))
    if spec.startswith("random:"):
        chosen = np.random.default_rng(seed + 1).choice(
            count, int(spec.split(":")[1]), replace=False
        )
        return sorted(chosen.tolist())
    neurons = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        neurons.extend(range(int(first), int(last) + 1) if last else [int(first)])
    if any(not 0 <= neuron < count for neuron in neurons):
        raise ValueError(f"neurons must lie in [0, {count})")
    return neurons


def _column_norms(layer):
    weight = layer.mlp.down_proj.weight.astype(mx.float32)
    return np.array(mx.linalg.norm(weight, axis=0))


class ActivationSurvey:
    """Streams token windows through the full model and keeps per-neuron extremes.

    Position 0 of every window is an attention sink with outsized activations; it is
    excluded from extremes and residual norms alike. Residual norms are kept for every
    layer so the same corpus also yields a complete steering calibration.
    """

    def __init__(self, runner, layers):
        self.runner = runner
        self.model = runner._model
        neurons = runner.config.intermediate_size
        self.extremes = {
            layer: {
                name: mx.zeros(neurons, mx.float32)
                for name in ("pp_max", "nn_max", "pn_min", "np_min")
            }
            for layer in layers
        }
        self.residual_norms = [[] for _ in self.model.model.layers]
        self.token_count = 0

    def update(self, window):
        """Accumulate one ``[batch, length]`` window of token ids."""
        window = np.asarray(window)
        if window.ndim != 2 or window.shape[1] < 2:
            raise ValueError("windows must be [batch, length] with length >= 2")
        hidden = self.model.model.embed_tokens(mx.array(window))
        positions = mx.arange(window.shape[1])[None]
        norms = []
        for index, layer in enumerate(self.model.model.layers):
            inputs = [] if index in self.extremes else None
            hidden = layer(hidden, positions, None, None, mlp_inputs=inputs)
            norms.append(
                mx.linalg.norm(hidden[:, 1:].astype(mx.float32), axis=-1).reshape(-1)
            )
            if inputs:
                self._update_extremes(index, layer.mlp, inputs[0][:, 1:])
            # Evaluating per layer bounds peak memory to one layer's activations.
            mx.eval(norms[-1], *self.extremes.get(index, {}).values())
        for index, row in enumerate(norms):
            self.residual_norms[index].append(np.array(row))
        self.token_count += window.size

    def _update_extremes(self, index, mlp, hidden):
        gate, up = mlp.gate_proj(hidden), mlp.up_proj(hidden)
        activation = (nn.silu(gate) * up).reshape(-1, gate.shape[-1])
        activation = activation.astype(mx.float32)
        positive_gate = gate.reshape(activation.shape) > 0
        positive_up = up.reshape(activation.shape) > 0
        extremes = self.extremes[index]
        for name, mask, maximum in (
            ("pp_max", positive_gate & positive_up, True),
            ("nn_max", ~positive_gate & ~positive_up, True),
            ("pn_min", positive_gate & ~positive_up, False),
            ("np_min", ~positive_gate & positive_up, False),
        ):
            if maximum:
                extremes[name] = mx.maximum(
                    extremes[name], mx.where(mask, activation, -math.inf).max(0)
                )
            else:
                extremes[name] = mx.minimum(
                    extremes[name], mx.where(mask, activation, math.inf).min(0)
                )

    def layer_statistics(self, layer, model_name):
        """Arrays of the source ``stats/<name>_L{layer}.npz`` file."""
        if not self.token_count:
            raise ValueError("no tokens have been surveyed")
        extremes = {
            name: np.array(value) for name, value in self.extremes[layer].items()
        }
        down = self.model.model.layers[layer].mlp.down_proj.weight
        return {
            **extremes,
            "abs_max": np.maximum.reduce(
                [
                    extremes["pp_max"],
                    extremes["nn_max"],
                    -extremes["pn_min"],
                    -extremes["np_min"],
                ]
            ),
            "w_down": np.array(down.astype(mx.float32)).T.astype(np.float16),
            "resid_norm_q10_50_90": np.quantile(
                np.concatenate(self.residual_norms[layer]), [0.1, 0.5, 0.9]
            ),
            "model": model_name,
            "layer": layer,
            "tokens": self.token_count,
        }

    def calibration(self):
        """Steering calibration from the same corpus: median residual norm per layer."""
        return SteeringCalibration(
            self.runner.model_identity,
            self.token_count,
            tuple(
                float(np.median(np.concatenate(rows))) for rows in self.residual_norms
            ),
        )


class SteeringSweep:
    """``main_rel`` rollouts for one layer, run as Engine requests with per-request steering.

    Every unit (a neuron or a random-direction control) gets each ``(prompt, dose)`` of
    the design; each prompt also gets one unsteered baseline. A dose ``d`` moves the
    residual stream by ``d`` times the layer's median residual norm along the unit's
    direction: for a neuron, ``d * median / |W_down[:, n]|`` added to its activation.
    """

    def __init__(
        self,
        runner,
        calibration,
        layer,
        neurons,
        prompt_ids,
        *,
        random_controls,
        seed,
        design=MAIN_REL_DESIGN,
    ):
        if not 0 <= layer < runner.config.num_hidden_layers:
            raise ValueError("sweep layer is outside the model")
        self.runner = runner
        self.calibration = calibration
        self.layer = layer
        self.prompt_ids = {name: tuple(ids) for name, ids in prompt_ids.items()}
        self.design = tuple(design)
        self.median = calibration.residual_medians[layer]
        self.seed = seed
        self.down_norms = _column_norms(runner._model.model.layers[layer])
        rng = np.random.default_rng(seed)
        self.control_references = rng.integers(
            len(self.down_norms), size=random_controls
        )
        self.control_directions = rng.standard_normal(
            (random_controls, runner.config.hidden_size)
        ).astype(np.float32)
        self.control_directions *= (
            self.down_norms[self.control_references]
            / np.linalg.norm(self.control_directions, axis=1)
        )[:, None]
        units = [("neuron", neuron) for neuron in neurons]
        # Spread controls evenly so any prefix of the run carries its share of the null.
        for control in range(random_controls):
            units.insert(
                round((control + 1) * len(units) / (random_controls + 1)),
                ("random", control),
            )
        self.units = units

    def jobs(self):
        baselines = [
            {"kind": "baseline", "neuron": -1, "prompt": prompt, "dose": 0.0}
            for prompt in self.prompt_ids
        ]
        return baselines + [
            {"kind": kind, "neuron": unit, "prompt": prompt, "dose": dose}
            for kind, unit in self.units
            for prompt, dose in self.design
        ]

    def metadata(self, prompts, **provenance):
        """Fields of the source ``runs/<run>.meta.json``.

        ``prompts`` maps prompt names to their text; ``provenance`` adds the model,
        statistics, calibration and generation limits.
        """
        return {
            **provenance,
            "engine": "mlx-resin",
            "layer": self.layer,
            "prompts": prompts,
            "design_name": "main_rel",
            "design": [list(entry) for entry in self.design],
            "resid_norm_p50": self.median,
            "dose_unit": "rel",
            # Neuron pushes add to the activation, which equals adding the push times
            # W_down[:, n] to the MLP output; controls add there directly.
            "hook": "mlp_out",
            "capture_hook": "mlp_in",
            "capture_positions": "last_prompt",
            "random_controls": len(self.control_references),
            "seed": self.seed,
            "prompt_ids": {name: list(ids) for name, ids in self.prompt_ids.items()},
        }

    def request(self, job):
        """Return Engine inputs ``(token_ids, media)`` and the job's push fields."""
        kind, unit, prompt, dose = (
            job["kind"],
            job["neuron"],
            job["prompt"],
            job["dose"],
        )
        tokens = self.prompt_ids[prompt]
        if kind == "baseline":
            reference, media = None, None
        elif kind == "neuron":
            reference = unit
            media = self.runner.steer(
                tokens, [(self.layer, unit, dose)], self.calibration
            )
        elif kind == "random":
            reference = int(self.control_references[unit])
            media = self.runner.steer(
                tokens,
                [],
                self.calibration,
                directions=[(self.layer, self.control_directions[unit], dose)],
            )
        else:
            raise ValueError(f"unknown sweep unit kind {kind!r}")
        if reference is None:
            addition, relative_push = 0.0, 0.0
        else:
            norm = float(self.down_norms[reference])
            addition = dose * self.median / norm
            relative_push = abs(addition) * norm / self.median
        capture_id = f"L{self.layer}_{kind}{unit}_{prompt}_{dose:+g}"
        fields = {
            "ref_neuron": reference,
            "add": addition,
            "rel_push": relative_push,
            "capture_id": capture_id.replace("+", "p").replace("-", "m"),
        }
        return (tokens if media is None else None), media, fields

    def residual_push(self, record):
        """The residual-stream vector a recorded rollout adds at the steered layer."""
        if record["kind"] == "neuron":
            column = self.runner._model.model.layers[self.layer].mlp.down_proj.weight[
                :, record["neuron"]
            ]
            return np.array(column.astype(mx.float32)) * record["add"]
        if record["kind"] == "random":
            return self.control_directions[record["neuron"]] * record["add"]
        return np.zeros(self.runner.config.hidden_size, np.float32)

    def run(self, engine, jobs, *, decode, max_tokens, top_k, stop_token_ids, queue):
        """Yield one source-format rollout record per job as each request finishes.

        At most ``queue`` requests are submitted ahead; the Engine batches them.
        """
        pending = iter(jobs)
        active = {}
        sampling = SamplingParams(
            temperature=0,
            max_output_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            top_logprobs=top_k,
        )

        def submit():
            job = next(pending, None)
            if job is None:
                return False
            tokens, media, fields = self.request(job)
            lease = engine.submit(tokens, media=media, sampling_params=sampling)
            active[lease.request_id] = (job | fields, lease, [], [], [], [])
            return True

        while len(active) < queue and submit():
            pass
        while active:
            for event in engine.step():
                job, _, token_ids, chosen, top_ids, top_scores = active[
                    event.request_id
                ]
                token_ids.append(event.token_id)
                scores = event.logprobs
                chosen.append(round(scores.logprob, 4))
                ranked = sorted(scores.top_logprobs, key=lambda pair: -pair[1])[:top_k]
                top_ids.append([token for token, _ in ranked])
                top_scores.append([round(score, 4) for _, score in ranked])
                if event.finished:
                    del active[event.request_id]
                    yield {
                        "layer": self.layer,
                        "kind": job["kind"],
                        "neuron": job["neuron"],
                        "ref_neuron": job["ref_neuron"],
                        "prompt": job["prompt"],
                        "dose": job["dose"],
                        "add": job["add"],
                        "rel_push": job["rel_push"],
                        "text": decode(token_ids),
                        "finish": event.finish_reason,
                        "n_gen": len(token_ids),
                        "token_ids": token_ids,
                        "logprob": chosen,
                        "top_ids": top_ids,
                        "top_lps": top_scores,
                        "capture_id": job["capture_id"],
                    }
                    submit()


def capture_mlp_inputs(runner, layer, pushes, prompt_ids, capture_layers):
    """MLP inputs at the last prompt token, with each row's residual push at ``layer``.

    ``pushes`` is ``[rows, hidden]``; the push applies at every prompt position, so a
    single prefill reproduces the state just before the model answers. Returns
    float32 ``[rows, len(capture_layers), hidden]``.
    """
    model = runner._model
    if not capture_layers or min(capture_layers) < 0:
        raise ValueError("capture layers must be nonempty and inside the model")
    rows = len(pushes)
    hidden = mx.broadcast_to(
        model.model.embed_tokens(mx.array(prompt_ids)[None]),
        (rows, len(prompt_ids), runner.config.hidden_size),
    )
    positions = mx.arange(len(prompt_ids))[None]
    captured = {}
    for index, decoder in enumerate(model.model.layers[: max(capture_layers) + 1]):
        inputs = [] if index in capture_layers else None
        addition = (None, mx.array(pushes, mx.float32)) if index == layer else None
        hidden = decoder(hidden, positions, None, addition, mlp_inputs=inputs)
        if inputs:
            captured[index] = inputs[0][:, -1].astype(mx.float32)
    values = mx.stack([captured[index] for index in capture_layers], axis=1)
    return np.array(values)


def _log_probabilities(logits):
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _top_indices(values, k):
    candidates = mx.argpartition(-values, kth=k - 1, axis=-1)[:, :k]
    order = mx.argsort(-mx.take_along_axis(values, candidates, axis=-1), axis=-1)
    return mx.take_along_axis(candidates, order, axis=-1)


def steer_lens(
    runner, contexts, layer, calibration, *, dose=0.8, k=12, neurons=None, batch=64
):
    """Next-token effect of pushing each neuron by ``±dose`` residual-norm units.

    The push applies at every position of each context; layers before ``layer`` are
    unaffected, so their output is computed once per context. Returns the ``k`` tokens
    whose log-probability rises most (averaged over contexts) and the mean
    KL(steered || unsteered), per neuron and sign.
    """
    model = runner._model
    decoders = model.model.layers
    count = runner.config.intermediate_size
    neurons = np.arange(count) if neurons is None else np.asarray(neurons)
    units = calibration.residual_medians[layer] / _column_norms(decoders[layer])
    prefixes = []
    for tokens in contexts:
        hidden = model.model.embed_tokens(mx.array(tokens)[None])
        positions = mx.arange(len(tokens))[None]
        for decoder in decoders[:layer]:
            hidden = decoder(hidden, positions, None, None)
        base = _log_probabilities(
            model.logits(model(mx.array(tokens)[None], positions, None)[:, -1])
        )
        mx.eval(hidden, base)
        prefixes.append((hidden, positions, base))
    results = {name: [] for name in ("up_tok", "down_tok", "up_kl", "down_kl")}
    for start in range(0, len(neurons), batch):
        chunk = neurons[start : start + batch]
        rows = np.arange(len(chunk))
        values = np.zeros((2 * len(chunk), count), np.float32)
        values[rows, chunk] = dose * units[chunk]
        values[rows + len(chunk), chunk] = -dose * units[chunk]
        addition = (mx.array(values), None)
        delta, divergence = 0, 0
        for hidden, positions, base in prefixes:
            hidden = mx.broadcast_to(hidden, (len(values), *hidden.shape[1:]))
            for index in range(layer, len(decoders)):
                hidden = decoders[index](
                    hidden, positions, None, addition if index == layer else None
                )
            steered = _log_probabilities(model.logits(model.model.norm(hidden[:, -1])))
            delta = delta + (steered - base)
            divergence = divergence + (mx.exp(steered) * (steered - base)).sum(-1)
        delta, divergence = delta / len(prefixes), divergence / len(prefixes)
        top = _top_indices(delta, k).astype(mx.int32)
        mx.eval(top, divergence)
        top, divergence = np.array(top), np.array(divergence)
        results["up_tok"].append(top[: len(chunk)])
        results["down_tok"].append(top[len(chunk) :])
        results["up_kl"].append(divergence[: len(chunk)])
        results["down_kl"].append(divergence[len(chunk) :])
    return {name: np.concatenate(parts) for name, parts in results.items()}


def update_steer_index(path, lenses, *, neurons, shape, dose, k):
    """Write ``steer_lens`` results into the explorer's ``steer_index.npz``.

    ``lenses`` maps layers to ``steer_lens`` outputs over ``neurons`` (``None`` for
    every neuron); ``shape`` is ``(layers, intermediate size)``. An existing index
    keeps the rows not recomputed here; a new one must cover every layer and neuron.
    """
    path = os.fspath(path)
    if os.path.exists(path):
        index = dict(np.load(path))
        if float(index["dose"]) != dose or index["up"].shape != (*shape, k):
            raise ValueError(f"{path} holds a different dose, shape or k")
    elif neurons is not None or sorted(lenses) != list(range(shape[0])):
        raise ValueError(f"creating {path} needs every layer and neuron")
    else:
        index = {
            "up": np.zeros((*shape, k), np.int32),
            "down": np.zeros((*shape, k), np.int32),
            "up_kl": np.zeros(shape, np.float32),
            "down_kl": np.zeros(shape, np.float32),
            "dose": np.float64(dose),
        }
    rows = slice(None) if neurons is None else np.asarray(neurons)
    for layer, lens in lenses.items():
        index["up"][layer, rows] = lens["up_tok"]
        index["down"][layer, rows] = lens["down_tok"]
        index["up_kl"][layer, rows] = lens["up_kl"]
        index["down_kl"][layer, rows] = lens["down_kl"]
    # Replace atomically: the explorer may be reading the old index.
    temporary = path + ".tmp.npz"
    np.savez(temporary, **index)
    os.replace(temporary, path)


class DirectConnections:
    """Each neuron's strongest direct-path links to neurons in other layers.

    For layers ``i < j``, ``P[B, A] = (W_gate_j[B] . v) * (W_up_j[B] . v)`` with
    ``v = (1 + w_j) * W_down_i[:, A]`` and ``w_j`` the pre-MLP norm weight of layer
    ``j``. Neuron A's downstream links are column A of ``|P|`` z-scored over B; neuron
    B's upstream links are row B z-scored over A. Each neuron keeps its ``k`` strongest
    links per side across all layers, with ``sign(P)``. Call ``add_target_layer`` for
    every layer from 1 upward, then ``arrays``.
    """

    def __init__(self, runner, k=24):
        self.model = runner._model
        self.k = k
        layers, neurons = len(self.model.model.layers), runner.config.intermediate_size
        self.best = {
            side: (
                mx.full((layers, neurons, k), -1e9, mx.float32),
                mx.zeros((layers, neurons, k), mx.int16),
                mx.zeros((layers, neurons, k), mx.int16),
                mx.zeros((layers, neurons, k), mx.int8),
            )
            for side in ("down", "up")
        }

    def _merge(self, side, layer, scores, sources, source_layer, signs):
        best_scores, best_layers, best_neurons, best_signs = self.best[side]
        merged = (
            mx.concatenate((best_scores[layer], scores), axis=1),
            mx.concatenate(
                (best_layers[layer], mx.full(sources.shape, source_layer, mx.int16)),
                axis=1,
            ),
            mx.concatenate((best_neurons[layer], sources.astype(mx.int16)), axis=1),
            mx.concatenate((best_signs[layer], signs), axis=1),
        )
        keep = mx.argpartition(-merged[0], kth=self.k - 1, axis=1)[:, : self.k]
        for target, values in zip(self.best[side], merged, strict=True):
            target[layer] = mx.take_along_axis(values, keep, axis=1)

    def add_target_layer(self, target):
        """Score every pair ``(source < target, target)``."""
        decoders = self.model.model.layers
        mlp = decoders[target].mlp
        scale = 1.0 + decoders[target].post_attention_layernorm.weight.astype(
            mx.float32
        )
        scale = scale.astype(mlp.gate_proj.weight.dtype)
        k = self.k
        for source in range(target):
            vectors = decoders[source].mlp.down_proj.weight * scale[:, None]
            product = (mlp.gate_proj.weight @ vectors).astype(mx.float32) * (
                mlp.up_proj.weight @ vectors
            ).astype(mx.float32)
            magnitude = mx.abs(product)
            # Bessel-corrected deviation, as torch.std computes it in the source method.
            columns = (magnitude - magnitude.mean(0, keepdims=True)) / (
                mx.std(magnitude, axis=0, keepdims=True, ddof=1) + 1e-8
            )
            top = mx.argpartition(-columns, kth=k - 1, axis=0)[:k]
            signs = mx.where(mx.take_along_axis(product, top, axis=0) > 0, 1, -1)
            self._merge(
                "down",
                source,
                mx.take_along_axis(columns, top, axis=0).T,
                top.T,
                target,
                signs.astype(mx.int8).T,
            )
            rows = (magnitude - magnitude.mean(1, keepdims=True)) / (
                mx.std(magnitude, axis=1, keepdims=True, ddof=1) + 1e-8
            )
            top = mx.argpartition(-rows, kth=k - 1, axis=1)[:, :k]
            signs = mx.where(mx.take_along_axis(product, top, axis=1) > 0, 1, -1)
            self._merge(
                "up",
                target,
                mx.take_along_axis(rows, top, axis=1),
                top,
                source,
                signs.astype(mx.int8),
            )
            mx.eval(self.best)

    def arrays(self):
        """Arrays of the source ``connections.npz``, strongest link first."""
        out = {}
        for side, (scores, layers, neurons, signs) in self.best.items():
            order = mx.argsort(-scores, axis=-1)
            out[f"{side}_z"] = np.array(
                mx.maximum(mx.take_along_axis(scores, order, axis=-1), -60000)
            ).astype(np.float16)
            out[f"{side}_layer"] = np.array(
                mx.take_along_axis(layers, order, axis=-1)
            ).astype(np.int8)
            out[f"{side}_n"] = np.array(mx.take_along_axis(neurons, order, axis=-1))
            out[f"{side}_sign"] = np.array(mx.take_along_axis(signs, order, axis=-1))
        out["k"] = self.k
        return out
