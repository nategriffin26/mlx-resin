# Qwen3.5 neuron sweeps

`scripts/qwen35_neuron_sweep.py` runs the neuron-sweep method of lyra's
[neuron explorer](https://github.com/lyramakesmusic/neuron-explorer) on Apple
Silicon and writes the files that explorer reads. The method and formats are
reimplemented from its README, research log and loaders; none of its code is copied,
because that repository carries no license. Computation lives in
`models/mlx/resin_models_mlx/qwen3_5/neurons.py`. Layer and neuron indices are
zero-based; a neuron's activation is `silu(gate) * up`, the input to `down_proj`.

| Command | Writes | Method |
| --- | --- | --- |
| `stats` | `<out>_L{l}.npz`, `<out>.calibration.json` | Streams FineWeb-Edu `sample-10BT` documents alternating with UltraChat conversations (seeded shuffles, EOS-separated, packed into `--batch` × `--seq-len` windows) through the full model for `--tokens` tokens. Per neuron: the extreme activation of each sign regime of `(gate, up)`, `abs_max`, the down-projection column, and residual-norm quantiles leaving the layer. Position 0 of each window is excluded. The same pass writes a steering calibration for every layer. |
| `sweep` | `runs/<run>.jsonl`, `runs/<run>.meta.json` | `main_rel`: every neuron (and `--random-controls` random directions) gets {who, snake, orange} × {−0.7, +0.5, +0.8} and {free} × {+3}; each prompt gets one unsteered baseline. Rollouts are greedy Engine requests with per-request steering, up to `--max-tokens`, with the chosen token's logprob and the top `--top-k` alternatives per step. Rerunning resumes. |
| `lens` | `<out>_L{l}.npz`, optionally `--index steer_index.npz` | Pushes each neuron by ±`--dose` on two chat and two plain contexts and keeps the 12 tokens whose next-token log-probability rises most and the KL from the unsteered prediction, averaged over contexts. |
| `connections` | `connections.npz` | Direct-path links: for layers `i < j`, `P[B, A] = (W_gate_j[B]·v)(W_up_j[B]·v)` with `v = (1 + w_j) ⊙ W_down_i[:, A]`; `|P|` z-scored per column (downstream of A) and per row (upstream of B); each neuron keeps its 24 strongest links per side across layers. |
| `capture` | `captures/<run>.mlp_in.npy` + `.json` | MLP input at the last prompt token at layers L, L+1, L+4, L+8, with each rollout's push, rebuilt by prefill; rows follow the jsonl. |

## Dose

A dose `d` moves the residual stream by `d × median(‖residual leaving L‖)` along
the unit's direction: for neuron `n`, `d × median / ‖W_down[:, n]‖` is added to its
activation at every position, prompt included ([steering contract](steering.md)).
A random control is a Gaussian direction rescaled to a randomly chosen neuron's
column norm and added to the MLP output with the same push size
(`runner.steer(..., directions=[(layer, vector, dose)])`). Each record's `add` is
the activation-space value and `rel_push` the push in residual medians. The
median comes from the `stats` corpus through its calibration file, so sweep,
lens and live steering share one unit. The source lens takes its median from the
four lens contexts instead.

Seeds reproduce the source's neuron subsets (`random:K`) and control directions.

## Explorer

Place `runs/` and `stats/` (named `q35-4b_L{l}.npz`) in the explorer checkout,
and `connections.npz` / `steer_index.npz` in its `explorer/cache/`. The explorer
derives `runs/<run>.units.jsonl` (word statistics, CPU only) from the run on first
load. Its live chat, reads-you and logit lens run its own PyTorch model; set
`EXPLORER_NO_MODEL=1` to browse cached files without loading it.
