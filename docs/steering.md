# Qwen3.5 neuron steering

`Qwen35MlxRunner.steer(token_ids, [(layer, neuron, dose), ...], calibration)` creates
an immutable `Qwen35Steering` payload implementing `MediaInput`. Submit it through
`Engine.submit(media=payload)`. `token_ids` is the original prompt, `token_spans`
is empty, and the runner reads each row's payload from `BatchPlan.media`. A steered
row and an unsteered row can therefore share a decode batch without sharing the
intervention.

## Dose

Layer and neuron indices are zero-based. At every token position, including
prompt positions, add this scalar to neuron `n` in `silu(gate_proj(x)) * up_proj(x)`
**before** `down_proj` in layer `L`:

```text
addition = dose × median(norm(residual leaving layer L), tokens 1:)
                / norm(down_proj.weight[:, n])
```

`runner.calibrate(tokens)` records each median and the input token count. It uses
unsteered layer outputs before final normalization, excludes the first token, and
keeps the residual norms in FP32. `SteeringCalibration.save/load` persists the
values. Calibration belongs to a specific checkpoint path and execution dtype;
the runner rejects a mismatch. Duplicate `(layer, neuron)` entries are summed,
zero totals omitted, and entries sorted before hashing. Positive and negative
doses use the same norm scale.

## Prefix identity

The first `min(8, prompt_length)` cache-key words hash the configuration,
calibration, calibrated additions, original token, and position. Their high bits
are set, placing them outside Qwen's vocabulary. Later words are raw token IDs.
The rooted prefix trie thus separates different interventions from each other
and from ordinary text, while identical configurations reuse the same pages.

Eight salted words provide 248 bits of digest separation. Shorter prompts carry
only 31 bits per salted token. Hash collisions remain theoretically possible;
the Engine's same-length uint32 identity channel cannot represent an unbounded,
collision-free configuration namespace.

The Engine publishes `cache_prompt + generated + sampled` in
`crates/resin-core/src/engine.rs::publish_batch_prefixes`. Generated tokens are
raw **below the salted root**, never published at an unsalted root. Keeping the
suffix raw also lets a same-steering continuation reuse generated-prefix pages
when the original prompt has at least eight tokens. A shorter original prompt
has fewer salted words, so extending it past that initial salt changes the
identity path and misses the old generated suffix. This affects reuse, not
isolation. No Engine or Rust changes are needed.

Steering is a model-runner feature, not vision media, an HTTP extension, or a
microscope UI. See the [Qwen runner](../models/mlx/resin_models_mlx/qwen3_5/README.md)
for the chat demo and calibration command.
