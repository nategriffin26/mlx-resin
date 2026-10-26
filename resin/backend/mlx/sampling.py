"""MLX sampling and raw-distribution score columns for Resin's output ABI."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from resin import ForwardOutput
from resin.backend.sampling import batch_sampling_inputs


def _offsets(counts):
    return np.concatenate((np.zeros(1, np.uint32), np.cumsum(counts, dtype=np.uint32)))


@dataclass(frozen=True)
class ScoreColumns:
    counts: np.ndarray
    token_ids: mx.array
    logprobs: mx.array

    def output(self, prefix):
        return {
            f"{prefix}_indptr": _offsets(self.counts),
            f"{prefix}_token_ids": np.array(self.token_ids, dtype=np.uint32),
            f"{prefix}_logprobs": np.array(self.logprobs, dtype=np.float32),
        }


@dataclass(frozen=True)
class PromptScores:
    logprobs: mx.array
    top: ScoreColumns | None = None
    targets: ScoreColumns | None = None

    @classmethod
    def concat(cls, parts):
        if not parts:
            return None
        columns = []
        for name in ("top", "targets"):
            present = [
                getattr(part, name) for part in parts if getattr(part, name) is not None
            ]
            columns.append(
                ScoreColumns(
                    np.concatenate(
                        [
                            getattr(part, name).counts
                            if getattr(part, name) is not None
                            else np.zeros(len(part.logprobs), np.uint32)
                            for part in parts
                        ]
                    ),
                    mx.concatenate([part.token_ids for part in present]),
                    mx.concatenate([part.logprobs for part in present]),
                )
                if present
                else None
            )
        return cls(mx.concatenate([part.logprobs for part in parts]), *columns)

    def output(self, plan):
        result = {
            "prompt_logprobs": np.array(self.logprobs, dtype=np.float32),
            "prompt_logprob_indptr": np.asarray(
                plan.prompt_score_indptr, dtype=np.uint32
            ),
        }
        if self.top is not None:
            result.update(self.top.output("prompt_top"))
        if self.targets is not None:
            result.update(self.targets.output("prompt_target"))
        return result


def _score_distribution(
    logits, selected, top_counts, target_counts=None, target_ids=None
):
    vocab = logits.shape[-1]
    if target_ids is not None and np.any(target_ids >= vocab):
        raise ValueError(
            f"prompt target token IDs must be less than vocabulary size {vocab}"
        )
    values = logits.astype(mx.float32)
    values = values - mx.logsumexp(values, axis=-1, keepdims=True)
    chosen = mx.take_along_axis(values, selected[:, None], axis=-1)[:, 0]
    top_counts = np.minimum(top_counts, vocab).astype(np.uint32)
    top = None
    maximum = int(top_counts.max()) if len(top_counts) else 0
    if maximum:
        ids = mx.argsort(-values, axis=-1)[:, :maximum]
        row_indices, columns = np.nonzero(
            np.arange(maximum)[None, :] < top_counts[:, None]
        )
        row_indices, columns = mx.array(row_indices), mx.array(columns)
        ids = ids[row_indices, columns]
        top = ScoreColumns(top_counts, ids, values[row_indices, ids])
    targets = None
    if target_ids is not None and len(target_ids):
        rows = mx.array(np.repeat(np.arange(len(selected)), target_counts))
        ids = mx.array(target_ids)
        targets = ScoreColumns(target_counts, ids, values[rows, ids])
    return PromptScores(chosen, top, targets)


def prompt_scores(plan, hidden, project=None, *, token_start=0):
    """Score scheduled prompt predecessors in vocabulary projections of 128 rows."""
    if not plan.needs_prompt_logprobs:
        return None
    indices = plan.prompt_score_indices
    keep = (indices >= token_start) & (indices < token_start + hidden.shape[0])
    rows = np.flatnonzero(keep)
    selected = indices[keep] - token_start
    targets = plan.prompt_score_token_ids[keep]
    top_counts = plan.prompt_score_top_counts[keep]
    indptr = plan.prompt_target_indptr
    target_counts = np.diff(indptr)[keep]
    parts = []
    for start in range(0, len(selected), 128):
        end = start + 128
        values = hidden[mx.array(selected[start:end])]
        values = project(values) if project is not None else values
        ids = np.concatenate(
            [
                plan.prompt_target_token_ids[indptr[row] : indptr[row + 1]]
                for row in rows[start:end]
            ]
        )
        parts.append(
            _score_distribution(
                values,
                mx.array(targets[start:end]),
                top_counts[start:end],
                target_counts[start:end],
                ids,
            )
        )
    return (
        PromptScores.concat(parts) if parts else PromptScores(mx.array([], mx.float32))
    )


class MlxSampler:
    """Stateless request-local draws; nucleus filtering precedes the top-k limit.

    Seeds and absolute positions determine MLX draws independently of batch
    ordering. RNG streams are reproducible within MLX, not across frameworks.
    """

    def __call__(self, logits, plan=None):
        if plan is None:
            return mx.argmax(logits, axis=-1)
        if logits.ndim != 2 or logits.shape[0] != plan.num_sequences:
            raise ValueError(
                "sampling logits must have shape [plan sequences, vocabulary]"
            )
        inputs = batch_sampling_inputs(plan, logits.shape[-1])
        result = mx.argmax(logits, axis=-1)
        for row, (temperature, sample) in enumerate(
            zip(
                inputs["temperatures"],
                inputs["sampling_mask"],
                strict=True,
            )
        ):
            if not sample or temperature == 0:
                continue
            scores = logits[row].astype(mx.float32) / temperature
            ids = mx.argsort(-scores)
            scores = scores[ids]
            top_p, top_k = inputs["top_ps"][row], inputs["top_ks"][row]
            if top_p < 1:
                probabilities = mx.softmax(scores)
                keep = mx.cumsum(probabilities) - probabilities < top_p
                scores = mx.where(keep, scores, -float("inf"))
            scores, ids = scores[:top_k], ids[:top_k]
            seed = inputs["seeds"][row]
            offset = inputs["offsets"][row]
            mixed = (seed + offset * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
            mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
            mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
            mixed ^= mixed >> 31
            key = mx.array([mixed >> 32, mixed & 0xFFFFFFFF], mx.uint32)
            result[row] = ids[mx.random.categorical(scores, key=key)]
        return result


def format_output(logits, token_ids, plan, *, prompt_logprobs=None):
    """Return one token per sequence and optional untempered model logprobs."""
    ids = np.array(token_ids, dtype=np.uint32)
    columns = {
        "token_ids": ids,
        "token_indptr": np.arange(len(ids) + 1, dtype=np.uint32),
    }
    if plan.needs_prompt_logprobs:
        if prompt_logprobs is None:
            raise ValueError("prefill did not provide requested prompt scores")
        columns.update(prompt_logprobs.output(plan))
    if plan.needs_logprobs:
        requested = np.array(
            [
                sample and params.logprobs
                for sample, params in zip(
                    plan.sampling_mask, plan.sampling_params, strict=True
                )
            ]
        )
        rows = np.flatnonzero(requested)
        counts = np.array(
            [plan.sampling_params[row].top_logprobs for row in rows], np.uint32
        )
        scores = _score_distribution(
            logits[mx.array(rows)], mx.array(ids[rows]), counts
        )
        columns.update(
            logprobs=np.array(scores.logprobs, dtype=np.float32),
            logprob_indptr=_offsets(requested),
        )
        if scores.top is not None:
            columns.update(scores.top.output("top"))
    return ForwardOutput(**columns)
