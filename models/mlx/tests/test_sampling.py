"""Analytic distributions through the real Engine isolate sampler semantics."""

import sys

import numpy as np
import pytest

from resin import Engine, RunnerConfig, SamplingParams, SchedulerConfig

if sys.platform == "darwin":
    mx = pytest.importorskip("mlx.core")
    from resin.backend.mlx import MlxSampler, format_output, prompt_scores
else:
    pytest.skip("MLX requires Apple Silicon", allow_module_level=True)


class DistributionRunner:
    runner_config = RunnerConfig(continuation_prefill=True)

    def __init__(self):
        self.sampler = MlxSampler()

    def forward(self, plan):
        # These probabilities have exact top-p boundaries at 0.5 and 0.75.
        logits = mx.log(mx.array([0.125, 0.25, 0.5, 0.125]))
        rows = mx.broadcast_to(logits, (plan.num_sequences, 4))
        prompt = mx.broadcast_to(logits, (len(plan.token_ids), 4))
        return format_output(
            rows,
            self.sampler(rows, plan),
            plan,
            prompt_logprobs=prompt_scores(plan, prompt),
        )


def collect(params, prompts=None, *, chunk=128):
    prompts = prompts or [[0, 1, 2]] * len(params)
    with Engine(
        DistributionRunner(), scheduler_config=SchedulerConfig(prefill_chunk_size=chunk)
    ) as engine:
        requests = [
            engine.submit(prompt, sampling_params=policy)
            for prompt, policy in zip(prompts, params, strict=True)
        ]
        events = []
        while engine.has_work():
            events.extend(engine.step())
        return [request.take_result() for request in requests], requests, events


@pytest.mark.parametrize(
    "policy",
    [
        {"temperature": 0},
        {"temperature": 1, "top_k": 1},
        {"temperature": 1, "top_p": 0.5},
        {"temperature": 2, "top_p": 0.01},
    ],
)
def test_filters_cannot_sample_outside_the_retained_mass(policy):
    result, _, _ = collect([SamplingParams(max_output_tokens=8, seed=31, **policy)])
    assert result == [[2] * 8]


def test_sampling_seed_is_independent_of_batch_order_and_prefill_chunks():
    policies = [
        SamplingParams(
            max_output_tokens=12, temperature=0.8, seed=seed, top_k=3, top_p=0.9
        )
        for seed in (17, 99, 123)
    ]
    prompts = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 0]]
    together, _, _ = collect(policies, prompts)
    reversed_rows, _, _ = collect(policies[::-1], prompts[::-1], chunk=2)
    separate = [
        collect([policy], [prompt], chunk=2)[0][0]
        for policy, prompt in zip(policies, prompts, strict=True)
    ]
    assert together == reversed_rows[::-1] == separate
    assert len({tuple(row) for row in together}) > 1
    same_positions, _, _ = collect(policies)
    assert len({tuple(row) for row in same_positions}) > 1


def test_scores_remain_untempered_and_targets_keep_requested_order():
    params = SamplingParams(
        max_output_tokens=2,
        temperature=4,
        top_k=1,
        seed=0,
        logprobs=True,
        top_logprobs=2,
        prompt_logprobs=True,
        prompt_top_logprobs=2,
    )
    with Engine(
        DistributionRunner(), scheduler_config=SchedulerConfig(prefill_chunk_size=2)
    ) as engine:
        request = engine.submit(
            [0, 1, 2, 3],
            sampling_params=params,
            prompt_target_logprobs=[[], [3, 0], [1], [2, 1]],
        )
        events = []
        while engine.has_work():
            events.extend(engine.step())
        assert request.take_result() == [2, 2]
        np.testing.assert_allclose(
            [event.logprobs.logprob for event in events],
            [-0.69314718056] * 2,
            atol=1e-6,
        )
        scores = request.prompt_logprobs
        assert scores[0] is None
        np.testing.assert_allclose(
            [score.logprob for score in scores[1:]],
            [-1.38629436112, -0.69314718056, -2.07944154168],
            atol=1e-6,
        )
        assert [token for token, _ in scores[1].target_logprobs] == [3, 0]
        assert [token for token, _ in scores[3].target_logprobs] == [2, 1]
        assert [token for token, _ in events[0].logprobs.top_logprobs] == [2, 1]


def test_importing_resin_does_not_import_mlx():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import resin; assert 'mlx' not in sys.modules; assert 'mlx.core' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
