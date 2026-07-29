"""Engine-free tests for the matched-seed paired A/B evaluation harness.

The statistics helpers (``_paired_mean_and_se``, ``_mcnemar_exact_p``) are tested
directly against hand-computed values. ``paired_evaluate`` is tested with
``_PairedScriptedEnv``, a double whose per-seed terminal outcome depends only on
the reset order and seed (not on the policy's actions), so the pairing and delta
math are isolated from policy behaviour and asserted against known numbers.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.env.spaces import build_observation_space, build_spaces
from sts_rl.eval import PairedEvalReport, paired_evaluate
from sts_rl.interface import ACTION_DIM, InterfaceError

# Private helpers live on the module; fetch it directly (the package re-export
# shadows the submodule attribute, mirroring test_evaluate.py).
eval_mod = importlib.import_module("sts_rl.eval.evaluate")

# A scripted outcome: (won, floor, hp, length, ret).
_Outcome = tuple[bool, int, int, int, float]


class _PairedScriptedEnv:
    """Serves ``outcomes_a`` for the first ``n`` resets, ``outcomes_b`` for the next ``n``.

    ``paired_evaluate`` runs ``policy_a`` over all ``n`` seeds, then ``policy_b``
    over the same seeds, so reset index ``k < n`` is policy A on ``seeds[k]`` and
    reset index ``n + k`` is policy B on ``seeds[k]``. The outcome depends only on
    that phase and the reset seed, never on the action taken, which decouples the
    harness math from the policy. The env is single-use per ``paired_evaluate``
    call (its reset counter advances monotonically); build a fresh one to re-run.
    """

    def __init__(
        self,
        outcomes_a: dict[int, _Outcome],
        outcomes_b: dict[int, _Outcome],
        acts_a: dict[int, int],
        acts_b: dict[int, int],
    ) -> None:
        assert len(outcomes_a) == len(outcomes_b), "both phases must cover the same seeds"
        self.observation_space, self.action_space = build_spaces()
        self._obs_space = build_observation_space()
        self._obs_space.seed(0)
        self._phase_outcomes = (outcomes_a, outcomes_b)
        self._phase_acts = (acts_a, acts_b)
        self._n = len(outcomes_a)
        self._resets = 0
        self._phase = 0
        self._seed = 0
        self._steps = 0

    @staticmethod
    def _legal_mask() -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        mask[0] = True  # END_TURN always legal here; greedy argmax must pick it
        return mask

    def reset(self, *, seed=None, options=None):
        assert seed is not None, "the harness always resets with an explicit seed"
        # First n resets are policy A's episodes, the next n are policy B's.
        self._phase = 0 if self._resets < self._n else 1
        self._resets += 1
        self._seed = seed
        self._steps = 0
        return self._obs_space.sample(), {"action_mask": self._legal_mask()}

    def step(self, action):
        self._steps += 1
        won, floor, hp, length, ret = self._phase_outcomes[self._phase][self._seed]
        terminated = self._steps >= length
        obs = self._obs_space.sample()
        info: dict = {"action_mask": self._legal_mask()}
        if terminated:
            info.update(won=won, floor=floor, hp=hp, episode={"r": ret, "l": self._steps})
            info["act"] = self._phase_acts[self._phase][self._seed]
        reward = ret if terminated else 0.0
        return obs, reward, terminated, False, info


def _policy() -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic()


# --- _paired_mean_and_se ----------------------------------------------------


def test_paired_mean_and_se_zero_variance() -> None:
    mean, se = eval_mod._paired_mean_and_se([1.0, 1.0, 1.0, 1.0])
    assert mean == pytest.approx(1.0)
    assert se == pytest.approx(0.0)


def test_paired_mean_and_se_matches_sample_formula() -> None:
    # diffs [1,-1,1,-1]: mean 0, sample var (ddof=1) = 4/3, se = sqrt(var/n) = sqrt(1/3).
    mean, se = eval_mod._paired_mean_and_se([1.0, -1.0, 1.0, -1.0])
    assert mean == pytest.approx(0.0)
    assert se == pytest.approx((1.0 / 3.0) ** 0.5)


def test_paired_mean_and_se_singleton_has_zero_se() -> None:
    # n < 2 has no spread to estimate: SE is reported as 0.0, not a raise.
    mean, se = eval_mod._paired_mean_and_se([0.5])
    assert mean == pytest.approx(0.5)
    assert se == pytest.approx(0.0)


# --- _mcnemar_exact_p -------------------------------------------------------


def test_mcnemar_no_discordant_pairs_is_one() -> None:
    assert eval_mod._mcnemar_exact_p(0, 0) == pytest.approx(1.0)


def test_mcnemar_balanced_discordant_caps_at_one() -> None:
    # b01 == b10: the doubled lower tail exceeds 1 and is capped.
    assert eval_mod._mcnemar_exact_p(5, 5) == pytest.approx(1.0)


def test_mcnemar_extreme_discordant_is_small() -> None:
    # All 10 discordant seeds favour one policy: two-sided p = 2 * 0.5^10.
    assert eval_mod._mcnemar_exact_p(10, 0) == pytest.approx(2.0 * 0.5**10)


def test_mcnemar_is_symmetric() -> None:
    assert eval_mod._mcnemar_exact_p(2, 7) == pytest.approx(eval_mod._mcnemar_exact_p(7, 2))


# --- paired_evaluate --------------------------------------------------------


def _known_difference_env() -> _PairedScriptedEnv:
    # A wins seeds 0,1,2 and loses 3; B wins only seed 0. Discordant (exactly one
    # won): seeds 1 and 2, both in A's favour -> win_a_only=2, win_b_only=0.
    outcomes_a: dict[int, _Outcome] = {
        0: (True, 50, 20, 3, 1.0),
        1: (True, 40, 15, 4, 1.0),
        2: (True, 45, 10, 2, 1.0),
        3: (False, 16, 0, 5, -1.0),
    }
    outcomes_b: dict[int, _Outcome] = {
        0: (True, 50, 25, 3, 1.0),
        1: (False, 16, 0, 4, -1.0),
        2: (False, 16, 0, 2, -1.0),
        3: (False, 16, 0, 5, -1.0),
    }
    acts_a = {0: 3, 1: 3, 2: 3, 3: 1}
    acts_b = {0: 3, 1: 1, 2: 1, 3: 1}
    return _PairedScriptedEnv(outcomes_a, outcomes_b, acts_a, acts_b)


def test_paired_evaluate_known_difference() -> None:
    seeds = [0, 1, 2, 3]
    report = paired_evaluate(
        _policy(), _policy(), _known_difference_env(), seeds, device=torch.device("cpu")
    )

    assert isinstance(report, PairedEvalReport)
    assert report.n_episodes == 4
    assert report.win_rate_a == pytest.approx(0.75)
    assert report.win_rate_b == pytest.approx(0.25)
    assert report.win_rate_delta == pytest.approx(0.5)
    # delta is exactly the difference of the two rates.
    assert report.win_rate_delta == pytest.approx(report.win_rate_a - report.win_rate_b)
    # won_diffs [0,1,1,0]: var (ddof=1) = 1/3, se = sqrt(var/4).
    assert report.win_rate_delta_se == pytest.approx(((1.0 / 3.0) / 4.0) ** 0.5)

    assert report.act1_clear_rate_a == pytest.approx(0.75)
    assert report.act1_clear_rate_b == pytest.approx(0.25)
    assert report.act1_clear_rate_delta == pytest.approx(0.5)

    assert report.mean_return_a == pytest.approx(0.5)
    assert report.mean_return_b == pytest.approx(-0.5)
    assert report.return_delta == pytest.approx(1.0)

    assert report.win_a_only == 2
    assert report.win_b_only == 0
    # McNemar exact two-sided for (2, 0): 2 * 0.5^2 = 0.5.
    assert report.mcnemar_p == pytest.approx(0.5)


def test_paired_evaluate_identical_outcomes_give_zero_delta() -> None:
    # Both phases share the same outcomes, so every paired difference is zero.
    outcomes: dict[int, _Outcome] = {
        0: (True, 50, 20, 3, 1.0),
        1: (False, 16, 0, 4, -1.0),
    }
    acts = {0: 3, 1: 1}
    env = _PairedScriptedEnv(outcomes, dict(outcomes), acts, dict(acts))
    report = paired_evaluate(_policy(), _policy(), env, [0, 1], device=torch.device("cpu"))

    assert report.win_rate_delta == pytest.approx(0.0)
    assert report.win_rate_delta_se == pytest.approx(0.0)
    assert report.act1_clear_rate_delta == pytest.approx(0.0)
    assert report.return_delta == pytest.approx(0.0)
    assert report.win_a_only == 0
    assert report.win_b_only == 0
    assert report.mcnemar_p == pytest.approx(1.0)


def test_paired_evaluate_is_deterministic() -> None:
    # A fresh env per call (the double is single-use), so two runs must match.
    seeds = [0, 1, 2, 3]
    first = paired_evaluate(
        _policy(), _policy(), _known_difference_env(), seeds, device=torch.device("cpu")
    )
    second = paired_evaluate(
        _policy(), _policy(), _known_difference_env(), seeds, device=torch.device("cpu")
    )
    assert first == second


def test_paired_evaluate_rejects_empty_seed_set() -> None:
    env = _known_difference_env()
    with pytest.raises(InterfaceError):
        paired_evaluate(_policy(), _policy(), env, seeds=[], device=torch.device("cpu"))


def test_paired_evaluate_infers_device_per_policy() -> None:
    # Omitting device infers cpu from each policy's parameters (no raise).
    report = paired_evaluate(_policy(), _policy(), _known_difference_env(), [0, 1, 2, 3])
    assert report.n_episodes == 4


def test_paired_report_as_dict_exposes_all_metrics_as_floats() -> None:
    report = PairedEvalReport(
        n_episodes=4,
        win_rate_a=0.75,
        win_rate_b=0.25,
        win_rate_delta=0.5,
        win_rate_delta_se=0.1,
        act1_clear_rate_a=0.75,
        act1_clear_rate_b=0.25,
        act1_clear_rate_delta=0.5,
        act1_clear_rate_delta_se=0.1,
        mean_return_a=0.5,
        mean_return_b=-0.5,
        return_delta=1.0,
        return_delta_se=0.2,
        win_a_only=2,
        win_b_only=0,
        mcnemar_p=0.5,
    )
    flat = report.as_dict()
    assert flat["win_rate_delta"] == pytest.approx(0.5)
    assert flat["mcnemar_p"] == pytest.approx(0.5)
    # Int-typed count fields are surfaced as floats for uniform logging.
    assert isinstance(flat["win_a_only"], float)
    assert all(isinstance(v, float) for v in flat.values())
