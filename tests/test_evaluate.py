"""Engine-free tests for the seeded holdout evaluation harness.

Two env backends are used, neither needing the C++ engine:

* ``_ScriptedEnv`` returns a fixed per-seed terminal outcome regardless of the
  action taken, so the aggregation math (win rate, averages) can be asserted
  against hand-computed values, decoupled from the policy.
* :class:`~sts_rl.env.stub_env.StubEnv` exercises the real gymnasium step path
  and confirms greedy eval is reproducible and that truncation counts as a loss.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from torch import nn

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.env.spaces import build_observation_space, build_spaces
from sts_rl.env.stub_env import StubEnv
from sts_rl.eval import (
    EvalReport,
    evaluate,
    make_holdout_seeds,
    run_episode,
)
from sts_rl.interface import ACTION_DIM, InterfaceError

# The package re-exports the `evaluate` function under the name `sts_rl.eval.evaluate`,
# shadowing the submodule attribute; fetch the module itself for monkeypatching.
eval_mod = importlib.import_module("sts_rl.eval.evaluate")

# A scripted outcome: (won, floor, hp, length, ret).
_Outcome = tuple[bool, int, int, int, float]


class _ScriptedEnv:
    """Minimal gymnasium-style env with a fixed per-seed terminal outcome.

    Emits interface-valid observations and a single-legal-action mask so the
    real policy runs, but the outcome depends only on the reset seed, not on the
    actions chosen. This isolates the harness's aggregation from policy
    behaviour.
    """

    def __init__(self, outcomes: dict[int, _Outcome]) -> None:
        self.observation_space, self.action_space = build_spaces()
        self._outcomes = outcomes
        self._obs_space = build_observation_space()
        self._obs_space.seed(0)
        self._seed = 0
        self._steps = 0

    @staticmethod
    def _legal_mask() -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        mask[0] = True  # END_TURN is always legal here
        return mask

    def reset(self, *, seed=None, options=None):
        assert seed is not None, "the harness always resets with an explicit seed"
        self._seed = seed
        self._steps = 0
        return self._obs_space.sample(), {"action_mask": self._legal_mask()}

    def step(self, action):
        self._steps += 1
        won, floor, hp, length, ret = self._outcomes[self._seed]
        terminated = self._steps >= length
        obs = self._obs_space.sample()
        info: dict = {"action_mask": self._legal_mask()}
        if terminated:
            info.update(
                won=won,
                floor=floor,
                hp=hp,
                episode={"r": ret, "l": self._steps},
            )
        reward = ret if terminated else 0.0
        return obs, reward, terminated, False, info


def _policy() -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic()


def test_evaluate_aggregation_matches_hand_computed() -> None:
    # seed -> (won, floor, hp, length, ret)
    outcomes: dict[int, _Outcome] = {
        0: (True, 5, 30, 4, 1.0),
        1: (False, 2, 0, 7, -1.0),
        2: (True, 9, 12, 3, 1.0),
        3: (False, 1, 0, 2, -1.0),
    }
    env = _ScriptedEnv(outcomes)
    report = evaluate(_policy(), env, seeds=list(outcomes), device=torch.device("cpu"))

    assert isinstance(report, EvalReport)
    assert report.n_episodes == 4
    assert report.win_rate == pytest.approx(2 / 4)
    assert report.avg_floor == pytest.approx((5 + 2 + 9 + 1) / 4)
    assert report.avg_hp == pytest.approx((30 + 0 + 12 + 0) / 4)
    assert report.avg_ep_len == pytest.approx((4 + 7 + 3 + 2) / 4)
    assert report.avg_return == pytest.approx((1.0 - 1.0 + 1.0 - 1.0) / 4)


def test_run_episode_reads_scripted_terminal_outcome() -> None:
    env = _ScriptedEnv({42: (True, 8, 21, 5, 1.0)})
    result = run_episode(_policy(), env, seed=42, device=torch.device("cpu"))

    assert result.seed == 42
    assert result.won is True
    assert result.floor == 8
    assert result.hp == 21
    assert result.length == 5
    assert result.ret == pytest.approx(1.0)


def test_evaluate_is_reproducible_under_greedy() -> None:
    # Greedy selection is deterministic in the weights, and StubEnv outcomes are
    # fixed per reset seed, so two evals of one policy over one seed set match.
    policy = _policy()
    seeds = make_holdout_seeds(base_seed=100, count=8)

    def make_env() -> StubEnv:
        return StubEnv(reward_mode="random", terminate_prob=0.3, max_episode_steps=8)

    first = evaluate(policy, make_env(), seeds, device=torch.device("cpu"))
    second = evaluate(policy, make_env(), seeds, device=torch.device("cpu"))
    assert first == second


def test_evaluate_counts_one_episode_per_seed() -> None:
    env = StubEnv(reward_mode="random", terminate_prob=0.3, max_episode_steps=8)
    seeds = make_holdout_seeds(base_seed=0, count=6)
    report = evaluate(_policy(), env, seeds, device=torch.device("cpu"))
    assert report.n_episodes == len(seeds)


def test_truncation_counts_as_loss() -> None:
    # terminate_prob=0 guarantees every episode ends by truncation, which the
    # interface reports as won=False, so the win rate must be exactly 0.
    env = StubEnv(reward_mode="random", terminate_prob=0.0, max_episode_steps=3)
    seeds = make_holdout_seeds(base_seed=7, count=5)
    report = evaluate(_policy(), env, seeds, device=torch.device("cpu"))
    assert report.win_rate == 0.0
    assert report.avg_ep_len == pytest.approx(3.0)  # all truncate at the cap


def test_win_rate_is_a_fraction() -> None:
    env = StubEnv(reward_mode="random", terminate_prob=0.5, max_episode_steps=6)
    report = evaluate(_policy(), env, make_holdout_seeds(0, 16), device=torch.device("cpu"))
    assert 0.0 <= report.win_rate <= 1.0


def test_evaluate_rejects_empty_seed_set() -> None:
    env = StubEnv(reward_mode="random")
    with pytest.raises(InterfaceError):
        evaluate(_policy(), env, seeds=[], device=torch.device("cpu"))


def test_evaluate_infers_device_from_policy() -> None:
    # Omitting device should infer cpu from the network's parameters (no raise).
    env = _ScriptedEnv({0: (True, 1, 1, 1, 1.0)})
    report = evaluate(_policy(), env, seeds=[0])
    assert report.n_episodes == 1


def test_as_dict_exposes_all_metrics_as_floats() -> None:
    report = EvalReport(
        n_episodes=3,
        win_rate=0.5,
        avg_floor=4.0,
        avg_hp=10.0,
        avg_ep_len=6.0,
        avg_return=0.25,
    )
    flat = report.as_dict()
    assert set(flat) == {
        "n_episodes",
        "win_rate",
        "avg_floor",
        "avg_hp",
        "avg_ep_len",
        "avg_return",
    }
    assert all(isinstance(v, float) for v in flat.values())
    assert flat["win_rate"] == pytest.approx(0.5)


def test_make_holdout_seeds_is_reproducible_and_distinct() -> None:
    a = make_holdout_seeds(base_seed=1000, count=32)
    b = make_holdout_seeds(base_seed=1000, count=32)
    assert a == b
    assert len(a) == 32
    assert len(set(a)) == 32  # distinct
    assert a[0] == 1000


def test_make_holdout_seeds_rejects_nonpositive_count() -> None:
    with pytest.raises(InterfaceError):
        make_holdout_seeds(base_seed=0, count=0)


class _RotatingMaskEnv:
    """Env whose single legal action alternates every step.

    Each :meth:`step` validates the received action against the mask returned on
    the immediately preceding call, counting any mismatch in ``illegal_actions``.
    A harness that reuses the reset mask (instead of refreshing it each step)
    picks the wrong index from step two onward, so ``illegal_actions`` is only
    zero when the mask is refreshed every step - the regression guard.
    """

    def __init__(self, length: int) -> None:
        self.observation_space, self.action_space = build_spaces()
        self._obs_space = build_observation_space()
        self._obs_space.seed(0)
        self._length = length
        self._steps = 0
        self._legal_index = 0
        self.illegal_actions = 0

    @staticmethod
    def _mask_for(index: int) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        mask[index] = True  # exactly one legal action -> greedy argmax must pick it
        return mask

    def reset(self, *, seed=None, options=None):
        self._steps = 0
        self._legal_index = 0
        self.illegal_actions = 0
        return self._obs_space.sample(), {"action_mask": self._mask_for(self._legal_index)}

    def step(self, action):
        if action != self._legal_index:
            self.illegal_actions += 1
        self._steps += 1
        terminated = self._steps >= self._length
        self._legal_index = 1 - self._legal_index  # rotate for the next presented mask
        obs = self._obs_space.sample()
        info: dict = {"action_mask": self._mask_for(self._legal_index)}
        if terminated:
            info.update(won=True, floor=1, hp=1, episode={"r": 1.0, "l": self._steps})
        return obs, 0.0, terminated, False, info


class _NeverEndsEnv:
    """Env that never terminates or truncates, to exercise the step ceiling."""

    def __init__(self) -> None:
        self.observation_space, self.action_space = build_spaces()
        self._obs_space = build_observation_space()
        self._obs_space.seed(0)

    @staticmethod
    def _mask() -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        mask[0] = True
        return mask

    def reset(self, *, seed=None, options=None):
        return self._obs_space.sample(), {"action_mask": self._mask()}

    def step(self, action):
        return self._obs_space.sample(), 0.0, False, False, {"action_mask": self._mask()}


def test_mask_is_refreshed_each_step() -> None:
    # Regression: the policy must act on the CURRENT legal set, not the reset
    # mask. With a stale mask the rotating env sees illegal actions from step 2.
    env = _RotatingMaskEnv(length=5)
    run_episode(_policy(), env, seed=0, device=torch.device("cpu"))
    assert env.illegal_actions == 0


def test_nonterminating_env_hits_step_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(eval_mod, "_EPISODE_STEP_CEILING", 3)
    with pytest.raises(InterfaceError):
        run_episode(_policy(), _NeverEndsEnv(), seed=0, device=torch.device("cpu"))


def test_evaluate_requires_device_for_paramless_policy() -> None:
    # A module with no parameters gives no device to infer, so evaluate must
    # raise rather than guess. Reaches the guard before any policy.act call.
    bare = nn.Module()
    env = _ScriptedEnv({0: (True, 1, 1, 1, 1.0)})
    with pytest.raises(ValueError):
        evaluate(bare, env, seeds=[0])
