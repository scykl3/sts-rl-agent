"""Tests for the combat environment adapter (StsEnv).

Requires the built engine; skips cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env.adapter import StsEnv
from sts_rl.env.reward import RewardConfig, beta
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    INFO_KEYS_ALWAYS,
    INTERFACE_VERSION,
    SHAPING_TERMS,
    TERMINAL_LOSS_REWARD,
    TERMINAL_WIN_REWARD,
)

REGRESSION_SEED = 42
MAX_SCRIPTED_STEPS = 400
_END_TURN = ACTION_BLOCK_BY_NAME["END_TURN"].start
_PROCEED = ACTION_BLOCK_BY_NAME["PROCEED"].start  # a non-combat index, always illegal here


def _greedy_to_terminal(env: StsEnv, obs_info):
    """Play any legal card each step, else end turn, until the episode ends."""
    _, info = obs_info
    for _ in range(MAX_SCRIPTED_STEPS):
        mask = info["action_mask"]
        legal = np.flatnonzero(mask)
        # Prefer a card play (anything that is not end turn) to make progress; fall
        # back to end turn.
        play = [i for i in legal if i != _END_TURN]
        action = int(play[0]) if play else _END_TURN
        _, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return reward, terminated, truncated, info
    raise AssertionError("combat did not end within the step cap")


def test_reset_returns_contract_obs_and_info() -> None:
    env = StsEnv()
    obs, info = env.reset(seed=REGRESSION_SEED)
    assert env.observation_space.contains(obs)
    assert env.interface_version == INTERFACE_VERSION
    assert info["action_mask"].shape == (env.action_space.n,)
    for key in INFO_KEYS_ALWAYS:
        assert key in info
    assert info["combat"].player_hp > 0
    env.close()


def test_scripted_combat_reaches_terminal_with_win_or_loss() -> None:
    env = StsEnv()
    obs_info = env.reset(seed=REGRESSION_SEED)
    reward, terminated, truncated, info = _greedy_to_terminal(env, obs_info)
    assert terminated and not truncated
    # The terminal step also carries a bounded shaping delta (|shaping| < 1), so
    # the terminal +1/-1 dominates the sign.
    assert info["won"] == (reward > 0)
    assert info["episode"]["l"] > 0
    env.close()


def test_reward_equals_terminal_plus_annealed_shaping() -> None:
    # Ties the adapter's per-step reward to the reward module: on every step,
    # reward == terminal + beta(t) * sum(info['shaping_terms']).
    cfg = RewardConfig()
    env = StsEnv(reward_config=cfg)
    _, info = env.reset(seed=REGRESSION_SEED)
    assert set(info["shaping_terms"]) == set(SHAPING_TERMS)
    assert all(v == 0.0 for v in info["shaping_terms"].values())  # no delta at reset

    # t stays in lockstep with the env clock because every step below is legal
    # (each step advances the env's global step by exactly one); the invalid
    # path is covered separately by test_invalid_action_advances_anneal_clock.
    t = 0
    saw_shaping = False
    for _ in range(MAX_SCRIPTED_STEPS):
        legal = np.flatnonzero(info["action_mask"])
        play = [i for i in legal if i != _END_TURN]
        action = int(play[0]) if play else _END_TURN
        _, reward, terminated, truncated, info = env.step(action)
        terms = info["shaping_terms"]
        assert set(terms) == set(SHAPING_TERMS)
        terminal = 0.0
        if terminated:
            terminal = TERMINAL_WIN_REWARD if info["won"] else TERMINAL_LOSS_REWARD
        assert reward == pytest.approx(terminal + beta(t, cfg) * sum(terms.values()))
        saw_shaping = saw_shaping or any(v != 0.0 for v in terms.values())
        t += 1
        if terminated or truncated:
            break
    assert saw_shaping  # a real combat moves HP, so shaping must fire at least once
    env.close()


def _first_playable_action(info: dict) -> int:
    legal = np.flatnonzero(info["action_mask"])
    play = [i for i in legal if i != _END_TURN]
    return int(play[0]) if play else _END_TURN


def test_invalid_action_advances_anneal_clock() -> None:
    # An illegal step takes no engine action but counts as one env interaction,
    # so the next legal step's beta index reflects it (beta(1), not beta(0)).
    cfg = RewardConfig()
    env = StsEnv(reward_config=cfg)
    _, info = env.reset(seed=REGRESSION_SEED)
    _, _, _, _, info = env.step(_PROCEED)  # illegal in combat: state unchanged
    assert info["invalid_action"] is True

    action = _first_playable_action(info)
    _, reward, terminated, _, info = env.step(action)
    terminal = 0.0
    if terminated:
        terminal = TERMINAL_WIN_REWARD if info["won"] else TERMINAL_LOSS_REWARD
    assert reward == pytest.approx(terminal + beta(1, cfg) * sum(info["shaping_terms"].values()))
    env.close()


def test_set_global_step_overrides_anneal_clock() -> None:
    cfg = RewardConfig()
    env = StsEnv(reward_config=cfg)
    _, info = env.reset(seed=REGRESSION_SEED)
    env.set_global_step(1000)
    action = _first_playable_action(info)
    _, reward, terminated, _, info = env.step(action)
    terminal = 0.0
    if terminated:
        terminal = TERMINAL_WIN_REWARD if info["won"] else TERMINAL_LOSS_REWARD
    assert reward == pytest.approx(terminal + beta(1000, cfg) * sum(info["shaping_terms"].values()))
    env.close()


def test_set_global_step_rejects_negative() -> None:
    from sts_rl.interface import InterfaceError

    env = StsEnv()
    env.reset(seed=REGRESSION_SEED)
    with pytest.raises(InterfaceError):
        env.set_global_step(-1)
    env.close()


def test_illegal_action_is_flagged_and_not_executed() -> None:
    env = StsEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    turn_before = info["combat"].turn
    hp_before = info["combat"].player_hp
    obs, reward, terminated, truncated, info = env.step(_PROCEED)  # illegal in combat
    assert info["invalid_action"] is True
    assert reward == 0.0
    assert not terminated
    # Engine state is untouched by an illegal action.
    assert info["combat"].turn == turn_before
    assert info["combat"].player_hp == hp_before
    env.close()


def test_strict_mode_raises_on_illegal_action() -> None:
    from sts_rl.interface import InterfaceError

    env = StsEnv(strict=True)
    env.reset(seed=REGRESSION_SEED)
    with pytest.raises(InterfaceError):
        env.step(_PROCEED)
    env.close()


def test_same_seed_is_deterministic() -> None:
    env_a, env_b = StsEnv(), StsEnv()
    _, info_a = env_a.reset(seed=REGRESSION_SEED)
    _, info_b = env_b.reset(seed=REGRESSION_SEED)
    assert info_a["combat"] == info_b["combat"]
    assert np.array_equal(info_a["action_mask"], info_b["action_mask"])
    env_a.close()
    env_b.close()


def test_truncation_when_step_cap_hit() -> None:
    # A tiny cap and a no-op-ish policy (repeated end turn) forces truncation before
    # a win: end turn each step until the cap trips.
    env = StsEnv(max_episode_steps=2)
    env.reset(seed=REGRESSION_SEED)
    env.step(_END_TURN)
    _, reward, terminated, truncated, info = env.step(_END_TURN)
    assert truncated and not terminated
    assert "episode" in info
    env.close()
