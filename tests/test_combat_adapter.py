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
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    INFO_KEYS_ALWAYS,
    INTERFACE_VERSION,
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
    assert reward in (TERMINAL_WIN_REWARD, TERMINAL_LOSS_REWARD)
    assert info["won"] == (reward == TERMINAL_WIN_REWARD)
    assert info["episode"]["l"] > 0
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
