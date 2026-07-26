"""Tests for the full-run environment adapter (StsRunEnv).

Requires the built engine; skips cleanly otherwise. A passive-combat scripted
policy (end turn only) reliably dies in Act 1, so these drive real runs to a
terminal loss without needing a trained agent; a full-win drive is left to the
end-to-end validation.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env.reward import RewardConfig, beta
from sts_rl.env.run import overworld_actions
from sts_rl.env.run_adapter import StsRunEnv
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    INFO_KEYS_ALWAYS,
    INFO_KEYS_TERMINAL,
    INTERFACE_VERSION,
    InterfaceError,
)

REGRESSION_SEED = 42
# Generous drive cap; a passive-combat run dies far sooner. Well under the env's
# own truncation cap so a driven episode ends by terminating, not truncating.
MAX_DRIVE_STEPS = 2000
_END_TURN = ACTION_BLOCK_BY_NAME["END_TURN"].start
_COMBAT = "combat"


def _greedy_action(info: dict) -> int:
    """Play any non-end-turn combat action if available; else the first legal action.

    In combat this makes progress (plays a card) rather than passing the turn; on
    the overworld it takes the first legal move, enough to walk a run forward.
    """
    legal = np.flatnonzero(info["action_mask"])
    if info["screen"] == _COMBAT:
        plays = [int(i) for i in legal if i != _END_TURN]
        return plays[0] if plays else _END_TURN
    return int(legal[0])


def _passive_action(info: dict) -> int:
    """End the turn in combat (never attack), first legal move on the overworld.

    A run driven this way takes no offensive combat action, so the player dies in
    the first unavoidable fight: a deterministic terminal loss.
    """
    legal = np.flatnonzero(info["action_mask"])
    if info["screen"] == _COMBAT:
        return _END_TURN if _END_TURN in legal else int(legal[0])
    return int(legal[0])


def test_reset_returns_interface_obs_and_info() -> None:
    env = StsRunEnv()
    obs, info = env.reset(seed=REGRESSION_SEED)
    assert env.observation_space.contains(obs)
    assert env.interface_version == INTERFACE_VERSION
    assert info["action_mask"].shape == (ACTION_DIM,)
    assert info["action_mask"].any()  # at least one legal action to start
    for key in INFO_KEYS_ALWAYS:
        assert key in info
    # Terminal-only keys must be absent on a non-terminal reset.
    for key in INFO_KEYS_TERMINAL:
        assert key not in info
    # The run opens on an overworld decision (Neow), not combat.
    assert info["screen"] != _COMBAT
    assert info["run"].outcome == "UNDECIDED"
    env.close()


def test_forced_continue_leaves_only_real_overworld_choices() -> None:
    """Every overworld decision the agent is asked to make offers >= 2 legal moves.

    A screen with exactly one legal action is a forced continue and is auto-advanced
    by the env, so the agent only ever sees genuine overworld choices. (Combat is not
    subject to this: a turn may legitimately offer a single legal move.)
    """
    env = StsRunEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    checked = 0
    for _ in range(MAX_DRIVE_STEPS):
        if info["screen"] != _COMBAT:
            assert len(overworld_actions(env._gc)) >= 2
            checked += 1
        _, _, terminated, truncated, info = env.step(_greedy_action(info))
        if terminated or truncated:
            break
    assert checked > 0  # the drive did reach overworld decision points
    env.close()


def test_drive_crosses_combat_and_overworld() -> None:
    """A driven run enters combat (combat obs + info) and returns to the overworld."""
    env = StsRunEnv()
    obs, info = env.reset(seed=REGRESSION_SEED)
    saw_combat = False
    combat_to_overworld = 0
    prev_screen = info["screen"]
    for _ in range(MAX_DRIVE_STEPS):
        obs, _, terminated, truncated, info = env.step(_greedy_action(info))
        assert env.observation_space.contains(obs)
        if info["screen"] == _COMBAT:
            saw_combat = True
            assert "combat" in info  # combat debug snapshot present in combat mode
            # A live combat exposes enemies in the observation.
            assert obs["enemy_alive"].sum() >= 1
        elif prev_screen == _COMBAT and not (terminated or truncated):
            combat_to_overworld += 1
        prev_screen = info["screen"]
        if terminated or truncated:
            break
    assert saw_combat
    assert combat_to_overworld >= 1  # at least one combat resolved back to the overworld
    env.close()


def test_scripted_run_reaches_terminal_loss() -> None:
    env = StsRunEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    terminated = truncated = False
    reward = 0.0
    for _ in range(MAX_DRIVE_STEPS):
        _, reward, terminated, truncated, info = env.step(_passive_action(info))
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["won"] is False
    # The terminal -1 dominates the bounded shaping delta, so the last step is negative.
    assert reward < 0
    assert info["run"].outcome == "PLAYER_LOSS"
    assert info["episode"]["l"] > 0
    for key in INFO_KEYS_TERMINAL:
        assert key in info
    env.close()


def test_reward_equals_terminal_plus_annealed_shaping() -> None:
    """On every non-terminal step, reward == beta(t) * sum(info['shaping_terms'])."""
    cfg = RewardConfig()
    env = StsRunEnv(reward_config=cfg)
    _, info = env.reset(seed=REGRESSION_SEED)
    for _ in range(MAX_DRIVE_STEPS):
        t_before = env._global_step
        _, reward, terminated, truncated, info = env.step(_greedy_action(info))
        if terminated or truncated:
            break
        expected = beta(t_before, cfg) * sum(info["shaping_terms"].values())
        assert reward == pytest.approx(expected)
    env.close()


def test_shaping_terms_are_mode_appropriate() -> None:
    """Combat steps carry only combat shaping; overworld steps only run shaping."""
    env = StsRunEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    saw_combat_step = saw_overworld_step = False
    for _ in range(MAX_DRIVE_STEPS):
        acted_in_combat = info["screen"] == _COMBAT
        _, _, terminated, truncated, info = env.step(_greedy_action(info))
        terms = info["shaping_terms"]
        if acted_in_combat:
            saw_combat_step = True
            assert terms["floor_progress"] == 0.0
            assert terms["boss_kill"] == 0.0
        else:
            saw_overworld_step = True
            assert terms["enemy_hp_removed"] == 0.0
            assert terms["damage_taken"] == 0.0
        if terminated or truncated:
            break
    assert saw_combat_step and saw_overworld_step
    env.close()


def test_illegal_action_is_noop_and_flagged() -> None:
    env = StsRunEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    illegal = int(np.flatnonzero(~info["action_mask"])[0])
    floor_before, screen_before = info["floor"], info["screen"]
    mask_before = info["action_mask"].copy()
    _, reward, terminated, truncated, info2 = env.step(illegal)
    assert info2["invalid_action"] is True
    assert reward == 0.0
    assert not terminated and not truncated
    # The engine was not touched: the decision point is unchanged.
    assert info2["floor"] == floor_before
    assert info2["screen"] == screen_before
    assert np.array_equal(info2["action_mask"], mask_before)
    env.close()


def test_out_of_range_action_is_illegal() -> None:
    env = StsRunEnv()
    env.reset(seed=REGRESSION_SEED)
    _, reward, terminated, _, info = env.step(ACTION_DIM + 5)
    assert info["invalid_action"] is True
    assert reward == 0.0
    assert not terminated
    env.close()


def test_strict_mode_raises_on_illegal_action() -> None:
    env = StsRunEnv(strict=True)
    _, info = env.reset(seed=REGRESSION_SEED)
    illegal = int(np.flatnonzero(~info["action_mask"])[0])
    with pytest.raises(InterfaceError):
        env.step(illegal)
    env.close()


def test_truncation_sets_flags_and_episode() -> None:
    env = StsRunEnv(max_episode_steps=5)
    _, info = env.reset(seed=REGRESSION_SEED)
    terminated = truncated = False
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(_greedy_action(info))
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert info["episode"]["l"] == 5
    for key in INFO_KEYS_TERMINAL:
        assert key in info
    env.close()


def test_deterministic_under_fixed_policy() -> None:
    """Same seed and policy yield identical reward, termination, and floor streams."""

    def run() -> list[tuple[float, bool, bool, int]]:
        env = StsRunEnv()
        _, info = env.reset(seed=REGRESSION_SEED)
        trace: list[tuple[float, bool, bool, int]] = []
        for _ in range(MAX_DRIVE_STEPS):
            _, reward, terminated, truncated, info = env.step(_passive_action(info))
            trace.append((round(reward, 6), terminated, truncated, info["floor"]))
            if terminated or truncated:
                break
        env.close()
        return trace

    assert run() == run()


def test_legal_actions_returns_mask_copy() -> None:
    env = StsRunEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    mask = env.legal_actions()
    assert np.array_equal(mask, info["action_mask"])
    mask[:] = False
    # Mutating the returned mask must not affect the env's internal mask.
    assert env.legal_actions().any()
    assert np.array_equal(env.legal_actions(), env.action_masks())
    env.close()


def test_set_global_step_validates_and_sets() -> None:
    env = StsRunEnv()
    env.reset(seed=REGRESSION_SEED)
    with pytest.raises(InterfaceError):
        env.set_global_step(-1)
    env.set_global_step(1000)
    assert env._global_step == 1000
    env.close()


@pytest.mark.parametrize("bad_steps", [0, -1])
def test_invalid_max_episode_steps_rejected(bad_steps: int) -> None:
    with pytest.raises(InterfaceError):
        StsRunEnv(max_episode_steps=bad_steps)


def test_invalid_render_mode_rejected() -> None:
    with pytest.raises(InterfaceError):
        StsRunEnv(render_mode="rgb_array")
