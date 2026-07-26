"""Tests for the full-run environment adapter (StsRunEnv).

Requires the built engine; skips cleanly otherwise. A passive-combat scripted
policy (end turn only) reliably dies in Act 1, driving real runs to a terminal
loss without a trained agent; a search-based reference driver
(test_reference_driver_wins_full_run_end_to_end) drives a complete run to a
terminal victory across all three acts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.actions import decode_action
from sts_rl.env.reward import RewardConfig, beta
from sts_rl.env.run import overworld_actions
from sts_rl.env.run_actions import decode_overworld_action
from sts_rl.env.run_adapter import _MAX_SEED, StsRunEnv
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    INFO_KEYS_ALWAYS,
    INFO_KEYS_TERMINAL,
    INTERFACE_VERSION,
    TERMINAL_LOSS_REWARD,
    TERMINAL_WIN_REWARD,
    InterfaceError,
)

REGRESSION_SEED = 42
# A seed whose greedy drive clears the first act's boss and crosses into the next
# act, so boss_kill / act-transition shaping is exercised non-vacuously.
ACT_CROSSING_SEED = 10
# Generous drive cap; a passive-combat run dies far sooner. Well under the env's
# own truncation cap so a driven episode ends by terminating, not truncating.
MAX_DRIVE_STEPS = 2000
_END_TURN = ACTION_BLOCK_BY_NAME["END_TURN"].start
_COMBAT = "combat"

# A seed the search-based reference driver below drives to a full-run victory
# (through the Act 3 boss). The win depends on the pinned engine commit and the
# driver's search budgets; if an engine bump changes play, re-scan low seeds for a
# new winner and update this.
FULL_WIN_SEED = 3
# Per-move combat search budget and per-decision overworld search budget for the
# reference driver. Large enough to win FULL_WIN_SEED deterministically, small
# enough to keep the end-to-end drive near a second.
_COMBAT_SEARCH_SIMS = 400
_OVERWORLD_SEARCH_SIMS = 200
# Ironclad runs cover Acts 1-3 (no Act 4), so a victory ends in the third act.
_FINAL_ACT = 3


def _drive(env: StsRunEnv, first_info: dict, policy) -> list[dict]:
    """Drive ``env`` with ``policy`` until the episode ends; return per-step records.

    Each record captures the state the action was taken in and the step's outcome:
    ``acted_in_combat``, ``reward``, ``terminated``, ``truncated``, ``shaping`` (the
    step's ``shaping_terms``), ``info``, and ``t_before`` (the anneal clock read
    before the step, for reconstructing ``beta(t)``).
    """
    info = first_info
    trace: list[dict] = []
    for _ in range(MAX_DRIVE_STEPS):
        acted_in_combat = info["screen"] == _COMBAT
        t_before = env._global_step
        _, reward, terminated, truncated, info = env.step(policy(info))
        trace.append(
            {
                "acted_in_combat": acted_in_combat,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "shaping": info["shaping_terms"],
                "info": info,
                "t_before": t_before,
            }
        )
        if terminated or truncated:
            return trace
    raise AssertionError("run did not end within the drive cap")


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


def _combat_action_key(action: Any) -> tuple[int, int, int, int]:
    """Identity of a combat ``search::Action``, used to match a searcher's chosen
    move to the interface index that decodes to it: (type, source, target, select)."""
    return (
        int(action.get_action_type()),
        action.get_source_idx(),
        action.get_target_idx(),
        action.get_select_idx(),
    )


class _ReferenceDriver:
    """Search-based reference policy over the interface action space.

    Picks engine-strength moves and maps each to the interface index that decodes
    to it, so it drives ``StsRunEnv.step()`` through the same decode / mask path an
    agent uses, not a private engine channel. Combat runs a ``BattleSearcher`` on
    the env's live ``BattleContext``; the overworld uses the engine's out-of-combat
    search (``pick_gameaction``). :meth:`action` returns ``None`` if a chosen move
    has no interface index (the layout does not cover that decision).

    The overworld agent is stateful - its search RNG advances across calls - so one
    is held per driver (per run); combat uses a fresh searcher per move. Both are
    deterministic for a fixed seed, so a run's outcome is reproducible.
    """

    def __init__(self) -> None:
        self._overworld_agent = sts.Agent()
        self._overworld_agent.simulation_count_base = _OVERWORLD_SEARCH_SIMS

    def action(self, env: StsRunEnv, info: dict) -> int | None:
        if info["screen"] == _COMBAT:
            return self._combat_index(env._bc, info["action_mask"])
        return self._overworld_index(env._gc, info["action_mask"])

    def _combat_index(self, bc: Any, mask: np.ndarray) -> int | None:
        searcher = sts.BattleSearcher(bc)
        searcher.search(_COMBAT_SEARCH_SIMS)
        want = _combat_action_key(searcher.get_best_action())
        for i in np.flatnonzero(mask):
            candidate = decode_action(int(i), bc)
            if candidate is not None and _combat_action_key(candidate) == want:
                return int(i)
        return None

    def _overworld_index(self, gc: Any, mask: np.ndarray) -> int | None:
        want = self._overworld_agent.pick_gameaction(gc)
        for i in np.flatnonzero(mask):
            candidate = decode_overworld_action(int(i), gc)
            if candidate is not None and candidate == want:
                return int(i)
        return None


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


def test_run_shaping_credited_exactly_once_across_combats() -> None:
    """Floor / boss shaping is credited once per floor / act gained, never on combat.

    The run-shaping baseline persists across an intervening combat and updates only
    on overworld steps, so the summed floor_progress / boss_kill contributions
    telescope to the observed floor / act delta - no double count at the boundary,
    none missed. Combat steps contribute zero to either term.
    """
    cfg = RewardConfig()
    env = StsRunEnv(reward_config=cfg)
    _, info0 = env.reset(seed=ACT_CROSSING_SEED)
    init_floor, init_act = info0["floor"], info0["act"]
    trace = _drive(env, info0, _greedy_action)
    final = trace[-1]["info"]

    floor_credit = sum(rec["shaping"]["floor_progress"] for rec in trace)
    boss_credit = sum(rec["shaping"]["boss_kill"] for rec in trace)
    assert floor_credit == pytest.approx(cfg.floor_progress * (final["floor"] - init_floor))
    assert boss_credit == pytest.approx(cfg.boss_kill * (final["act"] - init_act))
    # Non-vacuous: this drive actually cleared an act boss, so boss_kill fired.
    assert final["act"] > init_act
    # No run-shaping is ever credited on a combat step (that is combat shaping's job).
    for rec in trace:
        if rec["acted_in_combat"]:
            assert rec["shaping"]["floor_progress"] == 0.0
            assert rec["shaping"]["boss_kill"] == 0.0
    env.close()


def test_terminal_reward_added_once_on_terminal_step() -> None:
    """The terminal -1 is applied exactly once, only on the terminal step.

    Every non-terminal step's reward is pure shaping (no terminal component); the
    terminal step's reward minus its shaping equals exactly one TERMINAL_LOSS_REWARD.
    """
    cfg = RewardConfig()
    env = StsRunEnv(reward_config=cfg)
    _, info0 = env.reset(seed=REGRESSION_SEED)
    trace = _drive(env, info0, _passive_action)

    assert trace[-1]["terminated"] and not trace[-1]["truncated"]
    for rec in trace:
        shaping = beta(rec["t_before"], cfg) * sum(rec["shaping"].values())
        terminal_component = rec["reward"] - shaping
        if rec is trace[-1]:
            assert terminal_component == pytest.approx(TERMINAL_LOSS_REWARD)
        else:
            assert terminal_component == pytest.approx(0.0)
    env.close()


def test_reference_driver_wins_full_run_end_to_end() -> None:
    """End-to-end: a reference driver carries a full run to a terminal victory.

    The win-side counterpart to test_scripted_run_reaches_terminal_loss. It drives
    StsRunEnv through every screen type and combat of a complete Ironclad run,
    across all three acts, to a terminal PLAYER_VICTORY - exercising the adapter's
    win branch, the act transitions, and full decode / mask coverage together over
    a real winning run rather than a loss. The engine-strength moves are applied
    through StsRunEnv.step() via the shared decode / mask path (see
    _ReferenceDriver), so a win here validates the interface pipeline, not a
    private engine channel.
    """
    cfg = RewardConfig()
    env = StsRunEnv(reward_config=cfg)
    driver = _ReferenceDriver()
    obs, info = env.reset(seed=FULL_WIN_SEED)

    trace: list[tuple[int, float, dict[str, float]]] = []
    terminated = truncated = False
    for _ in range(MAX_DRIVE_STEPS):
        # Every step is a genuine decision with a valid observation, and the
        # driver's engine pick maps to an interface index - so a full winning run
        # never hits an unrepresentable decision.
        assert env.observation_space.contains(obs)
        assert info["action_mask"].any()
        t_before = env._global_step
        idx = driver.action(env, info)
        assert idx is not None, "engine pick has no interface index on this decision"
        obs, reward, terminated, truncated, info = env.step(idx)
        trace.append((t_before, reward, info["shaping_terms"]))
        if terminated or truncated:
            break
    else:
        raise AssertionError("run did not reach a terminal within the drive cap")

    assert terminated and not truncated
    assert info["won"] is True
    assert info["run"].outcome == "PLAYER_VICTORY"
    # Non-vacuous: the drive crossed both act boundaries to win in the final act.
    assert info["act"] == _FINAL_ACT
    for key in INFO_KEYS_TERMINAL:
        assert key in info
    assert info["episode"]["l"] == len(trace)
    # Cumulative episode return equals the summed per-step rewards (guards the
    # env's _ep_return accumulation, not just the per-step reward formula).
    assert info["episode"]["r"] == pytest.approx(sum(reward for _, reward, _ in trace))
    # The terminal +1 is credited exactly once, on the terminal step; every prior
    # step's reward is pure annealed shaping (the win-side mirror of
    # test_terminal_reward_added_once_on_terminal_step).
    for k, (t_before, reward, shaping) in enumerate(trace):
        terminal_component = reward - beta(t_before, cfg) * sum(shaping.values())
        expected = TERMINAL_WIN_REWARD if k == len(trace) - 1 else 0.0
        assert terminal_component == pytest.approx(expected)
    env.close()


def test_forced_continues_and_autoresolves_do_not_count_as_steps() -> None:
    """Only agent step() calls advance the step counter, not env-driven advances.

    reset settles past forced continues before the first decision without counting a
    step, and mid-episode forced continues / combat auto-resolves never inflate the
    count: the terminal episode length equals the number of step() calls made.
    """
    env = StsRunEnv()
    _, info0 = env.reset(seed=REGRESSION_SEED)
    assert env._steps == 0  # settle advanced forced continues but counted no step
    trace = _drive(env, info0, _passive_action)
    assert trace[-1]["info"]["episode"]["l"] == len(trace)
    env.close()


def test_reset_without_seed_is_valid_and_bounded() -> None:
    """reset(seed=None) draws an in-range episode seed and yields a valid observation."""
    env = StsRunEnv()
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["action_mask"].any()
    assert 0 <= env._episode_seed < _MAX_SEED
    env.close()
