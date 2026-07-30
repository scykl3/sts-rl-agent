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
from sts_rl.env.reward import RewardConfig, state_potentials
from sts_rl.env.run import overworld_actions
from sts_rl.env.run_actions import decode_overworld_action
from sts_rl.env.run_adapter import _MAX_SEED, StsRunEnv
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    INFO_KEYS_ALWAYS,
    INFO_KEYS_TERMINAL,
    INTERFACE_VERSION,
    SHAPING_TERMS,
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
    step's ``shaping_terms``), and ``info``.
    """
    info = first_info
    trace: list[dict] = []
    for _ in range(MAX_DRIVE_STEPS):
        acted_in_combat = info["screen"] == _COMBAT
        _, reward, terminated, truncated, info = env.step(policy(info))
        trace.append(
            {
                "acted_in_combat": acted_in_combat,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "shaping": info["shaping_terms"],
                "info": info,
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


def test_reward_equals_terminal_plus_potential_shaping() -> None:
    """On every non-terminal step, reward == sum(info['shaping_terms']).

    The shaping is the potential delta gamma * Phi(s') - Phi(s), un-annealed, so a
    non-terminal step's reward is exactly that sum (no terminal component).
    """
    env = StsRunEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    for _ in range(MAX_DRIVE_STEPS):
        _, reward, terminated, truncated, info = env.step(_greedy_action(info))
        if terminated or truncated:
            break
        assert reward == pytest.approx(sum(info["shaping_terms"].values()))
    env.close()


def test_mid_combat_steps_have_no_progress_shaping() -> None:
    """A step that both starts and ends inside a combat carries no floor/act shaping.

    The run-level floor and act potentials are constant while a fight is live, and
    gamma is 1.0 by default, so their potential delta is exactly 0 on a mid-combat
    step. (The act-boss-winning step ends in the overworld, so it is excluded and
    may credit boss_kill.)
    """
    env = StsRunEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    saw_mid_combat = False
    for _ in range(MAX_DRIVE_STEPS):
        acted_in_combat = info["screen"] == _COMBAT
        _, _, terminated, truncated, info = env.step(_greedy_action(info))
        stayed_in_combat = acted_in_combat and info["screen"] == _COMBAT
        if stayed_in_combat and not (terminated or truncated):
            saw_mid_combat = True
            assert info["shaping_terms"]["floor_progress"] == 0.0
            assert info["shaping_terms"]["boss_kill"] == 0.0
        if terminated or truncated:
            break
    assert saw_mid_combat  # the drive spent at least one step inside a fight
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


def test_run_potential_shaping_telescopes_per_term() -> None:
    """Each term's summed shaping over a full drive telescopes to -Phi_term(s_0).

    Potential-based shaping is policy-invariant: with gamma = 1 (the default) the
    undiscounted sum of a term's per-step delta over a trajectory ending at a
    terminal (Phi := 0 there) is Phi_term(terminal) - Phi_term(s_0) = -Phi_term(s_0),
    independent of the path. This is the potential analogue of "credited exactly
    once" and holds across the intervening combats: floor / act progress rewarded
    along the way is returned at the terminal.
    """
    cfg = RewardConfig()
    env = StsRunEnv(reward_config=cfg)  # gamma defaults to 1.0
    _, info0 = env.reset(seed=ACT_CROSSING_SEED)
    phi0 = state_potentials(cfg, run=info0["run"])  # overworld start: no combat term
    trace = _drive(env, info0, _greedy_action)

    assert trace[-1]["terminated"] or trace[-1]["truncated"]  # Phi was cashed out
    for term in SHAPING_TERMS:
        total = sum(rec["shaping"][term] for rec in trace)
        assert total == pytest.approx(-phi0[term], abs=1e-6)
    # Non-vacuous: the run starts in act 1, so the boss potential at s_0 is nonzero
    # and the boss term's total telescopes to a nonzero value, not a trivial 0.
    assert phi0["boss_kill"] != 0.0
    env.close()


def test_truncation_cashes_out_potential_not_true_successor() -> None:
    # Regression (mirror of the combat-env test): on a truncated step Phi(s') := 0,
    # so the per-term shaping is -Phi(s_t), NOT gamma*Phi(s'_true) - Phi(s_t).
    # Reverting `terminated or truncated` to `terminated` in the run adapter would
    # emit the latter and fail this test.
    cfg = RewardConfig()
    # A small step cap so the run truncates a few decisions in; a full run is
    # hundreds of decisions, so it cannot terminate this fast.
    env = StsRunEnv(max_episode_steps=5, reward_config=cfg)  # gamma defaults to 1.0
    _, info = env.reset(seed=REGRESSION_SEED)
    pre_trunc_info = info
    terminated = truncated = False
    for _ in range(MAX_DRIVE_STEPS):
        pre_trunc_info = info  # the decision (s_t) we are about to act on
        _, reward, terminated, truncated, info = env.step(_greedy_action(info))
        if terminated or truncated:
            break

    assert truncated and not terminated
    # Reconstruct Phi(s_t) from the pre-truncation decision's snapshots (combat when a
    # fight is live, else the run view) - the same source the adapter's baseline uses.
    prev_phi = state_potentials(cfg, combat=pre_trunc_info.get("combat"), run=pre_trunc_info["run"])
    # Non-vacuous: act >= 1 at any run state, so the boss potential is nonzero and
    # -Phi(s_t) genuinely differs from the buggy gamma*Phi(s'_true) - Phi(s_t).
    assert prev_phi["boss_kill"] != 0.0
    for name in SHAPING_TERMS:
        assert info["shaping_terms"][name] == pytest.approx(-prev_phi[name])
    assert reward == pytest.approx(sum(info["shaping_terms"].values()))
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
        shaping = sum(rec["shaping"].values())
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

    trace: list[tuple[float, dict[str, float]]] = []
    terminated = truncated = False
    for _ in range(MAX_DRIVE_STEPS):
        # Every step is a genuine decision with a valid observation, and the
        # driver's engine pick maps to an interface index - so a full winning run
        # never hits an unrepresentable decision.
        assert env.observation_space.contains(obs)
        assert info["action_mask"].any()
        idx = driver.action(env, info)
        assert idx is not None, "engine pick has no interface index on this decision"
        obs, reward, terminated, truncated, info = env.step(idx)
        trace.append((reward, info["shaping_terms"]))
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
    assert info["episode"]["r"] == pytest.approx(sum(reward for reward, _ in trace))
    # The terminal +1 is credited exactly once, on the terminal step; every prior
    # step's reward is pure potential shaping (the win-side mirror of
    # test_terminal_reward_added_once_on_terminal_step).
    for k, (reward, shaping) in enumerate(trace):
        terminal_component = reward - sum(shaping.values())
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


# gc.act is 1-based and beating the Act 1 boss advances the run to Act 2, so a
# terminal act >= this means Act 1 was cleared (matches eval's act-clear threshold).
_ACT2 = 2


def test_stop_after_act_terminates_at_act_clear() -> None:
    """stop_after_act=1 ends the episode at the Act 1 boss defeat with a win terminal.

    Driving greedy on ACT_CROSSING_SEED clears the Act 1 boss (gc.act advances to 2).
    With stop_after_act=1 the episode terminates there - terminated (not truncated),
    won True, terminal act >= 2 - instead of continuing into Act 2, and the +1 terminal
    win reward is credited on that step.
    """
    env = StsRunEnv(stop_after_act=1)
    _, info0 = env.reset(seed=ACT_CROSSING_SEED)
    trace = _drive(env, info0, _greedy_action)

    last = trace[-1]
    assert last["terminated"] and not last["truncated"]
    assert last["info"]["won"] is True
    assert last["info"]["act"] >= _ACT2  # cleared Act 1 -> advanced into Act 2
    terminal_component = last["reward"] - sum(last["shaping"].values())
    assert terminal_component == pytest.approx(TERMINAL_WIN_REWARD)
    env.close()


def test_stop_after_act_shortens_episode_vs_full_run() -> None:
    """stop_after_act=1 ends earlier (a win) than the full-run default on the same drive.

    The default (stop_after_act=None) runs the whole episode: greedy on
    ACT_CROSSING_SEED clears Act 1 but dies later, a full-run loss. stop_after_act=1
    stops at the Act 1 clear with a win. Same seed and policy, so the shared prefix is
    byte-identical and the only divergence is the early terminal - this also guards
    that stop_after_act=None is unchanged (the full-run loss still ends where it did).
    """
    full = StsRunEnv()  # stop_after_act defaults to None
    _, full_info0 = full.reset(seed=ACT_CROSSING_SEED)
    full_trace = _drive(full, full_info0, _greedy_action)
    full.close()

    stopped = StsRunEnv(stop_after_act=1)
    _, stopped_info0 = stopped.reset(seed=ACT_CROSSING_SEED)
    stopped_trace = _drive(stopped, stopped_info0, _greedy_action)
    stopped.close()

    # Default full run: continues past the Act 1 clear to a natural (loss) terminal.
    assert full_trace[-1]["info"]["won"] is False
    # Early stop: ends sooner, at the Act 1 clear, as a win.
    assert stopped_trace[-1]["info"]["won"] is True
    assert len(stopped_trace) < len(full_trace)
    # Every pre-terminal step is identical between the two runs (same seed/policy, and
    # the early-terminal branch has not fired yet), so their rewards match up to the
    # step before stop. They differ only on stop's final step (which zeroes the
    # potential and adds the +1 terminal).
    for i in range(len(stopped_trace) - 1):
        assert stopped_trace[i]["reward"] == pytest.approx(full_trace[i]["reward"])


def test_stop_after_act_preserves_shaping_telescoping() -> None:
    """With stop_after_act=1, each term's summed shaping still telescopes to -Phi_term(s_0).

    Potential-based shaping is policy- AND horizon-invariant: ending the episode early
    at the Act 1 clear (Phi := 0 at that terminal) leaves the undiscounted per-term sum
    at Phi_term(terminal) - Phi_term(s_0) = -Phi_term(s_0), exactly as for a full run.
    Guards that the early terminal did not break the telescoping invariant.
    """
    cfg = RewardConfig()
    env = StsRunEnv(reward_config=cfg, stop_after_act=1)  # gamma defaults to 1.0
    _, info0 = env.reset(seed=ACT_CROSSING_SEED)
    phi0 = state_potentials(cfg, run=info0["run"])  # overworld start: no combat term
    trace = _drive(env, info0, _greedy_action)

    assert trace[-1]["terminated"] and not trace[-1]["truncated"]
    assert trace[-1]["info"]["won"] is True  # ended at the Act 1 clear, not a loss
    for term in SHAPING_TERMS:
        total = sum(rec["shaping"][term] for rec in trace)
        assert total == pytest.approx(-phi0[term], abs=1e-6)
    # Non-vacuous: a run starts in act 1, so the boss potential at s_0 is nonzero.
    assert phi0["boss_kill"] != 0.0
    env.close()


@pytest.mark.parametrize("bad", [0, -1])
def test_stop_after_act_rejects_nonpositive(bad: int) -> None:
    """stop_after_act must be a 1-based act index when set; <= 0 fails at construction."""
    with pytest.raises(InterfaceError):
        StsRunEnv(stop_after_act=bad)
