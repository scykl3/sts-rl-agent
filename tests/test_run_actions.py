"""Tests for overworld (run-mode) action decode and the legal-action mask.

Safety-critical, mirroring the combat mask suite: a mask that marks an illegal
overworld move legal would let the adapter hand an invalid ``GameAction`` to the
engine, whose behavior on an invalid action is undefined. The core guarantee is
that every index the mask marks legal decodes to a ``GameAction`` the engine
confirms valid, checked by fuzzing agent-driven runs across seeds and screens.

Requires the built engine; skips cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.actions import auto_resolve, build_mask, decode_action
from sts_rl.env.engine import start_combat
from sts_rl.env.run import describe_action, is_run_over, overworld_actions, start_run
from sts_rl.env.run_actions import (
    _RT_CARD,
    _RT_CARD_REMOVE,
    _RT_GOLD,
    _RT_KEY,
    _RT_POTION,
    _RT_RELIC,
    _RT_SKIP,
    auto_resolve_overworld,
    build_overworld_mask,
    decode_overworld_action,
)
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    REWARD_CARD_OFFSET,
    REWARD_GOLD_OFFSET,
    REWARD_KEY_OFFSET,
    REWARD_POTION_OFFSET,
    REWARD_RELIC_OFFSET,
    REWARD_SINGING_BOWL_OFFSET,
    REWARD_SKIP_OFFSET,
    InterfaceError,
)

REGRESSION_SEED = 42
# A spread of seeds driven to (roughly) full runs; competent battle play reaches
# the reward / shop / rest / event / boss screens a random combat policy rarely would.
RUN_SEEDS = (1, 42, 2024)
# Per-decision battle-search budget: enough to win Act 1 fights and progress a
# run, small enough to keep the suite quick.
BATTLE_SIM_COUNT = 40
# Safety caps on the drive loops.
MAX_RUN_STEPS = 4000

_EVENT = ACTION_BLOCK_BY_NAME["EVENT_SELECT"]
_PROCEED = ACTION_BLOCK_BY_NAME["PROCEED"]
# Engine GameAction::getValidEventSelectBits enumerates an event's options as a
# bitmask; the highest bit any event sets is option index 6 (Cursed Tome's final
# phase returns 0x3 << 5, options 5 and 6). EVENT_SELECT slots at or beyond index 7
# are therefore dead headroom the engine can never make legal.
_ENGINE_MAX_EVENT_OPTION_IDX = 6
# Modest combat fuzz for the dead-slot guard: combat build_mask only sets combat
# blocks, so a small random-legal sweep suffices to catch a relayout collision.
DEAD_SLOT_COMBAT_SEEDS = range(30)
MAX_STEPS_PER_COMBAT = 500


def _battle_agent() -> object:
    agent = sts.Agent()
    agent.simulation_count_base = BATTLE_SIM_COUNT
    return agent


def _assert_reward_placement(gc: object) -> None:
    """Each reward action lands in its designated REWARD_SELECT sub-slot.

    The shared mask/decode map cannot catch a transposed sub-offset on its own -- a
    wrong-but-in-bounds index still decodes to a valid move -- so cross-check the
    engine's own action description (independent of the module's bit logic) against
    the slot each index falls in.
    """
    start = ACTION_BLOCK_BY_NAME["REWARD_SELECT"].start
    for index in np.flatnonzero(build_overworld_mask(gc)):
        rel = int(index) - start
        desc = describe_action(gc, decode_overworld_action(int(index), gc))
        if desc.startswith("gold"):
            assert rel == REWARD_GOLD_OFFSET
        elif desc.startswith("potion"):
            assert REWARD_POTION_OFFSET <= rel < REWARD_RELIC_OFFSET
        elif desc.startswith("relic"):
            assert REWARD_RELIC_OFFSET <= rel < REWARD_KEY_OFFSET
        elif desc.startswith("key"):
            assert rel == REWARD_KEY_OFFSET
        elif desc.startswith("card"):
            # Card choices (incl. the Singing Bowl option at idx2 5) share the card range.
            assert REWARD_CARD_OFFSET <= rel <= REWARD_SINGING_BOWL_OFFSET
        elif desc.startswith("skip"):
            assert rel == REWARD_SKIP_OFFSET


def test_neow_start_offers_event_options() -> None:
    """The opening Neow screen masks legal options inside the EVENT_SELECT block."""
    gc = start_run(seed=REGRESSION_SEED)
    assert gc.screen_state == sts.ScreenState.EVENT_SCREEN
    mask = build_overworld_mask(gc)
    legal = np.flatnonzero(mask)
    assert legal.size >= 1
    # Every legal Neow index lives in the EVENT_SELECT block and decodes to a
    # valid engine move.
    for index in legal:
        assert _EVENT.start <= index < _EVENT.stop
        action = decode_overworld_action(int(index), gc)
        assert action is not None
        assert action.isValidAction(gc)


def test_no_overworld_mask_in_battle() -> None:
    """In a battle screen the overworld mask is all-False (combat is handled elsewhere)."""
    gc = start_run(seed=REGRESSION_SEED)
    for _ in range(MAX_RUN_STEPS):
        if gc.screen_state == sts.ScreenState.BATTLE:
            break
        auto_resolve_overworld(gc)
        if is_run_over(gc) or gc.screen_state == sts.ScreenState.BATTLE:
            break
        legal = np.flatnonzero(build_overworld_mask(gc))
        decode_overworld_action(int(legal[0]), gc).execute(gc)
    assert gc.screen_state == sts.ScreenState.BATTLE
    assert overworld_actions(gc) == ()
    assert not build_overworld_mask(gc).any()


def test_reward_type_constants_match_engine() -> None:
    """The hand-copied RewardsActionType values match the engine enum.

    Reward/shop/boss decode reads the type from GameAction.bits against these
    constants; pinning them against the live enum makes a future engine reorder
    fail here rather than mis-decode silently on a path the fuzz may not reach.
    """
    rt = sts.RewardsActionType
    assert _RT_CARD == int(rt.CARD)
    assert _RT_GOLD == int(rt.GOLD)
    assert _RT_KEY == int(rt.KEY)
    assert _RT_POTION == int(rt.POTION)
    assert _RT_RELIC == int(rt.RELIC)
    assert _RT_CARD_REMOVE == int(rt.CARD_REMOVE)
    assert _RT_SKIP == int(rt.SKIP)


def test_decode_out_of_range_raises() -> None:
    gc = start_run(seed=REGRESSION_SEED)
    with pytest.raises(InterfaceError):
        decode_overworld_action(ACTION_DIM, gc)
    with pytest.raises(InterfaceError):
        decode_overworld_action(-1, gc)


def test_run_navigation_no_false_positives() -> None:
    """Every masked overworld index decodes to an engine-valid move, across screens.

    Drives agent-played runs (MCTS battles, random-legal overworld choices) so the
    walk passes through map, event, reward, shop, rest, treasure, and boss-relic
    screens. Because the engine aborts on an invalid ``execute``, completing many
    steps without a crash, plus the per-index validity assertion and the
    non-empty-mask assertion, is the zero-false-positive evidence.
    """
    agent = _battle_agent()
    seen_screens: set[str] = set()
    states_checked = 0
    for seed in RUN_SEEDS:
        gc = start_run(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(MAX_RUN_STEPS):
            if is_run_over(gc):
                break
            if gc.screen_state == sts.ScreenState.BATTLE:
                agent.playout_battle(gc)
                continue
            auto_resolve_overworld(gc)
            if is_run_over(gc) or gc.screen_state == sts.ScreenState.BATTLE:
                continue
            mask = build_overworld_mask(gc)
            legal = np.flatnonzero(mask)
            assert legal.size > 0, f"empty overworld mask on {gc.screen_state} (seed {seed})"
            seen_screens.add(gc.screen_state.name)
            if gc.screen_state == sts.ScreenState.REWARDS:
                _assert_reward_placement(gc)
            for index in legal:
                action = decode_overworld_action(int(index), gc)
                assert action is not None, f"masked index {index} decodes to None"
                assert action.isValidAction(gc), f"mask false positive at index {index}"
            states_checked += 1
            decode_overworld_action(int(rng.choice(legal)), gc).execute(gc)

    assert states_checked > 50
    # Neow (an event) and the map are traversed on every run; competent battle
    # play (search seeded from seed+floor) deterministically reaches the
    # post-combat reward screen, so requiring REWARDS also guarantees the
    # reward-placement parity check above actually ran.
    assert {"EVENT_SCREEN", "MAP_SCREEN", "REWARDS"} <= seen_screens


def _dead_action_indices() -> set[int]:
    """Action indices no screen can ever legally map to, derived from the live layout.

    PROCEED maps to no screen at all; EVENT_SELECT is sized past the engine's max
    event option index, leaving trailing headroom slots. Both ranges come from
    ACTION_BLOCK_BY_NAME, so a relayout moves them automatically (no literals).
    """
    dead = set(range(_PROCEED.start, _PROCEED.stop))
    dead |= set(range(_EVENT.start + _ENGINE_MAX_EVENT_OPTION_IDX + 1, _EVENT.stop))
    return dead


def test_dead_action_slots_never_masked_overworld() -> None:
    """PROCEED and the EVENT_SELECT dead headroom are never set legal in overworld masks.

    PROCEED is a reserved slot no screen maps to, and EVENT_SELECT is sized past the
    engine's max event option index, so both are dead today. This guards them: if a
    future layout shift made a dead index overlap a live action, the navigation fuzz
    below (map / event / reward / shop / rest / treasure / boss screens) would set it
    and this fails. Every index is derived from the interface, so the guard tracks a
    relayout rather than pinning current literals.
    """
    dead = _dead_action_indices()
    assert dead  # sanity: the layout has a PROCEED slot and EVENT_SELECT headroom
    agent = _battle_agent()
    seen_event_option = False
    for seed in RUN_SEEDS:
        gc = start_run(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(MAX_RUN_STEPS):
            if is_run_over(gc):
                break
            if gc.screen_state == sts.ScreenState.BATTLE:
                agent.playout_battle(gc)
                continue
            auto_resolve_overworld(gc)
            if is_run_over(gc) or gc.screen_state == sts.ScreenState.BATTLE:
                continue
            legal = np.flatnonzero(build_overworld_mask(gc))
            assert legal.size > 0, f"empty overworld mask on {gc.screen_state} (seed {seed})"
            for index in legal:
                assert (
                    int(index) not in dead
                ), f"dead slot {int(index)} masked legal on {gc.screen_state} (seed {seed})"
                if _EVENT.start <= int(index) < _EVENT.stop:
                    seen_event_option = True
            decode_overworld_action(int(rng.choice(legal)), gc).execute(gc)
    # Neow (the opening screen) is an event, so EVENT_SELECT is always exercised; this
    # keeps the headroom half of the guard from passing vacuously.
    assert seen_event_option


def test_dead_action_slots_never_masked_combat() -> None:
    """The overworld-only dead slots are never set in combat masks either.

    Combat build_mask sets only combat blocks, so PROCEED and the EVENT_SELECT
    headroom must stay unset throughout a combat. Fuzzing random-legal combats guards
    against a future relayout that let a combat index collide with one of them.
    """
    dead = _dead_action_indices()
    states_checked = 0
    for seed in DEAD_SLOT_COMBAT_SEEDS:
        rng = np.random.default_rng(seed)
        _, bc = start_combat(seed=seed)
        for _ in range(MAX_STEPS_PER_COMBAT):
            auto_resolve(bc)
            if bc.outcome != sts.BattleOutcome.UNDECIDED:
                break
            legal = np.flatnonzero(build_mask(bc))
            assert legal.size > 0, f"empty combat mask on non-terminal state (seed {seed})"
            for index in legal:
                assert (
                    int(index) not in dead
                ), f"dead slot {int(index)} masked legal in combat (seed {seed})"
            states_checked += 1
            decode_action(int(rng.choice(legal)), bc).execute(bc)
    assert states_checked > 0


def test_overworld_mask_rechecks_isvalidaction(monkeypatch) -> None:
    """build_overworld_mask gates each enumerated action on isValidAction (combat parity).

    getAllActionsInState already returns only legal moves, so the recheck is
    defense-in-depth, matching the combat build_mask contract. A stubbed action whose
    isValidAction is False must be dropped from the mask even though it was
    enumerated, proving the per-bit gate is actually applied (not just trusted).
    """
    from sts_rl.env import run_actions

    map_start = ACTION_BLOCK_BY_NAME["MAP_SELECT"].start

    class _FakeAction:
        """Minimal stand-in exposing the fields _gameaction_to_index / the recheck read."""

        def __init__(self, valid: bool) -> None:
            self.bits = 0  # not a potion action; reward-type bits unused on MAP_SCREEN
            self.idx1 = 0  # maps to MAP_SELECT slot 0
            self.idx2 = 0
            self._valid = valid

        def isValidAction(self, gc: object) -> bool:  # noqa: N802 - mirrors the engine API
            return self._valid

    class _FakeGC:
        screen_state = sts.ScreenState.MAP_SCREEN
        outcome = sts.GameOutcome.UNDECIDED  # is_run_over -> False

    # A valid enumerated action on the map screen maps to MAP_SELECT slot 0 and is set.
    monkeypatch.setattr(run_actions, "overworld_actions", lambda gc: (_FakeAction(True),))
    mask_valid = build_overworld_mask(_FakeGC())
    assert mask_valid[map_start]
    assert mask_valid.sum() == 1

    # The same enumerated action, now isValidAction False, is dropped by the recheck.
    monkeypatch.setattr(run_actions, "overworld_actions", lambda gc: (_FakeAction(False),))
    mask_invalid = build_overworld_mask(_FakeGC())
    assert not mask_invalid.any()
