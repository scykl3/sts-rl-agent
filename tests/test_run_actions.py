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
from sts_rl.env.run import describe_action, is_run_over, overworld_actions, start_run
from sts_rl.env.run_actions import (
    _PROCEED,
    _RT_CARD,
    _RT_CARD_REMOVE,
    _RT_GOLD,
    _RT_KEY,
    _RT_POTION,
    _RT_RELIC,
    _RT_SKIP,
    _gameaction_to_index,
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


class _StubGameAction:
    """Minimal stand-in for an engine ``GameAction`` for mask-logic unit tests.

    Carries only the fields the decode/mask logic reads (``bits``/``idx1``/``idx2``
    and ``isValidAction``), so a bit can be enumerated without reaching the state
    where the engine would naturally offer it.
    """

    def __init__(self, idx1: int = 0, idx2: int = 0, bits: int = 0, valid: bool = True) -> None:
        self.idx1 = idx1
        self.idx2 = idx2
        self.bits = bits
        self._valid = valid

    def isValidAction(self, gc: object) -> bool:  # noqa: N802 - matches engine method name
        return self._valid


class _StubGC:
    def __init__(self, screen: object) -> None:
        self.screen_state = screen


def test_overworld_mask_gates_on_isvalidaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mapped action that fails ``isValidAction`` is not set legal.

    ``getAllActionsInState`` already pre-filters by legality, so the per-bit
    re-check is defense-in-depth (matching combat ``build_mask`` and the gate in
    ``execute_overworld_action``); this pins that the mask builder itself drops a
    bit whose action reports invalid.
    """
    from sts_rl.env import run_actions as ra

    map_start = ACTION_BLOCK_BY_NAME["MAP_SELECT"].start
    gc = _StubGC(sts.ScreenState.MAP_SCREEN)
    monkeypatch.setattr(ra, "is_run_over", lambda _gc: False)

    monkeypatch.setattr(ra, "overworld_actions", lambda _gc: (_StubGameAction(idx1=0, valid=True),))
    assert build_overworld_mask(gc)[map_start]

    monkeypatch.setattr(
        ra, "overworld_actions", lambda _gc: (_StubGameAction(idx1=0, valid=False),)
    )
    assert not build_overworld_mask(gc).any()


def test_event_option_slots_all_representable() -> None:
    """Event options across the whole EVENT_SELECT block map to distinct slots.

    Several events use option indices above 3 (GOLDEN_IDOL, DESIGNER_IN_SPIRE, and
    CURSED_TOME, which reaches 6), so options 4..6 are legal-capable, not dead. This
    guards against re-tightening the cap on the mistaken belief that only 0..3 occur.
    """
    screen = sts.ScreenState.EVENT_SCREEN
    for idx1 in range(_EVENT.count):
        assert _gameaction_to_index(_StubGameAction(idx1=idx1), screen) == _EVENT.start + idx1
    # An option index at/beyond the block count is unrepresentable (returns None).
    assert _gameaction_to_index(_StubGameAction(idx1=_EVENT.count), screen) is None
    # A two-card MATCH_AND_KEEP pick (idx2 != 0) has no single-index form.
    assert _gameaction_to_index(_StubGameAction(idx1=0, idx2=1), screen) is None


def test_proceed_slot_guard_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_overworld_mask`` asserts the reserved PROCEED slot is never set.

    No ``_gameaction_to_index`` branch emits PROCEED, so it can never be legal; the
    assertion makes a future mapping regression that emitted it fail here instead of
    handing the policy an index that decodes to nothing. Simulate that regression by
    forcing the mapper onto the PROCEED slot and confirm the guard trips.
    """
    from sts_rl.env import run_actions as ra

    gc = _StubGC(sts.ScreenState.MAP_SCREEN)
    monkeypatch.setattr(ra, "is_run_over", lambda _gc: False)
    monkeypatch.setattr(ra, "overworld_actions", lambda _gc: (_StubGameAction(),))
    monkeypatch.setattr(ra, "_gameaction_to_index", lambda _action, _screen: _PROCEED.start)
    with pytest.raises(AssertionError):
        build_overworld_mask(gc)
