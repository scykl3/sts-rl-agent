"""Tests for the raw overworld (run-mode) engine wrapper.

These require the native engine module to be built (scripts/build_engine.sh). The
guard below skips the whole module cleanly when it is absent, so the suite stays
green on machines without a native build (for example CI).
"""

from __future__ import annotations

import pytest

# Skip the entire module unless the compiled engine is importable. Importing
# _engine raises EngineNotBuiltError (an ImportError) when the native module is
# absent; catch it and skip at module load. (pytest.importorskip only skips on a
# missing module, not on an ImportError raised during import, so guard explicitly.)
try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.engine import EngineError
from sts_rl.env.run import (
    RunSnapshot,
    describe_action,
    execute_overworld_action,
    is_run_over,
    overworld_actions,
    read_run,
    start_run,
)

# Ironclad Ascension 0 starting values, fixed by the character definition.
IRONCLAD_MAX_HP = 80
START_ACT = 1
START_FLOOR = 0

# A spread of seeds that should each start a valid run.
REACHABLE_SEEDS = (1, 7, 42, 123, 2024)

# The engine screen name for an in-combat state; overworld enumeration is empty there.
BATTLE_SCREEN = "BATTLE"

# Safety cap when driving the overworld to the first battle: a normal path is a
# handful of actions, so this only backstops a state machine that never fights.
MAX_DRIVE_ACTIONS = 500

# An action index far above any screen's option count (map/shop/rest top out at 7),
# so it is illegal in every non-terminal overworld state. idx1 == bits & 0xFF.
OUT_OF_RANGE_ACTION_BITS = 200


def test_start_run_reads_raw_state() -> None:
    """A seeded run starts non-terminal at full HP with a legal overworld action."""
    gc = start_run(seed=42)
    snap = read_run(gc)

    assert isinstance(snap, RunSnapshot)
    assert snap.outcome == "UNDECIDED"
    assert not is_run_over(gc)
    assert snap.player_hp == snap.player_max_hp == IRONCLAD_MAX_HP
    assert snap.act == START_ACT
    assert snap.floor == START_FLOOR
    assert snap.ascension == 0
    assert snap.gold >= 0
    assert snap.deck_size > 0
    # The run opens on an agent decision, so at least one legal move is offered.
    assert snap.action_count >= 1
    assert snap.screen and snap.screen != BATTLE_SCREEN


def test_start_run_is_deterministic() -> None:
    """The same seed produces an identical initial run readout."""
    assert read_run(start_run(seed=42)) == read_run(start_run(seed=42))


def test_start_run_propagates_ascension() -> None:
    """The ascension argument reaches the engine and is reflected in the readout."""
    assert read_run(start_run(seed=42, ascension=10)).ascension == 10


@pytest.mark.parametrize("seed", REACHABLE_SEEDS)
def test_various_seeds_start(seed: int) -> None:
    """Several seeds each start a valid, drivable run (structural invariants only)."""
    gc = start_run(seed=seed)
    snap = read_run(gc)
    assert not is_run_over(gc)
    assert 0 < snap.player_hp <= snap.player_max_hp
    actions = overworld_actions(gc)
    assert len(actions) == snap.action_count >= 1


def test_overworld_actions_are_all_legal() -> None:
    """Every enumerated overworld action passes the engine's own legality check."""
    gc = start_run(seed=42)
    actions = overworld_actions(gc)
    assert actions
    assert all(action.isValidAction(gc) for action in actions)
    # Each action carries a non-empty engine description.
    assert all(describe_action(gc, action) for action in actions)


def test_execute_advances_state() -> None:
    """Executing a legal overworld action mutates the run into a new state."""
    gc = start_run(seed=42)
    before = read_run(gc)
    execute_overworld_action(gc, overworld_actions(gc)[0])
    after = read_run(gc)
    assert after != before


def test_execute_illegal_action_raises() -> None:
    """An out-of-range action is rejected before it reaches the engine."""
    gc = start_run(seed=42)
    illegal = sts.GameAction(OUT_OF_RANGE_ACTION_BITS)
    assert not illegal.isValidAction(gc)
    with pytest.raises(EngineError):
        execute_overworld_action(gc, illegal)


def test_can_drive_overworld_to_first_battle() -> None:
    """Enumerate-then-execute drives the overworld screens into the first combat.

    Taking the first legal action at each screen walks Neow -> map -> ... -> a
    monster room, exercising the enumeration and safety-gated execution across
    every intervening non-combat screen exactly as a run adapter would.
    """
    gc = start_run(seed=42)
    for _ in range(MAX_DRIVE_ACTIONS):
        if gc.screen_state.name == BATTLE_SCREEN:
            break
        assert not is_run_over(gc)
        actions = overworld_actions(gc)
        assert actions, f"no legal action at screen {gc.screen_state}"
        execute_overworld_action(gc, actions[0])
    assert gc.screen_state.name == BATTLE_SCREEN
    # A battle state exposes no overworld actions; combat moves are handled elsewhere.
    assert overworld_actions(gc) == ()
