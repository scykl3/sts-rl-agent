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
from sts_rl.env.run import is_run_over, overworld_actions, start_run
from sts_rl.env.run_actions import (
    auto_resolve_overworld,
    build_overworld_mask,
    decode_overworld_action,
)
from sts_rl.interface import ACTION_BLOCK_BY_NAME, ACTION_DIM, InterfaceError

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
            for index in legal:
                action = decode_overworld_action(int(index), gc)
                assert action is not None, f"masked index {index} decodes to None"
                assert action.isValidAction(gc), f"mask false positive at index {index}"
            states_checked += 1
            decode_overworld_action(int(rng.choice(legal)), gc).execute(gc)

    assert states_checked > 50
    # Neow (an event) and the map are traversed on every run; competent battle
    # play also reaches the post-combat reward screen.
    assert {"EVENT_SCREEN", "MAP_SCREEN"} <= seen_screens
    assert len(seen_screens) >= 3
