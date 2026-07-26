"""Tests for the legal-action mask and action decode.

Safety-critical: a mask that marks an illegal action legal would let the adapter
hand an invalid move to the engine, whose behavior on an invalid action is
undefined. The core guarantee here is that every index the mask marks legal
decodes to an action the engine confirms valid, checked by fuzzing random-legal
rollouts across many seeds.

Requires the built engine; skips cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env.actions import auto_resolve, build_mask, decode_action
from sts_rl.env.engine import start_combat
from sts_rl.interface import ACTION_BLOCK_BY_NAME, ACTION_DIM, MAX_ENEMIES

# Wide enough to exercise the sequential multi-select (EXHAUST_MANY) card-select
# states, which first appear well past the low seeds.
FUZZ_SEEDS = range(220)
MAX_STEPS_PER_COMBAT = 500
REGRESSION_SEED = 42
# Seeds whose first combat reaches an EXHAUST_MANY card-select, exercised as an
# agent-driven pick-then-confirm sequence.
MULTI_SELECT_SEEDS = (150, 207, 344, 349)

_TARGETED = ACTION_BLOCK_BY_NAME["PLAY_CARD_TARGETED"]
_UNTARGETED = ACTION_BLOCK_BY_NAME["PLAY_CARD_UNTARGETED"]


def test_initial_mask_is_valid_and_offers_play_and_end_turn() -> None:
    _, bc = start_combat(seed=REGRESSION_SEED)
    mask = build_mask(bc)
    assert mask.shape == (ACTION_DIM,)
    assert mask.dtype == np.bool_
    assert mask[ACTION_BLOCK_BY_NAME["END_TURN"].start]  # end turn always legal in normal play
    assert mask.sum() > 1  # at least one card is also playable on turn 1


def test_targeted_and_untargeted_blocks_are_mutually_exclusive_per_slot() -> None:
    """A card routes to exactly one block: targeted iff the engine says it targets."""
    _, bc = start_combat(seed=REGRESSION_SEED)
    mask = build_mask(bc)
    for slot in range(bc.cards.cardsInHand):
        targeted_bits = mask[
            _TARGETED.start + slot * MAX_ENEMIES : _TARGETED.start + (slot + 1) * MAX_ENEMIES
        ]
        untargeted_bit = mask[_UNTARGETED.start + slot]
        requires_target = bc.cards.hand[slot].requiresTarget()
        if requires_target:
            assert not untargeted_bit
        else:
            assert not targeted_bits.any()


def test_no_false_positives_over_random_legal_rollouts() -> None:
    """Every masked-legal index decodes to an engine-valid action, across many states.

    Drives uniform-random legal play to terminal on each seed, auto-resolving
    unrepresentable states exactly as the env does. Because the engine aborts on
    an invalid execute, completing thousands of steps without a crash, plus the
    explicit per-index validity assertion and the non-empty-mask assertion,
    is the zero-false-positive evidence.
    """
    import slaythespire as sts

    states_checked = 0
    terminal_reached = 0
    for seed in FUZZ_SEEDS:
        rng = np.random.default_rng(seed)
        _, bc = start_combat(seed=seed)
        for _ in range(MAX_STEPS_PER_COMBAT):
            auto_resolve(bc)
            if bc.outcome != sts.BattleOutcome.UNDECIDED:
                terminal_reached += 1
                break
            mask = build_mask(bc)
            legal = np.flatnonzero(mask)
            assert legal.size > 0, f"empty mask on non-terminal state (seed {seed})"
            for index in legal:
                action = decode_action(int(index), bc)
                assert action is not None
                assert action.is_valid_action(bc), f"mask false positive at index {index}"
            states_checked += 1
            choice = int(rng.choice(legal))
            decode_action(choice, bc).execute(bc)
    assert states_checked > 100
    assert terminal_reached > 0


def _reach_multi_select(bc, rng):
    """Drive random legal play until an EXHAUST_MANY/GAMBLE state, or return None."""
    import slaythespire as sts

    for _ in range(MAX_STEPS_PER_COMBAT):
        auto_resolve(bc)
        if bc.outcome != sts.BattleOutcome.UNDECIDED:
            return None
        if bc.input_state == sts.InputState.CARD_SELECT and bc.card_select_task in (
            sts.CardSelectTask.EXHAUST_MANY,
            sts.CardSelectTask.GAMBLE,
        ):
            return bc
        legal = np.flatnonzero(build_mask(bc))
        decode_action(int(rng.choice(legal)), bc).execute(bc)
    return None


@pytest.mark.parametrize("seed", MULTI_SELECT_SEEDS)
def test_multi_select_exposes_single_picks_and_confirm(seed: int) -> None:
    """In an EXHAUST_MANY/GAMBLE state the agent sees single picks plus a confirm,
    a picked index becomes illegal on the next mask while the rest stay pickable,
    and confirm decodes to a valid MULTI_CARD_SELECT applying the running selection.

    The exclusion invariant (an already-picked index is masked off on the following
    step) is the load-bearing safety property for agent-driven multi-select: the
    mask and the execute-gate both call ``is_valid_action`` on the same move, so a
    broken exclusion would be a false positive invisible to the fuzz rollout. This
    test re-reads the mask *after* a pick to catch that directly.
    """
    import slaythespire as sts

    from sts_rl.interface import ACTION_BLOCK_BY_NAME

    bc = _reach_multi_select(start_combat(seed=seed)[1], np.random.default_rng(0))
    if bc is None:
        pytest.skip("did not reach a multi-select state for this seed")

    card_select = ACTION_BLOCK_BY_NAME["CARD_SELECT"]
    confirm = ACTION_BLOCK_BY_NAME["CONFIRM_SELECT"].start
    mask = build_mask(bc)

    assert mask[confirm]  # confirm is available in a multi-select
    pickable = np.flatnonzero(mask[card_select.start : card_select.stop])
    # Two distinct picks are needed to exercise pick -> re-check -> pick.
    if pickable.size < 2:
        pytest.skip("multi-select state has fewer than two pickable cards")
    # Confirm decodes to a MULTI applying the current selection, and is valid.
    assert decode_action(confirm, bc).is_valid_action(bc)

    # Pick the first card. Multi-select is fully agent-driven, so no auto-resolution
    # happens between picks; the state stays in the same CARD_SELECT task.
    first, second = int(pickable[0]), int(pickable[1])
    decode_action(card_select.start + first, bc).execute(bc)
    assert auto_resolve(bc) == 0

    # Re-read the mask: the picked index is now illegal, while every other
    # previously-legal pick and the confirm stay legal.
    mask_after = build_mask(bc)
    assert not mask_after[card_select.start + first]  # already picked -> excluded
    assert mask_after[confirm]
    for other in pickable:
        if int(other) != first:
            assert mask_after[card_select.start + int(other)]

    # Pick a second card, then confirm; the task resolves and combat continues.
    decode_action(card_select.start + second, bc).execute(bc)
    assert auto_resolve(bc) == 0
    confirm_action = decode_action(confirm, bc)
    assert confirm_action.is_valid_action(bc)
    confirm_action.execute(bc)
    assert bc.input_state != sts.InputState.CARD_SELECT or bc.card_select_task not in (
        sts.CardSelectTask.EXHAUST_MANY,
        sts.CardSelectTask.GAMBLE,
    )


def test_potions_are_routed_by_targeting_and_are_decodable() -> None:
    """A targeting potion fills its targeted block, an untargeted one its untargeted
    index; both are discardable, and every set potion bit decodes to a valid action."""
    import slaythespire as sts

    from sts_rl.interface import ACTION_BLOCK_BY_NAME

    gc, _ = start_combat(seed=REGRESSION_SEED)
    gc.obtain_potion(sts.Potion.FIRE_POTION)  # requires a target
    gc.obtain_potion(sts.Potion.BLOCK_POTION)  # untargeted
    bc = gc.create_battle_context()
    mask = build_mask(bc)

    targeted = ACTION_BLOCK_BY_NAME["USE_POTION_TARGETED"].start
    untargeted = ACTION_BLOCK_BY_NAME["USE_POTION_UNTARGETED"].start
    discard = ACTION_BLOCK_BY_NAME["DISCARD_POTION"].start

    # Slot 0 = Fire (targeted): a targeted bit is set, the untargeted bit is not.
    assert mask[targeted : targeted + MAX_ENEMIES].any()
    assert not mask[untargeted + 0]
    # Slot 1 = Block (untargeted): the untargeted bit is set, no targeted bits.
    assert mask[untargeted + 1]
    assert not mask[targeted + MAX_ENEMIES : targeted + 2 * MAX_ENEMIES].any()
    # Both occupied slots are discardable.
    assert mask[discard + 0] and mask[discard + 1]

    for index in np.flatnonzero(mask):
        action = decode_action(int(index), bc)
        assert action is not None and action.is_valid_action(bc)


def test_empty_belt_sets_no_potion_bits() -> None:
    from sts_rl.interface import ACTION_BLOCK_BY_NAME

    _, bc = start_combat(seed=REGRESSION_SEED)  # combats start with an empty belt
    mask = build_mask(bc)
    for name in ("USE_POTION_TARGETED", "USE_POTION_UNTARGETED", "DISCARD_POTION"):
        block = ACTION_BLOCK_BY_NAME[name]
        assert not mask[block.start : block.stop].any()


def test_decode_rejects_out_of_range_index() -> None:
    from sts_rl.interface import InterfaceError

    _, bc = start_combat(seed=REGRESSION_SEED)
    with pytest.raises(InterfaceError):
        decode_action(ACTION_DIM, bc)
    with pytest.raises(InterfaceError):
        decode_action(-1, bc)


def test_noncombat_indices_do_not_map_in_combat() -> None:
    """Out-of-combat blocks (map, shop, proceed, ...) decode to None during combat."""
    _, bc = start_combat(seed=REGRESSION_SEED)
    for name in (
        "MAP_SELECT",
        "SHOP_SELECT",
        "REST_SELECT",
        "TREASURE_SELECT",
        "PROCEED",
        "REWARD_SELECT",
    ):
        block = ACTION_BLOCK_BY_NAME[name]
        assert decode_action(block.start, bc) is None
