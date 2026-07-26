"""Decode overworld (run-mode) action indices to engine ``GameAction`` moves and
build the legal-action mask, the run-mode sibling of :mod:`sts_rl.env.actions`.

Where the combat module maps interface indices to combat ``search::Action``
moves over a ``BattleContext``, this one maps the overworld action blocks (map,
shop, rest, treasure, event, boss-relic, and the combat reward screen) to engine
``GameAction`` moves over a ``GameContext``.

The mask and decode share one map. :func:`_gameaction_to_index` assigns each
engine-enumerated legal ``GameAction`` its interface index; the mask sets exactly
those indices, and decode returns the enumerated ``GameAction`` for a requested
index. So a set mask bit always corresponds to a move the engine already
enumerated as legal, and decoding it returns that same move -- which the caller
still gates on ``isValidAction`` before executing, since ``execute`` on an
invalid ``GameAction`` is undefined under the engine's asserts.

Reward-screen action types are read out of ``GameAction.bits`` -- the same field
the potion flag and the bound ``idx1``/``idx2`` come from -- so the whole mapping
reads one source, matching the engine's ``GameAction`` bit layout
(``type << 27 | idx2 << 8 | idx1``).

Scope / deferrals:
- Overworld potion use is not represented: the engine's ``getAllActionsInState``
  does not enumerate potion drink/discard on overworld screens, so neither does
  this mask.
- A card-select screen offering more than ``CHOICE_MAX`` candidates (deck-wide
  event selects, large pile searches) exposes only the first ``CHOICE_MAX``; a
  state whose only legal picks lie beyond that is advanced by
  :func:`auto_resolve_overworld`.
- The two-card ``MATCH_AND_KEEP`` event grid is not representable as a single
  index and is auto-resolved.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.run import execute_overworld_action, is_run_over, overworld_actions
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    CHOICE_MAX,
    MAX_REWARD_CARD_GROUPS,
    MAX_REWARD_CARDS_PER_GROUP,
    MAX_REWARD_POTIONS,
    MAX_REWARD_RELICS,
    REWARD_CARD_OFFSET,
    REWARD_GOLD_OFFSET,
    REWARD_KEY_OFFSET,
    REWARD_POTION_OFFSET,
    REWARD_RELIC_OFFSET,
    REWARD_SINGING_BOWL_OFFSET,
    REWARD_SKIP_OFFSET,
    InterfaceError,
    Mask,
)

# RewardsActionType values (engine GameAction.h). Read from GameAction.bits via
# ``(bits >> 27) & 0x7`` -- the same field the potion flag and idx1/idx2 use.
_RT_CARD = 0
_RT_GOLD = 1
_RT_KEY = 2
_RT_POTION = 3
_RT_RELIC = 4
_RT_CARD_REMOVE = 5
_RT_SKIP = 6

# GameAction.bits layout: high bit flags a potion action; the reward type sits in
# bits 27-29; idx1/idx2 are the low two bytes (mirrored by the bound idx1/idx2).
_POTION_ACTION_BIT = 0x80000000
_REWARD_TYPE_SHIFT = 27
_REWARD_TYPE_MASK = 0x7

# Card-reward option index the engine reserves for the Singing Bowl "+2 max HP
# instead of a card" choice (GameAction::SINGING_BOWL_CARD_IDX).
_SINGING_BOWL_OPTION = 5

# Shop sub-layout within SHOP_SELECT: 7 cards, 3 relics, 3 potions, remove, leave.
_SHOP_CARDS = 7
_SHOP_RELICS = 3
_SHOP_POTIONS = 3
_SHOP_RELIC_OFFSET = _SHOP_CARDS
_SHOP_POTION_OFFSET = _SHOP_RELIC_OFFSET + _SHOP_RELICS
_SHOP_REMOVE_OFFSET = _SHOP_POTION_OFFSET + _SHOP_POTIONS
_SHOP_SKIP_OFFSET = _SHOP_REMOVE_OFFSET + 1

# Boss-relic sub-layout within BOSS_RELIC_SELECT: 3 relics then skip.
_BOSS_RELIC_COUNT = 3

# Bound on auto-resolution steps between two agent actions; far above any real
# chain of unrepresentable overworld screens, so it only backstops a loop bug.
_RESOLVE_CAP = 32

# Block starts, resolved once from the shared action layout.
_MAP = ACTION_BLOCK_BY_NAME["MAP_SELECT"]
_SHOP = ACTION_BLOCK_BY_NAME["SHOP_SELECT"]
_REST = ACTION_BLOCK_BY_NAME["REST_SELECT"]
_TREASURE = ACTION_BLOCK_BY_NAME["TREASURE_SELECT"]
_EVENT = ACTION_BLOCK_BY_NAME["EVENT_SELECT"]
_BOSS = ACTION_BLOCK_BY_NAME["BOSS_RELIC_SELECT"]
_REWARD = ACTION_BLOCK_BY_NAME["REWARD_SELECT"]
_CARD_SELECT = ACTION_BLOCK_BY_NAME["CARD_SELECT"]


def _reward_type(bits: int) -> int:
    return (bits >> _REWARD_TYPE_SHIFT) & _REWARD_TYPE_MASK


def _reward_index(bits: int, idx1: int, idx2: int) -> int | None:
    """Interface index for a ``GameAction`` on the combat REWARDS screen, or None."""
    rt = _reward_type(bits)
    if rt == _RT_GOLD:
        # Every gold pile is enumerated at idx1 0, so one slot re-collects them.
        return _REWARD.start + REWARD_GOLD_OFFSET
    if rt == _RT_POTION and idx1 < MAX_REWARD_POTIONS:
        return _REWARD.start + REWARD_POTION_OFFSET + idx1
    if rt == _RT_RELIC and idx1 < MAX_REWARD_RELICS:
        return _REWARD.start + REWARD_RELIC_OFFSET + idx1
    if rt == _RT_KEY:
        return _REWARD.start + REWARD_KEY_OFFSET
    if rt == _RT_CARD:
        if idx2 == _SINGING_BOWL_OPTION:
            return _REWARD.start + REWARD_SINGING_BOWL_OFFSET
        if idx1 < MAX_REWARD_CARD_GROUPS and idx2 < MAX_REWARD_CARDS_PER_GROUP:
            return _REWARD.start + REWARD_CARD_OFFSET + idx1 * MAX_REWARD_CARDS_PER_GROUP + idx2
        return None
    if rt == _RT_SKIP:
        return _REWARD.start + REWARD_SKIP_OFFSET
    return None


def _shop_index(bits: int, idx1: int) -> int | None:
    rt = _reward_type(bits)
    if rt == _RT_CARD and idx1 < _SHOP_CARDS:
        return _SHOP.start + idx1
    if rt == _RT_RELIC and idx1 < _SHOP_RELICS:
        return _SHOP.start + _SHOP_RELIC_OFFSET + idx1
    if rt == _RT_POTION and idx1 < _SHOP_POTIONS:
        return _SHOP.start + _SHOP_POTION_OFFSET + idx1
    if rt == _RT_CARD_REMOVE:
        return _SHOP.start + _SHOP_REMOVE_OFFSET
    if rt == _RT_SKIP:
        return _SHOP.start + _SHOP_SKIP_OFFSET
    return None


def _gameaction_to_index(action: Any, screen: Any) -> int | None:
    """Map an engine ``GameAction`` (legal in ``screen``) to its interface index.

    Returns ``None`` for a legal move the interface cannot represent (overworld
    potion use, a match-and-keep pair, or a card/pile pick beyond its cap); such
    states are advanced by :func:`auto_resolve_overworld`.
    """
    bits = action.bits
    if bits & _POTION_ACTION_BIT:
        return None  # overworld potion use: not enumerated by the engine, not represented here
    idx1 = action.idx1
    idx2 = action.idx2
    screen_state = sts.ScreenState

    if screen == screen_state.MAP_SCREEN:
        return _MAP.start + idx1 if idx1 < _MAP.count else None
    if screen == screen_state.REST_ROOM:
        return _REST.start + idx1 if idx1 < _REST.count else None
    if screen == screen_state.TREASURE_ROOM:
        return _TREASURE.start + idx1 if idx1 < _TREASURE.count else None
    if screen == screen_state.EVENT_SCREEN:
        # Single-option events carry idx2 == 0; a nonzero idx2 is a MATCH_AND_KEEP
        # card pair, which has no single-index representation.
        if idx2 != 0:
            return None
        return _EVENT.start + idx1 if idx1 < _EVENT.count else None
    if screen == screen_state.CARD_SELECT:
        return _CARD_SELECT.start + idx1 if idx1 < CHOICE_MAX else None
    if screen == screen_state.BOSS_RELIC_REWARDS:
        rt = _reward_type(bits)
        if rt == _RT_RELIC and idx1 < _BOSS_RELIC_COUNT:
            return _BOSS.start + idx1
        if rt == _RT_SKIP:
            return _BOSS.start + _BOSS_RELIC_COUNT
        return None
    if screen == screen_state.SHOP_ROOM:
        return _shop_index(bits, idx1)
    if screen == screen_state.REWARDS:
        return _reward_index(bits, idx1, idx2)
    return None


def decode_overworld_action(index: int, gc: Any) -> Any | None:
    """Return the engine ``GameAction`` an overworld action index maps to, or None.

    Re-enumerates the engine's legal moves for the current screen and returns the
    one whose :func:`_gameaction_to_index` equals ``index`` (the first, if several
    map to the same slot, as with gold). Returns ``None`` for an index no legal
    move maps to (illegal, unmapped, or a combat block). Raises
    :class:`InterfaceError` if ``index`` is outside the action space. The returned
    action is not re-validated here; callers must check ``isValidAction`` before
    executing it.
    """
    if not 0 <= index < ACTION_DIM:
        raise InterfaceError(f"action index {index} out of range [0, {ACTION_DIM})")
    screen = gc.screen_state
    for action in overworld_actions(gc):
        if _gameaction_to_index(action, screen) == index:
            return action
    return None


def build_overworld_mask(gc: Any) -> Mask:
    """Return the ``(ACTION_DIM,)`` bool mask of representable legal overworld moves.

    A bit is set only for an engine-enumerated legal ``GameAction`` that maps to
    an interface index, so decoding and executing any set index is safe. May be
    all-False on a non-terminal screen whose only legal moves are unrepresentable
    (match-and-keep, a card-select beyond ``CHOICE_MAX``); callers advance past
    such states with :func:`auto_resolve_overworld` before handing the mask to an
    agent. Does not mutate ``gc`` or assert the mask is non-empty.
    """
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    if is_run_over(gc):
        return mask
    screen = gc.screen_state
    for action in overworld_actions(gc):
        index = _gameaction_to_index(action, screen)
        if index is not None:
            mask[index] = True
    return mask


def auto_resolve_overworld(gc: Any) -> int:
    """Advance ``gc`` past non-terminal screens with no representable overworld move.

    Executes the first engine-legal move until the screen has a representable
    legal action or the run ends, so the agent is never handed an all-False mask
    (e.g. a match-and-keep grid or a card-select whose only picks lie beyond
    ``CHOICE_MAX``). Returns the number of fallback steps taken. Raises
    :class:`InterfaceError` if a non-terminal screen offers no engine move or the
    step cap is exceeded. Mutates ``gc``.
    """
    steps = 0
    for _ in range(_RESOLVE_CAP):
        if is_run_over(gc) or build_overworld_mask(gc).any():
            return steps
        actions = overworld_actions(gc)
        if not actions:
            raise InterfaceError(
                f"no representable or resolvable overworld action at screen {gc.screen_state}"
            )
        # Route through the legality-gated executor rather than execute() directly,
        # for defense-in-depth against the undefined-execute abort this module avoids.
        execute_overworld_action(gc, actions[0])
        steps += 1
    raise InterfaceError(
        f"auto_resolve_overworld exceeded {_RESOLVE_CAP} steps; possible state loop"
    )
