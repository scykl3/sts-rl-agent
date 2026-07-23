"""Decode flat action indices to engine moves and build the legal-action mask.

Both the decode and the mask are derived from a single mapping from a contract
action index to a concrete engine ``search::Action``. That shared mapping is the
safety guarantee: the mask marks an index legal only when the exact action it
maps to passes the engine's ``is_valid_action``, and decode returns that same
action, so an executed action was always validated first. This matters because
the engine's ``execute`` on an invalid action is undefined (it can abort the
process), so an action is never executed without first checking legality.

Coverage in combat: end turn, playing a card (targeted or untargeted), using or
discarding a potion, an in-combat single-card selection, and confirming a
sequential multi-select (exhaust-many / gamble) via CONFIRM_SELECT. A potion is
routed to the targeted or untargeted block by the engine's own
``potion_requires_target``, so nothing about potion targeting is hardcoded here.
The non-combat blocks (card reward, map, shop, rest, event, boss relic, proceed)
belong to a later run-mode adapter and are masked off here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sts_rl.env._engine import slaythespire as sts
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    CHOICE_MAX,
    MAX_ENEMIES,
    PILE_MAX,
    POTION_SLOTS,
    InterfaceError,
    Mask,
)

# The engine requires a card-play action to carry a target index in [0, 5) even
# for untargeted cards, where it is ignored. Slot 0 is always a valid placeholder.
_UNTARGETED_PLACEHOLDER_TARGET = 0

# A potion target index above the enemy range signals "discard" rather than
# "drink" to the engine (it checks target_idx > 5).
_DISCARD_TARGET = 6

# Block start offsets, resolved once from the shared action layout.
_END_TURN = ACTION_BLOCK_BY_NAME["END_TURN"].start
_PLAY_TARGETED = ACTION_BLOCK_BY_NAME["PLAY_CARD_TARGETED"].start
_PLAY_UNTARGETED = ACTION_BLOCK_BY_NAME["PLAY_CARD_UNTARGETED"].start
_USE_POTION_TARGETED = ACTION_BLOCK_BY_NAME["USE_POTION_TARGETED"].start
_USE_POTION_UNTARGETED = ACTION_BLOCK_BY_NAME["USE_POTION_UNTARGETED"].start
_DISCARD_POTION = ACTION_BLOCK_BY_NAME["DISCARD_POTION"].start
_CARD_SELECT = ACTION_BLOCK_BY_NAME["CARD_SELECT"].start
_CONFIRM_SELECT = ACTION_BLOCK_BY_NAME["CONFIRM_SELECT"].start

# Sequential multi-pick card-select tasks: the engine resolves these by
# accumulating single picks (each a SINGLE_CARD_SELECT that re-opens the screen)
# and then confirming with a MULTI_CARD_SELECT carrying the running selection.
# The agent drives them directly: single picks plus the CONFIRM_SELECT action,
# whose engine move is MULTI_CARD_SELECT(card_select_selected_bits).
_MULTI_SELECT_TASKS = (sts.CardSelectTask.EXHAUST_MANY, sts.CardSelectTask.GAMBLE)

# Bound on auto-resolution steps between two agent actions; far above any real
# chain of engine-driven card-select screens, so it only backstops a loop bug.
_RESOLVE_CAP = 32


def decode_action(index: int, bc: Any) -> Any | None:
    """Map a combat action index to an engine ``search::Action``.

    Returns ``None`` for indices this combat adapter does not map (potion and
    non-combat blocks), which the caller treats as illegal. Raises
    :class:`InterfaceError` if ``index`` is outside the action space. The
    returned action is not guaranteed legal; callers must check
    ``is_valid_action`` before executing it.
    """
    if not 0 <= index < ACTION_DIM:
        raise InterfaceError(f"action index {index} out of range [0, {ACTION_DIM})")

    action_type = sts.ActionType
    if index == _END_TURN:
        return sts.Action(action_type.END_TURN)
    if _PLAY_TARGETED <= index < _PLAY_TARGETED + ACTION_BLOCK_BY_NAME["PLAY_CARD_TARGETED"].count:
        offset = index - _PLAY_TARGETED
        hand_slot, enemy = divmod(offset, MAX_ENEMIES)
        return sts.Action(action_type.CARD, hand_slot, enemy)
    if (
        _PLAY_UNTARGETED
        <= index
        < _PLAY_UNTARGETED + ACTION_BLOCK_BY_NAME["PLAY_CARD_UNTARGETED"].count
    ):
        hand_slot = index - _PLAY_UNTARGETED
        return sts.Action(action_type.CARD, hand_slot, _UNTARGETED_PLACEHOLDER_TARGET)
    if (
        _USE_POTION_TARGETED
        <= index
        < _USE_POTION_TARGETED + ACTION_BLOCK_BY_NAME["USE_POTION_TARGETED"].count
    ):
        offset = index - _USE_POTION_TARGETED
        slot, enemy = divmod(offset, MAX_ENEMIES)
        return sts.Action(action_type.POTION, slot, enemy)
    if (
        _USE_POTION_UNTARGETED
        <= index
        < _USE_POTION_UNTARGETED + ACTION_BLOCK_BY_NAME["USE_POTION_UNTARGETED"].count
    ):
        slot = index - _USE_POTION_UNTARGETED
        return sts.Action(action_type.POTION, slot, _UNTARGETED_PLACEHOLDER_TARGET)
    if _DISCARD_POTION <= index < _DISCARD_POTION + ACTION_BLOCK_BY_NAME["DISCARD_POTION"].count:
        slot = index - _DISCARD_POTION
        return sts.Action(action_type.POTION, slot, _DISCARD_TARGET)
    if _CARD_SELECT <= index < _CARD_SELECT + ACTION_BLOCK_BY_NAME["CARD_SELECT"].count:
        return sts.Action(action_type.SINGLE_CARD_SELECT, index - _CARD_SELECT)
    if index == _CONFIRM_SELECT:
        # Confirm a sequential multi-select by applying the engine's running
        # selection. Only legal in an EXHAUST_MANY/GAMBLE state; the caller gates
        # on is_valid_action, so a stray confirm elsewhere is rejected.
        return sts.Action(action_type.MULTI_CARD_SELECT, bc.card_select_selected_bits)
    return None


def _mask_potions(bc: Any, mask: Mask) -> None:
    """Set the legal potion-action bits for the current belt (in-place).

    Each occupied belt slot is routed by the engine's ``potion_requires_target``:
    a targeting potion fills its ``USE_POTION_TARGETED`` sub-indices for legal
    enemies, an untargeted one its ``USE_POTION_UNTARGETED`` index; every occupied
    slot also gets its ``DISCARD_POTION`` index. Empty slots, and slots beyond the
    contract's ``POTION_SLOTS`` cap, contribute nothing. Every bit is gated on
    ``is_valid_action``.
    """
    for slot, potion in enumerate(bc.potions):
        if slot >= POTION_SLOTS:
            break
        if potion in (sts.Potion.EMPTY_POTION_SLOT, sts.Potion.INVALID):
            continue
        if sts.potion_requires_target(potion):
            base = _USE_POTION_TARGETED + slot * MAX_ENEMIES
            for enemy in range(MAX_ENEMIES):
                action = sts.Action(sts.ActionType.POTION, slot, enemy)
                mask[base + enemy] = action.is_valid_action(bc)
        else:
            action = sts.Action(sts.ActionType.POTION, slot, _UNTARGETED_PLACEHOLDER_TARGET)
            mask[_USE_POTION_UNTARGETED + slot] = action.is_valid_action(bc)
        discard = sts.Action(sts.ActionType.POTION, slot, _DISCARD_TARGET)
        mask[_DISCARD_POTION + slot] = discard.is_valid_action(bc)


def build_mask(bc: Any) -> Mask:
    """Return the ``(ACTION_DIM,)`` bool mask of agent-representable legal actions.

    A bit is set only when the action it decodes to passes the engine's
    ``is_valid_action``, so decoding and executing any set index is safe.

    The mask may legitimately be all-False on a *non-terminal* state whose only
    legal engine moves are not representable in the contract action space (a
    pile-select pick beyond ``CHOICE_MAX``). Callers that hand the mask to an
    agent must first call :func:`auto_resolve` to advance past such states; this
    function does not mutate ``bc`` or assert the mask is non-empty.
    """
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)

    if bc.outcome != sts.BattleOutcome.UNDECIDED:
        return mask

    input_state = bc.input_state
    if input_state == sts.InputState.PLAYER_NORMAL:
        end_turn = sts.Action(sts.ActionType.END_TURN)
        mask[_END_TURN] = end_turn.is_valid_action(bc)
        for hand_slot in range(bc.cards.cardsInHand):
            card = bc.cards.hand[hand_slot]
            if card.requiresTarget():
                base = _PLAY_TARGETED + hand_slot * MAX_ENEMIES
                for enemy in range(MAX_ENEMIES):
                    action = sts.Action(sts.ActionType.CARD, hand_slot, enemy)
                    mask[base + enemy] = action.is_valid_action(bc)
            else:
                action = sts.Action(sts.ActionType.CARD, hand_slot, _UNTARGETED_PLACEHOLDER_TARGET)
                mask[_PLAY_UNTARGETED + hand_slot] = action.is_valid_action(bc)
        _mask_potions(bc, mask)
    elif input_state == sts.InputState.CARD_SELECT:
        for choice in range(CHOICE_MAX):
            action = sts.Action(sts.ActionType.SINGLE_CARD_SELECT, choice)
            mask[_CARD_SELECT + choice] = action.is_valid_action(bc)
        # Sequential multi-select tasks also offer a confirm that applies the
        # running selection; it is always legal (an empty selection is allowed).
        if bc.card_select_task in _MULTI_SELECT_TASKS:
            confirm = sts.Action(sts.ActionType.MULTI_CARD_SELECT, bc.card_select_selected_bits)
            mask[_CONFIRM_SELECT] = confirm.is_valid_action(bc)

    return mask


def _fallback_action(bc: Any) -> Any | None:
    """Return an engine action that resolves a state with no representable move.

    Multi-select tasks are agent-representable (single picks plus CONFIRM_SELECT),
    so this only handles a single-select task whose only legal picks lie beyond
    the representable ``CHOICE_MAX`` range: it takes the first engine-valid pick
    across the full pile. Returns ``None`` if no fallback applies.
    """
    if bc.input_state != sts.InputState.CARD_SELECT:
        return None
    for choice in range(PILE_MAX):
        action = sts.Action(sts.ActionType.SINGLE_CARD_SELECT, choice)
        if action.is_valid_action(bc):
            return action
    return None


def auto_resolve(bc: Any) -> int:
    """Advance ``bc`` past non-terminal states with no agent-representable action.

    Executes a fallback action (see :func:`_fallback_action`) until the state has
    a representable legal move or the battle ends, so the agent is never handed
    an all-False mask. Returns the number of fallback steps taken. Raises
    :class:`InterfaceError` if a non-terminal state cannot be resolved (an
    unexpected input state) or the step cap is exceeded. Mutates ``bc``.
    """
    steps = 0
    for _ in range(_RESOLVE_CAP):
        if bc.outcome != sts.BattleOutcome.UNDECIDED or build_mask(bc).any():
            return steps
        action = _fallback_action(bc)
        if action is None or not action.is_valid_action(bc):
            raise InterfaceError(
                f"no representable or resolvable action: input_state={bc.input_state}, "
                f"card_select_task={bc.card_select_task}"
            )
        action.execute(bc)
        steps += 1
    raise InterfaceError(f"auto_resolve exceeded {_RESOLVE_CAP} steps; possible state loop")
