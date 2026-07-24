"""Observation encoder: raw engine state -> the interface's observation dict.

Reads a live ``GameContext`` + ``BattleContext`` and emits the
:data:`~sts_rl.interface.OBS_FIELDS`-shaped observation the agent consumes. All
field widths and id bounds come from :mod:`sts_rl.interface`, so an engine enum
bump propagates here instead of silently desyncing.

Scope is combat: ``map_context`` (a run-mode feature) is left zero and filled
when run-mode screens land. Every other field is populated from the live state.

Conventions:

- Empty slots (unused hand/pile/potion/enemy positions) stay at ``PAD_ID`` 0 for
  id fields and 0.0 for real/unit fields, so the returned dict is always a valid
  member of :func:`~sts_rl.env.spaces.build_observation_space`.
- Power vectors are dense amounts indexed by the engine's status id: entry ``j``
  holds ``getStatus(status_j)`` (0 when absent), matching the embedding-table
  layout the agent's encoder expects.
- When intents are hidden (Runic Dome), each enemy's move id and predicted
  intent damage/hits are masked to 0 and ``enemy_intent_hidden`` is set; HP,
  block, and powers stay visible, matching the player's information set.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sts_rl.env._engine import slaythespire as sts
from sts_rl.interface import (
    HAND_MAX,
    MAX_ENEMIES,
    N_MONSTER_MOVE_IDS,
    N_MONSTER_POWER_IDS,
    N_PLAYER_POWER_IDS,
    N_RELIC_IDS,
    N_SCREENS,
    OBS_FIELDS,
    PAD_ID,
    PILE_MAX,
    POTION_SLOTS,
    Obs,
)

# Precomputed (table_index, status) pairs, filtered to those that fit their
# embedding table. Built once at import: the enum membership is fixed for a given
# engine build. INVALID (player id 0) is skipped -- it is the "no status"
# sentinel, not a real power. MonsterStatus has no INVALID member.
_PLAYER_STATUSES: tuple[tuple[int, Any], ...] = tuple(
    (int(status), status)
    for status in sts.PlayerStatus.__members__.values()
    if status != sts.PlayerStatus.INVALID and int(status) < N_PLAYER_POWER_IDS
)
_MONSTER_STATUSES: tuple[tuple[int, Any], ...] = tuple(
    (int(status), status)
    for status in sts.MonsterStatus.__members__.values()
    if int(status) < N_MONSTER_POWER_IDS
)

# CardType ids for the hand one-hot features (bound enum: ATTACK/SKILL/POWER).
_CARD_TYPE_ATTACK = int(sts.CardType.ATTACK)
_CARD_TYPE_SKILL = int(sts.CardType.SKILL)
_CARD_TYPE_POWER = int(sts.CardType.POWER)

# Potion slot sentinels the engine uses for "no potion here".
_POTION_EMPTY = sts.Potion.EMPTY_POTION_SLOT
_POTION_INVALID = sts.Potion.INVALID


def _empty_obs() -> Obs:
    """Zero-filled observation with every field's contract dtype and shape."""
    return {field.name: np.zeros(field.shape, dtype=field.dtype) for field in OBS_FIELDS}


def encode_observation(gc: Any, bc: Any) -> Obs:
    """Encode a live ``GameContext``/``BattleContext`` into the observation dict.

    ``gc`` supplies run-level scalars (gold, floor, act, ascension, screen);
    ``bc`` supplies the combat state (player vitals, powers, piles, enemies).
    Reads only; no engine mutation.
    """
    obs = _empty_obs()
    player = bc.player
    monsters = bc.monsters

    # -- player_scalars: hp_cur, hp_max, block, energy, gold, floor, ascension, turn
    obs["player_scalars"][:] = (
        player.curHp,
        player.maxHp,
        player.block,
        player.energy,
        gc.gold,
        gc.floor_num,
        gc.ascension,
        bc.turn,
    )

    # -- relics_multihot: one bit per owned relic id
    relics = obs["relics_multihot"]
    for relic in gc.relics:
        rid = int(relic.id)
        if 0 <= rid < N_RELIC_IDS:
            relics[rid] = 1.0

    # -- player_powers: dense amount per player status id
    powers = obs["player_powers"]
    for idx, status in _PLAYER_STATUSES:
        powers[idx] = player.getStatus(status)

    # -- potions: id + usable per belt slot (empty slots stay PAD/0)
    potion_ids = obs["potion_ids"]
    potion_usable = obs["potion_usable"]
    potions = bc.potions
    for i in range(min(len(potions), POTION_SLOTS)):
        potion = potions[i]
        if potion == _POTION_EMPTY or potion == _POTION_INVALID:
            continue
        potion_ids[i] = int(potion)
        potion_usable[i] = 1.0

    # -- hand: id + per-card features (upgraded, cost, type one-hot, ethereal)
    hand_ids = obs["hand_ids"]
    hand_feats = obs["hand_feats"]
    hand = bc.cards.hand
    for i in range(min(len(hand), HAND_MAX)):
        card = hand[i]
        hand_ids[i] = int(card.id)
        card_type = int(card.getType())
        hand_feats[i, :] = (
            float(card.upgraded),
            float(card.cost),
            1.0 if card_type == _CARD_TYPE_ATTACK else 0.0,
            1.0 if card_type == _CARD_TYPE_SKILL else 0.0,
            1.0 if card_type == _CARD_TYPE_POWER else 0.0,
            1.0 if card.isEthereal() else 0.0,
        )

    # -- draw / discard / exhaust piles: order-agnostic id sets, PAD-padded
    _fill_pile_ids(obs["draw_ids"], bc.cards.drawPile)
    _fill_pile_ids(obs["discard_ids"], bc.cards.discardPile)
    _fill_pile_ids(obs["exhaust_ids"], bc.cards.exhaustPile)

    # -- enemies: id, scalars, intent, powers, alive (per fixed slot)
    intents_hidden = bool(bc.intents_hidden)
    enemy_ids = obs["enemy_ids"]
    enemy_scalars = obs["enemy_scalars"]
    enemy_move_ids = obs["enemy_move_ids"]
    enemy_intent_hidden = obs["enemy_intent_hidden"]
    enemy_powers = obs["enemy_powers"]
    enemy_alive = obs["enemy_alive"]
    for i in range(min(monsters.monsterCount, MAX_ENEMIES)):
        monster = monsters[i]
        enemy_ids[i] = int(monster.id)
        enemy_alive[i] = 1.0 if monster.isAlive() else 0.0

        intent_damage, intent_hits = monster.get_move_base_damage(bc)
        # moveHistory is a fixed-size array; index 0 is the current committed move
        # (0 / INVALID when none is rolled yet).
        move_id = int(monster.moveHistory[0])
        if intents_hidden:
            enemy_intent_hidden[i] = 1.0
            intent_damage = 0
            intent_hits = 0
            move_id = PAD_ID
        if 0 <= move_id < N_MONSTER_MOVE_IDS:
            enemy_move_ids[i] = move_id
        enemy_scalars[i, :] = (
            monster.curHp,
            monster.maxHp,
            monster.block,
            intent_damage,
            intent_hits,
        )
        for idx, status in _MONSTER_STATUSES:
            enemy_powers[i, idx] = monster.getStatus(status)

    # -- screen_onehot: current overworld/battle screen
    screen_idx = int(gc.screen_state)
    if 0 <= screen_idx < N_SCREENS:
        obs["screen_onehot"][screen_idx] = 1.0

    # map_context stays zero: it is a run-mode feature, filled with non-combat
    # screens later.
    return obs


def _fill_pile_ids(out: np.ndarray, pile: Any) -> None:
    """Write up to ``PILE_MAX`` card ids from ``pile`` into ``out`` (rest PAD).

    The draw/discard/exhaust piles are order-agnostic sets downstream, so slot
    order carries no meaning; a pile larger than ``PILE_MAX`` is truncated to the
    fixed width.
    """
    for i in range(min(len(pile), PILE_MAX)):
        out[i] = int(pile[i].id)
