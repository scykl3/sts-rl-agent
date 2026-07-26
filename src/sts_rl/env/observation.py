"""Observation encoder: raw engine state -> the interface's observation dict.

Reads a live ``GameContext`` + ``BattleContext`` and emits the
:data:`~sts_rl.interface.OBS_FIELDS`-shaped observation the agent consumes. All
field widths and id bounds come from :mod:`sts_rl.interface`, so an engine enum
bump propagates here instead of silently desyncing.

One encoder serves both modes. Pass a live ``BattleContext`` for combat, or
``bc=None`` for an overworld (run-mode) state: combat-only fields (hand, piles,
enemies, powers, potions) then stay zero and ``map_context`` is filled from the
run's map. ``map_context`` is left zero during combat.

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
    ACTION_BLOCK_BY_NAME,
    HAND_MAX,
    MAP_CONTEXT_DIM,
    MAX_ENEMIES,
    N_MONSTER_MOVE_IDS,
    N_MONSTER_POWER_IDS,
    N_NODE_TYPES,
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
# engine build. Keyed by index so enum aliases (a second name for the same id)
# collapse to one entry, avoiding a redundant duplicate write per encode. INVALID
# (player id 0) is skipped -- it is the "no status" sentinel, not a real power.
# MonsterStatus has no INVALID member.
_PLAYER_STATUSES: tuple[tuple[int, Any], ...] = tuple(
    {
        int(status): status
        for status in sts.PlayerStatus.__members__.values()
        if status != sts.PlayerStatus.INVALID and int(status) < N_PLAYER_POWER_IDS
    }.items()
)
_MONSTER_STATUSES: tuple[tuple[int, Any], ...] = tuple(
    {
        int(status): status
        for status in sts.MonsterStatus.__members__.values()
        if int(status) < N_MONSTER_POWER_IDS
    }.items()
)

# CardType ids for the hand one-hot features (bound enum: ATTACK/SKILL/POWER).
_CARD_TYPE_ATTACK = int(sts.CardType.ATTACK)
_CARD_TYPE_SKILL = int(sts.CardType.SKILL)
_CARD_TYPE_POWER = int(sts.CardType.POWER)

# Potion slot sentinels the engine uses for "no potion here".
_POTION_EMPTY = sts.Potion.EMPTY_POTION_SLOT
_POTION_INVALID = sts.Potion.INVALID

# --- map_context layout (run mode) -----------------------------------------
# The act map is a grid MAP_COLS wide and MAP_ROWS tall (verified against the
# engine). MAP_SELECT chooses a target column, so the per-column features below
# align 1:1 with that action block: per-column feature j describes the node
# reachable at map column j.
MAP_COLS = ACTION_BLOCK_BY_NAME["MAP_SELECT"].count  # engine map width (x = 0..MAP_COLS-1)
# Grid rows are 0..MAP_ROWS-1 (verified against the engine: get_room_type past the
# top row is INVALID). The act boss is not stored in the grid; it sits above the
# top row and is reached by the edges out of it. MAP_ROWS scales cur_y.
MAP_ROWS = 15
MAX_ACTS = 3  # Acts 1-3 (no Act 4 at Ascension 0); scales act
# floor_num is a cumulative counter across acts (not the per-act row), so it needs its
# own scale: an upper bound on the floors in a 3-act run, giving floor_norm ~[0, 1].
RUN_MAX_FLOOR = 56

# map_context is split into a "current" block (where the run stands) and a
# per-column "next" block (each reachable next-row node). The named widths below
# must sum to MAP_CONTEXT_DIM; the check makes a layout drift a hard error.
_MAP_CUR_ROOM_ONEHOT = N_NODE_TYPES  # current node room type (real types 0..N-1)
_MAP_CUR_POS = 2  # cur_x, cur_y normalized (both negative before the first row)
_MAP_PROGRESS = 2  # act, floor normalized
_MAP_PER_COL_FEATS = 4  # per column: reachable, is_combat, is_elite, room_type_norm
_MAP_CUR_BLOCK = _MAP_CUR_ROOM_ONEHOT + _MAP_CUR_POS + _MAP_PROGRESS
_MAP_NEXT_BLOCK = MAP_COLS * _MAP_PER_COL_FEATS
if _MAP_CUR_BLOCK + _MAP_NEXT_BLOCK != MAP_CONTEXT_DIM:
    raise AssertionError(
        f"map_context layout ({_MAP_CUR_BLOCK} + {_MAP_NEXT_BLOCK}) must sum to "
        f"MAP_CONTEXT_DIM ({MAP_CONTEXT_DIM})"
    )

# Room ids that gate the per-column combat/elite flags (a live enum, not magic ints).
_ROOM_MONSTER = int(sts.Room.MONSTER)
_ROOM_ELITE = int(sts.Room.ELITE)
_ROOM_BOSS = int(sts.Room.BOSS)
# A combat node from the map's perspective: normal fight, elite, or act boss.
_COMBAT_ROOMS = (_ROOM_MONSTER, _ROOM_ELITE, _ROOM_BOSS)


def _empty_obs() -> Obs:
    """Zero-filled observation with every field's interface dtype and shape."""
    return {field.name: np.zeros(field.shape, dtype=field.dtype) for field in OBS_FIELDS}


def _read_status(player: Any, status: Any) -> float:
    """Read one player status as a float, tolerating bit-only powers."""
    try:
        return float(player.getStatus(status))
    except IndexError:
        # Engine quirk: a bit-only status (e.g. BARRICADE) sets its presence bit
        # but has no statusMap entry, so getStatus (statusMap.at) throws. Encode
        # such a binary power as presence 1.0; amount-bearing statuses (STRENGTH,
        # etc.) take the map value, absent ones 0.0.
        return 1.0 if player.hasStatus(status) else 0.0


def encode_observation(gc: Any, bc: Any) -> Obs:
    """Encode a live ``GameContext`` (+ optional ``BattleContext``) into the obs dict.

    ``gc`` supplies run-level scalars (gold, floor, act, ascension, screen) and,
    in run mode, the map. ``bc`` supplies the combat state (player vitals, powers,
    piles, enemies); pass ``bc=None`` for an overworld screen, where combat-only
    fields stay zero and ``map_context`` is filled instead. Reads only; no engine
    mutation.
    """
    obs = _empty_obs()
    # relics and the screen one-hot are read from gc in both modes.
    _fill_relics_multihot(obs, gc)
    _fill_screen_onehot(obs, gc)

    if bc is None:
        _fill_run_scalars(obs, gc)
        _fill_map_context(obs, gc)
        return obs

    player = bc.player
    monsters = bc.monsters

    # Card, monster, and potion ids below are written straight from engine enums
    # with no per-write clamp: validate_engine_enums asserts each enum's max id
    # fits its table (max_id < N) at startup, so a held id is always in bounds.
    # The enemy move id guards explicitly because its raw value can be an
    # out-of-range sentinel (a not-yet-rolled move).

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

    # -- player_powers: dense amount per player status id
    powers = obs["player_powers"]
    for idx, status in _PLAYER_STATUSES:
        powers[idx] = _read_status(player, status)

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

    # map_context stays zero in combat: it is a run-mode feature, filled only when
    # bc is None (overworld encoding).
    return obs


def _fill_relics_multihot(obs: Obs, gc: Any) -> None:
    """Set one bit per owned relic id (guards the INVALID sentinel out of range)."""
    relics = obs["relics_multihot"]
    for relic in gc.relics:
        rid = int(relic.id)
        if 0 <= rid < N_RELIC_IDS:
            relics[rid] = 1.0


def _fill_screen_onehot(obs: Obs, gc: Any) -> None:
    """Set the one-hot bit for the current screen (guards out-of-range screen ids)."""
    screen_idx = int(gc.screen_state)
    if 0 <= screen_idx < N_SCREENS:
        obs["screen_onehot"][screen_idx] = 1.0


def _fill_run_scalars(obs: Obs, gc: Any) -> None:
    """Player scalars for an overworld state (no combat: block/energy/turn are 0)."""
    obs["player_scalars"][:] = (
        gc.cur_hp,
        gc.max_hp,
        0.0,  # block: no combat block out of battle
        0.0,  # energy: refilled at combat start
        gc.gold,
        gc.floor_num,
        gc.ascension,
        0.0,  # turn: combat-only counter
    )


def _fill_map_context(obs: Obs, gc: Any) -> None:
    """Fill ``map_context`` from the run's map: current node + reachable next nodes.

    Layout (sums to ``MAP_CONTEXT_DIM``): a current block -- room-type one-hot,
    normalized ``(x, y)`` position, normalized ``(act, floor)`` -- followed by one
    per-column block for each of the ``MAP_COLS`` map columns, aligned to the
    ``MAP_SELECT`` action index. Each per-column block is
    ``(reachable, is_combat, is_elite, room_type_norm)`` for the node reachable at
    that column in the next row; unreachable columns stay zero.

    Before the first row the engine reports ``cur == (-1, -1)``; the normalized
    position is then negative (a distinct pre-map signal) and the reachable set is
    every column holding a real room in row 0.
    """
    ctx = obs["map_context"]
    spire_map = gc.map
    cur_x = int(gc.cur_map_node_x)
    cur_y = int(gc.cur_map_node_y)

    # current node room type (NONE/INVALID, e.g. pre-map, leave the one-hot zero)
    cur_room_id = int(gc.cur_room)
    if 0 <= cur_room_id < N_NODE_TYPES:
        ctx[cur_room_id] = 1.0

    pos = _MAP_CUR_ROOM_ONEHOT
    ctx[pos] = cur_x / (MAP_COLS - 1)
    ctx[pos + 1] = cur_y / (MAP_ROWS - 1)
    progress = pos + _MAP_CUR_POS
    ctx[progress] = (int(gc.act) - 1) / (MAX_ACTS - 1)
    ctx[progress + 1] = int(gc.floor_num) / RUN_MAX_FLOOR

    # Columns reachable from the current node in the next row. Pre-map (cur_y < 0),
    # any row-0 column holding a real room is a legal first step; otherwise the
    # engine's edge list gives the reachable next-row columns.
    next_row = cur_y + 1
    if cur_y < 0:
        reachable = {
            c for c in range(MAP_COLS) if 0 <= int(spire_map.get_room_type(c, 0)) < N_NODE_TYPES
        }
    else:
        reachable = {int(c) for c in spire_map.edges(cur_x, cur_y)}

    # Edges out of the top grid row lead to the act boss, which is not stored in the
    # grid (next_row is past the grid, so get_room_type would return INVALID). On
    # that boundary the reachable columns are the boss; encode it as BOSS so it keeps
    # its combat signal instead of reading as SHOP (room id 0). Within the grid, read
    # the real next-row room type (the norm guard keeps any stray sentinel at zero).
    on_boss_boundary = next_row >= MAP_ROWS
    base = progress + _MAP_PROGRESS
    for col in reachable:
        room_id = _ROOM_BOSS if on_boss_boundary else int(spire_map.get_room_type(col, next_row))
        slot = base + col * _MAP_PER_COL_FEATS
        ctx[slot] = 1.0  # reachable
        ctx[slot + 1] = 1.0 if room_id in _COMBAT_ROOMS else 0.0
        ctx[slot + 2] = 1.0 if room_id == _ROOM_ELITE else 0.0
        if 0 <= room_id < N_NODE_TYPES:
            ctx[slot + 3] = room_id / (N_NODE_TYPES - 1)


def _fill_pile_ids(out: np.ndarray, pile: Any) -> None:
    """Write up to ``PILE_MAX`` card ids from ``pile`` into ``out`` (rest PAD).

    The draw/discard/exhaust piles are order-agnostic sets downstream, so slot
    order carries no meaning; a pile larger than ``PILE_MAX`` is truncated to the
    fixed width.
    """
    for i in range(min(len(pile), PILE_MAX)):
        out[i] = int(pile[i].id)
