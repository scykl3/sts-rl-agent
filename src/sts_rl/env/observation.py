"""Observation encoder: raw engine state -> the interface's observation dict.

Reads a live ``GameContext`` + ``BattleContext`` and emits the
:data:`~sts_rl.interface.OBS_FIELDS`-shaped observation the agent consumes. All
field widths and id bounds come from :mod:`sts_rl.interface`, so an engine enum
bump propagates here instead of silently desyncing.

One encoder serves both modes. Pass a live ``BattleContext`` for combat, or
``bc=None`` for an overworld (run-mode) state: combat-only fields (hand, piles,
enemies, powers) then stay zero, while ``map_context`` is filled from the run's
map, ``deck_ids`` from the run deck, the potion belt from the overworld, and the
``reward_*`` fields on a REWARDS screen. ``map_context``, ``deck_ids``, and the
``reward_*`` fields are left zero during combat; the potion belt and ``keys_act``
(act plus owned ruby / emerald / sapphire keys) are filled in both modes.

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
    CHOICE_MAX,
    DECK_MAX,
    HAND_MAX,
    MAP_CONTEXT_DIM,
    MAX_BOSS_RELICS,
    MAX_ENEMIES,
    MAX_REWARD_CARD_GROUPS,
    MAX_REWARD_CARDS_PER_GROUP,
    MAX_REWARD_POTIONS,
    MAX_REWARD_RELICS,
    MAX_SHOP_CARDS,
    MAX_SHOP_POTIONS,
    MAX_SHOP_RELICS,
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

# Empty-slot marker for reward_relic_ids. Unlike cards / potions, RelicId 0 (AKABEKO)
# is a real relic, so PAD 0 cannot mark "empty" here; the relic INVALID id does. It
# equals N_RELIC_IDS (one past the last real relic) -- the field's id_high and the
# column the encoder drops -- so assert that coupling holds (raise, not bare assert, so
# it survives -O), mirroring the map_context layout guard below.
_RELIC_INVALID = int(sts.RelicId.INVALID)
if _RELIC_INVALID != N_RELIC_IDS:
    raise AssertionError(
        f"relic empty-slot sentinel RelicId.INVALID ({_RELIC_INVALID}) must equal "
        f"N_RELIC_IDS ({N_RELIC_IDS}); reward_relic_ids id_high and the encoder's "
        "dropped INVALID column both depend on it"
    )

# Relic-offer id fields, all defaulting to the INVALID empty sentinel (above) rather
# than PAD 0: RelicId 0 (AKABEKO) is a real relic, so an all-zero relic-offer field
# would read as "AKABEKO offered in every slot". _empty_obs seeds each; the encoder
# drops the INVALID column so an empty slot contributes nothing.
_RELIC_OFFER_FIELDS = ("reward_relic_ids", "shop_relic_ids", "boss_relic_ids")

# The combat/elite/chest REWARDS screen: the only screen whose rewardsContainer
# holds the live offered rewards (openCombatRewardScreen sets both together). The
# separate BOSS_RELIC_REWARDS screen is covered by its own action block, not here.
_SCREEN_REWARDS = int(sts.ScreenState.REWARDS)

# The deck-wide / pile card-select screen (event removes and transforms, large pile
# searches). Its screen_state_info.to_select_cards holds the offered candidates,
# order-aligned with the CARD_SELECT action block; PAD on every other screen.
_SCREEN_CARD_SELECT = int(sts.ScreenState.CARD_SELECT)

# The shop screen (SHOP_ROOM) and the act-boss relic reward screen
# (BOSS_RELIC_REWARDS). screen_state_info exposes the shop grid and the three offered
# boss relics on these screens; the shop / boss id and price fields stay empty on
# every other screen, so the screen guards make them meaningful exactly when the
# SHOP_SELECT / BOSS_RELIC_SELECT blocks are legal.
_SCREEN_SHOP = int(sts.ScreenState.SHOP_ROOM)
_SCREEN_BOSS_RELIC = int(sts.ScreenState.BOSS_RELIC_REWARDS)

# Shop.prices layout (engine Shop.h): cards [0..6], relics [7..9], potions [10..12].
# A slot's price is read from the same index the SHOP_SELECT action buys --
# isValidShopAction gates a card / relic / potion buy on prices[idx1] /
# prices[7+idx1] / prices[10+idx1] != -1 -- so the observation slot and the action
# slot stay aligned.
_SHOP_PRICE_RELIC_BASE = MAX_SHOP_CARDS
_SHOP_PRICE_POTION_BASE = MAX_SHOP_CARDS + MAX_SHOP_RELICS

# Gold-price normalization for the shop real fields (item prices and remove cost).
# Shop prices span roughly 20-300 gold (cards / potions cheaper, relics dearer);
# dividing by a fixed high-end price keeps the normalized value O(1) without clipping,
# matching the raw-but-scaled style of the map_context progress features. An absent
# price (engine -1 / None) encodes as 0.0; the paired id / multihot slot already marks
# the slot empty (PAD / INVALID), so a 0.0 price is never confused with a free item.
SHOP_PRICE_SCALE = 300.0

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
    """Empty observation with every field's interface dtype and shape.

    Fields are zero-filled (PAD_ID 0 for id fields, 0.0 for real / unit), EXCEPT the
    relic-offer fields (reward / shop / boss relic ids), which are seeded with the
    relic INVALID sentinel: RelicId 0 (AKABEKO) is a real relic, so an all-zero
    relic-offer field would read as "AKABEKO offered in every slot". INVALID marks "no
    relic here" -- an empty offer, and every non-offer / combat state where the field
    is left untouched -- which the encoder maps to a zero contribution.
    """
    obs: Obs = {field.name: np.zeros(field.shape, dtype=field.dtype) for field in OBS_FIELDS}
    for name in _RELIC_OFFER_FIELDS:
        obs[name].fill(_RELIC_INVALID)
    return obs


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
    # relics, the screen one-hot, and keys/act are read from gc in both modes.
    _fill_relics_multihot(obs, gc)
    _fill_screen_onehot(obs, gc)
    _fill_keys_act(obs, gc)

    if bc is None:
        _fill_run_scalars(obs, gc)
        _fill_map_context(obs, gc)
        _fill_reward_ids(obs, gc)
        _fill_card_select_ids(obs, gc)
        _fill_deck_ids(obs, gc)
        _fill_overworld_potions(obs, gc)
        _fill_shop(obs, gc)
        _fill_boss_relic_ids(obs, gc)
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

    # -- card-select candidates: set only when an in-combat card-select is active
    # (Headbutt / Exhume / Discovery / ...); PAD otherwise, slot-aligned to CARD_SELECT.
    _fill_combat_card_select_ids(obs, bc)

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
    # any row-0 column holding a real room is a legal first step; within the grid the
    # engine's edge list gives the reachable next-row columns; at the boss node there
    # are none (see below).
    next_row = cur_y + 1
    if cur_y < 0:
        reachable = {
            c for c in range(MAP_COLS) if 0 <= int(spire_map.get_room_type(c, 0)) < N_NODE_TYPES
        }
    elif 0 <= cur_x < MAP_COLS and cur_y < MAP_ROWS:
        reachable = {int(c) for c in spire_map.edges(cur_x, cur_y)}
    else:
        # The engine parks the run on the act boss node at cur_y == MAP_ROWS (above
        # the grid) through the boss reward / relic screens. edges() indexes the grid
        # and throws there; the run leaves via the act transition, not a map choice,
        # so no next-row column is reachable.
        reachable = set()

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


def _fill_reward_ids(obs: Obs, gc: Any) -> None:
    """Fill ``reward_card_ids`` / ``reward_relic_ids`` / ``reward_potion_ids`` from
    the live REWARDS screen, slot-aligned with the ``REWARD_SELECT`` action block.

    Only the combat/elite/chest REWARDS screen carries a live ``rewardsContainer``;
    on every other screen these fields stay PAD (0), so the guard makes them
    meaningful exactly when the ``REWARD_SELECT`` block is legal.

    Cards keep their choice-group structure: the engine offers one group per card
    reward (a second appears with Prayer Wheel), and each group lets the player
    take one card. Slot ``group * MAX_REWARD_CARDS_PER_GROUP + j`` holds the j-th
    card of group ``g``, matching the group layout of the ``REWARD_SELECT`` card
    sub-block. Relics and potions are flat: one slot per offered item.
    """
    if int(gc.screen_state) != _SCREEN_REWARDS:
        return
    rewards = gc.screen_state_info.rewards_container

    # Cards: pre-filtered of INVALID by the binding, so ids are written directly
    # (like hand_ids) -- startup enum validation guarantees each fits its table.
    card_ids = obs["reward_card_ids"]
    groups = rewards.cards
    for g in range(min(len(groups), MAX_REWARD_CARD_GROUPS)):
        group = groups[g]
        base = g * MAX_REWARD_CARDS_PER_GROUP
        for j in range(min(len(group), MAX_REWARD_CARDS_PER_GROUP)):
            card_ids[base + j] = int(group[j].id)

    # Relics: RelicId 0 (AKABEKO) is a REAL relic, so an empty slot is the relic INVALID
    # sentinel (RelicId.INVALID == N_RELIC_IDS, a legal value of this field per its
    # id_high), NOT PAD 0 the way cards / potions mark empty. _empty_obs seeded every
    # slot with that sentinel, so write only real offered relics (0..N_RELIC_IDS-1,
    # AKABEKO included) as their id; an unoffered or out-of-range slot stays INVALID
    # (empty), which the encoder drops so it contributes nothing.
    relic_ids = obs["reward_relic_ids"]
    relics = rewards.relics
    for i in range(min(len(relics), MAX_REWARD_RELICS)):
        rid = int(relics[i])
        if 0 <= rid < N_RELIC_IDS:
            relic_ids[i] = rid

    # Potions: skip the engine's "no potion" sentinels (same as the belt encoder);
    # written per-index, so a skipped slot stays PAD.
    potion_ids = obs["reward_potion_ids"]
    potions = rewards.potions
    for i in range(min(len(potions), MAX_REWARD_POTIONS)):
        potion = potions[i]
        if potion == _POTION_EMPTY or potion == _POTION_INVALID:
            continue
        potion_ids[i] = int(potion)


def _fill_card_select_ids(obs: Obs, gc: Any) -> None:
    """Fill ``card_select_ids`` from the live CARD_SELECT screen, slot-aligned with
    the ``CARD_SELECT`` action block.

    Only the deck-wide / pile card-select screen (event removes and transforms,
    large pile searches) carries a live ``to_select_cards`` candidate list; on every
    other screen this field stays PAD (0), so the guard makes it meaningful exactly
    when the ``CARD_SELECT`` block is legal. The binding preserves candidate order,
    so ``to_select_cards[i]`` is the card the overworld ``CARD_SELECT`` action at
    index ``i`` picks (see :mod:`sts_rl.env.run_actions`); candidates past
    ``CHOICE_MAX`` are truncated to the field width.

    This is the overworld (``GameContext``) card-select path: a combat
    (``BattleContext``) card-select exposes only its selected bitmask, not its
    candidate list, so a combat card-select leaves this field PAD.

    Candidate ids are written directly (like ``hand_ids`` / ``reward_card_ids``):
    they are real deck cards, and startup enum validation guarantees each fits its
    embedding table.
    """
    if int(gc.screen_state) != _SCREEN_CARD_SELECT:
        return
    card_select_ids = obs["card_select_ids"]
    candidates = gc.screen_state_info.to_select_cards
    for i in range(min(len(candidates), CHOICE_MAX)):
        card_select_ids[i] = int(candidates[i].id)


def _fill_combat_card_select_ids(obs: Obs, bc: Any) -> None:
    """Fill ``card_select_ids`` from an active in-combat card-select, slot-aligned
    with the ``CARD_SELECT`` action block.

    The combat counterpart of :func:`_fill_card_select_ids`. Cards a combat select
    picks FROM live in a pile (hand / draw / discard / exhaust) or are generated
    (Discovery / Codex), so the engine returns ``(pick_index, card_id)`` pairs where
    ``pick_index`` is the absolute pile index a ``SINGLE_CARD_SELECT`` action
    consumes -- the same index the ``CARD_SELECT`` block uses (see
    :mod:`sts_rl.env.actions`). Each candidate is written at its own index (not
    compacted), so ``card_select_ids[i]`` is the card that ``SINGLE_CARD_SELECT(i)``
    picks. The list is empty whenever no card-select is active, so a normal combat
    step leaves the field PAD; indices past ``CHOICE_MAX`` are dropped to the width.
    """
    card_select_ids = obs["card_select_ids"]
    for idx, card_id in bc.card_select_candidate_ids():
        if 0 <= idx < CHOICE_MAX:
            card_select_ids[idx] = card_id


def _fill_shop(obs: Obs, gc: Any) -> None:
    """Fill the shop id + price fields from the live SHOP_ROOM screen, slot-aligned
    with the ``SHOP_SELECT`` action block.

    Only the SHOP_ROOM screen carries a live ``shop``; on every other screen these
    fields stay empty (PAD id / INVALID relic / 0.0 price), so the guard makes them
    meaningful exactly when the ``SHOP_SELECT`` block is legal.

    A slot's price is the engine's ``prices`` entry for that slot (cards 0..6, relics
    7..9, potions 10..12); ``-1`` marks a bought / empty slot, which
    ``isValidShopAction`` also rejects, so such a slot is left empty here (its paired
    id stays PAD / INVALID and its price 0.0). Prices present but unaffordable are
    shown -- the agent should see an item it cannot yet afford -- so the fill gates on
    presence (``price != -1``), not on gold. Prices normalize by ``SHOP_PRICE_SCALE``.

    Cards: the ``cards`` accessor is INVALID-filtered, but the engine never sets a
    shop card INVALID on purchase (``buyCard`` only clears the price unless The
    Courier restocks the slot), so all seven slots stay populated and position-aligned
    with ``prices[0..6]`` and the ``SHOP_SELECT`` card slots in live play.
    """
    if int(gc.screen_state) != _SCREEN_SHOP:
        return
    shop = gc.screen_state_info.shop
    prices = shop.prices

    # Cards: embedded per slot downstream; show only still-purchasable slots
    # (price != -1), a bought slot stays PAD. Written directly (like reward_card_ids)
    # -- startup enum validation guarantees each id fits card_embed.
    shop_card_ids = obs["shop_card_ids"]
    shop_card_prices = obs["shop_card_prices"]
    cards = shop.cards
    for i in range(min(len(cards), MAX_SHOP_CARDS)):
        price = prices[i]
        if price < 0:
            continue
        shop_card_ids[i] = int(cards[i].id)
        shop_card_prices[i] = price / SHOP_PRICE_SCALE

    # Relics: a fixed 3 slots, order-agnostic multihot downstream. Empty / bought
    # slots (price -1) or the INVALID sentinel keep the seeded INVALID empty marker,
    # which the encoder drops; only a real offered relic (0..N_RELIC_IDS-1, AKABEKO
    # included) is written, so an empty slot is never a phantom AKABEKO.
    shop_relic_ids = obs["shop_relic_ids"]
    shop_relic_prices = obs["shop_relic_prices"]
    relics = shop.relics
    for i in range(min(len(relics), MAX_SHOP_RELICS)):
        price = prices[_SHOP_PRICE_RELIC_BASE + i]
        rid = int(relics[i])
        if price < 0 or not (0 <= rid < N_RELIC_IDS):
            continue
        shop_relic_ids[i] = rid
        shop_relic_prices[i] = price / SHOP_PRICE_SCALE

    # Potions: a fixed 3 slots, embedded per slot downstream. Skip the engine's empty
    # / invalid potion sentinels and bought slots (price -1); a skipped slot stays PAD.
    shop_potion_ids = obs["shop_potion_ids"]
    shop_potion_prices = obs["shop_potion_prices"]
    potions = shop.potions
    for i in range(min(len(potions), MAX_SHOP_POTIONS)):
        price = prices[_SHOP_PRICE_POTION_BASE + i]
        potion = potions[i]
        if price < 0 or potion == _POTION_EMPTY or potion == _POTION_INVALID:
            continue
        shop_potion_ids[i] = int(potion)
        shop_potion_prices[i] = price / SHOP_PRICE_SCALE

    # Card-removal service cost: Optional[int], None (engine -1) once used this visit;
    # left 0.0 then. The remove action carries no id slot, so the price alone marks it.
    remove_cost = shop.remove_cost
    if remove_cost is not None and remove_cost >= 0:
        obs["shop_remove_cost"][0] = remove_cost / SHOP_PRICE_SCALE


def _fill_boss_relic_ids(obs: Obs, gc: Any) -> None:
    """Fill ``boss_relic_ids`` from the live BOSS_RELIC_REWARDS screen (the three
    offered act-boss relics), mirroring the reward-relic offer field.

    Only the BOSS_RELIC_REWARDS screen carries live ``boss_relics``; on every other
    screen the field stays at the INVALID sentinel (seeded in ``_empty_obs``), which
    the encoder drops. Written per slot as a real relic id (an order-agnostic multihot
    downstream); RelicId 0 (AKABEKO) is a real relic, so empty is the INVALID marker,
    not PAD 0. The three boss relics are free, so there is no paired price field.
    """
    if int(gc.screen_state) != _SCREEN_BOSS_RELIC:
        return
    relic_ids = obs["boss_relic_ids"]
    relics = gc.screen_state_info.boss_relics
    for i in range(min(len(relics), MAX_BOSS_RELICS)):
        rid = int(relics[i])
        if 0 <= rid < N_RELIC_IDS:
            relic_ids[i] = rid


def _fill_keys_act(obs: Obs, gc: Any) -> None:
    """Fill ``keys_act`` = [act, ruby, emerald, sapphire] for the current state.

    ``act`` complements ``player_scalars`` (which carries floor but not act); the
    three owned-key flags (ruby == red, emerald == green, sapphire == blue) track
    Act-3 boss-door progress. Known in both overworld and combat, so this runs for
    both. A raw passthrough block (like ``player_scalars``): written as-is, not
    embedded.
    """
    obs["keys_act"][:] = (
        gc.act,
        1.0 if gc.red_key else 0.0,
        1.0 if gc.green_key else 0.0,
        1.0 if gc.blue_key else 0.0,
    )


def _fill_deck_ids(obs: Obs, gc: Any) -> None:
    """Fill ``deck_ids`` from the overworld deck (order-agnostic id set), PAD tail.

    Each card's id is written up to ``DECK_MAX``; a deck larger than the cap is
    truncated to the fixed width. Overworld-only: in combat the draw / discard /
    hand / exhaust piles already cover every card, so the field stays PAD there.
    Card ids are written directly (like ``hand_ids`` / pile ids) -- startup enum
    validation guarantees each fits ``card_embed``.
    """
    deck_ids = obs["deck_ids"]
    deck = gc.deck
    for i in range(min(len(deck), DECK_MAX)):
        deck_ids[i] = int(deck[i].id)


def _fill_overworld_potions(obs: Obs, gc: Any) -> None:
    """Fill ``potion_ids`` / ``potion_usable`` from the overworld potion belt.

    The combat encoder reads ``bc.potions``; out of combat the belt lives on the
    ``GameContext``, which the binding does not expose as a direct attribute. The
    read-only ``getNNRepresentation`` accessor is the bound path that surfaces it
    (the belt up to the current capacity), so an overworld obs shows held potions
    instead of reading all-PAD. Skips the engine's empty / invalid slot sentinels,
    exactly like the combat and reward-belt fills; a skipped slot stays PAD.
    """
    belt = sts.getNNRepresentation(gc).potions
    potion_ids = obs["potion_ids"]
    potion_usable = obs["potion_usable"]
    for i in range(min(len(belt), POTION_SLOTS)):
        potion = int(belt[i])
        if potion == int(_POTION_EMPTY) or potion == int(_POTION_INVALID):
            continue
        potion_ids[i] = potion
        potion_usable[i] = 1.0


def _fill_pile_ids(out: np.ndarray, pile: Any) -> None:
    """Write up to ``PILE_MAX`` card ids from ``pile`` into ``out`` (rest PAD).

    The draw/discard/exhaust piles are order-agnostic sets downstream, so slot
    order carries no meaning; a pile larger than ``PILE_MAX`` is truncated to the
    fixed width.
    """
    for i in range(min(len(pile), PILE_MAX)):
        out[i] = int(pile[i].id)
