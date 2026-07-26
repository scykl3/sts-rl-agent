"""Tests for the observation encoder (raw engine state -> interface obs dict).

Requires the built engine; skips cleanly otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.adapter import StsEnv
from sts_rl.env.engine import start_combat
from sts_rl.env.observation import (
    MAP_COLS,
    MAP_ROWS,
    _MAP_CUR_BLOCK,
    _MAP_CUR_POS,
    _MAP_CUR_ROOM_ONEHOT,
    _MAP_PER_COL_FEATS,
    _empty_obs,
    _fill_map_context,
    _fill_reward_ids,
    encode_observation,
)
from sts_rl.env.run import execute_overworld_action, overworld_actions, start_run
from sts_rl.env.spaces import build_observation_space
from sts_rl.interface import (
    MAX_REWARD_CARD_GROUPS,
    MAX_REWARD_CARDS_PER_GROUP,
    MAX_REWARD_POTIONS,
    MAX_REWARD_RELICS,
    N_NODE_TYPES,
    N_RELIC_IDS,
    OBS_FIELDS,
    OBS_FIELD_BY_NAME,
)

# reward_* fields: populated only on the REWARDS screen, PAD everywhere else.
_REWARD_ID_FIELDS = ("reward_card_ids", "reward_relic_ids", "reward_potion_ids")

REGRESSION_SEED = 42

# Overworld screen name and a safety cap for driving the run to its first map.
_MAP_SCREEN = "MAP_SCREEN"
_MAX_DRIVE = 50

# Fields populated only in combat; every one must stay zero in an overworld obs.
_COMBAT_ONLY_FIELDS = (
    "player_powers",
    "potion_ids",
    "potion_usable",
    "hand_ids",
    "hand_feats",
    "draw_ids",
    "discard_ids",
    "exhaust_ids",
    "enemy_ids",
    "enemy_scalars",
    "enemy_move_ids",
    "enemy_intent_hidden",
    "enemy_powers",
    "enemy_alive",
)


def _combat(seed: int = REGRESSION_SEED):
    return start_combat(seed, ascension=0)


def _drive_to_first_map_screen(seed: int = REGRESSION_SEED):
    """Return a run positioned at its first map screen (still before the first row)."""
    gc = start_run(seed=seed)
    for _ in range(_MAX_DRIVE):
        if gc.screen_state.name == _MAP_SCREEN:
            return gc
        execute_overworld_action(gc, overworld_actions(gc)[0])
    raise AssertionError("run never reached a map screen")


def _encoded_reachable_cols(map_context) -> set[int]:
    """Columns flagged reachable in a map_context vector."""
    return {
        col
        for col in range(MAP_COLS)
        if map_context[_MAP_CUR_BLOCK + col * _MAP_PER_COL_FEATS] == 1.0
    }


def test_encoded_obs_matches_interface_dtype_shape_and_is_finite() -> None:
    gc, bc = _combat()
    obs = encode_observation(gc, bc)
    # Every declared field is present, correctly typed/shaped, and finite.
    assert set(obs) == {field.name for field in OBS_FIELDS}
    for field in OBS_FIELDS:
        arr = obs[field.name]
        assert arr.shape == field.shape, field.name
        assert arr.dtype == field.dtype, field.name
        assert np.isfinite(arr).all(), field.name


def test_encoded_obs_is_member_of_observation_space() -> None:
    gc, bc = _combat()
    obs = encode_observation(gc, bc)
    assert build_observation_space().contains(obs)


def test_id_fields_stay_within_embedding_bounds() -> None:
    gc, bc = _combat()
    obs = encode_observation(gc, bc)
    for field in OBS_FIELDS:
        if field.bounds != "id":
            continue
        assert field.id_high is not None  # guaranteed by ObsField
        arr = obs[field.name]
        assert arr.min() >= 0, field.name
        assert arr.max() <= field.id_high, field.name


def test_combat_start_populates_expected_fields() -> None:
    gc, bc = _combat()
    obs = encode_observation(gc, bc)
    # player_scalars layout: hp_cur, hp_max, block, energy, gold, floor, asc, turn
    scalars = obs["player_scalars"]
    assert scalars[0] == bc.player.curHp > 0
    assert scalars[1] == bc.player.maxHp > 0
    assert scalars[3] == bc.player.energy
    assert scalars[5] == gc.floor_num
    # A fresh combat deals a hand and faces at least one live enemy.
    assert np.count_nonzero(obs["hand_ids"]) == bc.cards.cardsInHand > 0
    assert obs["enemy_alive"].sum() >= 1
    # Ironclad always carries Burning Blood; its bit is set in the multi-hot.
    assert obs["relics_multihot"][int(sts.RelicId.BURNING_BLOOD)] == 1.0
    # In combat the screen one-hot marks BATTLE and nothing else.
    battle = int(sts.ScreenState.BATTLE)
    assert obs["screen_onehot"][battle] == 1.0
    assert obs["screen_onehot"].sum() == 1.0


def test_padding_slots_beyond_live_enemies_are_zero() -> None:
    gc, bc = _combat()
    obs = encode_observation(gc, bc)
    n = bc.monsters.monsterCount
    # Slots past the actual enemy count are PAD: id 0, zeroed scalars/powers.
    assert np.all(obs["enemy_ids"][n:] == 0)
    assert np.all(obs["enemy_scalars"][n:] == 0.0)
    assert np.all(obs["enemy_powers"][n:] == 0.0)
    # No potions at run start: every belt slot is PAD and unusable.
    assert np.all(obs["potion_ids"] == 0)
    assert np.all(obs["potion_usable"] == 0.0)
    # map_context is a run-mode feature, unset during combat.
    assert np.all(obs["map_context"] == 0.0)


def test_player_powers_indexed_by_status_id() -> None:
    # Assert the delta from buffing, so the test gates the id->index mapping
    # without coupling to the seeded encounter having no turn-0 powers.
    gc, bc = _combat()
    before = encode_observation(gc, bc)["player_powers"].copy()
    bc.player.buff(sts.PlayerStatus.STRENGTH, 3)
    bc.player.buff(sts.PlayerStatus.RITUAL, 1)
    delta = encode_observation(gc, bc)["player_powers"] - before
    assert delta[int(sts.PlayerStatus.STRENGTH)] == 3.0
    assert delta[int(sts.PlayerStatus.RITUAL)] == 1.0
    changed = set(np.nonzero(delta)[0].tolist())
    assert changed == {int(sts.PlayerStatus.STRENGTH), int(sts.PlayerStatus.RITUAL)}


def test_bit_only_status_survives_and_encodes_as_presence() -> None:
    # Regression: a "bit-only" player status (e.g. BARRICADE) sets its presence
    # bit but has no statusMap entry, so the engine's getStatus (statusMap.at)
    # throws IndexError on it. encode_observation must survive and encode the
    # power as presence 1.0 rather than crashing the live-engine rollout.
    gc, bc = _combat()
    bc.player.buff(sts.PlayerStatus.BARRICADE, 1)
    assert bc.player.hasStatus(sts.PlayerStatus.BARRICADE)  # bit set, no map entry
    obs = encode_observation(gc, bc)  # must not raise IndexError
    assert obs["player_powers"][int(sts.PlayerStatus.BARRICADE)] == 1.0


def test_enemy_powers_indexed_by_status_id() -> None:
    gc, bc = _combat()
    monster = bc.monsters[0]
    before = encode_observation(gc, bc)["enemy_powers"][0].copy()
    monster.buff(sts.MonsterStatus.STRENGTH, 2)
    monster.addDebuff(sts.MonsterStatus.VULNERABLE, 2, False)
    delta = encode_observation(gc, bc)["enemy_powers"][0] - before
    assert delta[int(sts.MonsterStatus.STRENGTH)] == 2.0
    assert delta[int(sts.MonsterStatus.VULNERABLE)] == 2.0
    changed = set(np.nonzero(delta)[0].tolist())
    assert changed == {int(sts.MonsterStatus.STRENGTH), int(sts.MonsterStatus.VULNERABLE)}


def test_hand_feats_encode_card_attributes() -> None:
    # hand_feats layout per card: upgraded, cost, is_attack, is_skill, is_power, ethereal.
    gc, bc = _combat()
    obs = encode_observation(gc, bc)
    hand = bc.cards.hand
    n = bc.cards.cardsInHand
    assert n > 0
    for i in range(n):
        card = hand[i]
        feats = obs["hand_feats"][i]
        assert feats[0] == float(card.upgraded)
        assert feats[1] == float(card.cost)
        # Exactly one type one-hot is set, matching getType (all zero for curse/status).
        one_hot = feats[2:5]
        expected = np.zeros(3, dtype=np.float32)
        card_type = int(card.getType())
        for pos, type_id in enumerate(
            (sts.CardType.ATTACK, sts.CardType.SKILL, sts.CardType.POWER)
        ):
            if card_type == int(type_id):
                expected[pos] = 1.0
        assert np.array_equal(one_hot, expected)
        assert feats[5] == float(card.isEthereal())
    # Rows beyond the live hand are PAD (all zero).
    assert np.all(obs["hand_feats"][n:] == 0.0)


def test_hidden_intents_mask_move_and_intent_scalars() -> None:
    gc, bc = _combat()
    bc.intents_hidden = True
    obs = encode_observation(gc, bc)
    # For each live enemy: flagged hidden, move id PAD, intent damage/hits zeroed.
    for i in range(bc.monsters.monsterCount):
        if not obs["enemy_alive"][i]:
            continue
        assert obs["enemy_intent_hidden"][i] == 1.0
        assert obs["enemy_move_ids"][i] == 0
        assert obs["enemy_scalars"][i, 3] == 0.0  # intent damage
        assert obs["enemy_scalars"][i, 4] == 0.0  # intent hits
        # HP stays visible even under hidden intent.
        assert obs["enemy_scalars"][i, 0] == bc.monsters[i].curHp


def test_same_seed_encodes_identically() -> None:
    gc_a, bc_a = _combat()
    gc_b, bc_b = _combat()
    obs_a = encode_observation(gc_a, bc_a)
    obs_b = encode_observation(gc_b, bc_b)
    for name in OBS_FIELD_BY_NAME:
        assert np.array_equal(obs_a[name], obs_b[name]), name


def test_adapter_returns_encoded_not_placeholder_observation() -> None:
    # Regression: the adapter must return the real encoding, not the former
    # all-zero placeholder. A live combat has a non-empty hand and a live enemy.
    env = StsEnv()
    obs, _ = env.reset(seed=REGRESSION_SEED)
    assert env.observation_space.contains(obs)
    assert obs["player_scalars"][0] > 0  # current HP
    assert np.count_nonzero(obs["hand_ids"]) > 0
    assert obs["enemy_alive"].sum() >= 1
    env.close()


def test_overworld_obs_is_valid_with_bc_none() -> None:
    # bc=None yields a space-valid, finite observation whose run-level scalars come
    # from the GameContext and whose combat-only fields are all zero.
    gc = _drive_to_first_map_screen()
    obs = encode_observation(gc, None)

    assert build_observation_space().contains(obs)
    for field in OBS_FIELDS:
        assert np.isfinite(obs[field.name]).all(), field.name

    # player_scalars: hp_cur, hp_max, block, energy, gold, floor, ascension, turn
    scalars = obs["player_scalars"]
    assert scalars[0] == gc.cur_hp > 0
    assert scalars[1] == gc.max_hp > 0
    assert scalars[2] == 0.0  # no block out of combat
    assert scalars[3] == 0.0  # no energy out of combat
    assert scalars[4] == gc.gold
    assert scalars[5] == gc.floor_num
    assert scalars[7] == 0.0  # no turn counter out of combat

    # Screen one-hot marks exactly the map screen; Ironclad keeps Burning Blood.
    assert obs["screen_onehot"][int(sts.ScreenState.MAP_SCREEN)] == 1.0
    assert obs["screen_onehot"].sum() == 1.0
    assert obs["relics_multihot"][int(sts.RelicId.BURNING_BLOOD)] == 1.0

    for name in _COMBAT_ONLY_FIELDS:
        assert not np.any(obs[name]), name


def test_map_context_reachable_matches_legal_map_moves_pre_map() -> None:
    # Before the first row, the reachable columns encoded in map_context equal the
    # engine's legal map moves, and each column's features match its room type.
    gc = _drive_to_first_map_screen()
    assert gc.cur_map_node_y < 0  # standing before the first map row

    mc = encode_observation(gc, None)["map_context"]
    legal_cols = {action.idx1 for action in overworld_actions(gc)}
    assert _encoded_reachable_cols(mc) == legal_cols

    elite = int(sts.Room.ELITE)
    combat_rooms = (int(sts.Room.MONSTER), elite)
    for col in legal_cols:
        slot = _MAP_CUR_BLOCK + col * _MAP_PER_COL_FEATS
        room_id = int(gc.map.get_room_type(col, 0))
        assert mc[slot + 1] == (1.0 if room_id in combat_rooms else 0.0)
        assert mc[slot + 2] == (1.0 if room_id == elite else 0.0)
        assert mc[slot + 3] == room_id / (N_NODE_TYPES - 1)

    # Pre-map: current-room one-hot is empty and the position is the negative
    # sentinel (cur == (-1, -1)); act 1 / floor 0 give a zero progress block.
    assert not np.any(mc[:_MAP_CUR_ROOM_ONEHOT])
    assert mc[_MAP_CUR_ROOM_ONEHOT] < 0.0
    assert mc[_MAP_CUR_ROOM_ONEHOT + 1] < 0.0
    progress = _MAP_CUR_ROOM_ONEHOT + _MAP_CUR_POS
    assert mc[progress] == 0.0  # (act 1 - 1) / (MAX_ACTS - 1)
    assert mc[progress + 1] == 0.0  # floor 0


def test_map_context_edges_branch_after_first_choice() -> None:
    # After stepping onto the first row, the reachable set is the engine's edge list
    # from the current node, and the current-node block reflects the entered node.
    gc = _drive_to_first_map_screen()
    execute_overworld_action(gc, overworld_actions(gc)[0])
    cur_x, cur_y = gc.cur_map_node_x, gc.cur_map_node_y
    assert cur_y >= 0

    obs = _empty_obs()
    _fill_map_context(obs, gc)
    mc = obs["map_context"]

    assert _encoded_reachable_cols(mc) == {int(c) for c in gc.map.edges(cur_x, cur_y)}
    # Current-node one-hot marks the entered room; position is normalized in [0, 1].
    assert mc[int(gc.cur_room)] == 1.0
    assert mc[_MAP_CUR_ROOM_ONEHOT] == cur_x / (MAP_COLS - 1)
    assert mc[_MAP_CUR_ROOM_ONEHOT + 1] == cur_y / (MAP_ROWS - 1)


def test_map_context_encodes_act_boss_above_top_row() -> None:
    # The act boss is not stored in the map grid: it sits above the top row and is
    # reached by that row's edges, so the next-row room lookup returns INVALID. It
    # must still encode as a combat node (BOSS), not read as SHOP (room id 0).
    # Driving a real run to the top row means clearing a whole act, so we encode a
    # synthetic cursor over the real generated map (only _fill_map_context's inputs).
    spire_map = start_run(seed=REGRESSION_SEED).map

    def _real(x: int, y: int) -> bool:
        return 0 <= int(spire_map.get_room_type(x, y)) < N_NODE_TYPES

    top_row = max(y for y in range(MAP_ROWS) for x in range(MAP_COLS) if _real(x, y))
    src_x = next(
        x for x in range(MAP_COLS) if _real(x, top_row) and len(spire_map.edges(x, top_row)) > 0
    )
    boss_cols = {int(c) for c in spire_map.edges(src_x, top_row)}
    # Precondition the fix depends on: the boss node is genuinely unstored, so the
    # naive next-row lookup would misencode it as SHOP (room id 0 -> all-zero features).
    assert boss_cols and all(not _real(c, top_row + 1) for c in boss_cols)

    fake_gc = SimpleNamespace(
        map=spire_map,
        cur_map_node_x=src_x,
        cur_map_node_y=top_row,
        cur_room=int(spire_map.get_room_type(src_x, top_row)),
        act=1,
        floor_num=top_row,
    )
    obs = _empty_obs()
    _fill_map_context(obs, fake_gc)
    mc = obs["map_context"]

    assert _encoded_reachable_cols(mc) == boss_cols
    boss_norm = int(sts.Room.BOSS) / (N_NODE_TYPES - 1)
    for col in boss_cols:
        slot = _MAP_CUR_BLOCK + col * _MAP_PER_COL_FEATS
        assert mc[slot] == 1.0  # reachable
        assert mc[slot + 1] == 1.0  # is_combat: the boss is a fight
        assert mc[slot + 2] == 0.0  # is_elite: the boss is not an elite
        assert mc[slot + 3] == boss_norm  # room type is BOSS, not SHOP (id 0)


def test_map_context_on_boss_node_does_not_index_off_grid() -> None:
    # While the boss reward / relic screens are up, the engine parks the run ON the
    # boss node at cur_y == MAP_ROWS (above the grid). map_context must encode that
    # overworld state without indexing off the grid: edges() throws there, and the
    # run leaves via the act transition, not a map choice, so no column is reachable.
    spire_map = start_run(seed=REGRESSION_SEED).map
    # Precondition the guard exists for: querying edges at the boss row raises.
    with pytest.raises(IndexError):
        spire_map.edges(0, MAP_ROWS)

    fake_gc = SimpleNamespace(
        map=spire_map,
        cur_map_node_x=0,
        cur_map_node_y=MAP_ROWS,  # the boss node, one past the top grid row
        cur_room=int(sts.Room.BOSS),
        act=1,
        floor_num=MAP_ROWS,
    )
    obs = _empty_obs()
    _fill_map_context(obs, fake_gc)  # must not raise IndexError
    mc = obs["map_context"]

    assert np.isfinite(mc).all()
    assert _encoded_reachable_cols(mc) == set()  # no next-row node from the boss
    assert mc[int(sts.Room.BOSS)] == 1.0  # current-room one-hot marks the boss


# --- reward_* fields (REWARDS screen) --------------------------------------


def _reward_gc(screen, cards=(), relics=(), potions=()):
    """A minimal gc whose screen + rewardsContainer drive ``_fill_reward_ids`` alone.

    ``cards`` is a sequence of groups; each group is a sequence of card ids. Card
    objects only need an ``id`` attribute (that is all the encoder reads).
    """
    container = SimpleNamespace(
        cards=[[SimpleNamespace(id=cid) for cid in group] for group in cards],
        relics=list(relics),
        potions=list(potions),
    )
    return SimpleNamespace(
        screen_state=screen, screen_state_info=SimpleNamespace(rewards_container=container)
    )


def test_reward_fields_pad_off_reward_screen() -> None:
    # reward_* are reward-screen-only: PAD in combat (bc set) and on the map screen.
    gc, bc = _combat()
    combat_obs = encode_observation(gc, bc)
    map_obs = encode_observation(_drive_to_first_map_screen(), None)
    for name in _REWARD_ID_FIELDS:
        assert not np.any(combat_obs[name]), name
        assert not np.any(map_obs[name]), name


def test_reward_ids_populated_from_live_container() -> None:
    # Round-trip through a real engine Rewards container (proves the binding shape
    # and Card.id path), gated by a synthetic REWARDS screen. Two card groups, a
    # relic, and a potion land in their aligned slots; unused slots stay PAD.
    info = start_run(seed=REGRESSION_SEED).screen_state_info
    rc = info.rewards_container
    rc.clear()
    group0 = [sts.CardId.ACCURACY, sts.CardId.ADRENALINE, sts.CardId.ANGER]
    group1 = [sts.CardId.BASH, sts.CardId.CLEAVE]
    rc.add_card_reward([sts.Card(cid, 0) for cid in group0])
    rc.add_card_reward([sts.Card(cid, 0) for cid in group1])
    rc.add_relic(sts.RelicId.ART_OF_WAR)
    rc.add_potion(sts.Potion.AMBROSIA)

    gc = SimpleNamespace(screen_state=sts.ScreenState.REWARDS, screen_state_info=info)
    obs = _empty_obs()
    _fill_reward_ids(obs, gc)

    card_ids = obs["reward_card_ids"]
    # Group 0 occupies slots 0..len-1; group 1 starts at MAX_REWARD_CARDS_PER_GROUP.
    for j, cid in enumerate(group0):
        assert card_ids[j] == int(cid)
    for j, cid in enumerate(group1):
        assert card_ids[MAX_REWARD_CARDS_PER_GROUP + j] == int(cid)
    # The gap slot after group 0 and the tail after group 1 stay PAD.
    assert card_ids[len(group0)] == 0
    assert np.all(card_ids[MAX_REWARD_CARDS_PER_GROUP + len(group1) :] == 0)

    assert obs["reward_relic_ids"][0] == int(sts.RelicId.ART_OF_WAR)
    assert np.all(obs["reward_relic_ids"][1:] == 0)
    assert obs["reward_potion_ids"][0] == int(sts.Potion.AMBROSIA)
    assert np.all(obs["reward_potion_ids"][1:] == 0)

    # The populated obs stays a valid member of the observation space.
    assert build_observation_space().contains(obs)


def test_reward_ids_enforce_caps_and_skip_sentinels() -> None:
    # Groups past MAX_REWARD_CARD_GROUPS and cards past MAX_REWARD_CARDS_PER_GROUP
    # are dropped; the INVALID relic sentinel (id past the table) and the potion
    # sentinels are guarded to PAD without shifting the surviving slots.
    over_group = list(range(1, MAX_REWARD_CARDS_PER_GROUP + 3))  # more cards than fit
    extra_group = [7, 8]  # a 3rd group, beyond MAX_REWARD_CARD_GROUPS
    relics = [
        int(sts.RelicId.ART_OF_WAR),
        int(sts.RelicId.INVALID),
        int(sts.RelicId.BIRD_FACED_URN),
    ]
    potions = [sts.Potion.INVALID, sts.Potion.AMBROSIA, sts.Potion.EMPTY_POTION_SLOT]
    gc = _reward_gc(
        sts.ScreenState.REWARDS,
        cards=[over_group, [9], extra_group][: MAX_REWARD_CARD_GROUPS + 1],
        relics=relics,
        potions=potions,
    )
    obs = _empty_obs()
    _fill_reward_ids(obs, gc)

    card_ids = obs["reward_card_ids"]
    # Group 0 is truncated to the per-group cap; group 1's single card follows at
    # the next group block; every other slot stays PAD.
    kept0 = over_group[:MAX_REWARD_CARDS_PER_GROUP]
    expected = [0] * len(card_ids)
    expected[: len(kept0)] = kept0
    expected[MAX_REWARD_CARDS_PER_GROUP] = 9
    assert card_ids.tolist() == expected
    # Cards past group 0's per-group cap and the whole 3rd group (past the group
    # cap) are dropped: none of their ids appear anywhere in the field.
    dropped = over_group[MAX_REWARD_CARDS_PER_GROUP:] + extra_group
    assert not np.isin(dropped, card_ids).any()

    # Relic slot 1 (INVALID=180, past N_RELIC_IDS) is guarded to PAD; 0 and 2 kept.
    assert obs["reward_relic_ids"][0] == int(sts.RelicId.ART_OF_WAR)
    assert obs["reward_relic_ids"][1] == 0
    assert obs["reward_relic_ids"][2] == int(sts.RelicId.BIRD_FACED_URN)
    assert int(sts.RelicId.INVALID) >= N_RELIC_IDS  # the guard's precondition

    # Potion sentinels at slots 0 and 2 are skipped; the real potion at slot 1 kept.
    assert obs["reward_potion_ids"][0] == 0
    assert obs["reward_potion_ids"][1] == int(sts.Potion.AMBROSIA)
    assert obs["reward_potion_ids"][2] == 0
    assert build_observation_space().contains(obs)


def test_reward_ids_truncate_relics_and_potions_to_caps() -> None:
    # More relics / potions than fit are truncated to their caps (no overflow).
    relics = [int(sts.RelicId.ART_OF_WAR)] * (MAX_REWARD_RELICS + 2)
    potions = [sts.Potion.AMBROSIA] * (MAX_REWARD_POTIONS + 2)
    gc = _reward_gc(sts.ScreenState.REWARDS, relics=relics, potions=potions)
    obs = _empty_obs()
    _fill_reward_ids(obs, gc)
    assert obs["reward_relic_ids"].shape == (MAX_REWARD_RELICS,)
    assert obs["reward_potion_ids"].shape == (MAX_REWARD_POTIONS,)
    assert np.all(obs["reward_relic_ids"] == int(sts.RelicId.ART_OF_WAR))
    assert np.all(obs["reward_potion_ids"] == int(sts.Potion.AMBROSIA))


def test_fill_reward_ids_noop_off_reward_screen() -> None:
    # A populated container on a non-REWARDS screen leaves every reward field PAD.
    gc = _reward_gc(
        sts.ScreenState.MAP_SCREEN,
        cards=[[sts.CardId.ACCURACY]],
        relics=[int(sts.RelicId.ART_OF_WAR)],
        potions=[sts.Potion.AMBROSIA],
    )
    obs = _empty_obs()
    _fill_reward_ids(obs, gc)
    for name in _REWARD_ID_FIELDS:
        assert not np.any(obs[name]), name
