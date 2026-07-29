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
from sts_rl.env.actions import build_mask
from sts_rl.env.adapter import StsEnv
from sts_rl.env.engine import start_combat
from sts_rl.env.observation import (
    MAP_COLS,
    MAP_LOOKAHEAD_ELITE_CAP,
    MAP_ROWS,
    SHOP_PRICE_SCALE,
    _CARD_TYPE_ATTACK,
    _CARD_TYPE_POWER,
    _MAP_AGG_GLOBAL,
    _MAP_AGG_PER_COL,
    _MAP_CUR_BLOCK,
    _MAP_CUR_POS,
    _MAP_CUR_ROOM_ONEHOT,
    _MAP_NO_REST_DIST,
    _MAP_PER_COL_FEATS,
    _empty_obs,
    _fill_boss_relic_ids,
    _fill_card_select_ids,
    _fill_deck_ids,
    _fill_keys_act,
    _fill_map_context,
    _fill_neow_event,
    _fill_reward_ids,
    _fill_shop,
    _map_dp,
    encode_observation,
)
from sts_rl.env.run import execute_overworld_action, overworld_actions, start_run
from sts_rl.env.spaces import build_observation_space
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    CHOICE_MAX,
    DECK_MAX,
    MAX_BOSS_RELICS,
    MAX_NEOW_OPTIONS,
    MAX_REWARD_CARD_GROUPS,
    MAX_REWARD_CARDS_PER_GROUP,
    MAX_REWARD_POTIONS,
    MAX_REWARD_RELICS,
    MAX_SHOP_CARDS,
    N_EVENT_IDS,
    N_NEOW_BONUS,
    N_NEOW_DRAWBACK,
    N_NODE_TYPES,
    N_RELIC_IDS,
    OBS_FIELDS,
    OBS_FIELD_BY_NAME,
)

# reward_* id fields: populated only on the REWARDS screen. Off it, cards / potions rest
# at PAD 0 and reward_relic_ids at its INVALID empty marker (RelicId 0, AKABEKO, is a
# real relic, so PAD 0 cannot mark an empty relic slot).
_REWARD_ID_FIELDS = ("reward_card_ids", "reward_relic_ids", "reward_potion_ids")

REGRESSION_SEED = 42

# Overworld screen name and a safety cap for driving the run to its first map.
_MAP_SCREEN = "MAP_SCREEN"
_MAX_DRIVE = 50

# Fields populated only in combat; every one must stay zero in an overworld obs.
# potion_ids / potion_usable are intentionally NOT here: the belt is now filled in
# both modes. deck_ids and keys_act are also excluded (overworld / both).
_COMBAT_ONLY_FIELDS = (
    "player_powers",
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
    # map_context / map_lookahead are run-mode features, unset during combat.
    assert np.all(obs["map_context"] == 0.0)
    assert np.all(obs["map_lookahead"] == 0.0)


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

    # map_lookahead on the boss boundary: the next node is the act boss (0 elites, no
    # rest), so the global summary and every boss-col forward cone reflect that.
    look = obs["map_lookahead"]
    assert look[0] == 0.0  # global min elites-to-boss
    assert look[1] == 0.0  # global max elites-to-boss
    assert look[2] == pytest.approx(1.0)  # no rest reachable -> distance saturates
    for col in boss_cols:
        slot = _MAP_AGG_GLOBAL + col * _MAP_AGG_PER_COL
        assert look[slot] == 0.0  # 0 elites through the boss
        assert look[slot + 1] == pytest.approx(1.0)  # no rest through the boss


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


# --- map lookahead aggregates ----------------------------------------------


class _FakeMap:
    """Hand-built act map for lookahead-DP tests: ``get_room_type`` + ``edges`` only.

    ``rooms`` maps ``(x, y)`` to a Room id for populated nodes (absent cells read as
    INVALID, i.e. not a real node); ``adj`` maps ``(x, y)`` to the next-row columns
    reachable from it.
    """

    def __init__(
        self, rooms: dict[tuple[int, int], int], adj: dict[tuple[int, int], list[int]]
    ) -> None:
        self._rooms = rooms
        self._adj = adj

    def get_room_type(self, x: int, y: int) -> int:
        return self._rooms.get((x, y), int(sts.Room.INVALID))

    def edges(self, x: int, y: int) -> list[int]:
        return list(self._adj.get((x, y), []))


def _diamond_map() -> _FakeMap:
    """A small map: a monster forking to a rest (left) and an elite (right).

    (3, TOP-2) MONSTER --> (2, TOP-1) REST  --> (3, TOP) MONSTER --> boss
                      \\--> (4, TOP-1) ELITE --> (3, TOP) MONSTER --> boss
    """
    top = MAP_ROWS - 1
    rooms = {
        (3, top): int(sts.Room.MONSTER),
        (2, top - 1): int(sts.Room.REST),
        (4, top - 1): int(sts.Room.ELITE),
        (3, top - 2): int(sts.Room.MONSTER),
    }
    adj = {
        (3, top - 2): [2, 4],
        (2, top - 1): [3],
        (4, top - 1): [3],
    }
    return _FakeMap(rooms, adj)


def test_map_dp_computes_elites_and_rest_distance() -> None:
    top = MAP_ROWS - 1
    emin, emax, drest = _map_dp(_diamond_map())

    # Top monster -> boss: no elite, no rest ahead.
    assert emin[top][3] == 0.0 and emax[top][3] == 0.0
    assert drest[top][3] == _MAP_NO_REST_DIST
    # The rest node itself is distance 0; the elite node counts one elite.
    assert drest[top - 1][2] == 0.0 and emin[top - 1][2] == 0.0
    assert emin[top - 1][4] == 1.0 and emax[top - 1][4] == 1.0
    assert drest[top - 1][4] == _MAP_NO_REST_DIST
    # The fork: best path has 0 elites (via the rest), worst has 1 (via the elite);
    # the nearest rest is one row ahead.
    assert emin[top - 2][3] == 0.0
    assert emax[top - 2][3] == 1.0
    assert drest[top - 2][3] == 1.0


def test_map_context_lookahead_forward_cone_and_global() -> None:
    top = MAP_ROWS - 1
    fake_gc = SimpleNamespace(
        map=_diamond_map(),
        cur_map_node_x=3,
        cur_map_node_y=top - 2,
        cur_room=int(sts.Room.MONSTER),
        act=1,
        floor_num=top - 2,
    )
    obs = _empty_obs()
    _fill_map_context(obs, fake_gc)
    look = obs["map_lookahead"]

    per_col = _MAP_AGG_GLOBAL
    # Global: best-case 0 elites, worst-case 1 elite (scaled), nearest rest 1 row (via col 2 -> 0).
    assert look[0] == pytest.approx(0.0)
    assert look[1] == pytest.approx(1.0 / MAP_LOOKAHEAD_ELITE_CAP)
    assert look[2] == pytest.approx(0.0)
    # Column 2 leads to the rest (0 elites, rest at distance 0).
    s2 = per_col + 2 * _MAP_AGG_PER_COL
    assert look[s2] == pytest.approx(0.0)
    assert look[s2 + 1] == pytest.approx(0.0)
    # Column 4 leads to the elite (1 elite, no rest reachable -> saturates to 1.0).
    s4 = per_col + 4 * _MAP_AGG_PER_COL
    assert look[s4] == pytest.approx(1.0 / MAP_LOOKAHEAD_ELITE_CAP)
    assert look[s4 + 1] == pytest.approx(1.0)
    # An unreachable column carries no forward cone.
    s0 = per_col + 0 * _MAP_AGG_PER_COL
    assert look[s0] == 0.0 and look[s0 + 1] == 0.0


def test_map_context_lookahead_is_finite_and_normalized_on_real_map() -> None:
    # On a real generated map the whole lookahead block must stay finite and scaled
    # to [0, 1]; at the run's first decision there is a reachable next row, so the
    # global summary is populated (not all zero).
    gc = start_run(seed=REGRESSION_SEED)
    obs = _empty_obs()
    _fill_map_context(obs, gc)
    block = obs["map_lookahead"]

    assert np.isfinite(block).all()
    assert (block >= 0.0).all() and (block <= 1.0).all()
    assert block.any()  # non-vacuous: some lookahead feature fired


def test_map_context_lookahead_zero_when_parked_on_boss() -> None:
    # Parked on the boss node (cur_y past the top grid row) there is no next-row
    # node, so the lookahead block stays entirely zero (mirrors the empty reachable set).
    spire_map = start_run(seed=REGRESSION_SEED).map
    fake_gc = SimpleNamespace(
        map=spire_map,
        cur_map_node_x=0,
        cur_map_node_y=MAP_ROWS,
        cur_room=int(sts.Room.BOSS),
        act=1,
        floor_num=MAP_ROWS,
    )
    obs = _empty_obs()
    _fill_map_context(obs, fake_gc)
    assert np.allclose(obs["map_lookahead"], 0.0)


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
    # reward_* carry no offer off the REWARDS screen (combat with bc set, and the map
    # screen): cards / potions rest at PAD 0, reward_relic_ids at the relic INVALID
    # sentinel (its empty marker, since RelicId 0 (AKABEKO) is a real relic).
    gc, bc = _combat()
    combat_obs = encode_observation(gc, bc)
    map_obs = encode_observation(_drive_to_first_map_screen(), None)
    for name in _REWARD_ID_FIELDS:
        empty = N_RELIC_IDS if name == "reward_relic_ids" else 0
        assert np.all(combat_obs[name] == empty), name
        assert np.all(map_obs[name] == empty), name


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
    # Unused relic slots rest at the INVALID empty marker (N_RELIC_IDS), not PAD 0.
    assert np.all(obs["reward_relic_ids"][1:] == N_RELIC_IDS)
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

    # Relic slot 1 (INVALID=180, past N_RELIC_IDS) fails the valid-id guard, so it keeps
    # the INVALID empty marker (not PAD 0, which would collide with AKABEKO); 0 and 2 kept.
    assert obs["reward_relic_ids"][0] == int(sts.RelicId.ART_OF_WAR)
    assert obs["reward_relic_ids"][1] == N_RELIC_IDS
    assert obs["reward_relic_ids"][2] == int(sts.RelicId.BIRD_FACED_URN)
    assert int(sts.RelicId.INVALID) == N_RELIC_IDS  # empty marker == the field's id_high

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
    # A populated container on a non-REWARDS screen is a noop: every reward field stays
    # at its _empty_obs default (cards / potions PAD 0, relics the INVALID empty marker).
    gc = _reward_gc(
        sts.ScreenState.MAP_SCREEN,
        cards=[[sts.CardId.ACCURACY]],
        relics=[int(sts.RelicId.ART_OF_WAR)],
        potions=[sts.Potion.AMBROSIA],
    )
    obs = _empty_obs()
    _fill_reward_ids(obs, gc)
    for name in _REWARD_ID_FIELDS:
        empty = N_RELIC_IDS if name == "reward_relic_ids" else 0
        assert np.all(obs[name] == empty), name


# --- card_select_ids (CARD_SELECT screen) ----------------------------------


def test_card_select_ids_populated_from_live_container() -> None:
    # Round-trip through a real engine card-select candidate list (proves the
    # binding shape and Card.id path), gated by a synthetic CARD_SELECT screen. The
    # candidates land in their aligned slots with order preserved; the tail past the
    # candidate count stays PAD.
    info = start_run(seed=REGRESSION_SEED).screen_state_info
    info.clear_to_select_cards()
    candidates = [sts.CardId.BASH, sts.CardId.CLEAVE, sts.CardId.ANGER]
    for cid in candidates:
        info.add_to_select_card(sts.Card(cid, 0))

    gc = SimpleNamespace(screen_state=sts.ScreenState.CARD_SELECT, screen_state_info=info)
    obs = _empty_obs()
    _fill_card_select_ids(obs, gc)

    csi = obs["card_select_ids"]
    for i, cid in enumerate(candidates):
        assert csi[i] == int(cid)
    assert np.all(csi[len(candidates) :] == 0)  # tail past the candidates stays PAD
    assert build_observation_space().contains(obs)


def test_card_select_ids_truncate_to_choice_max() -> None:
    # More candidates than fit are truncated to CHOICE_MAX (no overflow past the
    # field width), mirroring the reward-slot truncation.
    info = start_run(seed=REGRESSION_SEED).screen_state_info
    info.clear_to_select_cards()
    for _ in range(CHOICE_MAX + 3):
        info.add_to_select_card(sts.Card(sts.CardId.ANGER, 0))
    gc = SimpleNamespace(screen_state=sts.ScreenState.CARD_SELECT, screen_state_info=info)
    obs = _empty_obs()
    _fill_card_select_ids(obs, gc)
    assert obs["card_select_ids"].shape == (CHOICE_MAX,)
    assert np.all(obs["card_select_ids"] == int(sts.CardId.ANGER))
    assert build_observation_space().contains(obs)


def test_card_select_ids_pad_off_card_select_screen() -> None:
    # A populated candidate list on a non-CARD_SELECT screen leaves the field PAD;
    # so do combat (bc set) and the map screen, where the field is meaningless.
    info = start_run(seed=REGRESSION_SEED).screen_state_info
    info.clear_to_select_cards()
    info.add_to_select_card(sts.Card(sts.CardId.BASH, 0))
    gc = SimpleNamespace(screen_state=sts.ScreenState.MAP_SCREEN, screen_state_info=info)
    obs = _empty_obs()
    _fill_card_select_ids(obs, gc)
    assert not np.any(obs["card_select_ids"])

    # And in real encodes off the card-select screen: combat and the map screen.
    gc_combat, bc = _combat()
    assert not np.any(encode_observation(gc_combat, bc)["card_select_ids"])
    map_obs = encode_observation(_drive_to_first_map_screen(), None)
    assert not np.any(map_obs["card_select_ids"])


def _card_select_obs_and_legal(gc: object, bc: object):
    """Encode ``bc`` and return ``(card_select_ids, populated_indices, legal_indices)``.

    ``legal_indices`` are the ``SINGLE_CARD_SELECT`` choices ``build_mask`` marks
    legal; ``populated_indices`` are the non-PAD ``card_select_ids`` slots. The
    encode must be a valid observation.
    """
    obs = encode_observation(gc, bc)
    assert build_observation_space().contains(obs)
    csi = obs["card_select_ids"]
    start = ACTION_BLOCK_BY_NAME["CARD_SELECT"].start
    legal = set(np.flatnonzero(build_mask(bc)[start : start + CHOICE_MAX]).tolist())
    populated = set(np.flatnonzero(csi).tolist())
    return csi, populated, legal


def test_combat_card_select_ids_unfiltered_pick() -> None:
    """``EXHAUST_ONE`` picks from hand with no filter: every hand slot is a legal
    pick, shown at its own index and nowhere else, matching ``build_mask``."""
    gc, bc = _combat()
    bc.open_card_select(sts.CardSelectTask.EXHAUST_ONE, 1)
    hand = [int(bc.cards.hand[i].id) for i in range(bc.cards.cardsInHand)]
    assert hand, "combat start should deal a non-empty hand"

    csi, populated, legal = _card_select_obs_and_legal(gc, bc)
    for i, card_id in enumerate(hand):
        assert csi[i] == card_id  # each hand card at its own pick index
    assert not np.any(csi[len(hand) :])  # PAD past the hand
    assert populated == legal


def test_combat_card_select_ids_reflect_engine_filter() -> None:
    """``DUAL_WIELD`` offers only ATTACK/POWER cards, so a filtered subset of the
    hand is legal. The populated slots must equal exactly the attack/power hand
    indices and match ``build_mask`` -- guarding the binding's pile+filter mapping
    against drift from the engine's own enumeration.
    """
    gc, bc = _combat()
    bc.open_card_select(sts.CardSelectTask.DUAL_WIELD, 1)
    attack_power = {_CARD_TYPE_ATTACK, _CARD_TYPE_POWER}
    expected = {
        i for i in range(bc.cards.cardsInHand) if int(bc.cards.hand[i].getType()) in attack_power
    }
    assert expected, "Ironclad opening hand always has at least one attack"

    csi, populated, legal = _card_select_obs_and_legal(gc, bc)
    assert populated == expected  # the engine's ATTACK/POWER filter is reflected
    assert populated == legal  # and agrees with the mask
    for idx in populated:
        assert csi[idx] == int(bc.cards.hand[idx].id)


def test_combat_card_select_ids_draw_pile_source() -> None:
    """``SECRET_WEAPON`` picks attacks from the DRAW pile, so it exercises a non-hand
    pile source (the discard / exhaust / draw branches share this pile-indexed
    mapping). The populated slots must be the draw-pile attack indices at their
    absolute positions, matching ``build_mask`` -- guarding the binding's pile choice
    and its sparse-index alignment, not just the hand path.
    """
    gc, bc = _combat()
    bc.open_card_select(sts.CardSelectTask.SECRET_WEAPON, 1)  # draw pile, ATTACK filter
    draw = bc.cards.drawPile
    expected = {i for i in range(len(draw)) if int(draw[i].getType()) == _CARD_TYPE_ATTACK}
    assert expected, "opening draw pile should hold at least one attack"

    csi, populated, legal = _card_select_obs_and_legal(gc, bc)
    assert populated == expected  # only draw-pile attacks, at their draw-pile indices
    assert populated == legal
    for idx in populated:
        assert csi[idx] == int(draw[idx].id)


def test_combat_card_select_ids_generated_source() -> None:
    """A generated card-select (``DISCOVERY``) surfaces the offered cards at slots
    0..2 -- the non-pile branch, reading ``cardSelectInfo.cards`` -- matching the mask.
    """
    gc, bc = _combat()
    bc.open_discovery_select([sts.CardId.ANGER, sts.CardId.CLEAVE, sts.CardId.CLOTHESLINE], 1, True)
    truth = dict(bc.card_select_candidate_ids())  # engine's (idx -> id) for the offered cards
    assert truth, "discovery should offer candidates"

    csi, populated, legal = _card_select_obs_and_legal(gc, bc)
    assert populated == legal
    for idx, card_id in truth.items():
        assert csi[idx] == card_id


# --- deck_ids (overworld deck) ---------------------------------------------


def test_deck_ids_populated_from_live_deck_overworld() -> None:
    # deck_ids mirrors the live gc.deck (order preserved up to DECK_MAX), with the
    # tail past the deck size left PAD. Proves the gc.deck -> Card.id binding path.
    gc = _drive_to_first_map_screen()
    deck_ids = [int(card.id) for card in gc.deck]
    assert deck_ids  # a real run always carries a starting deck
    obs = encode_observation(gc, None)

    assert obs["deck_ids"][: len(deck_ids)].tolist() == deck_ids
    assert np.all(obs["deck_ids"][len(deck_ids) :] == 0)  # PAD tail
    # Deck ids stay within the card embedding table.
    assert obs["deck_ids"].max() <= OBS_FIELD_BY_NAME["deck_ids"].id_high
    assert build_observation_space().contains(obs)


def test_deck_ids_pad_in_combat() -> None:
    # In combat the draw / discard / hand / exhaust piles cover the deck, so
    # deck_ids stays PAD (the field is overworld-only).
    gc, bc = _combat()
    assert not np.any(encode_observation(gc, bc)["deck_ids"])


def test_deck_ids_truncate_to_deck_max() -> None:
    # A deck larger than DECK_MAX is truncated to the fixed width (no overflow),
    # mirroring the pile / card-select truncation.
    over = [SimpleNamespace(id=int(sts.CardId.STRIKE_RED)) for _ in range(DECK_MAX + 5)]
    gc = SimpleNamespace(deck=over)
    obs = _empty_obs()
    _fill_deck_ids(obs, gc)
    assert obs["deck_ids"].shape == (DECK_MAX,)
    assert np.all(obs["deck_ids"] == int(sts.CardId.STRIKE_RED))


# --- overworld potions (belt visible off-combat) ---------------------------


def test_overworld_potions_visible() -> None:
    # Regression for I4: a held potion must be visible OUT of combat (previously the
    # belt was filled only in the combat branch, so overworld potions read all-PAD).
    gc = _drive_to_first_map_screen()
    gc.obtain_potion(sts.Potion.FIRE_POTION)
    obs = encode_observation(gc, None)

    ids = obs["potion_ids"].tolist()
    assert int(sts.Potion.FIRE_POTION) in ids
    slot = ids.index(int(sts.Potion.FIRE_POTION))
    assert obs["potion_usable"][slot] == 1.0
    assert build_observation_space().contains(obs)


# --- keys_act (act + owned keys, both modes) -------------------------------


def test_keys_act_populated_overworld_and_combat() -> None:
    # keys_act = [act, ruby, emerald, sapphire] is filled in both modes (act is
    # known in both; player_scalars carries floor but not act).
    gc = _drive_to_first_map_screen()
    obs = encode_observation(gc, None)
    assert obs["keys_act"][0] == gc.act
    assert obs["keys_act"][1] == float(gc.red_key)
    assert obs["keys_act"][2] == float(gc.green_key)
    assert obs["keys_act"][3] == float(gc.blue_key)

    gc_combat, bc = _combat()
    combat_obs = encode_observation(gc_combat, bc)
    assert combat_obs["keys_act"][0] == gc_combat.act


def test_keys_act_maps_keys_to_slots() -> None:
    # The key -> slot mapping is ruby == red, emerald == green, sapphire == blue,
    # with act in slot 0. A fake gc pins the mapping without engine key mutation.
    obs = _empty_obs()
    fake_gc = SimpleNamespace(act=3, red_key=True, green_key=False, blue_key=True)
    _fill_keys_act(obs, fake_gc)
    assert obs["keys_act"].tolist() == [3.0, 1.0, 0.0, 1.0]


# --- shop fields (SHOP_ROOM screen) ----------------------------------------


def _shop_gc(cards=(), relics=(), potions=(), prices=None, remove_cost=None, screen=None):
    """A minimal gc whose screen + shop drive ``_fill_shop`` alone.

    ``cards`` is a sequence of card ids (Card objects only need an ``id``). ``relics``
    / ``potions`` are the per-slot id / Potion sequences (fixed 3 slots each). ``prices``
    is the engine's 13-entry price array (cards 0..6, relics 7..9, potions 10..12);
    defaults to all -1 (every slot empty). ``screen`` defaults to SHOP_ROOM.
    """
    if prices is None:
        prices = [-1] * 13
    shop = SimpleNamespace(
        cards=[SimpleNamespace(id=cid) for cid in cards],
        relics=list(relics),
        potions=list(potions),
        prices=list(prices),
        remove_cost=remove_cost,
    )
    return SimpleNamespace(
        screen_state=sts.ScreenState.SHOP_ROOM if screen is None else screen,
        screen_state_info=SimpleNamespace(shop=shop),
    )


def test_shop_fields_populated_from_live_shop() -> None:
    # A stocked shop: cards / relics / potions with prices land in their aligned slots
    # (normalized by SHOP_PRICE_SCALE); a bought slot (price -1) stays empty, and its
    # neighbors keep their original-slot alignment (the engine does not compact bought
    # cards, so shop.cards[i] stays slot i).
    cards = [sts.CardId.STRIKE_RED, sts.CardId.BASH, sts.CardId.CLEAVE]
    # Slot 0 = AKABEKO (RelicId 0, a real relic, offered as id 0), slot 1 a real relic
    # but bought (price -1), slot 2 an INVALID id (must stay the empty marker).
    relics = [0, int(sts.RelicId.ART_OF_WAR), int(sts.RelicId.INVALID)]
    potions = [sts.Potion.FIRE_POTION, sts.Potion.EMPTY_POTION_SLOT, sts.Potion.AMBROSIA]
    prices = [-1] * 13
    prices[0], prices[1], prices[2] = 50, -1, 75  # card slot 1 bought
    prices[7], prices[8], prices[9] = 150, -1, 200  # relic slot 1 bought
    prices[10], prices[11], prices[12] = 60, -1, 40  # potion slot 1 empty
    gc = _shop_gc(cards=cards, relics=relics, potions=potions, prices=prices, remove_cost=90)
    obs = _empty_obs()
    _fill_shop(obs, gc)

    # Cards: slots 0 and 2 populated; slot 1 (price -1) stays PAD with 0.0 price, while
    # slot 2 (CLEAVE) keeps its original-slot alignment past the bought slot.
    assert obs["shop_card_ids"][0] == int(sts.CardId.STRIKE_RED)
    assert obs["shop_card_ids"][1] == 0
    assert obs["shop_card_ids"][2] == int(sts.CardId.CLEAVE)
    assert obs["shop_card_prices"][0] == 50 / SHOP_PRICE_SCALE
    assert obs["shop_card_prices"][1] == 0.0
    assert obs["shop_card_prices"][2] == 75 / SHOP_PRICE_SCALE

    # Relics: AKABEKO (id 0) kept and distinguishable from empty; the bought slot
    # (price -1) and the INVALID id both stay the INVALID empty marker with 0.0 price.
    assert obs["shop_relic_ids"][0] == 0  # AKABEKO, not the empty marker
    assert obs["shop_relic_ids"][1] == N_RELIC_IDS  # bought -> empty
    assert obs["shop_relic_ids"][2] == N_RELIC_IDS  # INVALID id -> empty
    assert obs["shop_relic_prices"][0] == 150 / SHOP_PRICE_SCALE
    assert obs["shop_relic_prices"][1] == 0.0
    assert obs["shop_relic_prices"][2] == 0.0

    # Potions: slots 0 and 2 populated; the EMPTY sentinel slot 1 stays PAD.
    assert obs["shop_potion_ids"][0] == int(sts.Potion.FIRE_POTION)
    assert obs["shop_potion_ids"][1] == 0
    assert obs["shop_potion_ids"][2] == int(sts.Potion.AMBROSIA)
    assert obs["shop_potion_prices"][0] == 60 / SHOP_PRICE_SCALE
    assert obs["shop_potion_prices"][2] == 40 / SHOP_PRICE_SCALE

    # Remove cost normalized into the single scalar slot.
    assert obs["shop_remove_cost"][0] == 90 / SHOP_PRICE_SCALE
    assert build_observation_space().contains(obs)


def test_shop_remove_cost_none_encodes_zero() -> None:
    # remove_cost None (engine -1, already used this visit) leaves the field 0.0.
    gc = _shop_gc(remove_cost=None)
    obs = _empty_obs()
    _fill_shop(obs, gc)
    assert obs["shop_remove_cost"][0] == 0.0


def test_shop_cards_truncate_to_cap() -> None:
    # More card slots than fit are truncated to MAX_SHOP_CARDS (no overflow).
    cards = [sts.CardId.STRIKE_RED] * (MAX_SHOP_CARDS + 3)
    prices = [100] * 13
    gc = _shop_gc(cards=cards, prices=prices)
    obs = _empty_obs()
    _fill_shop(obs, gc)
    assert obs["shop_card_ids"].shape == (MAX_SHOP_CARDS,)
    assert np.all(obs["shop_card_ids"] == int(sts.CardId.STRIKE_RED))


def test_fill_shop_noop_off_shop_screen() -> None:
    # A populated shop on a non-SHOP_ROOM screen is a noop: card / potion fields PAD,
    # relic fields the INVALID empty marker, prices 0.0.
    gc = _shop_gc(
        cards=[sts.CardId.STRIKE_RED],
        relics=[int(sts.RelicId.ART_OF_WAR), int(sts.RelicId.INVALID), int(sts.RelicId.INVALID)],
        potions=[sts.Potion.FIRE_POTION, sts.Potion.AMBROSIA, sts.Potion.EMPTY_POTION_SLOT],
        prices=[100] * 13,
        remove_cost=90,
        screen=sts.ScreenState.MAP_SCREEN,
    )
    obs = _empty_obs()
    _fill_shop(obs, gc)
    assert np.all(obs["shop_card_ids"] == 0)
    assert np.all(obs["shop_card_prices"] == 0.0)
    assert np.all(obs["shop_relic_ids"] == N_RELIC_IDS)
    assert np.all(obs["shop_relic_prices"] == 0.0)
    assert np.all(obs["shop_potion_ids"] == 0)
    assert np.all(obs["shop_potion_prices"] == 0.0)
    assert obs["shop_remove_cost"][0] == 0.0


def test_shop_fields_empty_off_shop_screen_real_encode() -> None:
    # In real encodes off the shop screen (combat and the map screen), the shop fields
    # stay empty: cards / potions PAD, relics the INVALID marker, prices 0.0.
    gc_combat, bc = _combat()
    combat_obs = encode_observation(gc_combat, bc)
    map_obs = encode_observation(_drive_to_first_map_screen(), None)
    for obs in (combat_obs, map_obs):
        assert np.all(obs["shop_card_ids"] == 0)
        assert np.all(obs["shop_potion_ids"] == 0)
        assert np.all(obs["shop_relic_ids"] == N_RELIC_IDS)
        assert np.all(obs["shop_card_prices"] == 0.0)
        assert np.all(obs["shop_relic_prices"] == 0.0)
        assert np.all(obs["shop_potion_prices"] == 0.0)
        assert obs["shop_remove_cost"][0] == 0.0


# --- boss_relic_ids (BOSS_RELIC_REWARDS screen) ----------------------------


def _boss_gc(boss_relics=(), screen=None):
    """A minimal gc whose screen + boss_relics drive ``_fill_boss_relic_ids`` alone."""
    return SimpleNamespace(
        screen_state=sts.ScreenState.BOSS_RELIC_REWARDS if screen is None else screen,
        screen_state_info=SimpleNamespace(boss_relics=list(boss_relics)),
    )


def test_boss_relic_ids_populated_from_live_screen() -> None:
    # The three offered boss relics land per slot; AKABEKO (id 0) is kept (a real
    # relic, distinguishable from empty), and an INVALID slot stays the empty marker.
    boss_relics = [0, int(sts.RelicId.ART_OF_WAR), int(sts.RelicId.INVALID)]
    gc = _boss_gc(boss_relics=boss_relics)
    obs = _empty_obs()
    _fill_boss_relic_ids(obs, gc)
    assert obs["boss_relic_ids"][0] == 0  # AKABEKO kept, not the empty marker
    assert obs["boss_relic_ids"][1] == int(sts.RelicId.ART_OF_WAR)
    assert obs["boss_relic_ids"][2] == N_RELIC_IDS  # INVALID stays the empty marker
    assert build_observation_space().contains(obs)


def test_boss_relic_ids_truncate_to_cap() -> None:
    # More offered relics than fit are truncated to MAX_BOSS_RELICS (no overflow).
    gc = _boss_gc(boss_relics=[int(sts.RelicId.ART_OF_WAR)] * (MAX_BOSS_RELICS + 2))
    obs = _empty_obs()
    _fill_boss_relic_ids(obs, gc)
    assert obs["boss_relic_ids"].shape == (MAX_BOSS_RELICS,)
    assert np.all(obs["boss_relic_ids"] == int(sts.RelicId.ART_OF_WAR))


def test_fill_boss_relic_ids_noop_off_boss_screen() -> None:
    # Populated boss relics on a non-BOSS_RELIC_REWARDS screen is a noop: the field
    # stays at the INVALID empty marker (a phantom AKABEKO would fail this).
    gc = _boss_gc(
        boss_relics=[int(sts.RelicId.ART_OF_WAR)] * MAX_BOSS_RELICS,
        screen=sts.ScreenState.MAP_SCREEN,
    )
    obs = _empty_obs()
    _fill_boss_relic_ids(obs, gc)
    assert np.all(obs["boss_relic_ids"] == N_RELIC_IDS)


# --- Neow-event one-hots (EVENT_SCREEN) ------------------------------------


def _neow_gc(options, cur_event=None, screen=None):
    """A minimal gc whose screen + cur_event + neowRewards drive ``_fill_neow_event`` alone.

    ``options`` is a sequence of ``(bonus_id, drawback_id)`` pairs; each becomes a
    NeowOption-like object with ``.r`` / ``.d`` (all the fill reads). ``cur_event``
    defaults to ``Event.NEOW`` and ``screen`` to ``EVENT_SCREEN``.
    """
    ssi = SimpleNamespace(neowRewards=[SimpleNamespace(r=r, d=d) for r, d in options])
    return SimpleNamespace(
        screen_state=sts.ScreenState.EVENT_SCREEN if screen is None else screen,
        cur_event=sts.Event.NEOW if cur_event is None else cur_event,
        screen_state_info=ssi,
    )


def test_neow_event_populated_from_live_run() -> None:
    # A fresh run starts on the Neow event screen; encode overworld (bc=None) and confirm
    # the event / Neow one-hots match the live screen. Proves the gc.cur_event and
    # screen_state_info.neowRewards -> .r / .d binding path (not just a fake gc).
    gc = start_run(seed=REGRESSION_SEED)
    assert gc.screen_state == sts.ScreenState.EVENT_SCREEN
    assert int(gc.cur_event) == int(sts.Event.NEOW)
    obs = encode_observation(gc, None)

    # event_onehot marks exactly the NEOW event.
    assert obs["event_onehot"][int(sts.Event.NEOW)] == 1.0
    assert obs["event_onehot"].sum() == 1.0

    # Each offered option's NeowBonus / NeowDrawback sets its own per-option span.
    options = gc.screen_state_info.neowRewards
    assert len(options) == MAX_NEOW_OPTIONS
    for k, opt in enumerate(options):
        assert obs["neow_bonus"][k * N_NEOW_BONUS + int(opt.r)] == 1.0
        assert obs["neow_drawback"][k * N_NEOW_DRAWBACK + int(opt.d)] == 1.0
    assert obs["neow_bonus"].sum() == MAX_NEOW_OPTIONS
    assert obs["neow_drawback"].sum() == MAX_NEOW_OPTIONS
    assert build_observation_space().contains(obs)


def test_fill_neow_event_sets_event_and_option_onehots() -> None:
    # On the Neow screen: event_onehot marks NEOW, and each option's NeowBonus /
    # NeowDrawback sets exactly one bit in that option's span. The populated blocks must
    # differ from the empty all-zero baseline (revert guard: a no-op fill fails here).
    options = [
        (int(sts.NeowBonus.TRANSFORM_CARD), int(sts.NeowDrawback.NONE)),
        (int(sts.NeowBonus.RANDOM_COMMON_RELIC), int(sts.NeowDrawback.NONE)),
        (int(sts.NeowBonus.TWO_FIFTY_GOLD), int(sts.NeowDrawback.PERCENT_DAMAGE)),
        (int(sts.NeowBonus.BOSS_RELIC), int(sts.NeowDrawback.LOSE_STARTER_RELIC)),
    ]
    gc = _neow_gc(options)
    obs = _empty_obs()
    _fill_neow_event(obs, gc)

    # event_onehot: exactly the NEOW bit is set.
    expected_event = np.zeros(N_EVENT_IDS, dtype=np.float32)
    expected_event[int(sts.Event.NEOW)] = 1.0
    assert np.array_equal(obs["event_onehot"], expected_event)

    # Each option sets exactly one bit in its own per-option span, at the .r / .d id.
    for k, (bonus, drawback) in enumerate(options):
        assert obs["neow_bonus"][k * N_NEOW_BONUS + bonus] == 1.0
        assert obs["neow_drawback"][k * N_NEOW_DRAWBACK + drawback] == 1.0
    assert obs["neow_bonus"].sum() == MAX_NEOW_OPTIONS  # one bit per option
    assert obs["neow_drawback"].sum() == MAX_NEOW_OPTIONS

    # Revert guard: the populated one-hots differ from the empty baseline (a no-op fill,
    # e.g. if _fill_neow_event were reverted, would leave them equal and fail this).
    empty = _empty_obs()
    for name in ("event_onehot", "neow_bonus", "neow_drawback"):
        assert not np.array_equal(obs[name], empty[name]), name
    assert build_observation_space().contains(obs)


def test_fill_neow_event_non_neow_event_sets_only_event_onehot() -> None:
    # A non-Neow event on the event screen sets only event_onehot (its cur_event bit);
    # the per-option Neow blocks stay zero, since the option identities are Neow-specific.
    other = next(
        e
        for e in sts.Event.__members__.values()
        if int(e) != int(sts.Event.NEOW) and 0 <= int(e) < N_EVENT_IDS
    )
    gc = _neow_gc([], cur_event=other)
    obs = _empty_obs()
    _fill_neow_event(obs, gc)
    assert obs["event_onehot"][int(other)] == 1.0
    assert obs["event_onehot"].sum() == 1.0
    assert not np.any(obs["neow_bonus"])
    assert not np.any(obs["neow_drawback"])


def test_fill_neow_event_truncates_options_to_cap() -> None:
    # More options than fit are truncated to MAX_NEOW_OPTIONS (no overflow past the
    # per-option one-hot spans), mirroring the reward / boss-relic slot truncation.
    options = [(int(sts.NeowBonus.TWO_FIFTY_GOLD), int(sts.NeowDrawback.NONE))] * (
        MAX_NEOW_OPTIONS + 2
    )
    gc = _neow_gc(options)
    obs = _empty_obs()
    _fill_neow_event(obs, gc)
    assert obs["neow_bonus"].shape == (MAX_NEOW_OPTIONS * N_NEOW_BONUS,)
    assert obs["neow_drawback"].shape == (MAX_NEOW_OPTIONS * N_NEOW_DRAWBACK,)
    assert obs["neow_bonus"].sum() == MAX_NEOW_OPTIONS  # one bit per kept option
    assert obs["neow_drawback"].sum() == MAX_NEOW_OPTIONS


def test_fill_neow_event_noop_off_event_screen() -> None:
    # Off the event screen the fill is a no-op: every block stays all-zero, so a stale
    # cur_event never leaks a phantom event bit.
    options = [(int(sts.NeowBonus.TWO_FIFTY_GOLD), int(sts.NeowDrawback.PERCENT_DAMAGE))]
    gc = _neow_gc(options, screen=sts.ScreenState.MAP_SCREEN)
    obs = _empty_obs()
    _fill_neow_event(obs, gc)
    for name in ("event_onehot", "neow_bonus", "neow_drawback"):
        assert not np.any(obs[name]), name


def test_neow_event_fields_empty_off_event_screen_real_encode() -> None:
    # In real encodes off the event screen (combat and the map screen), the Neow-event
    # one-hots stay all-zero.
    gc_combat, bc = _combat()
    combat_obs = encode_observation(gc_combat, bc)
    map_obs = encode_observation(_drive_to_first_map_screen(), None)
    for obs in (combat_obs, map_obs):
        assert not np.any(obs["event_onehot"])
        assert not np.any(obs["neow_bonus"])
        assert not np.any(obs["neow_drawback"])


def test_fill_neow_event_drops_out_of_range_event_id() -> None:
    # A cur_event id past the table (>= N_EVENT_IDS) or negative is dropped by the bounds
    # guard: no event_onehot bit is set and nothing overflows. Unreachable from the live
    # engine (max id is exactly N_EVENT_IDS - 1), so this guards the invariant directly.
    for bad_event in (N_EVENT_IDS, N_EVENT_IDS + 5, -1):
        gc = _neow_gc([], cur_event=bad_event)
        obs = _empty_obs()
        _fill_neow_event(obs, gc)  # must not raise
        assert not np.any(obs["event_onehot"]), bad_event
        assert not np.any(obs["neow_bonus"])
        assert not np.any(obs["neow_drawback"])


def test_fill_neow_event_drops_out_of_range_option_ids() -> None:
    # On the Neow screen, an option whose NeowBonus / NeowDrawback id is out of range is
    # dropped, not written. Placed last so that without the per-index guard the write
    # (k * N_NEOW_BONUS + N_NEOW_BONUS) would land one past the option's span / the array
    # end; the guard keeps every write inside its own per-option one-hot span.
    valid = (int(sts.NeowBonus.TWO_FIFTY_GOLD), int(sts.NeowDrawback.NONE))
    options = [valid, valid, valid, (N_NEOW_BONUS, N_NEOW_DRAWBACK)]  # last is out of range
    assert len(options) == MAX_NEOW_OPTIONS
    gc = _neow_gc(options)
    obs = _empty_obs()
    _fill_neow_event(obs, gc)  # must not raise / overflow

    # event_onehot still marks NEOW; only the three in-range options set a bit.
    assert obs["event_onehot"][int(sts.Event.NEOW)] == 1.0
    assert obs["event_onehot"].sum() == 1.0
    assert obs["neow_bonus"].sum() == MAX_NEOW_OPTIONS - 1
    assert obs["neow_drawback"].sum() == MAX_NEOW_OPTIONS - 1
    # The dropped option's own span is entirely zero (no overflow into it).
    last = MAX_NEOW_OPTIONS - 1
    assert not np.any(obs["neow_bonus"][last * N_NEOW_BONUS : (last + 1) * N_NEOW_BONUS])
    assert not np.any(obs["neow_drawback"][last * N_NEOW_DRAWBACK : (last + 1) * N_NEOW_DRAWBACK])
