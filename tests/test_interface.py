from __future__ import annotations

import numpy as np
import gymnasium as gym
import pytest

import sts_rl.interface as interface
from sts_rl.env import spaces

# ---------------------------------------------------------------------------
# Version and action dimension
# ---------------------------------------------------------------------------


def test_interface_version():
    assert interface.INTERFACE_VERSION == "0.8.0"


def test_action_dim_is_257():
    assert interface.ACTION_DIM == 257


def test_action_dim_equals_sum_of_block_counts():
    assert interface.ACTION_DIM == sum(b.count for b in interface.ACTION_BLOCKS)


# ---------------------------------------------------------------------------
# Action blocks
# ---------------------------------------------------------------------------


def test_action_blocks_are_contiguous():
    blocks = interface.ACTION_BLOCKS
    assert blocks[0].start == 0
    for prev, nxt in zip(blocks, blocks[1:]):
        assert nxt.start == prev.stop
    assert blocks[-1].stop == interface.ACTION_DIM


def _expected_block_spec():
    """(name, count) in declaration order, expressed via the shared constants."""
    return [
        ("END_TURN", 1),
        ("PLAY_CARD_TARGETED", interface.HAND_MAX * interface.MAX_ENEMIES),
        ("PLAY_CARD_UNTARGETED", interface.HAND_MAX),
        ("USE_POTION_TARGETED", interface.POTION_SLOTS * interface.MAX_ENEMIES),
        ("USE_POTION_UNTARGETED", interface.POTION_SLOTS),
        ("DISCARD_POTION", interface.POTION_SLOTS),
        ("CARD_SELECT", interface.CHOICE_MAX),
        ("CONFIRM_SELECT", 1),
        ("REWARD_SELECT", interface.REWARD_SELECT_COUNT),
        ("MAP_SELECT", 7),
        ("SHOP_SELECT", 15),
        ("REST_SELECT", 7),
        ("TREASURE_SELECT", 2),
        ("EVENT_SELECT", 10),
        ("BOSS_RELIC_SELECT", 4),
        ("PROCEED", 1),
    ]


def test_action_block_offsets_match_spec():
    expected = _expected_block_spec()
    blocks = interface.ACTION_BLOCKS
    assert len(blocks) == len(expected)

    start = 0
    for block, (name, count) in zip(blocks, expected):
        assert block.name == name
        assert block.start == start
        assert block.count == count
        assert block.stop == start + count
        assert interface.ACTION_BLOCK_BY_NAME[block.name] is block
        start += count

    assert interface.ACTION_BLOCK_BY_NAME["END_TURN"].start == 0
    # PROCEED is the tail block, so its stop equals the full action dim.
    assert interface.ACTION_BLOCK_BY_NAME["PROCEED"].stop == interface.ACTION_DIM
    # CONFIRM_SELECT closes the combat prefix, ahead of the overworld blocks.
    assert interface.ACTION_BLOCK_BY_NAME["CONFIRM_SELECT"].stop == 193


def test_action_block_contains():
    block = interface.ACTION_BLOCK_BY_NAME["PLAY_CARD_TARGETED"]
    assert block.contains(block.start) is True
    assert block.contains(block.stop - 1) is True
    assert block.contains(block.stop) is False
    assert block.contains(block.start - 1) is False


def test_reward_select_sublayout_tiles_the_block():
    # The REWARD_SELECT block is sub-divided by the REWARD_*_OFFSET slots; the
    # offsets must tile the block exactly (no gaps, no overlap) and sum to its count.
    i = interface
    assert i.REWARD_GOLD_OFFSET == 0
    assert i.REWARD_POTION_OFFSET == i.REWARD_GOLD_OFFSET + i.MAX_REWARD_GOLD
    assert i.REWARD_RELIC_OFFSET == i.REWARD_POTION_OFFSET + i.MAX_REWARD_POTIONS
    assert i.REWARD_KEY_OFFSET == i.REWARD_RELIC_OFFSET + i.MAX_REWARD_RELICS
    assert i.REWARD_CARD_OFFSET == i.REWARD_KEY_OFFSET + 1
    assert i.REWARD_SINGING_BOWL_OFFSET == i.REWARD_CARD_OFFSET + i.MAX_REWARD_CARD_SLOTS
    assert i.REWARD_SKIP_OFFSET == i.REWARD_SINGING_BOWL_OFFSET + 1
    assert i.REWARD_SELECT_COUNT == i.REWARD_SKIP_OFFSET + 1
    assert i.REWARD_SELECT_COUNT == 18
    assert i.ACTION_BLOCK_BY_NAME["REWARD_SELECT"].count == i.REWARD_SELECT_COUNT
    assert i.MAX_REWARD_CARD_SLOTS == i.MAX_REWARD_CARD_GROUPS * i.MAX_REWARD_CARDS_PER_GROUP


def test_reward_obs_fields_slot_aligned_with_block():
    # The reward id observation fields must match the block's per-category slot
    # counts, so obs slot i and action slot i refer to the same reward item.
    by = interface.OBS_FIELD_BY_NAME
    assert by["reward_card_ids"].shape == (interface.MAX_REWARD_CARD_SLOTS,)
    assert by["reward_relic_ids"].shape == (interface.MAX_REWARD_RELICS,)
    assert by["reward_potion_ids"].shape == (interface.MAX_REWARD_POTIONS,)


def test_card_select_obs_field_aligned_with_block():
    # card_select_ids must be exactly as wide as the CARD_SELECT action block, so
    # obs slot i and action index i refer to the same candidate card.
    field = interface.OBS_FIELD_BY_NAME["card_select_ids"]
    assert field.shape == (interface.CHOICE_MAX,)
    assert field.shape[0] == interface.ACTION_BLOCK_BY_NAME["CARD_SELECT"].count


def test_deck_and_keys_act_obs_fields():
    # deck_ids is a DECK_MAX-wide card-id field (the overworld deck as an
    # order-agnostic set); keys_act is a KEYS_ACT_DIM-wide real passthrough
    # ([act, ruby, emerald, sapphire]). Both map to no action slot.
    by = interface.OBS_FIELD_BY_NAME

    deck = by["deck_ids"]
    assert deck.shape == (interface.DECK_MAX,)
    assert deck.dtype == np.int32
    assert deck.bounds == "id"
    assert deck.id_high == interface.N_CARD_IDS - 1

    keys_act = by["keys_act"]
    assert interface.KEYS_ACT_DIM == 4
    assert keys_act.shape == (interface.KEYS_ACT_DIM,)
    assert keys_act.dtype == np.float32
    assert keys_act.bounds == "real"
    assert keys_act.id_high is None

    # deck_ids / keys_act keep their relative order after card_select_ids (the 0.6.0
    # tail); the 0.8.0 shop / boss / event-neow blocks are appended after keys_act, so
    # this triple is now interior rather than the final three fields.
    names = [f.name for f in interface.OBS_FIELDS]
    ka = names.index("keys_act")
    assert names[ka - 2 : ka + 1] == ["card_select_ids", "deck_ids", "keys_act"]


def test_shop_boss_event_neow_obs_fields():
    # The 0.8.0 shop / boss-relic / event-neow offer fields: shapes, dtypes, bounds,
    # and their append-last placement after keys_act (so the prior layout stays a clean
    # prefix for warm-start migration).
    by = interface.OBS_FIELD_BY_NAME

    assert by["shop_card_ids"].shape == (interface.SHOP_CARD_SLOTS,)
    assert by["shop_relic_ids"].shape == (interface.SHOP_RELIC_SLOTS,)
    assert by["shop_potion_ids"].shape == (interface.SHOP_POTION_SLOTS,)
    assert by["shop_prices"].shape == (interface.SHOP_PRICE_SLOTS,)
    assert by["shop_remove_cost"].shape == (1,)
    assert by["boss_relic_ids"].shape == (interface.BOSS_RELIC_SLOTS,)
    assert by["event_onehot"].shape == (interface.N_EVENT_IDS,)
    assert by["neow_bonus_onehot"].shape == (interface.NEOW_OPTION_SLOTS, interface.N_NEOW_BONUS)
    assert by["neow_drawback_onehot"].shape == (
        interface.NEOW_OPTION_SLOTS,
        interface.N_NEOW_DRAWBACK,
    )

    # Prices / remove-cost are raw real passthroughs; the one-hots are unit.
    assert by["shop_prices"].bounds == "real"
    assert by["shop_remove_cost"].bounds == "real"
    for name in ("event_onehot", "neow_bonus_onehot", "neow_drawback_onehot"):
        assert by[name].bounds == "unit"
        assert by[name].dtype == np.float32

    # Shop price vector covers cards + relics + potions (the buyable-item slots).
    assert interface.SHOP_PRICE_SLOTS == (
        interface.SHOP_CARD_SLOTS + interface.SHOP_RELIC_SLOTS + interface.SHOP_POTION_SLOTS
    )

    # Appended last, in fixed order, after keys_act.
    names = [f.name for f in interface.OBS_FIELDS]
    assert names[-9:] == [
        "shop_card_ids",
        "shop_relic_ids",
        "shop_potion_ids",
        "shop_prices",
        "shop_remove_cost",
        "boss_relic_ids",
        "event_onehot",
        "neow_bonus_onehot",
        "neow_drawback_onehot",
    ]


def test_event_neow_enums_in_table_sizes():
    # The event / Neow one-hot widths are validated at startup like the other enum
    # tables (their max id must fit), so they must be registered in EXPECTED_TABLE_SIZES.
    for key in ("N_EVENT_IDS", "N_NEOW_BONUS", "N_NEOW_DRAWBACK"):
        assert key in interface.EXPECTED_TABLE_SIZES
        assert interface.EXPECTED_TABLE_SIZES[key] == getattr(interface, key)


# ---------------------------------------------------------------------------
# Observation fields
# ---------------------------------------------------------------------------


def test_obs_fields_count_and_unique_names():
    fields = interface.OBS_FIELDS
    assert len(fields) == 33
    names = [f.name for f in fields]
    assert len(names) == len(set(names))
    for f in fields:
        assert interface.OBS_FIELD_BY_NAME[f.name] is f


def test_obs_field_shapes_match_constants():
    by_name = interface.OBS_FIELD_BY_NAME
    assert by_name["hand_ids"].shape == (interface.HAND_MAX,)
    assert by_name["enemy_scalars"].shape == (interface.MAX_ENEMIES, 5)
    assert by_name["draw_ids"].shape == (interface.PILE_MAX,)
    assert by_name["player_scalars"].shape == (8,)
    assert by_name["enemy_move_ids"].shape == (interface.MAX_ENEMIES,)
    assert by_name["enemy_intent_hidden"].shape == (interface.MAX_ENEMIES,)
    assert by_name["enemy_powers"].shape == (interface.MAX_ENEMIES, interface.N_MONSTER_POWER_IDS)
    assert by_name["relics_multihot"].shape == (interface.N_RELIC_IDS,)
    assert by_name["map_context"].shape == (40,)
    # Pin the per-card feature width (6) directly; the space-vs-registry test is
    # tautological here since both sides read the same registry.
    assert by_name["hand_feats"].shape == (interface.HAND_MAX, 6)


def test_obs_dim_constants_match_expected_literals():
    # The named feature-width constants must equal their expected literal
    # values, so lifting them to constants (vs bare ints in OBS_FIELDS) cannot
    # drift.
    assert interface.PLAYER_SCALAR_DIM == 8
    assert interface.ENEMY_SCALAR_DIM == 5
    assert interface.HAND_FEAT_DIM == 6
    assert interface.MAP_CONTEXT_DIM == 40
    # And OBS_FIELDS must actually use them.
    by_name = interface.OBS_FIELD_BY_NAME
    assert by_name["player_scalars"].shape == (interface.PLAYER_SCALAR_DIM,)
    assert by_name["enemy_scalars"].shape == (interface.MAX_ENEMIES, interface.ENEMY_SCALAR_DIM)
    assert by_name["hand_feats"].shape == (interface.HAND_MAX, interface.HAND_FEAT_DIM)
    assert by_name["map_context"].shape == (interface.MAP_CONTEXT_DIM,)


def test_id_fields_have_id_high():
    expected_id_high = {
        "hand_ids": interface.N_CARD_IDS - 1,
        "draw_ids": interface.N_CARD_IDS - 1,
        "discard_ids": interface.N_CARD_IDS - 1,
        "exhaust_ids": interface.N_CARD_IDS - 1,
        "potion_ids": interface.N_POTION_IDS - 1,
        "enemy_ids": interface.N_MONSTER_IDS - 1,
        "enemy_move_ids": interface.N_MONSTER_MOVE_IDS - 1,
        "reward_card_ids": interface.N_CARD_IDS - 1,
        # Relics uniquely allow the INVALID sentinel (== N_RELIC_IDS) as a legal empty
        # marker, since RelicId 0 (AKABEKO) is a real relic; hence N_RELIC_IDS, not - 1.
        "reward_relic_ids": interface.N_RELIC_IDS,
        "reward_potion_ids": interface.N_POTION_IDS - 1,
        "card_select_ids": interface.N_CARD_IDS - 1,
        "deck_ids": interface.N_CARD_IDS - 1,
        "shop_card_ids": interface.N_CARD_IDS - 1,
        # Shop / boss relic offers reuse the INVALID-sentinel empty marker (== N_RELIC_IDS),
        # like reward_relic_ids, since RelicId 0 (AKABEKO) is a real relic.
        "shop_relic_ids": interface.N_RELIC_IDS,
        "shop_potion_ids": interface.N_POTION_IDS - 1,
        "boss_relic_ids": interface.N_RELIC_IDS,
    }
    # Pin the exact set of id fields, so silently switching one to another
    # bounds value (which __post_init__ would happily accept) is caught.
    assert {f.name for f in interface.OBS_FIELDS if f.bounds == "id"} == set(expected_id_high)
    for f in interface.OBS_FIELDS:
        if f.bounds == "id":
            assert f.id_high is not None
            assert f.id_high == expected_id_high[f.name]
        else:
            assert f.id_high is None


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


def test_observation_space_matches_registry():
    os = spaces.build_observation_space()
    assert set(os.spaces.keys()) == {f.name for f in interface.OBS_FIELDS}
    for f in interface.OBS_FIELDS:
        sub = os.spaces[f.name]
        assert isinstance(sub, gym.spaces.Box)
        assert sub.shape == f.shape
        if f.bounds == "id":
            assert sub.dtype == np.int32
            assert int(sub.high.max()) == f.id_high
            assert int(sub.low.min()) == 0
        else:
            assert sub.dtype == np.float32


def test_action_space_is_discrete_257():
    space = spaces.build_action_space()
    assert space == gym.spaces.Discrete(interface.ACTION_DIM)
    assert space.n == 257


def test_build_spaces_returns_pair():
    obs_space, act_space = spaces.build_spaces()
    assert isinstance(obs_space, gym.spaces.Dict)
    assert isinstance(act_space, gym.spaces.Discrete)


def test_observation_space_sample_is_contained():
    os = spaces.build_observation_space()
    assert os.contains(os.sample()) is True


# ---------------------------------------------------------------------------
# Mask validation
# ---------------------------------------------------------------------------


def test_assert_valid_mask_accepts_legal():
    mask = np.zeros(interface.ACTION_DIM, dtype=bool)
    mask[0] = True
    assert interface.assert_valid_mask(mask) is None


def test_assert_valid_mask_rejects_all_false():
    mask = np.zeros(interface.ACTION_DIM, dtype=bool)
    with pytest.raises(interface.InterfaceError):
        interface.assert_valid_mask(mask)


def test_assert_valid_mask_rejects_wrong_shape():
    mask = np.zeros(interface.ACTION_DIM + 1, dtype=bool)
    mask[0] = True
    with pytest.raises(interface.InterfaceError):
        interface.assert_valid_mask(mask)


def test_assert_valid_mask_rejects_wrong_dtype():
    mask = np.ones(interface.ACTION_DIM, dtype=np.float32)
    with pytest.raises(interface.InterfaceError):
        interface.assert_valid_mask(mask)


def test_mask_shape_and_dtype_constants():
    assert interface.MASK_SHAPE == (interface.ACTION_DIM,)
    assert interface.MASK_DTYPE == np.dtype(bool)


# ---------------------------------------------------------------------------
# Rewards, shaping, info keys
# ---------------------------------------------------------------------------


def test_terminal_rewards():
    assert interface.TERMINAL_WIN_REWARD == 1.0
    assert interface.TERMINAL_LOSS_REWARD == -1.0
    assert interface.SHAPING_TERMS == (
        "enemy_hp_removed",
        "damage_taken",
        "floor_progress",
        "boss_kill",
    )


def test_info_keys_present():
    required = [
        "action_mask",
        "screen",
        "turn",
        "floor",
        "act",
        "hp",
        "ascension",
        "shaping_terms",
        "seed",
        "interface_version",
        "won",
        "episode",
        "rng_state",
        "engine_commit",
        "invalid_action",
    ]
    for key in required:
        assert key in interface.INFO_KEYS


def test_info_keys_always_vs_terminal_split():
    # Terminal-only keys must be exactly {won, episode} and must NOT appear in
    # the always-present set (the agent reads info every step).
    assert set(interface.INFO_KEYS_TERMINAL) == {"won", "episode"}
    assert set(interface.INFO_KEYS_ALWAYS).isdisjoint(interface.INFO_KEYS_TERMINAL)
    # INFO_KEYS is the union, with no key lost or duplicated.
    assert interface.INFO_KEYS == interface.INFO_KEYS_ALWAYS + interface.INFO_KEYS_TERMINAL
    assert len(set(interface.INFO_KEYS)) == len(interface.INFO_KEYS)
    # Always-present keys the agent relies on each step.
    for key in ("action_mask", "screen", "turn", "hp"):
        assert key in interface.INFO_KEYS_ALWAYS


# ---------------------------------------------------------------------------
# Engine enum validation
# ---------------------------------------------------------------------------


def _max_ids_at_capacity() -> dict[str, int]:
    """The largest engine max-id per enum that still fits (N-1 for each)."""
    return {name: n - 1 for name, n in interface.EXPECTED_TABLE_SIZES.items()}


def test_validate_engine_enums_accepts_max_id_within_table():
    # max_id == N - 1 is the boundary that fits (ids run 0..N-1).
    assert interface.validate_engine_enums(_max_ids_at_capacity()) is None


def test_validate_engine_enums_rejects_overflow():
    # max_id == N overflows a table of N rows (valid ids are 0..N-1).
    max_ids = _max_ids_at_capacity()
    max_ids["N_CARD_IDS"] = interface.N_CARD_IDS  # one past the last valid row
    with pytest.raises(interface.InterfaceError) as excinfo:
        interface.validate_engine_enums(max_ids)
    msg = str(excinfo.value)
    assert "N_CARD_IDS" in msg
    # Error should state the minimum N that would fit.
    assert str(interface.N_CARD_IDS + 1) in msg


def test_validate_engine_enums_rejects_missing_key():
    max_ids = _max_ids_at_capacity()
    max_ids.pop(next(iter(max_ids)))
    with pytest.raises(interface.InterfaceError):
        interface.validate_engine_enums(max_ids)
