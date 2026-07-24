"""Tests for the observation encoder (raw engine state -> contract obs dict).

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
from sts_rl.env.adapter import StsEnv
from sts_rl.env.engine import start_combat
from sts_rl.env.observation import encode_observation
from sts_rl.env.spaces import build_observation_space
from sts_rl.interface import OBS_FIELDS, OBS_FIELD_BY_NAME

REGRESSION_SEED = 42


def _combat(seed: int = REGRESSION_SEED):
    return start_combat(seed, ascension=0)


def test_encoded_obs_matches_contract_dtype_shape_and_is_finite() -> None:
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
