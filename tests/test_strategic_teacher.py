"""Tests for the StrategicTeacher composite heuristic.

All tests are engine-free: card and potion id -> name resolution are stubbed, so
the suite runs without the C++ engine binding. Every heuristic is checked for the
one invariant that must never break - the returned action is legal under the mask
(or the teacher defers) - plus the intended choice on clear cases.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from sts_rl.agent.card_teacher import (
    CARD_PICK_START,
    CARD_SKIP_IDX,
    CardRewardTeacher,
)
from sts_rl.agent.strategic_teacher import (
    DEFENSIVE_UNTARGETED_POTIONS,
    MIN_REWARD_POTION_TAKE_TIER,
    REWARD_POTION_START,
    _MAP,
    _MAP_CTX_COL_BASE,
    _MAP_CTX_IS_ELITE,
    _MAP_CTX_PER_COL,
    _MAP_LA_ELITES,
    _MAP_LA_GLOBAL,
    _MAP_LA_PER_COL,
    _PS_HP_CUR,
    _PS_HP_MAX,
    _REST,
    _USE_POTION_UNTARGETED,
    REST_REST_OFFSET,
    REST_SMITH_OFFSET,
    StrategicTeacher,
)
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    MAX_REWARD_CARD_SLOTS,
    MAX_REWARD_POTIONS,
    OBS_FIELDS,
    PAD_ID,
)

# --- Stub id -> name resolution ---------------------------------------------

_STUB_CARD_NAMES: dict[int, str] = {
    1: "OFFERING",  # S-tier
    2: "INFLAME",  # A-tier
    3: "HEADBUTT",  # B-tier
    4: "ANGER",  # C-tier
    5: "CLASH",  # D-tier
}
_STUB_POTION_NAMES: dict[int, str] = {
    10: "BLOCK_POTION",  # HIGH, defensive/untargeted
    11: "FIRE_POTION",  # default MED, offensive (not in defensive set)
    12: "WEAK_POTION",  # LOW
}


def _stub_card_name(self: CardRewardTeacher, card_id: int) -> str | None:
    return None if card_id == PAD_ID else _STUB_CARD_NAMES.get(card_id)


def _stub_potion_name(self: StrategicTeacher, potion_id: int) -> str | None:
    return None if potion_id == PAD_ID else _STUB_POTION_NAMES.get(potion_id)


@pytest.fixture(autouse=True)
def _patch_resolution():
    with (
        patch.object(CardRewardTeacher, "_card_id_to_name", _stub_card_name),
        patch.object(StrategicTeacher, "_potion_id_to_name", _stub_potion_name),
    ):
        yield


# --- Builders ---------------------------------------------------------------


def _obs() -> dict[str, np.ndarray]:
    """A full, in-shape observation (all fields zero / PAD)."""
    return {f.name: np.zeros(f.shape, dtype=f.dtype) for f in OBS_FIELDS}


def _mask() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.bool_)


def _teacher() -> StrategicTeacher:
    return StrategicTeacher(validate=False)


def _set_hp(obs: dict[str, np.ndarray], cur: float, mx: float) -> None:
    obs["player_scalars"][_PS_HP_CUR] = cur
    obs["player_scalars"][_PS_HP_MAX] = mx


def _set_map_column(
    obs: dict[str, np.ndarray],
    col: int,
    *,
    is_elite: bool = False,
    elites_ahead: float = 0.0,
) -> None:
    ctx = obs["map_context"]
    ctx[_MAP_CTX_COL_BASE + col * _MAP_CTX_PER_COL + _MAP_CTX_IS_ELITE] = 1.0 if is_elite else 0.0
    look = obs["map_lookahead"]
    look[_MAP_LA_GLOBAL + col * _MAP_LA_PER_COL + _MAP_LA_ELITES] = elites_ahead


# --- Card delegation --------------------------------------------------------


class TestCardDelegationUnchanged:
    """Reward-card picks match CardRewardTeacher byte-for-byte."""

    @pytest.mark.parametrize(
        "card_ids,legal_slots,skip_legal",
        [
            ([1, 3, 0, 0, 0, 0, 0, 0], [0, 1], True),  # S over B
            ([4, 5, 0, 0, 0, 0, 0, 0], [0, 1], True),  # low tier -> skip/leave
            ([2, 3, 0, 0, 0, 0, 0, 0], [0, 1], False),  # skip illegal -> take best
        ],
    )
    def test_matches_card_teacher(self, card_ids, legal_slots, skip_legal) -> None:
        obs = _obs()
        for i, cid in enumerate(card_ids):
            obs["reward_card_ids"][i] = cid
        mask = _mask()
        for slot in legal_slots:
            mask[CARD_PICK_START + slot] = True
        if skip_legal:
            mask[CARD_SKIP_IDX] = True

        strategic = StrategicTeacher(card_teacher=CardRewardTeacher(validate=False), validate=False)
        card_only = CardRewardTeacher(validate=False)
        assert strategic.select_action(obs, mask) == card_only.select_action(obs, mask)
        assert mask[strategic.select_action(obs, mask)]


# --- Campfire ---------------------------------------------------------------


class TestCampfire:
    def test_rests_when_hp_low(self) -> None:
        obs = _obs()
        _set_hp(obs, cur=10, mx=80)  # ~0.125 fraction
        mask = _mask()
        mask[_REST.start + REST_REST_OFFSET] = True
        mask[_REST.start + REST_SMITH_OFFSET] = True
        action = _teacher().select_action(obs, mask)
        assert action == _REST.start + REST_REST_OFFSET
        assert mask[action]

    def test_smiths_when_hp_healthy(self) -> None:
        obs = _obs()
        _set_hp(obs, cur=80, mx=80)  # full HP
        mask = _mask()
        mask[_REST.start + REST_REST_OFFSET] = True
        mask[_REST.start + REST_SMITH_OFFSET] = True
        action = _teacher().select_action(obs, mask)
        assert action == _REST.start + REST_SMITH_OFFSET

    def test_rests_when_healthy_but_nothing_to_smith(self) -> None:
        obs = _obs()
        _set_hp(obs, cur=80, mx=80)
        mask = _mask()
        mask[_REST.start + REST_REST_OFFSET] = True  # only rest legal (2 legal? no, one)
        # add skip to make the block non-degenerate but smith illegal
        mask[_REST.stop - 1] = True  # skip
        action = _teacher().select_action(obs, mask)
        assert action == _REST.start + REST_REST_OFFSET

    def test_single_legal_option_is_returned(self) -> None:
        obs = _obs()
        mask = _mask()
        only = _REST.stop - 1  # e.g. only "skip" legal
        mask[only] = True
        action = _teacher().select_action(obs, mask)
        assert action == only

    def test_defers_when_only_relic_options(self) -> None:
        obs = _obs()
        _set_hp(obs, cur=80, mx=80)
        mask = _mask()
        # Two legal actions, neither rest (0) nor smith (1): recall (2) + skip (last).
        mask[_REST.start + 2] = True
        mask[_REST.stop - 1] = True
        assert _teacher().select_action(obs, mask) is None


# --- Map --------------------------------------------------------------------


class TestMap:
    def test_single_column_is_returned(self) -> None:
        obs = _obs()
        mask = _mask()
        mask[_MAP.start + 3] = True
        action = _teacher().select_action(obs, mask)
        assert action == _MAP.start + 3

    def test_avoids_elite_when_hp_low(self) -> None:
        obs = _obs()
        _set_hp(obs, cur=10, mx=80)  # low HP
        _set_map_column(obs, 0, is_elite=True)  # column 0 is an elite
        _set_map_column(obs, 1, is_elite=False)  # column 1 is safe
        mask = _mask()
        mask[_MAP.start + 0] = True
        mask[_MAP.start + 1] = True
        action = _teacher().select_action(obs, mask)
        assert action == _MAP.start + 1  # avoids the elite
        assert mask[action]

    def test_prefers_fewer_elites_ahead(self) -> None:
        obs = _obs()
        _set_hp(obs, cur=80, mx=80)
        # Neither column is an elite next node, but column 2 has more elites on the
        # forward path to the boss.
        _set_map_column(obs, 1, elites_ahead=0.1)
        _set_map_column(obs, 2, elites_ahead=0.9)
        mask = _mask()
        mask[_MAP.start + 1] = True
        mask[_MAP.start + 2] = True
        action = _teacher().select_action(obs, mask)
        assert action == _MAP.start + 1

    def test_tie_breaks_lowest_column(self) -> None:
        obs = _obs()
        mask = _mask()
        mask[_MAP.start + 2] = True
        mask[_MAP.start + 4] = True  # identical (zero) features -> tie
        action = _teacher().select_action(obs, mask)
        assert action == _MAP.start + 2


# --- Reward potion ----------------------------------------------------------


class TestRewardPotion:
    def test_takes_best_offered_potion(self) -> None:
        obs = _obs()
        obs["reward_potion_ids"][0] = 12  # WEAK_POTION (LOW)
        obs["reward_potion_ids"][1] = 10  # BLOCK_POTION (HIGH)
        mask = _mask()
        mask[REWARD_POTION_START + 0] = True
        mask[REWARD_POTION_START + 1] = True
        # No card slots legal -> the reward-potion branch owns this step.
        action = _teacher().select_action(obs, mask)
        assert action == REWARD_POTION_START + 1  # BLOCK_POTION
        assert mask[action]

    def test_defers_when_no_potion_present(self) -> None:
        obs = _obs()  # all reward_potion_ids are PAD
        mask = _mask()
        mask[REWARD_POTION_START + 0] = True
        assert _teacher().select_action(obs, mask) is None

    def test_take_bar_constant_is_met_by_default_potion(self) -> None:
        # Regression guard: the default potion tier must clear the take bar, else a
        # perfectly good offered potion would be skipped.
        obs = _obs()
        obs["reward_potion_ids"][0] = 11  # FIRE_POTION -> default tier
        mask = _mask()
        mask[REWARD_POTION_START + 0] = True
        teacher = _teacher()
        action = teacher.select_action(obs, mask)
        assert action == REWARD_POTION_START + 0
        assert teacher._potion_tier(11) >= MIN_REWARD_POTION_TAKE_TIER


# --- Combat potion ----------------------------------------------------------


class TestCombatPotion:
    def test_uses_defensive_potion_at_low_hp(self) -> None:
        obs = _obs()
        _set_hp(obs, cur=10, mx=80)  # emergency
        obs["potion_ids"][0] = 10  # BLOCK_POTION (defensive/untargeted)
        mask = _mask()
        mask[_USE_POTION_UNTARGETED.start + 0] = True
        action = _teacher().select_action(obs, mask)
        assert action == _USE_POTION_UNTARGETED.start + 0
        assert mask[action]

    def test_defers_at_healthy_hp(self) -> None:
        obs = _obs()
        _set_hp(obs, cur=80, mx=80)
        obs["potion_ids"][0] = 10  # BLOCK_POTION usable, but not an emergency
        mask = _mask()
        mask[_USE_POTION_UNTARGETED.start + 0] = True
        assert _teacher().select_action(obs, mask) is None

    def test_defers_when_only_offensive_potion(self) -> None:
        obs = _obs()
        _set_hp(obs, cur=10, mx=80)  # emergency
        obs["potion_ids"][0] = 11  # FIRE_POTION: not in the defensive set
        mask = _mask()
        mask[_USE_POTION_UNTARGETED.start + 0] = True
        assert _teacher().select_action(obs, mask) is None
        assert "FIRE_POTION" not in DEFENSIVE_UNTARGETED_POTIONS


# --- Defer / safety ---------------------------------------------------------


class TestDeferAndLegality:
    def test_defers_on_unsupported_screen(self) -> None:
        obs = _obs()
        mask = _mask()
        mask[ACTION_BLOCK_BY_NAME["END_TURN"].start] = True  # a combat play step
        assert _teacher().select_action(obs, mask) is None

    def test_defers_on_plain_combat_step(self) -> None:
        obs = _obs()
        mask = _mask()
        play = ACTION_BLOCK_BY_NAME["PLAY_CARD_UNTARGETED"]
        mask[play.start] = True
        mask[ACTION_BLOCK_BY_NAME["END_TURN"].start] = True
        assert _teacher().select_action(obs, mask) is None

    @pytest.mark.parametrize("seed", range(8))
    def test_returned_action_is_always_legal(self, seed: int) -> None:
        # Random single-block masks across the covered screens: any non-None result
        # must be legal, and the teacher must never return a masked-out action.
        rng = np.random.default_rng(seed)
        obs = _obs()
        _set_hp(obs, cur=float(rng.integers(1, 80)), mx=80.0)
        # Non-PAD ids (1..5) so a legal card slot always carries a real card, the
        # real-env invariant: this keeps the card-teacher delegate from raising on a
        # legal-slot-with-PAD-id malformed mask (see StrategicTeacher's contract).
        for name in ("reward_card_ids", "reward_potion_ids", "potion_ids"):
            obs[name][:] = rng.integers(1, 6, size=obs[name].shape).astype(obs[name].dtype)
        blocks = ["REWARD_SELECT", "REST_SELECT", "MAP_SELECT", "USE_POTION_UNTARGETED", "END_TURN"]
        block = ACTION_BLOCK_BY_NAME[blocks[int(rng.integers(len(blocks)))]]
        mask = _mask()
        # Unmask a random non-empty subset of the block.
        for i in range(block.start, block.stop):
            if rng.random() < 0.5:
                mask[i] = True
        if not mask.any():
            mask[block.start] = True
        action = _teacher().select_action(obs, mask)
        if action is not None:
            assert 0 <= action < ACTION_DIM
            assert mask[action], f"teacher returned illegal action {action}"


# --- Sanity on shared constants ---------------------------------------------


def test_card_pick_range_and_potion_constants() -> None:
    # Guards the imported layout the tests build against. The card-pick range ends
    # before skip (a Singing Bowl slot sits between them), and the reward-potion
    # slots lie inside the REWARD_SELECT block.
    reward = ACTION_BLOCK_BY_NAME["REWARD_SELECT"]
    assert CARD_PICK_START + MAX_REWARD_CARD_SLOTS <= CARD_SKIP_IDX
    assert reward.start <= REWARD_POTION_START
    assert REWARD_POTION_START + MAX_REWARD_POTIONS <= reward.stop
