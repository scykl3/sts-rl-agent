"""Tests for the CardRewardTeacher heuristic.

All tests are engine-free: card id -> name resolution is stubbed so the tests
run without the C++ engine binding.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from sts_rl.agent.card_teacher import (
    CARD_PICK_START,
    CARD_SKIP_IDX,
    TIER_A,
    TIER_B,
    TIER_C,
    TIER_D,
    TIER_S,
    CardRewardTeacher,
)
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    MAX_REWARD_CARD_SLOTS,
    PAD_ID,
    REWARD_GOLD_OFFSET,
)


def _make_obs(card_ids: list[int]) -> dict[str, np.ndarray]:
    """Build a minimal obs dict with just reward_card_ids."""
    ids = np.zeros(MAX_REWARD_CARD_SLOTS, dtype=np.int32)
    for i, cid in enumerate(card_ids):
        if i < MAX_REWARD_CARD_SLOTS:
            ids[i] = cid
    return {"reward_card_ids": ids}


def _make_mask(legal_card_slots: list[int], skip_legal: bool = True) -> np.ndarray:
    """Build a mask with specific card slots and optionally skip legal."""
    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    for slot in legal_card_slots:
        mask[CARD_PICK_START + slot] = True
    if skip_legal:
        mask[CARD_SKIP_IDX] = True
    return mask


def _make_teacher(
    tiers: dict[str, int] | None = None,
    min_keep_tier: int = TIER_C,
    default_tier: int = TIER_C,
) -> CardRewardTeacher:
    """Build a teacher with validation disabled (no engine needed)."""
    return CardRewardTeacher(
        tiers=tiers, min_keep_tier=min_keep_tier, default_tier=default_tier, validate=False
    )


# Stub card id -> name mapping for tests
_STUB_CARD_NAMES: dict[int, str] = {
    1: "OFFERING",  # S-tier
    2: "INFLAME",  # A-tier
    3: "HEADBUTT",  # B-tier
    4: "ANGER",  # C-tier
    5: "CLASH",  # D-tier
    6: "STRIKE_RED",  # AVOID
    7: "UNKNOWN_CARD",  # not in tier list
}


def _stub_card_id_to_name(self: CardRewardTeacher, card_id: int) -> str | None:
    """Stub that maps test card ids to names without the engine."""
    if card_id == PAD_ID:
        return None
    return _STUB_CARD_NAMES.get(card_id)


@pytest.fixture(autouse=True)
def _patch_card_resolution():
    """Patch _card_id_to_name for all tests in this module."""
    with patch.object(CardRewardTeacher, "_card_id_to_name", _stub_card_id_to_name):
        yield


class TestCardRewardTeacherPicksHighestTier:
    """Teacher picks the highest-tier legal card."""

    def test_picks_s_over_a(self) -> None:
        teacher = _make_teacher()
        obs = _make_obs([1, 2, 0, 0, 0, 0, 0, 0])  # OFFERING(S), INFLAME(A)
        mask = _make_mask([0, 1])
        action = teacher.select_action(obs, mask)
        assert action == CARD_PICK_START + 0  # slot 0 = OFFERING

    def test_picks_a_over_b(self) -> None:
        teacher = _make_teacher()
        obs = _make_obs([3, 2, 0, 0, 0, 0, 0, 0])  # HEADBUTT(B), INFLAME(A)
        mask = _make_mask([0, 1])
        action = teacher.select_action(obs, mask)
        assert action == CARD_PICK_START + 1  # slot 1 = INFLAME

    def test_respects_mask_skips_illegal_slots(self) -> None:
        teacher = _make_teacher()
        obs = _make_obs([1, 2, 0, 0, 0, 0, 0, 0])  # OFFERING(S), INFLAME(A)
        # Only slot 1 is legal
        mask = _make_mask([1])
        action = teacher.select_action(obs, mask)
        assert action == CARD_PICK_START + 1  # INFLAME (only legal card)


class TestCardRewardTeacherSkipsLowTier:
    """Teacher skips when all legal cards are below min_keep_tier."""

    def test_skips_when_best_below_threshold(self) -> None:
        teacher = _make_teacher(min_keep_tier=TIER_B)
        obs = _make_obs([4, 5, 0, 0, 0, 0, 0, 0])  # ANGER(C), CLASH(D)
        mask = _make_mask([0, 1], skip_legal=True)
        action = teacher.select_action(obs, mask)
        assert action == CARD_SKIP_IDX

    def test_takes_card_when_skip_not_legal(self) -> None:
        teacher = _make_teacher(min_keep_tier=TIER_B)
        obs = _make_obs([4, 5, 0, 0, 0, 0, 0, 0])  # ANGER(C), CLASH(D)
        mask = _make_mask([0, 1], skip_legal=False)
        action = teacher.select_action(obs, mask)
        # Takes best available (ANGER at C > CLASH at D)
        assert action == CARD_PICK_START + 0

    def test_takes_card_at_threshold(self) -> None:
        teacher = _make_teacher(min_keep_tier=TIER_C)
        obs = _make_obs([4, 0, 0, 0, 0, 0, 0, 0])  # ANGER(C)
        mask = _make_mask([0], skip_legal=True)
        action = teacher.select_action(obs, mask)
        # ANGER is at C == min_keep_tier, so should NOT skip (< is strict)
        assert action == CARD_PICK_START + 0


class TestCardRewardTeacherLegality:
    """Teacher always returns a legal action."""

    def test_action_is_always_legal(self) -> None:
        teacher = _make_teacher()
        # Various configurations
        configs = [
            ([1, 2, 3], [0, 1, 2], True),
            ([5, 6], [0, 1], True),
            ([5, 6], [0, 1], False),
            ([0, 0, 0, 0, 0, 0, 0, 0], [], True),  # all PAD, only skip
        ]
        for card_ids, legal_slots, skip_legal in configs:
            obs = _make_obs(card_ids + [0] * (MAX_REWARD_CARD_SLOTS - len(card_ids)))
            mask = _make_mask(legal_slots, skip_legal=skip_legal)
            if not mask.any():
                continue  # skip invalid mask
            action = teacher.select_action(obs, mask)
            assert mask[action], f"action {action} is illegal"
            # The teacher owns only the card-pick + skip sub-slice: it must never
            # return a gold / potion / relic action from the wider REWARD_SELECT
            # block (those would break bc._action_to_subslice_idx).
            is_card_or_skip = (
                CARD_PICK_START <= action < CARD_PICK_START + MAX_REWARD_CARD_SLOTS
                or action == CARD_SKIP_IDX
            )
            assert is_card_or_skip, f"action {action} is neither a card slot nor skip"


class TestCardRewardTeacherNoCardNoSkip:
    """Teacher refuses (raises) when invoked off a card-decision step."""

    def test_raises_when_no_card_and_no_skip(self) -> None:
        teacher = _make_teacher()
        obs = _make_obs([0] * MAX_REWARD_CARD_SLOTS)  # all PAD: no cards offered
        # A non-card REWARD_SELECT action (gold) is legal, but no card slot and
        # no skip. The teacher must refuse rather than fall back to the gold
        # action from the wider block.
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        reward_block = ACTION_BLOCK_BY_NAME["REWARD_SELECT"]
        mask[reward_block.start + REWARD_GOLD_OFFSET] = True
        with pytest.raises(ValueError, match="card-decision step"):
            teacher.select_action(obs, mask)


class TestCardRewardTeacherPADHandling:
    """PAD slots (id 0) are never picked."""

    def test_pad_slots_ignored(self) -> None:
        teacher = _make_teacher()
        # All slots are PAD except slot 2
        obs = _make_obs([0, 0, 2, 0, 0, 0, 0, 0])  # only INFLAME at slot 2
        mask = _make_mask([0, 1, 2], skip_legal=True)
        action = teacher.select_action(obs, mask)
        assert action == CARD_PICK_START + 2  # only non-PAD card

    def test_all_pad_skips(self) -> None:
        teacher = _make_teacher()
        obs = _make_obs([0, 0, 0, 0, 0, 0, 0, 0])
        mask = _make_mask([0, 1, 2], skip_legal=True)
        action = teacher.select_action(obs, mask)
        assert action == CARD_SKIP_IDX


class TestCardRewardTeacherTieBreak:
    """Deterministic tie-break: lowest slot index wins."""

    def test_same_tier_picks_lowest_slot(self) -> None:
        # Both slots have A-tier cards
        tiers = {"CARD_A": TIER_A, "CARD_B": TIER_A}
        stub_names = {10: "CARD_A", 11: "CARD_B"}
        teacher = _make_teacher(tiers=tiers)
        obs = _make_obs([10, 11, 0, 0, 0, 0, 0, 0])
        mask = _make_mask([0, 1])

        with patch.object(
            CardRewardTeacher,
            "_card_id_to_name",
            lambda self, cid: stub_names.get(cid),
        ):
            action = teacher.select_action(obs, mask)
        assert action == CARD_PICK_START + 0  # lowest slot


class TestCardRewardTeacherUnknownCards:
    """Unknown card ids use the default tier and increment the unknown counter."""

    def test_unknown_card_uses_default(self) -> None:
        teacher = _make_teacher(default_tier=TIER_B)
        obs = _make_obs([7, 0, 0, 0, 0, 0, 0, 0])  # UNKNOWN_CARD, not in tiers
        mask = _make_mask([0], skip_legal=True)
        action = teacher.select_action(obs, mask)
        # default_tier=B >= min_keep_tier=C, so it should take the card
        assert action == CARD_PICK_START + 0
        assert teacher.unknown_count == 1

    def test_unknown_card_skipped_when_below_threshold(self) -> None:
        teacher = _make_teacher(default_tier=TIER_D, min_keep_tier=TIER_B)
        obs = _make_obs([7, 0, 0, 0, 0, 0, 0, 0])
        mask = _make_mask([0], skip_legal=True)
        action = teacher.select_action(obs, mask)
        assert action == CARD_SKIP_IDX


class TestCardRewardTeacherValidation:
    """Construction rejects bogus tier keys when validation is enabled."""

    def test_bogus_key_raises(self) -> None:
        # Provide a fake engine that knows only OFFERING
        class FakeCardId:
            __members__ = {"OFFERING": None}

        class FakeSts:
            CardId = FakeCardId

        with patch.dict(
            "sys.modules",
            {"sts_rl.env._engine": type("m", (), {"slaythespire": FakeSts})()},
        ):
            with pytest.raises(ValueError, match="not valid CardId enum members"):
                CardRewardTeacher(
                    tiers={"OFFERING": TIER_S, "BOGUS_CARD": TIER_A},
                    validate=True,
                )
