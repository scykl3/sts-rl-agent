"""Heuristic teacher for the reward-card-pick decision (Ironclad).

Assigns tier scores to each Ironclad card and selects the highest-tier legal card
from a reward screen, or skips when no offered card meets a configurable quality
threshold. Used as the supervision signal for behavior-cloning pretraining of the
card-pick policy head before PPO fine-tuning.

The teacher operates ONLY over the card-pick + skip sub-slice of the
REWARD_SELECT action block, so gold / potion / relic / key decisions that may be
simultaneously legal are left to the base policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    MAX_REWARD_CARD_SLOTS,
    PAD_ID,
    REWARD_CARD_OFFSET,
    REWARD_SKIP_OFFSET,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# --- Tier scale (named constants) ------------------------------------------
TIER_S: int = 5
TIER_A: int = 4
TIER_B: int = 3
TIER_C: int = 2
TIER_D: int = 1
TIER_AVOID: int = 0

# --- Derived action indices from live interface constants -------------------
_RS = ACTION_BLOCK_BY_NAME["REWARD_SELECT"]
CARD_PICK_START: int = _RS.start + REWARD_CARD_OFFSET
CARD_PICK_END: int = CARD_PICK_START + MAX_REWARD_CARD_SLOTS
CARD_SKIP_IDX: int = _RS.start + REWARD_SKIP_OFFSET


def _has_legal_card_slot(mask: np.ndarray) -> bool:
    """True if at least one card-pick slot is legal in the full-space mask.

    The card-decision test both the BC collector and the composite teacher use to
    decide whether a step is a reward-card pick the card teacher should own.
    """
    return bool(mask[CARD_PICK_START:CARD_PICK_END].any())


# --- Ironclad card tier list -----------------------------------------------
# Keyed by CardId enum NAME (UPPER_SNAKE). Ratings derived from community
# consensus tier lists for Ascension 0 Ironclad. Cards not in this dict use
# the teacher's default_tier.
IRONCLAD_CARD_TIERS: dict[str, int] = {
    # S-tier: build-defining, always take
    "OFFERING": TIER_S,
    "CORRUPTION": TIER_S,
    "DEMON_FORM": TIER_S,
    "FEED": TIER_S,
    "IMMOLATE": TIER_S,
    "IMPERVIOUS": TIER_S,
    "REAPER": TIER_S,
    "BARRICADE": TIER_S,
    # A-tier: strong cards, take in most builds
    "BATTLE_TRANCE": TIER_A,
    "SHRUG_IT_OFF": TIER_A,
    "INFLAME": TIER_A,
    "METALLICIZE": TIER_A,
    "POMMEL_STRIKE": TIER_A,
    "FLAME_BARRIER": TIER_A,
    "FEEL_NO_PAIN": TIER_A,
    "DARK_EMBRACE": TIER_A,
    "SPOT_WEAKNESS": TIER_A,
    "DISARM": TIER_A,
    "SHOCKWAVE": TIER_A,
    "LIMIT_BREAK": TIER_A,
    "FIEND_FIRE": TIER_A,
    "BLUDGEON": TIER_A,
    "DOUBLE_TAP": TIER_A,
    "WHIRLWIND": TIER_A,
    "EXHUME": TIER_A,
    "BRUTALITY": TIER_A,
    # B-tier: good role-players
    "ARMAMENTS": TIER_B,
    "HEADBUTT": TIER_B,
    "UPPERCUT": TIER_B,
    "CARNAGE": TIER_B,
    "DROPKICK": TIER_B,
    "EVOLVE": TIER_B,
    "ENTRENCH": TIER_B,
    "BODY_SLAM": TIER_B,
    "SEEING_RED": TIER_B,
    "BURNING_PACT": TIER_B,
    "PUMMEL": TIER_B,
    "DUAL_WIELD": TIER_B,
    "POWER_THROUGH": TIER_B,
    "GHOSTLY_ARMOR": TIER_B,
    "HEAVY_BLADE": TIER_B,
    "IRON_WAVE": TIER_B,
    "BLOODLETTING": TIER_B,
    "SENTINEL": TIER_B,
    "SECOND_WIND": TIER_B,
    "RAGE": TIER_B,
    "COMBUST": TIER_B,
    "HEMOKINESIS": TIER_B,
    "FIRE_BREATHING": TIER_B,
    "TWIN_STRIKE": TIER_B,
    "CLOTHESLINE": TIER_B,
    "RAMPAGE": TIER_B,
    "BERSERK": TIER_B,
    "JUGGERNAUT": TIER_B,
    # C-tier: situational or filler
    "ANGER": TIER_C,
    "FLEX": TIER_C,
    "HAVOC": TIER_C,
    "THUNDERCLAP": TIER_C,
    "TRUE_GRIT": TIER_C,
    "WARCRY": TIER_C,
    "CLEAVE": TIER_C,
    "BLOOD_FOR_BLOOD": TIER_C,
    "INTIMIDATE": TIER_C,
    "INFERNAL_BLADE": TIER_C,
    "SEVER_SOUL": TIER_C,
    "RUPTURE": TIER_C,
    "SEARING_BLOW": TIER_C,
    # D-tier: generally weak
    "CLASH": TIER_D,
    "WILD_STRIKE": TIER_D,
    "RECKLESS_CHARGE": TIER_D,
    "SWORD_BOOMERANG": TIER_D,
    "PERFECTED_STRIKE": TIER_D,
    # AVOID: starter cards or actively harmful
    "STRIKE_RED": TIER_AVOID,
    "DEFEND_RED": TIER_AVOID,
}


def validate_tier_keys() -> None:
    """Validate that every key in IRONCLAD_CARD_TIERS is a real CardId member.

    Deferred to runtime because the engine binding may not be available in all
    environments (tests use stubs). Call this once at teacher construction when
    the engine IS available.
    """
    try:
        from sts_rl.env._engine import slaythespire as sts
    except ImportError:
        logger.warning("engine not available; skipping CardId validation of tier list keys")
        return
    members = set(sts.CardId.__members__.keys())
    bad_keys = sorted(k for k in IRONCLAD_CARD_TIERS if k not in members)
    if bad_keys:
        raise ValueError(
            f"IRONCLAD_CARD_TIERS contains {len(bad_keys)} key(s) that are not "
            f"valid CardId enum members: {bad_keys}"
        )


class CardRewardTeacher:
    """Deterministic heuristic teacher for the reward-card-pick decision.

    Selects the highest-tier legal card from the offered slots; ties broken by
    lowest slot index (deterministic). Skips when the best legal card is below
    ``min_keep_tier`` and skip is legal.

    Operates only over the card-pick + skip sub-slice of REWARD_SELECT, so other
    simultaneously-legal reward actions (gold, potions, relics) are ignored.
    """

    def __init__(
        self,
        tiers: dict[str, int] | None = None,
        *,
        min_keep_tier: int = TIER_C,
        default_tier: int = TIER_C,
        validate: bool = True,
    ) -> None:
        self.tiers = tiers if tiers is not None else IRONCLAD_CARD_TIERS
        self.min_keep_tier = min_keep_tier
        self.default_tier = default_tier
        self._unknown_count: int = 0
        if validate:
            validate_tier_keys()

    @property
    def unknown_count(self) -> int:
        """Aggregate count of unknown card ids encountered (not per-call spam)."""
        return self._unknown_count

    def _card_id_to_name(self, card_id: int) -> str | None:
        """Resolve a card integer id to its UPPER_SNAKE enum name, or None for PAD."""
        if card_id == PAD_ID:
            return None
        try:
            from sts_rl.env._engine import slaythespire as sts

            return sts.CardId(card_id).name
        except (ImportError, ValueError):
            return None

    def _resolve_tier(self, card_name: str | None) -> int | None:
        """Return the tier for a card name, or None if PAD/unresolvable."""
        if card_name is None:
            return None
        tier = self.tiers.get(card_name)
        if tier is None:
            self._unknown_count += 1
            return self.default_tier
        return tier

    def select_action(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        """Pick the best legal card or skip.

        Args:
            obs: Observation dict with at least ``reward_card_ids`` key,
                shape ``(MAX_REWARD_CARD_SLOTS,)`` int32.
            mask: Boolean action mask, shape ``(ACTION_DIM,)``.

        Returns:
            Action index within the full ACTION_DIM space.
        """
        reward_card_ids = obs["reward_card_ids"]
        assert reward_card_ids.shape == (MAX_REWARD_CARD_SLOTS,), (
            f"expected reward_card_ids shape ({MAX_REWARD_CARD_SLOTS},), "
            f"got {reward_card_ids.shape}"
        )

        best_tier = -1
        best_action = -1

        for slot in range(MAX_REWARD_CARD_SLOTS):
            action_idx = CARD_PICK_START + slot
            if not mask[action_idx]:
                continue
            card_id = int(reward_card_ids[slot])
            if card_id == PAD_ID:
                continue
            card_name = self._card_id_to_name(card_id)
            tier = self._resolve_tier(card_name)
            if tier is None:
                continue
            # Deterministic tie-break: lowest slot index wins (first encounter)
            if tier > best_tier:
                best_tier = tier
                best_action = action_idx

        # Skip if best card is below threshold and skip is legal
        skip_legal = bool(mask[CARD_SKIP_IDX])
        if best_action == -1 or (best_tier < self.min_keep_tier and skip_legal):
            if skip_legal:
                action = CARD_SKIP_IDX
            elif best_action != -1:
                # No skip available, take the best card even if below threshold
                action = best_action
            else:
                # No resolvable card slot is legal and skip is not legal. The
                # teacher owns only the card-pick + skip sub-slice, so it must
                # never fall back to a gold / potion / relic action in the wider
                # REWARD_SELECT block. Reaching here means the caller invoked the
                # teacher on a non-card-decision step (the composite teacher gates
                # this delegation on _has_legal_card_slot).
                raise ValueError(
                    "CardRewardTeacher.select_action found no legal card slot and "
                    "no legal skip; invoke the teacher only on a card-decision step"
                )
        else:
            action = best_action

        assert mask[
            action
        ], f"CardRewardTeacher returned illegal action {action}; mask[action]={mask[action]}"
        return action
