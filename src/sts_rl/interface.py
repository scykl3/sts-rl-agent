"""Shared interface definitions for the Slay the Spire RL environment and agent.

INTERFACE_VERSION 0.4.0.

This module is the single source of truth shared by the environment and the
agent. It defines the observation shapes, action-index layout, dtypes, mask
rules, and enum cardinalities that both sides depend on.

Changing any constant, shape, dtype, or action index is a breaking change:
bump ``INTERFACE_VERSION`` and update both the environment and the agent.

The enum-derived table sizes below (``N_CARD_IDS``, ``N_RELIC_IDS``, and
friends) are INITIAL placeholders except where confirmed against the engine.
They must be validated at startup via :func:`validate_engine_enums`, which
asserts each engine max id fits its table (``max_id < N``); an overflow is a
hard error, never a silent reshape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

INTERFACE_VERSION: str = "0.4.0"

# Sentinel id that fills empty pile / potion / enemy slots.
PAD_ID: int = 0

# --- Structural caps -------------------------------------------------------
HAND_MAX = 10
MAX_ENEMIES = 5
POTION_SLOTS = 5
PILE_MAX = 64
CHOICE_MAX = 10

# --- Reward-screen selection caps (combat / elite / chest REWARDS screen) ---
# The REWARDS screen offers a heterogeneous, variable set of items the agent
# takes one at a time -- taking one removes it and re-presents the rest -- until
# it leaves. The flat REWARD_SELECT block reserves a fixed slot per takeable
# item; each id-bearing category has a matching observation field so the agent
# sees which card / relic / potion occupies a slot (the slot index alone carries
# no meaning).
# One gold slot suffices even for two piles (base + stolen gold): the engine
# enumerates every gold reward at index 0, so the single slot, taken once per
# step, collects them all as the screen re-presents.
MAX_REWARD_GOLD = 1
MAX_REWARD_POTIONS = 3  # at most one potion per reward in practice; headroom
MAX_REWARD_RELICS = 3  # matches the engine's 3-relic rewards container
MAX_REWARD_KEY = 1  # at most one key (Sapphire or Emerald) offered at once
MAX_REWARD_CARD_GROUPS = 2  # card-choice groups (a second appears with Prayer Wheel)
MAX_REWARD_CARDS_PER_GROUP = 4  # cards per group == CardReward fixed_list<Card,4>
MAX_REWARD_CARD_SLOTS = MAX_REWARD_CARD_GROUPS * MAX_REWARD_CARDS_PER_GROUP  # 8

# --- Enum cardinalities (confirm against engine enums at startup) ----------
N_CARD_IDS = 380  # CardId; engine max id 370
N_RELIC_IDS = 180  # RelicId; engine max id 179
N_POTION_IDS = 44  # Potion; engine max id 43
# Player and monster statuses are two disjoint 0-based enums, so they get
# separate embedding tables (matching the player_powers / enemy_powers split)
# rather than one shared id space.
N_PLAYER_POWER_IDS = 87  # PlayerStatus; engine max id 86
N_MONSTER_POWER_IDS = 42  # MonsterStatus; engine max id 41
# N_MONSTER_IDS sizes the enemy embedding table for the engine's MonsterId enum
# (sts_lightspeed, include/constants/MonsterIds.h): INVALID=0 sentinel through
# WRITHING_MASS=65 (the max), contiguous, 66 members. Table size = max_id + 1
# = 66 so raw enum ids 0..65 index rows directly; INVALID=0 doubles as PAD.
N_MONSTER_IDS = 66
# The engine has no Intent enum; an enemy's next move is its raw MonsterMoveId,
# fed through a learned embedding (like card/relic/monster ids). Predicted base
# damage and hit count stay as scalars in enemy_scalars. MonsterMoveId INVALID=0
# so id 0 doubles as PAD (empty slot / hidden intent), like the other id tables.
N_MONSTER_MOVE_IDS = 197  # MonsterMoveId; engine max id 196
N_NODE_TYPES = 8  # Room (real node types); engine max id 7
N_SCREENS = 12  # ScreenState; engine max id 9

# --- Observation feature widths (named so OBS_FIELDS carries no magic ints) -
PLAYER_SCALAR_DIM = 8  # hp_cur, hp_max, block, energy, gold, floor, ascension, turn
HAND_FEAT_DIM = 6  # upgraded, cost, is_attack, is_skill, is_power, ethereal
ENEMY_SCALAR_DIM = 5  # hp_cur, hp_max, block, intent_val, intent_hits
MAP_CONTEXT_DIM = 40  # current + available next node types/positions (run mode)

# --- Type aliases ----------------------------------------------------------
Obs = dict[str, np.ndarray]
Info = dict[str, Any]
Mask = np.ndarray  # bool, shape (ACTION_DIM,)


class InterfaceError(Exception):
    """Raised when a value violates these shared definitions (bad mask, enum mismatch)."""


# --- Action blocks ---------------------------------------------------------
@dataclass(frozen=True)
class ActionBlock:
    name: str
    start: int
    count: int

    @property
    def stop(self) -> int:
        return self.start + self.count

    def contains(self, index: int) -> bool:
        return self.start <= index < self.stop


# --- REWARD_SELECT internal sub-layout --------------------------------------
# One flat block covers the whole REWARDS screen; these offsets (relative to the
# block start) carve it into per-item-type slots. Decode and the reward
# observation fields both index off these, so the slot an action takes and the
# slot the observation describes always line up.
REWARD_GOLD_OFFSET = 0
REWARD_POTION_OFFSET = REWARD_GOLD_OFFSET + MAX_REWARD_GOLD
REWARD_RELIC_OFFSET = REWARD_POTION_OFFSET + MAX_REWARD_POTIONS
REWARD_KEY_OFFSET = REWARD_RELIC_OFFSET + MAX_REWARD_RELICS
REWARD_CARD_OFFSET = REWARD_KEY_OFFSET + MAX_REWARD_KEY
REWARD_SINGING_BOWL_OFFSET = REWARD_CARD_OFFSET + MAX_REWARD_CARD_SLOTS
REWARD_SKIP_OFFSET = REWARD_SINGING_BOWL_OFFSET + 1  # Singing Bowl: one slot
REWARD_SELECT_COUNT = REWARD_SKIP_OFFSET + 1  # skip / leave: one slot


# Ordered (name, count) specs. Counts are expressed in terms of the caps above
# where applicable. Start values are derived contiguously below.
_ACTION_BLOCK_SPECS: tuple[tuple[str, int], ...] = (
    ("END_TURN", 1),
    ("PLAY_CARD_TARGETED", HAND_MAX * MAX_ENEMIES),  # 50
    ("PLAY_CARD_UNTARGETED", HAND_MAX),  # 10
    ("USE_POTION_TARGETED", POTION_SLOTS * MAX_ENEMIES),  # 25
    ("USE_POTION_UNTARGETED", POTION_SLOTS),  # 5
    ("DISCARD_POTION", POTION_SLOTS),  # 5
    ("CARD_SELECT", CHOICE_MAX),  # 10
    # Confirm the running selection of a sequential in-combat multi-select
    # (EXHAUST_MANY / GAMBLE). Kept with the combat blocks so every combat index
    # stays a fixed prefix as the overworld region grows.
    ("CONFIRM_SELECT", 1),
    # REWARDS screen: one flat block, sub-divided by the REWARD_*_OFFSET slots
    # above, covering gold / potions / relics / key / card choices / Singing
    # Bowl / skip. Taking an item removes it and the screen re-presents the rest.
    ("REWARD_SELECT", REWARD_SELECT_COUNT),  # 18
    ("MAP_SELECT", 7),
    ("SHOP_SELECT", 15),
    ("REST_SELECT", 7),  # rest / smith / recall / lift / toke / dig / skip
    ("TREASURE_SELECT", 2),  # open chest / skip
    ("EVENT_SELECT", 10),
    ("BOSS_RELIC_SELECT", 4),
    ("PROCEED", 1),
)


def _build_action_blocks() -> tuple[ActionBlock, ...]:
    blocks: list[ActionBlock] = []
    start = 0
    for name, count in _ACTION_BLOCK_SPECS:
        block = ActionBlock(name=name, start=start, count=count)
        blocks.append(block)
        start = block.stop
    return tuple(blocks)


ACTION_BLOCKS: tuple[ActionBlock, ...] = _build_action_blocks()

ACTION_DIM: int = sum(block.count for block in ACTION_BLOCKS)

_EXPECTED_ACTION_DIM = 171
if ACTION_DIM != _EXPECTED_ACTION_DIM:
    raise InterfaceError(
        f"ACTION_DIM miscount: computed {ACTION_DIM}, expected "
        f"{_EXPECTED_ACTION_DIM}. Check ACTION_BLOCKS counts."
    )

ACTION_BLOCK_BY_NAME: dict[str, ActionBlock] = {b.name: b for b in ACTION_BLOCKS}

# --- Mask rules -------------------------------------------------------------
MASK_SHAPE: tuple[int, ...] = (ACTION_DIM,)
MASK_DTYPE = np.bool_


def assert_valid_mask(mask: np.ndarray) -> None:
    """Raise :class:`InterfaceError` unless ``mask`` is a valid action mask.

    A valid mask has shape ``MASK_SHAPE``, boolean dtype, and at least one
    legal action set. An all-False mask is a hard error: it would NaN the
    ``-inf`` softmax downstream.
    """
    if mask.shape != MASK_SHAPE:
        raise InterfaceError(f"mask shape check failed: got {mask.shape}, expected {MASK_SHAPE}")
    if mask.dtype != np.bool_:
        raise InterfaceError(f"mask dtype check failed: got {mask.dtype}, expected {np.bool_}")
    if not mask.any():
        raise InterfaceError(
            "mask legality check failed: all-False mask has no legal action "
            "(would NaN the -inf softmax)"
        )


# --- Observation field registry -------------------------------------------
@dataclass(frozen=True)
class ObsField:
    name: str
    dtype: type  # np.float32 or np.int32
    shape: tuple[int, ...]
    bounds: str  # one of "unit" (0..1), "real" (-inf..inf), "id" (0..id_high)
    id_high: int | None = None  # required when bounds == "id"; else None

    def __post_init__(self) -> None:
        if (self.bounds == "id") != (self.id_high is not None):
            raise InterfaceError(
                f"ObsField {self.name!r}: id_high must be set iff bounds=='id' "
                f"(bounds={self.bounds!r}, id_high={self.id_high!r})"
            )


OBS_FIELDS: tuple[ObsField, ...] = (
    ObsField("player_scalars", np.float32, (PLAYER_SCALAR_DIM,), "real"),
    ObsField("relics_multihot", np.float32, (N_RELIC_IDS,), "unit"),
    ObsField("player_powers", np.float32, (N_PLAYER_POWER_IDS,), "real"),
    ObsField("potion_ids", np.int32, (POTION_SLOTS,), "id", id_high=N_POTION_IDS - 1),
    ObsField("potion_usable", np.float32, (POTION_SLOTS,), "unit"),
    ObsField("hand_ids", np.int32, (HAND_MAX,), "id", id_high=N_CARD_IDS - 1),
    ObsField("hand_feats", np.float32, (HAND_MAX, HAND_FEAT_DIM), "real"),
    ObsField("draw_ids", np.int32, (PILE_MAX,), "id", id_high=N_CARD_IDS - 1),
    ObsField("discard_ids", np.int32, (PILE_MAX,), "id", id_high=N_CARD_IDS - 1),
    ObsField("exhaust_ids", np.int32, (PILE_MAX,), "id", id_high=N_CARD_IDS - 1),
    ObsField("enemy_ids", np.int32, (MAX_ENEMIES,), "id", id_high=N_MONSTER_IDS - 1),
    ObsField("enemy_scalars", np.float32, (MAX_ENEMIES, ENEMY_SCALAR_DIM), "real"),
    ObsField("enemy_move_ids", np.int32, (MAX_ENEMIES,), "id", id_high=N_MONSTER_MOVE_IDS - 1),
    ObsField("enemy_intent_hidden", np.float32, (MAX_ENEMIES,), "unit"),
    ObsField("enemy_powers", np.float32, (MAX_ENEMIES, N_MONSTER_POWER_IDS), "real"),
    ObsField("enemy_alive", np.float32, (MAX_ENEMIES,), "unit"),
    ObsField("screen_onehot", np.float32, (N_SCREENS,), "unit"),
    ObsField("map_context", np.float32, (MAP_CONTEXT_DIM,), "real"),
    # Run-mode REWARDS-screen contents, slot-aligned with the REWARD_SELECT
    # block's card / relic / potion sub-slots so the agent sees which item each
    # takeable slot holds; the positional index alone is meaningless. Zero (PAD)
    # outside a reward screen.
    ObsField("reward_card_ids", np.int32, (MAX_REWARD_CARD_SLOTS,), "id", id_high=N_CARD_IDS - 1),
    ObsField("reward_relic_ids", np.int32, (MAX_REWARD_RELICS,), "id", id_high=N_RELIC_IDS - 1),
    ObsField("reward_potion_ids", np.int32, (MAX_REWARD_POTIONS,), "id", id_high=N_POTION_IDS - 1),
)

OBS_FIELD_BY_NAME: dict[str, ObsField] = {f.name: f for f in OBS_FIELDS}

# --- Reward / info surface -------------------------------------------------
# Reward is: reward = terminal + beta(t) * sum(shaping_terms).
# The terminal component is never annealed; only the shaping sum is scaled by
# the schedule beta(t). The tunable coefficients are part of the training
# configuration, not these shared definitions.
TERMINAL_WIN_REWARD: float = 1.0
TERMINAL_LOSS_REWARD: float = -1.0
SHAPING_TERMS: tuple[str, ...] = (
    "enemy_hp_removed",
    "damage_taken",
    "floor_progress",
    "boss_kill",
)
# Keys present in EVERY info dict (reset and every step). The agent may read
# these on any step without a terminal guard.
INFO_KEYS_ALWAYS: tuple[str, ...] = (
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
    "rng_state",
    "engine_commit",
    "invalid_action",
)
# Keys present ONLY on the terminal step (terminated or truncated). Reading
# these on a non-terminal step is a bug.
INFO_KEYS_TERMINAL: tuple[str, ...] = (
    "won",
    "episode",
)
# Full set of keys that may appear across a call (always + terminal-only).
INFO_KEYS: tuple[str, ...] = INFO_KEYS_ALWAYS + INFO_KEYS_TERMINAL

# --- Enum validation -------------------------------------------------------
# Embedding-table sizes (N). Each id field indexes a table of N rows, so valid
# ids run 0..N-1. The startup guard asserts the engine's highest id for each
# enum is < N: cardinality equality is NOT sufficient, because a non-contiguous
# or 1-based enum can have a max id >= its member count and still overflow the
# table.
EXPECTED_TABLE_SIZES: dict[str, int] = {
    "N_CARD_IDS": N_CARD_IDS,
    "N_RELIC_IDS": N_RELIC_IDS,
    "N_POTION_IDS": N_POTION_IDS,
    "N_PLAYER_POWER_IDS": N_PLAYER_POWER_IDS,
    "N_MONSTER_POWER_IDS": N_MONSTER_POWER_IDS,
    "N_MONSTER_IDS": N_MONSTER_IDS,
    "N_MONSTER_MOVE_IDS": N_MONSTER_MOVE_IDS,
    "N_NODE_TYPES": N_NODE_TYPES,
    "N_SCREENS": N_SCREENS,
}


def validate_engine_enums(engine_max_ids: Mapping[str, int]) -> None:
    """Assert every engine enum's highest id fits its embedding table.

    ``engine_max_ids`` maps each key in :data:`EXPECTED_TABLE_SIZES` to the
    highest id value the live engine can emit for that enum (the environment
    adapter reads these from the ``sts_lightspeed`` bindings at startup). Each
    id indexes an embedding table of ``N`` rows (0..N-1), so the invariant is
    ``max_id < N``.

    Raise :class:`InterfaceError` listing every enum whose ``max_id`` would
    overflow its table (with the minimum ``N`` that would fit) or that is
    missing from ``engine_max_ids``. An overflow is a hard error, never a
    silent reshape.
    """
    problems: list[str] = []
    for name, table_size in EXPECTED_TABLE_SIZES.items():
        if name not in engine_max_ids:
            problems.append(
                f"{name}: missing from engine_max_ids (expected table size {table_size})"
            )
            continue
        max_id = engine_max_ids[name]
        if max_id >= table_size:
            problems.append(
                f"{name}: engine max id {max_id} overflows expected table size "
                f"{table_size} (need max_id < N, i.e. N >= {max_id + 1})"
            )
    if problems:
        raise InterfaceError(
            "engine enum ids do not fit their embedding tables:\n  " + "\n  ".join(problems)
        )
