"""Composite heuristic teacher for the survival-critical strategic decisions.

The card-reward teacher (:class:`~sts_rl.agent.card_teacher.CardRewardTeacher`)
only supervises the reward-card pick, so campfire, map/path, and potion decisions
reach behavior cloning with no teaching signal and the warm-start policy plays
them blind. :class:`StrategicTeacher` broadens the teaching signal to those
decisions while delegating the reward-card pick VERBATIM to the card teacher, so
that behavior is unchanged.

Covered decisions (the teacher returns an action):
- Reward-card pick: delegated to :class:`CardRewardTeacher` (unchanged).
- Campfire (``REST_SELECT``): rest when HP is low, else smith to improve the deck,
  else rest to recover; only relic-specific options (recall / lift / toke / dig)
  fall through to the base policy.
- Map / path (``MAP_SELECT``): pick the lowest-risk reachable column - prefer
  non-combat nodes, avoid elites (heavily when HP is low), and favor fewer forced
  elites and a nearer rest on the way to the act boss.
- Reward potion (``REWARD_SELECT`` potion slots, once no card slot is left): take
  the best offered potion, since a free potion is generally worth the belt slot.
- Emergency combat potion (``USE_POTION_UNTARGETED``): at low HP, drink a clearly
  defensive / healing untargeted potion; otherwise the base policy plays combat.

Safety contract (mirrors :class:`CardRewardTeacher`):
- :meth:`select_action` returns a full-space action index that is LEGAL under the
  supplied mask on a screen it recognizes, or ``None`` to DEFER so the caller
  falls back to the base policy.
- It NEVER returns a masked-out action, and only overrides screens it recognizes;
  every other state (general combat play, shop, Neow, events, targeted potions,
  the which-card-to-upgrade select after smithing) defers.
- The one non-(legal-or-defer) exit is inherited from the card teacher: a mask
  with a legal card-pick slot whose ``reward_card_ids`` id is PAD and no legal
  skip makes ``CardRewardTeacher.select_action`` raise. This cannot arise from a
  real environment observation (a mask-legal card slot always carries a written,
  non-PAD id), so it surfaces a malformed mask rather than being handled.

Extension points (deliberately deferred - do not half-implement):
- TODO(shop): a ``SHOP_SELECT`` heuristic (buy high-value cards / relics, use the
  card-removal service on a starter card) once a shop value model exists.
- TODO(neow): an ``EVENT_SELECT`` heuristic for the opening Neow event, reading
  the ``neow_bonus`` / ``neow_drawback`` one-hots.
- TODO(potion-full): discarding a low-value held potion to make room for a better
  offered one is not representable - the engine enumerates no overworld potion
  discard, so a full belt simply cannot take a reward potion.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import numpy as np

from sts_rl.agent.card_teacher import (
    CardRewardTeacher,
    _has_legal_card_slot,
)
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    MAP_CONTEXT_DIM,
    MAP_LOOKAHEAD_DIM,
    MAX_REWARD_POTIONS,
    N_NODE_TYPES,
    PAD_ID,
    PLAYER_SCALAR_DIM,
    POTION_SLOTS,
    REWARD_POTION_OFFSET,
    InterfaceError,
)

logger = logging.getLogger(__name__)

# --- Action blocks (resolved once from the shared layout) -------------------
_REST = ACTION_BLOCK_BY_NAME["REST_SELECT"]
_MAP = ACTION_BLOCK_BY_NAME["MAP_SELECT"]
_REWARD = ACTION_BLOCK_BY_NAME["REWARD_SELECT"]
_USE_POTION_UNTARGETED = ACTION_BLOCK_BY_NAME["USE_POTION_UNTARGETED"]

# REST_SELECT sub-slots. The interface documents the block order as
# "rest / smith / recall / lift / toke / dig / skip"; only rest and smith carry a
# generic campfire decision, the rest are relic-specific and left to the policy.
REST_REST_OFFSET = 0
REST_SMITH_OFFSET = 1

REWARD_POTION_START = _REWARD.start + REWARD_POTION_OFFSET

# --- player_scalars layout (mirror the interface PLAYER_SCALAR_DIM doc) ------
# Positional, exactly as the env encoder writes them:
# hp_cur, hp_max, block, energy, gold, floor, ascension, turn. Read only for
# HP-fraction thresholds (a QUALITY signal, never safety - the returned action is
# always gated on the mask, so a layout drift can only degrade the choice).
_PS_HP_CUR = 0
_PS_HP_MAX = 1
if PLAYER_SCALAR_DIM < 2:  # pragma: no cover - guards the positional read above
    raise InterfaceError(
        f"player_scalars needs hp_cur/hp_max; PLAYER_SCALAR_DIM={PLAYER_SCALAR_DIM}"
    )

# --- map_context / map_lookahead layout (mirror sts_rl.env.observation) ------
# The env encoder owns the single source of truth for these layouts, but it imports
# the native engine at module load, so the agent side cannot import it. These named
# widths duplicate that layout; the sum assertions below catch gross drift. A subtle
# split change would only degrade map-choice QUALITY, never safety: the map action is
# always chosen from a legal mask column, so an illegal column can never be returned.
_MAP_COLS = _MAP.count
# map_context: [cur room-type one-hot | cur_x, cur_y | act, floor | per-column x cols].
_MAP_CTX_CUR_ROOM = N_NODE_TYPES
_MAP_CTX_CUR_POS = 2
_MAP_CTX_PROGRESS = 2
_MAP_CTX_PER_COL = 4  # per column: reachable, is_combat, is_elite, room_type_norm
_MAP_CTX_COL_BASE = _MAP_CTX_CUR_ROOM + _MAP_CTX_CUR_POS + _MAP_CTX_PROGRESS
if _MAP_CTX_COL_BASE + _MAP_COLS * _MAP_CTX_PER_COL != MAP_CONTEXT_DIM:
    raise InterfaceError(
        f"map_context layout drift: {_MAP_CTX_COL_BASE} + {_MAP_COLS} * {_MAP_CTX_PER_COL} "
        f"!= MAP_CONTEXT_DIM ({MAP_CONTEXT_DIM}); re-sync with sts_rl.env.observation"
    )
_MAP_CTX_IS_COMBAT = 1
_MAP_CTX_IS_ELITE = 2
# map_lookahead: [global (3) | per-column (2) x cols], per column = (min elites-to-boss,
# rows-to-nearest-rest), both normalized to [0, 1].
_MAP_LA_GLOBAL = 3
_MAP_LA_PER_COL = 2
if _MAP_LA_GLOBAL + _MAP_COLS * _MAP_LA_PER_COL != MAP_LOOKAHEAD_DIM:
    raise InterfaceError(
        f"map_lookahead layout drift: {_MAP_LA_GLOBAL} + {_MAP_COLS} * {_MAP_LA_PER_COL} "
        f"!= MAP_LOOKAHEAD_DIM ({MAP_LOOKAHEAD_DIM}); re-sync with sts_rl.env.observation"
    )
_MAP_LA_ELITES = 0
_MAP_LA_REST_DIST = 1

# --- Heuristic thresholds ---------------------------------------------------
DEFAULT_REST_HP_FRACTION = 0.5  # rest (not smith) when HP fraction is below this
DEFAULT_MAP_LOW_HP_FRACTION = 0.5  # below this, weight elite-avoidance more heavily
DEFAULT_POTION_EMERGENCY_HP_FRACTION = 0.35  # drink a defensive potion below this

# Map-column risk scoring (a higher score is preferred).
_MAP_NONCOMBAT_BONUS = 1.0  # rest / event / shop / treasure next node
_MAP_COMBAT_PENALTY = 0.5  # ordinary monster fight
_MAP_ELITE_PENALTY = 2.0  # elite at healthy HP
_MAP_ELITE_PENALTY_LOW_HP = 6.0  # elite at low HP: strongly avoid
_MAP_ELITES_AHEAD_WEIGHT = 1.0  # fewer forced elites toward the boss is better
_MAP_REST_DIST_WEIGHT = 0.5  # a nearer rest on the forward path is better

# --- Potion value model -----------------------------------------------------
# Keyed by the engine Potion enum NAME (UPPER_SNAKE). Best-effort: an unmatched
# name (a mis-guessed enum member, or a potion not listed) falls back to the
# default tier, and unknown ids never error. Not strictly validated - unlike the
# card tier list - because the potion enum names cannot be confirmed offline; the
# construction-time check only WARNS on unknown keys when the engine is present.
POTION_TIER_HIGH = 3
POTION_TIER_MED = 2
POTION_TIER_LOW = 1
POTION_TIER_DEFAULT = POTION_TIER_MED
# Take almost any offered potion: a free potion is worth an open belt slot.
MIN_REWARD_POTION_TAKE_TIER = POTION_TIER_LOW

POTION_VALUES: dict[str, int] = {
    # High: survival, healing, or build-defining.
    "BLOOD_POTION": POTION_TIER_HIGH,
    "FAIRY_POTION": POTION_TIER_HIGH,
    "BLOCK_POTION": POTION_TIER_HIGH,
    "ANCIENT_POTION": POTION_TIER_HIGH,
    "GHOST_IN_A_JAR": POTION_TIER_HIGH,
    "FRUIT_JUICE": POTION_TIER_HIGH,
    "HEART_OF_IRON": POTION_TIER_HIGH,
    "ESSENCE_OF_STEEL": POTION_TIER_HIGH,
    "ENTROPIC_BREW": POTION_TIER_HIGH,
    "CULTIST_POTION": POTION_TIER_HIGH,
    "REGEN_POTION": POTION_TIER_HIGH,
    # Low: situational / weak on their own.
    "WEAK_POTION": POTION_TIER_LOW,
    "FEAR_POTION": POTION_TIER_LOW,
    "SNECKO_OIL": POTION_TIER_LOW,
    # Everything else uses POTION_TIER_DEFAULT.
}

# Clearly defensive / healing potions that are usable UNTARGETED, so an emergency
# use needs no enemy-target decision. Offensive / targeted potions are left to the
# base policy (targeting is a combat-policy call, not a strategic one).
DEFENSIVE_UNTARGETED_POTIONS: frozenset[str] = frozenset(
    {
        "BLOCK_POTION",
        "BLOOD_POTION",
        "FAIRY_POTION",
        "GHOST_IN_A_JAR",
        "ESSENCE_OF_STEEL",
        "HEART_OF_IRON",
        "REGEN_POTION",
        "FRUIT_JUICE",
        "ANCIENT_POTION",
    }
)


def _warn_unknown_potion_keys(names: Iterable[str]) -> None:
    """Warn (never raise) on potion-table names absent from the engine enum.

    Deferred to runtime and non-fatal because the engine binding may be absent
    (tests) and the potion enum names cannot be confirmed offline; a mis-guessed
    name is harmless (it simply never matches a real potion).
    """
    try:
        from sts_rl.env._engine import slaythespire as sts
    except ImportError:
        return
    members = set(sts.Potion.__members__.keys())
    unknown = sorted(n for n in names if n not in members)
    if unknown:
        logger.warning(
            "StrategicTeacher potion table has %d name(s) not in the engine Potion "
            "enum (ignored as unknown): %s",
            len(unknown),
            unknown,
        )


class StrategicTeacher:
    """Composite heuristic teacher covering the survival-critical strategic screens.

    Reuses a :class:`CardRewardTeacher` for reward-card picks and adds campfire,
    map, and potion heuristics. :meth:`select_action` returns a mask-legal
    full-space action on a recognized screen, or ``None`` to defer to the base
    policy (see the module docstring for the full safety contract).
    """

    def __init__(
        self,
        card_teacher: CardRewardTeacher | None = None,
        *,
        rest_hp_fraction: float = DEFAULT_REST_HP_FRACTION,
        map_low_hp_fraction: float = DEFAULT_MAP_LOW_HP_FRACTION,
        potion_emergency_hp_fraction: float = DEFAULT_POTION_EMERGENCY_HP_FRACTION,
        potion_values: Mapping[str, int] | None = None,
        validate: bool = True,
    ) -> None:
        # Reuse (do not re-implement) the card teacher so its pick behavior is
        # byte-for-byte unchanged. Its own `validate` gates the CardId tier check.
        self.card_teacher = (
            card_teacher if card_teacher is not None else CardRewardTeacher(validate=validate)
        )
        self.rest_hp_fraction = rest_hp_fraction
        self.map_low_hp_fraction = map_low_hp_fraction
        self.potion_emergency_hp_fraction = potion_emergency_hp_fraction
        self.potion_values: dict[str, int] = (
            dict(potion_values) if potion_values is not None else dict(POTION_VALUES)
        )
        if validate:
            _warn_unknown_potion_keys(self.potion_values.keys())

    # -- dispatch -----------------------------------------------------------

    def select_action(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int | None:
        """Return a mask-legal action for a recognized screen, or ``None`` to defer.

        Screens are identified from the mask's legal action blocks (authoritative
        about what is legal), so no engine screen enum is needed. Overworld screens
        are mutually exclusive, so the dispatch order is a router, not a priority.
        """
        # Reward-card pick: delegate verbatim (byte-for-byte unchanged behavior).
        if _has_legal_card_slot(mask):
            return self.card_teacher.select_action(obs, mask)
        # Campfire.
        if self._block_has_legal(mask, _REST.start, _REST.stop):
            return self._campfire_action(obs, mask)
        # Map / path.
        if self._block_has_legal(mask, _MAP.start, _MAP.stop):
            return self._map_action(obs, mask)
        # Reward potion (a reward screen with no card slot left to take).
        if self._block_has_legal(
            mask, REWARD_POTION_START, REWARD_POTION_START + MAX_REWARD_POTIONS
        ):
            return self._reward_potion_action(obs, mask)
        # Emergency combat potion.
        if self._block_has_legal(mask, _USE_POTION_UNTARGETED.start, _USE_POTION_UNTARGETED.stop):
            return self._combat_potion_action(obs, mask)
        # Unrecognized / ambiguous screen: let the base policy decide.
        return None

    # -- campfire -----------------------------------------------------------

    def _campfire_action(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int | None:
        """Rest when HP is low, else smith; defer relic-specific / skip-only options."""
        legal_rest = [i for i in range(_REST.start, _REST.stop) if mask[i]]
        if len(legal_rest) == 1:
            # A forced single campfire option (e.g. only rest, or only skip): take it.
            return legal_rest[0]

        rest_idx = _REST.start + REST_REST_OFFSET
        smith_idx = _REST.start + REST_SMITH_OFFSET
        rest_legal = bool(mask[rest_idx])
        smith_legal = bool(mask[smith_idx])
        hp_frac = _hp_fraction(obs)

        # Survival first: rest when HP is low.
        if rest_legal and hp_frac is not None and hp_frac < self.rest_hp_fraction:
            return rest_idx
        # Deck-building: smith to upgrade a card (the engine then opens a card-select
        # for WHICH card - left to the base policy; see the module TODO).
        if smith_legal:
            return smith_idx
        # Nothing to smith: rest to recover if possible.
        if rest_legal:
            return rest_idx
        # Only relic-specific options (recall / lift / toke / dig) remain: defer.
        return None

    # -- map ----------------------------------------------------------------

    def _map_action(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int | None:
        """Pick the lowest-risk reachable map column; tie-break the lowest column."""
        legal_cols = [c for c in range(_MAP.count) if mask[_MAP.start + c]]
        if not legal_cols:  # pragma: no cover - block flagged legal implies a column
            return None
        if len(legal_cols) == 1:
            return _MAP.start + legal_cols[0]

        hp_frac = _hp_fraction(obs)
        low_hp = hp_frac is not None and hp_frac < self.map_low_hp_fraction
        best_col = legal_cols[0]
        best_score = self._map_column_score(obs, best_col, low_hp)
        for col in legal_cols[1:]:
            score = self._map_column_score(obs, col, low_hp)
            if score > best_score:  # strict: first (lowest) column wins on a tie
                best_score = score
                best_col = col
        return _MAP.start + best_col

    def _map_column_score(self, obs: dict[str, np.ndarray], col: int, low_hp: bool) -> float:
        """Risk-averse score for a reachable map column (higher is safer / better)."""
        ctx = obs["map_context"]
        look = obs["map_lookahead"]
        col_base = _MAP_CTX_COL_BASE + col * _MAP_CTX_PER_COL
        is_combat = float(ctx[col_base + _MAP_CTX_IS_COMBAT]) >= 0.5
        is_elite = float(ctx[col_base + _MAP_CTX_IS_ELITE]) >= 0.5
        la_base = _MAP_LA_GLOBAL + col * _MAP_LA_PER_COL
        elites_ahead = float(look[la_base + _MAP_LA_ELITES])  # normalized [0, 1]
        rest_dist = float(look[la_base + _MAP_LA_REST_DIST])  # normalized [0, 1]

        score = 0.0
        if is_elite:
            score -= _MAP_ELITE_PENALTY_LOW_HP if low_hp else _MAP_ELITE_PENALTY
        elif is_combat:
            score -= _MAP_COMBAT_PENALTY
        else:
            # Non-combat next node (rest / event / shop / treasure): lowest immediate risk.
            score += _MAP_NONCOMBAT_BONUS
        # Forward cone toward the act boss: fewer forced elites and a nearer rest are better.
        score -= elites_ahead * _MAP_ELITES_AHEAD_WEIGHT
        score -= rest_dist * _MAP_REST_DIST_WEIGHT
        return score

    # -- potions ------------------------------------------------------------

    def _reward_potion_action(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int | None:
        """Take the best offered reward potion; defer if none clears the take bar."""
        potion_ids = obs["reward_potion_ids"]
        best_idx: int | None = None
        best_tier = -1
        for i in range(MAX_REWARD_POTIONS):
            action_idx = REWARD_POTION_START + i
            if not mask[action_idx]:
                continue
            pid = int(potion_ids[i])
            if pid == PAD_ID:
                continue
            tier = self._potion_tier(pid)
            if tier > best_tier:  # strict: first (lowest) slot wins on a tie
                best_tier = tier
                best_idx = action_idx
        if best_idx is not None and best_tier >= MIN_REWARD_POTION_TAKE_TIER:
            return best_idx
        # No takeable/worthwhile potion: leave gold / relic / skip to the base policy.
        return None

    def _combat_potion_action(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int | None:
        """At low HP, drink the best defensive untargeted potion; otherwise defer.

        Fires only in an HP emergency with a clearly defensive / healing potion, so
        the warm-start policy still plays ordinary combat - this teaches the single
        unambiguous case (an emergency potion) without hijacking normal turns.
        """
        hp_frac = _hp_fraction(obs)
        if hp_frac is None or hp_frac > self.potion_emergency_hp_fraction:
            return None

        potion_ids = obs["potion_ids"]
        best_idx: int | None = None
        best_tier = -1
        for slot in range(POTION_SLOTS):
            action_idx = _USE_POTION_UNTARGETED.start + slot
            if not mask[action_idx]:
                continue
            pid = int(potion_ids[slot])
            if pid == PAD_ID:
                continue
            if not self._is_defensive_untargeted(pid):
                continue
            tier = self._potion_tier(pid)
            if tier > best_tier:  # strict: first (lowest) slot wins on a tie
                best_tier = tier
                best_idx = action_idx
        # None if no defensive untargeted potion is usable: defer to the base policy.
        return best_idx

    # -- potion helpers -----------------------------------------------------

    def _potion_id_to_name(self, potion_id: int) -> str | None:
        """Resolve a potion id to its UPPER_SNAKE enum name, or None for PAD/unknown."""
        if potion_id == PAD_ID:
            return None
        try:
            from sts_rl.env._engine import slaythespire as sts

            return sts.Potion(potion_id).name
        except (ImportError, ValueError):
            return None

    def _potion_tier(self, potion_id: int) -> int:
        """Value tier for a potion id (default tier for PAD / unknown / unmatched)."""
        name = self._potion_id_to_name(potion_id)
        if name is None:
            return POTION_TIER_DEFAULT
        return self.potion_values.get(name, POTION_TIER_DEFAULT)

    def _is_defensive_untargeted(self, potion_id: int) -> bool:
        """True if the potion is a clearly defensive / healing untargeted potion."""
        name = self._potion_id_to_name(potion_id)
        return name is not None and name in DEFENSIVE_UNTARGETED_POTIONS

    # -- shared -------------------------------------------------------------

    @staticmethod
    def _block_has_legal(mask: np.ndarray, start: int, stop: int) -> bool:
        """True if any action in the half-open ``[start, stop)`` range is legal."""
        return bool(mask[start:stop].any())


def _hp_fraction(obs: dict[str, np.ndarray]) -> float | None:
    """Current HP fraction from player_scalars, or None when max HP is unavailable."""
    scalars = obs["player_scalars"]
    hp_max = float(scalars[_PS_HP_MAX])
    if hp_max <= 0.0:
        return None
    return float(scalars[_PS_HP_CUR]) / hp_max
