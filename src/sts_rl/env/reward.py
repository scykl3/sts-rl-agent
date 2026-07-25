"""Reward computation: terminal signal plus annealed shaping.

``reward = terminal + beta(t) * sum(shaping_terms)``

The terminal component (+1 win / -1 loss) dominates and is never annealed. The
shaping terms are heuristic, NOT potential-based: they bias the optimal policy
while active, so ``beta(t)`` decays their weight toward ``beta_min`` over
training to remove that bias asymptotically. Coefficients and the anneal
schedule are tunable training configuration and live here, not in the shared
interface module; only the term names come from :data:`SHAPING_TERMS`.

The shaping values reported (in ``info['shaping_terms']`` and summed for the
reward) are the coefficient-applied, pre-``beta`` contributions, so that
``reward_shaping = beta(t) * sum(terms.values())`` reconstructs the shaped
reward exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sts_rl.interface import SHAPING_TERMS

if TYPE_CHECKING:
    # Only type hints; keep the runtime imports out so this module (and its
    # tests) do not depend on the built engine. The functions access snapshots
    # by attribute and work on any object with the same fields.
    from sts_rl.env.engine import CombatSnapshot
    from sts_rl.env.run import RunSnapshot

# Default shaping coefficients. Positive coefficients reward progress; the
# damage coefficient is negative so taking damage is penalized.
DEFAULT_ENEMY_HP_REMOVED_COEF = 0.05
DEFAULT_DAMAGE_TAKEN_COEF = -0.02
DEFAULT_FLOOR_PROGRESS_COEF = 0.02
DEFAULT_BOSS_KILL_COEF = 0.20

# beta(t) = max(BETA_MIN, 1 - t / T_ANNEAL): the shaping weight starts at 1.0
# and decays linearly to BETA_MIN over T_ANNEAL env steps.
DEFAULT_BETA_MIN = 0.0
DEFAULT_T_ANNEAL = 2e7


@dataclass(frozen=True)
class RewardConfig:
    """Tunable reward coefficients and the shaping-anneal schedule.

    Coefficients are keyed by the same names as :data:`SHAPING_TERMS`. The
    schedule fields drive :func:`beta`.
    """

    enemy_hp_removed: float = DEFAULT_ENEMY_HP_REMOVED_COEF
    damage_taken: float = DEFAULT_DAMAGE_TAKEN_COEF
    floor_progress: float = DEFAULT_FLOOR_PROGRESS_COEF
    boss_kill: float = DEFAULT_BOSS_KILL_COEF
    beta_min: float = DEFAULT_BETA_MIN
    t_anneal: float = DEFAULT_T_ANNEAL

    def __post_init__(self) -> None:
        if self.t_anneal <= 0:
            raise ValueError(f"t_anneal must be > 0, got {self.t_anneal}")
        if not 0.0 <= self.beta_min <= 1.0:
            raise ValueError(f"beta_min must be in [0, 1], got {self.beta_min}")


def beta(t: int, cfg: RewardConfig) -> float:
    """Shaping weight at env step ``t``: ``max(beta_min, 1 - t / t_anneal)``.

    Clamped to ``[beta_min, 1.0]`` so it is well defined for any ``t >= 0``.
    """
    return max(cfg.beta_min, min(1.0, 1.0 - t / cfg.t_anneal))


def zero_shaping_terms() -> dict[str, float]:
    """A fresh shaping-term dict with every term at ``0.0``.

    Used when no state change occurred (episode start, an invalid action that
    did not touch the engine), so ``info['shaping_terms']`` is always complete.
    """
    return {name: 0.0 for name in SHAPING_TERMS}


def _enemy_hp_fraction(snapshot: CombatSnapshot) -> float:
    """Total current enemy HP as a fraction of total enemy max HP, in [0, 1].

    Uses the current monster roster (dead monsters read as 0 HP). A mid-combat
    summon changes the roster and can perturb the per-step delta transiently;
    this is accepted because the shaping weight anneals to zero.
    """
    total_max = sum(m.max_hp for m in snapshot.monsters)
    if total_max <= 0:
        return 0.0
    total_cur = sum(max(0, m.hp) for m in snapshot.monsters)
    return total_cur / total_max


def _player_hp_fraction(snapshot: CombatSnapshot) -> float:
    """Player current HP as a fraction of max HP, in [0, 1]."""
    if snapshot.player_max_hp <= 0:
        return 0.0
    return max(0, snapshot.player_hp) / snapshot.player_max_hp


def combat_shaping_terms(
    prev: CombatSnapshot, curr: CombatSnapshot, cfg: RewardConfig
) -> dict[str, float]:
    """Coefficient-applied, pre-``beta`` shaping terms for one combat step.

    ``enemy_hp_removed`` rewards the drop in enemy HP fraction; ``damage_taken``
    penalizes the drop in player HP fraction (healing yields a positive
    contribution via the negative coefficient). ``floor_progress`` and
    ``boss_kill`` are run-mode signals and are ``0.0`` in combat mode.
    """
    terms = zero_shaping_terms()
    enemy_removed = _enemy_hp_fraction(prev) - _enemy_hp_fraction(curr)
    player_hp_lost = _player_hp_fraction(prev) - _player_hp_fraction(curr)
    terms["enemy_hp_removed"] = cfg.enemy_hp_removed * enemy_removed
    terms["damage_taken"] = cfg.damage_taken * player_hp_lost
    return terms


def run_shaping_terms(prev: RunSnapshot, curr: RunSnapshot, cfg: RewardConfig) -> dict[str, float]:
    """Coefficient-applied, pre-``beta`` shaping terms for one overworld step.

    ``floor_progress`` rewards descending to new floors (the run's floor number
    only increases), and ``boss_kill`` rewards each act advance, which happens
    exactly when the act boss is defeated. Both deltas are floored at zero so a
    non-progressing transition contributes nothing rather than a spurious
    penalty. On the step that crosses into a new act both terms fire, since the
    boss floor is also a new floor; the spec treats them as independent terms, so
    this double credit is intended. ``enemy_hp_removed`` and ``damage_taken`` are
    combat signals and stay ``0.0`` here; the run adapter applies combat shaping
    on battle steps and this on overworld steps.
    """
    terms = zero_shaping_terms()
    floors_gained = max(0, curr.floor - prev.floor)
    bosses_killed = max(0, curr.act - prev.act)
    terms["floor_progress"] = cfg.floor_progress * floors_gained
    terms["boss_kill"] = cfg.boss_kill * bosses_killed
    return terms


def shaping_reward(terms: dict[str, float], t: int, cfg: RewardConfig) -> float:
    """Annealed shaping reward: ``beta(t) * sum(terms.values())``."""
    return beta(t, cfg) * sum(terms.values())
