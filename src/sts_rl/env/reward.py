"""Reward computation: terminal signal plus potential-based shaping.

``reward = terminal + F``, where ``F(s, s') = gamma * Phi(s') - Phi(s)`` is
potential-based shaping (Ng, Harada and Russell 1999). Over any trajectory F
telescopes to ``gamma^T * Phi(s_T) - Phi(s_0)``, and with the convention
``Phi(terminal) := 0`` the total shaping is just ``-Phi(s_0)`` - a constant
independent of the path taken. So the shaping adds no bias to the optimal policy;
it only redistributes reward within an episode to give a denser learning signal,
and it needs no anneal. This differs from a heuristic additive bonus, which
biases the policy toward the shaped quantity: a potential term rewards *progress*
along the way but returns that credit at the terminal, so a run that reaches
floor 30 and one that reaches floor 5 earn the same total shaping.

``gamma`` is supplied by the env (single-sourced from the trainer's discount) so
F telescopes against the same return GAE bootstraps; a term shaped with a
different discount would not telescope and would bias the policy.

The potential ``Phi`` is a weighted sum of per-term potentials keyed by
:data:`SHAPING_TERMS`:

- ``enemy_hp_removed``: combat-local. ``w * (1 - enemy_hp_fraction)`` while a
  fight is live, ``0`` otherwise (enemies do not persist to the overworld). It
  therefore telescopes within a combat and cashes out at combat end, and cannot
  inject a spurious jump at the combat/overworld boundary (enemies start a fight
  at full HP, so its combat-entry value is ``0`` too).
- ``damage_taken``: ``w * (1 - player_hp_fraction)`` with ``w`` negative, so
  retaining HP raises the potential and losing it lowers it. Player HP is read
  from the live combat view mid-combat and the run view otherwise, so it is
  continuous across the combat/overworld boundary (the engine syncs combat HP
  back to the run on combat end).
- ``floor_progress``: ``w * floor``; ``boss_kill``: ``w * act``. Run-level and
  monotonic; both are ``0`` when there is no run view (the single-combat env).

``info['shaping_terms']`` reports the per-term F contribution
``gamma * Phi_term(s') - Phi_term(s)``, so ``sum(terms.values())`` reconstructs
the shaped reward exactly. Coefficients are tunable training configuration and
live here, not in the shared interface module; only the term names come from
:data:`SHAPING_TERMS`.
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

# Default potential weights. A positive weight rewards more of its quantity; the
# damage weight is negative so that losing player HP lowers the potential (and a
# per-step HP loss yields a negative shaping contribution). At gamma = 1 the
# per-step contribution of each term equals the old additive per-step delta, so
# these carry over the previously tuned magnitudes.
DEFAULT_ENEMY_HP_REMOVED_COEF = 0.05
DEFAULT_DAMAGE_TAKEN_COEF = -0.02
DEFAULT_FLOOR_PROGRESS_COEF = 0.02
DEFAULT_BOSS_KILL_COEF = 0.20

# Fallback discount for the potential term when a caller constructs an env
# without one. Callers that train should pass the trainer's gamma so the shaping
# telescopes against the same return (see the module docstring); this default is
# only for direct/test construction.
DEFAULT_SHAPING_GAMMA = 1.0


@dataclass(frozen=True)
class RewardConfig:
    """Tunable potential weights, keyed by the same names as :data:`SHAPING_TERMS`.

    These are potential weights for :func:`state_potentials`, not additive
    bonuses: the shaping reward is the telescoping difference
    ``gamma * Phi(s') - Phi(s)`` (see the module docstring), so there is no
    anneal schedule - a potential term is unbiased for any weight.
    """

    enemy_hp_removed: float = DEFAULT_ENEMY_HP_REMOVED_COEF
    damage_taken: float = DEFAULT_DAMAGE_TAKEN_COEF
    floor_progress: float = DEFAULT_FLOOR_PROGRESS_COEF
    boss_kill: float = DEFAULT_BOSS_KILL_COEF


def zero_shaping_terms() -> dict[str, float]:
    """A fresh shaping-term dict with every term at ``0.0``.

    Used for the potential of a terminal state (``Phi(terminal) := 0``) and for
    the reported shaping on the reset step and on an invalid action that did not
    touch the engine, so ``info['shaping_terms']`` is always complete.
    """
    return {name: 0.0 for name in SHAPING_TERMS}


def _enemy_hp_fraction(snapshot: CombatSnapshot) -> float:
    """Total current enemy HP as a fraction of total enemy max HP, in [0, 1].

    Uses the current monster roster (dead monsters read as 0 HP). A mid-combat
    summon changes the roster and can perturb the potential transiently; this is
    accepted because the enemy potential is combat-local and telescopes to zero
    over the fight regardless.
    """
    total_max = sum(m.max_hp for m in snapshot.monsters)
    if total_max <= 0:
        return 0.0
    total_cur = sum(max(0, m.hp) for m in snapshot.monsters)
    return total_cur / total_max


def _player_hp_fraction(snapshot: CombatSnapshot | RunSnapshot) -> float:
    """Player current HP as a fraction of max HP, in [0, 1].

    Works on either snapshot type: both a combat snapshot (live HP mid-fight) and
    a run snapshot (overworld HP) expose ``player_hp`` / ``player_max_hp``.
    """
    if snapshot.player_max_hp <= 0:
        return 0.0
    return max(0, snapshot.player_hp) / snapshot.player_max_hp


def state_potentials(
    cfg: RewardConfig,
    *,
    combat: CombatSnapshot | None = None,
    run: RunSnapshot | None = None,
) -> dict[str, float]:
    """Per-term potential ``Phi(s)`` for one state, keyed by :data:`SHAPING_TERMS`.

    Pass the live ``combat`` snapshot while a fight is active and/or the ``run``
    snapshot for the overworld view. A terminal state passes neither, giving an
    all-zero potential (the ``Phi(terminal) := 0`` convention). Player HP is taken
    from the combat view when present (authoritative mid-fight; the run view's HP
    is stale until combat syncs back), else the run view.
    """
    terms = zero_shaping_terms()
    if combat is not None:
        # Combat-local: 0 when no fight is live, so it cashes out at combat end.
        terms["enemy_hp_removed"] = cfg.enemy_hp_removed * (1.0 - _enemy_hp_fraction(combat))
    hp_source = combat if combat is not None else run
    if hp_source is not None:
        terms["damage_taken"] = cfg.damage_taken * (1.0 - _player_hp_fraction(hp_source))
    if run is not None:
        terms["floor_progress"] = cfg.floor_progress * float(run.floor)
        terms["boss_kill"] = cfg.boss_kill * float(run.act)
    return terms


def shaping_delta(prev: dict[str, float], curr: dict[str, float], gamma: float) -> dict[str, float]:
    """Per-term potential-based shaping ``F = gamma * Phi(s') - Phi(s)``.

    ``prev`` and ``curr`` are per-term potentials from :func:`state_potentials`
    (``curr`` is all-zero at a terminal). ``gamma`` must be the trainer's discount
    so the term telescopes against the return. ``sum(...values())`` is the scalar
    shaping reward added to the terminal signal.
    """
    return {name: gamma * curr[name] - prev[name] for name in SHAPING_TERMS}
