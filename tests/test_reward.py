"""Tests for potential-based reward shaping (engine-independent).

:mod:`sts_rl.env.reward` accesses snapshots by attribute only, so these tests use
lightweight stand-ins and run without a built engine. They cover the per-term
potential :func:`state_potentials`, the shaping delta
:func:`shaping_delta` (``gamma * Phi(s') - Phi(s)``), and the policy-invariance
(telescoping) property that motivates the potential-based form.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sts_rl.env.reward import (
    DEFAULT_BOSS_KILL_COEF,
    DEFAULT_DAMAGE_TAKEN_COEF,
    DEFAULT_ENEMY_HP_REMOVED_COEF,
    DEFAULT_FLOOR_PROGRESS_COEF,
    RewardConfig,
    shaping_delta,
    state_potentials,
    zero_shaping_terms,
)
from sts_rl.interface import SHAPING_TERMS


@dataclass
class _Monster:
    hp: int
    max_hp: int


@dataclass
class _Combat:
    player_hp: int
    player_max_hp: int
    monsters: tuple[_Monster, ...]


@dataclass
class _Run:
    floor: int
    act: int
    player_hp: int
    player_max_hp: int


def _combat(
    player_hp: int, monsters: tuple[tuple[int, int], ...], *, player_max: int = 80
) -> _Combat:
    # monsters is a tuple of (current_hp, max_hp); max_hp is fixed for the fight
    # and does not shrink when a monster is damaged or dies.
    return _Combat(player_hp, player_max, tuple(_Monster(hp, mx) for hp, mx in monsters))


def _run(floor: int, act: int, *, player_hp: int = 80, player_max: int = 80) -> _Run:
    return _Run(floor, act, player_hp, player_max)


# --- config / coverage -----------------------------------------------------


def test_zero_shaping_terms_covers_every_interface_term() -> None:
    terms = zero_shaping_terms()
    assert set(terms) == set(SHAPING_TERMS)
    assert all(v == 0.0 for v in terms.values())


def test_default_coefficients_match_constants() -> None:
    cfg = RewardConfig()
    assert cfg.enemy_hp_removed == DEFAULT_ENEMY_HP_REMOVED_COEF
    assert cfg.damage_taken == DEFAULT_DAMAGE_TAKEN_COEF
    assert cfg.floor_progress == DEFAULT_FLOOR_PROGRESS_COEF
    assert cfg.boss_kill == DEFAULT_BOSS_KILL_COEF


# --- state_potentials ------------------------------------------------------


def test_terminal_state_has_all_zero_potential() -> None:
    # No combat and no run view is the Phi(terminal) := 0 convention.
    terms = state_potentials(RewardConfig())
    assert set(terms) == set(SHAPING_TERMS)
    assert all(v == 0.0 for v in terms.values())


def test_enemy_potential_zero_at_full_enemy_hp() -> None:
    # Combat entry: all monsters at full HP -> (1 - 1) = 0, so no boundary jump
    # from the overworld (enemy potential 0 there too).
    cfg = RewardConfig()
    terms = state_potentials(cfg, combat=_combat(80, ((48, 48), (48, 48))))
    assert terms["enemy_hp_removed"] == 0.0


def test_enemy_potential_rises_as_enemies_are_damaged() -> None:
    cfg = RewardConfig()
    # current 24 / max 96 -> fraction 0.25, so potential = coef * (1 - 0.25).
    terms = state_potentials(cfg, combat=_combat(80, ((24, 48), (0, 48))))
    assert terms["enemy_hp_removed"] == pytest.approx(cfg.enemy_hp_removed * 0.75)


def test_enemy_potential_zero_without_a_combat_view() -> None:
    # Combat-local: the overworld has no enemies, so the term is absent (0).
    terms = state_potentials(RewardConfig(), run=_run(5, 1))
    assert terms["enemy_hp_removed"] == 0.0


def test_player_hp_potential_uses_combat_view_when_present() -> None:
    cfg = RewardConfig()
    # Live combat HP 40/80 = 0.5; a stale run view at full HP must be ignored.
    combat = _combat(40, ((48, 48),), player_max=80)
    run = _run(5, 1, player_hp=80, player_max=80)
    terms = state_potentials(cfg, combat=combat, run=run)
    assert terms["damage_taken"] == pytest.approx(cfg.damage_taken * (1 - 0.5))


def test_player_hp_potential_uses_run_view_without_combat() -> None:
    cfg = RewardConfig()
    terms = state_potentials(cfg, run=_run(5, 1, player_hp=60, player_max=80))
    assert terms["damage_taken"] == pytest.approx(cfg.damage_taken * (1 - 0.75))


def test_floor_and_act_potentials_come_from_the_run_view() -> None:
    cfg = RewardConfig()
    terms = state_potentials(cfg, run=_run(6, 2))
    assert terms["floor_progress"] == pytest.approx(cfg.floor_progress * 6)
    assert terms["boss_kill"] == pytest.approx(cfg.boss_kill * 2)


def test_floor_and_act_potentials_zero_without_a_run_view() -> None:
    # The single-combat env has no run view, so these run-level terms are 0.
    terms = state_potentials(RewardConfig(), combat=_combat(80, ((48, 48),)))
    assert terms["floor_progress"] == 0.0
    assert terms["boss_kill"] == 0.0


# --- degenerate inputs -----------------------------------------------------


def test_empty_roster_enemy_fraction_is_defined() -> None:
    # No monsters -> enemy fraction defined as 0.0 (no divide-by-zero), so the
    # potential is coef * (1 - 0) = coef (all enemies gone). In the adapters this
    # state is a combat end and is cashed out to 0, so it never leaks as a jump.
    cfg = RewardConfig()
    terms = state_potentials(cfg, combat=_combat(80, ()))
    assert terms["enemy_hp_removed"] == pytest.approx(cfg.enemy_hp_removed)


def test_zero_player_max_hp_does_not_divide_by_zero() -> None:
    cfg = RewardConfig()
    terms = state_potentials(cfg, combat=_combat(0, ((48, 48),), player_max=0))
    # fraction defined as 0.0, so damage potential = coef * (1 - 0); finite, no crash.
    assert terms["damage_taken"] == pytest.approx(cfg.damage_taken)


def test_overkill_negative_enemy_hp_is_clamped() -> None:
    # Overkill can report negative hp; it floors to 0, so the fraction is 0 and the
    # potential caps at coef (a monster cannot yield more than its full max hp).
    cfg = RewardConfig()
    terms = state_potentials(cfg, combat=_combat(80, ((-10, 48),)))
    assert terms["enemy_hp_removed"] == pytest.approx(cfg.enemy_hp_removed)


# --- shaping_delta ---------------------------------------------------------


def test_shaping_delta_is_gamma_curr_minus_prev() -> None:
    prev = {
        "enemy_hp_removed": 0.02,
        "damage_taken": -0.01,
        "floor_progress": 0.10,
        "boss_kill": 0.2,
    }
    curr = {
        "enemy_hp_removed": 0.05,
        "damage_taken": -0.02,
        "floor_progress": 0.12,
        "boss_kill": 0.2,
    }
    d1 = shaping_delta(prev, curr, gamma=1.0)
    assert d1["enemy_hp_removed"] == pytest.approx(0.05 - 0.02)
    assert d1["floor_progress"] == pytest.approx(0.12 - 0.10)
    d09 = shaping_delta(prev, curr, gamma=0.9)
    assert d09["floor_progress"] == pytest.approx(0.9 * 0.12 - 0.10)
    assert d09["boss_kill"] == pytest.approx(0.9 * 0.2 - 0.2)


def test_gamma_one_recovers_the_additive_enemy_delta() -> None:
    # At gamma = 1 the per-step F equals the old additive term: dealing 50% of
    # enemy HP yields coef * 0.5 (Phi(s')-Phi(s) = coef*((1-0.5)-(1-1))).
    cfg = RewardConfig()
    prev = state_potentials(cfg, combat=_combat(80, ((48, 48), (48, 48))))  # frac 1.0
    curr = state_potentials(cfg, combat=_combat(80, ((0, 48), (48, 48))))  # frac 0.5
    d = shaping_delta(prev, curr, gamma=1.0)
    assert d["enemy_hp_removed"] == pytest.approx(cfg.enemy_hp_removed * 0.5)


def test_gamma_one_recovers_the_additive_damage_delta() -> None:
    # Losing 25% of max HP yields coef_damage * 0.25 (coef_damage is negative).
    cfg = RewardConfig()
    prev = state_potentials(cfg, combat=_combat(80, ((48, 48),), player_max=80))  # frac 1.0
    curr = state_potentials(cfg, combat=_combat(60, ((48, 48),), player_max=80))  # frac 0.75
    d = shaping_delta(prev, curr, gamma=1.0)
    assert d["damage_taken"] == pytest.approx(cfg.damage_taken * 0.25)
    assert d["damage_taken"] < 0.0


# --- policy invariance (telescoping) ---------------------------------------


@pytest.mark.parametrize("gamma", [1.0, 0.99, 0.9])
def test_discounted_shaping_telescopes_to_minus_phi0(gamma: float) -> None:
    """sum_t gamma^t * F_t == gamma^T * Phi(s_T) - Phi(s_0) == -Phi(s_0).

    This is the potential-based policy-invariance guarantee: the discounted sum
    of the shaping over any trajectory depends only on the endpoints, and with
    Phi(terminal) := 0 it is the constant -Phi(s_0). Holds per term for any Phi
    sequence, so it locks the shaping to a pure telescoping difference.
    """
    cfg = RewardConfig()
    # An arbitrary trajectory: overworld -> combat -> more combat -> overworld -> terminal.
    phis = [
        state_potentials(cfg, run=_run(0, 1, player_hp=80, player_max=80)),
        state_potentials(cfg, combat=_combat(70, ((30, 48),)), run=_run(1, 1)),
        state_potentials(cfg, combat=_combat(60, ((10, 48),)), run=_run(1, 1)),
        state_potentials(cfg, run=_run(2, 1, player_hp=60, player_max=80)),
        zero_shaping_terms(),  # Phi(terminal) := 0
    ]
    totals = dict.fromkeys(SHAPING_TERMS, 0.0)
    for t, (prev, curr) in enumerate(zip(phis, phis[1:])):
        delta = shaping_delta(prev, curr, gamma)
        for name in SHAPING_TERMS:
            totals[name] += (gamma**t) * delta[name]
    for name in SHAPING_TERMS:
        assert totals[name] == pytest.approx(-phis[0][name])
