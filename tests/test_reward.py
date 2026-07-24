"""Tests for reward shaping and the anneal schedule (engine-independent).

:mod:`sts_rl.env.reward` accesses combat snapshots by attribute only, so these
tests use lightweight stand-ins and run without a built engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sts_rl.env.reward import (
    DEFAULT_DAMAGE_TAKEN_COEF,
    DEFAULT_ENEMY_HP_REMOVED_COEF,
    DEFAULT_T_ANNEAL,
    RewardConfig,
    beta,
    combat_shaping_terms,
    shaping_reward,
    zero_shaping_terms,
)
from sts_rl.interface import SHAPING_TERMS


@dataclass
class _Monster:
    hp: int
    max_hp: int


@dataclass
class _Snap:
    player_hp: int
    player_max_hp: int
    monsters: tuple[_Monster, ...]


def _snap(player_hp: int, monsters: tuple[tuple[int, int], ...], *, player_max: int = 80) -> _Snap:
    # monsters is a tuple of (current_hp, max_hp); max_hp is fixed for the fight
    # and does not shrink when a monster is damaged or dies.
    return _Snap(
        player_hp=player_hp,
        player_max_hp=player_max,
        monsters=tuple(_Monster(hp=hp, max_hp=max_hp) for hp, max_hp in monsters),
    )


# --- beta schedule ---------------------------------------------------------


def test_beta_starts_at_one() -> None:
    assert beta(0, RewardConfig()) == 1.0


def test_beta_reaches_beta_min_at_anneal_horizon() -> None:
    cfg = RewardConfig(beta_min=0.0, t_anneal=1000.0)
    assert beta(1000, cfg) == 0.0
    assert beta(5000, cfg) == 0.0  # clamped, never negative


def test_beta_is_monotone_non_increasing() -> None:
    cfg = RewardConfig(t_anneal=1000.0)
    vals = [beta(t, cfg) for t in range(0, 1200, 100)]
    assert all(later <= earlier for earlier, later in zip(vals, vals[1:]))


def test_beta_floor_respected() -> None:
    cfg = RewardConfig(beta_min=0.1, t_anneal=1000.0)
    assert beta(10_000, cfg) == pytest.approx(0.1)


def test_beta_halfway() -> None:
    cfg = RewardConfig(beta_min=0.0, t_anneal=1000.0)
    assert beta(250, cfg) == pytest.approx(0.75)


# --- RewardConfig validation ----------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_reward_config_rejects_nonpositive_t_anneal(bad: float) -> None:
    with pytest.raises(ValueError):
        RewardConfig(t_anneal=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_reward_config_rejects_out_of_range_beta_min(bad: float) -> None:
    with pytest.raises(ValueError):
        RewardConfig(beta_min=bad)


def test_default_t_anneal_matches_spec() -> None:
    assert RewardConfig().t_anneal == DEFAULT_T_ANNEAL


# --- shaping terms ---------------------------------------------------------


def test_zero_shaping_terms_covers_every_interface_term() -> None:
    terms = zero_shaping_terms()
    assert set(terms) == set(SHAPING_TERMS)
    assert all(v == 0.0 for v in terms.values())


def test_no_state_change_yields_zero_terms() -> None:
    snap = _snap(80, ((48, 48), (48, 48)))
    terms = combat_shaping_terms(snap, snap, RewardConfig())
    assert all(v == 0.0 for v in terms.values())


def test_dealing_enemy_damage_rewards_positive() -> None:
    cfg = RewardConfig()
    prev = _snap(80, ((48, 48), (48, 48)))  # total 96 / 96
    curr = _snap(80, ((0, 48), (48, 48)))  # total 48 / 96 -> fraction dropped by 0.5
    terms = combat_shaping_terms(prev, curr, cfg)
    assert terms["enemy_hp_removed"] == pytest.approx(DEFAULT_ENEMY_HP_REMOVED_COEF * 0.5)
    assert terms["damage_taken"] == 0.0


def test_taking_damage_penalizes() -> None:
    cfg = RewardConfig()
    prev = _snap(80, ((48, 48),))
    curr = _snap(60, ((48, 48),))  # lost 20/80 = 0.25 of max hp
    terms = combat_shaping_terms(prev, curr, cfg)
    # Negative coefficient on a positive hp-loss fraction -> negative reward.
    assert terms["damage_taken"] == pytest.approx(DEFAULT_DAMAGE_TAKEN_COEF * 0.25)
    assert terms["damage_taken"] < 0.0


def test_player_healing_gives_positive_contribution() -> None:
    cfg = RewardConfig()
    prev = _snap(40, ((48, 48),))
    curr = _snap(60, ((48, 48),))  # gained 20/80 -> hp-lost fraction is -0.25
    terms = combat_shaping_terms(prev, curr, cfg)
    assert terms["damage_taken"] == pytest.approx(-DEFAULT_DAMAGE_TAKEN_COEF * 0.25)
    assert terms["damage_taken"] > 0.0


def test_run_mode_terms_stay_zero_in_combat() -> None:
    prev = _snap(80, ((48, 48),))
    curr = _snap(60, ((0, 48),))
    terms = combat_shaping_terms(prev, curr, RewardConfig())
    assert terms["floor_progress"] == 0.0
    assert terms["boss_kill"] == 0.0


# --- degenerate inputs -----------------------------------------------------


def test_no_monsters_yields_zero_enemy_term() -> None:
    # Empty roster (or zero total max hp): the fraction is defined as 0.0, so
    # there is no delta and no division by zero.
    prev = _snap(80, ())
    curr = _snap(80, ())
    terms = combat_shaping_terms(prev, curr, RewardConfig())
    assert terms["enemy_hp_removed"] == 0.0


def test_zero_player_max_hp_yields_zero_damage_term() -> None:
    prev = _snap(0, ((48, 48),), player_max=0)
    curr = _snap(0, ((48, 48),), player_max=0)
    terms = combat_shaping_terms(prev, curr, RewardConfig())
    assert terms["damage_taken"] == 0.0


def test_negative_enemy_hp_is_clamped() -> None:
    # Overkill can report negative hp; it is floored to 0, so the removed
    # fraction caps at 1.0 (a monster cannot yield more than its full max hp).
    cfg = RewardConfig()
    prev = _snap(80, ((48, 48),))
    curr = _snap(80, ((-10, 48),))
    terms = combat_shaping_terms(prev, curr, cfg)
    assert terms["enemy_hp_removed"] == pytest.approx(cfg.enemy_hp_removed * 1.0)


def test_enemy_summon_perturbs_removed_term() -> None:
    # A mid-combat summon grows the roster, shifting the current-roster
    # denominator, so enemy_hp_removed can go negative with no damage dealt.
    # Documented, accepted behavior (annealed away by beta); locked here.
    cfg = RewardConfig()
    prev = _snap(80, ((24, 48),))  # fraction 24/48 = 0.5
    curr = _snap(80, ((24, 48), (40, 40)))  # fraction 64/88
    terms = combat_shaping_terms(prev, curr, cfg)
    assert terms["enemy_hp_removed"] < 0.0
    assert terms["enemy_hp_removed"] == pytest.approx(cfg.enemy_hp_removed * (0.5 - 64 / 88))


# --- shaping_reward assembly ----------------------------------------------


def test_shaping_reward_scales_sum_by_beta() -> None:
    cfg = RewardConfig(beta_min=0.0, t_anneal=1000.0)
    terms = {
        "enemy_hp_removed": 0.05,
        "damage_taken": -0.01,
        "floor_progress": 0.0,
        "boss_kill": 0.0,
    }
    # beta(250) = 0.75
    assert shaping_reward(terms, 250, cfg) == pytest.approx(0.75 * 0.04)


def test_shaping_reward_vanishes_after_anneal() -> None:
    cfg = RewardConfig(beta_min=0.0, t_anneal=1000.0)
    terms = {
        "enemy_hp_removed": 0.05,
        "damage_taken": -0.01,
        "floor_progress": 0.0,
        "boss_kill": 0.0,
    }
    assert shaping_reward(terms, 2000, cfg) == 0.0
