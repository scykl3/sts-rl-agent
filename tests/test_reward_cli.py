"""Tests for the shared reward-shaping CLI helper.

Both training entry points (``scripts/train_combat.py`` and
``scripts/train_run.py``) register the four shaping flags and map them back to a
:class:`RewardConfig` through :mod:`sts_rl.env.reward_cli`. These tests lock that
shared helper directly, so the two scripts only need to test that their parser
wires the flags in (they share this mapping, no longer duplicate it).

Engine-free: ``add_reward_shaping_args`` / ``reward_config_from_args`` use only
``argparse`` and :class:`RewardConfig`, so no native build is needed.
"""

from __future__ import annotations

import argparse

from sts_rl.env.reward import RewardConfig
from sts_rl.env.reward_cli import add_reward_shaping_args, reward_config_from_args


def _reward_ns(**overrides: float) -> argparse.Namespace:
    """A Namespace carrying the four reward-shaping flag dests, each defaulting to
    the matching ``RewardConfig()`` value; overrides replace individual coefficients.

    Lets ``reward_config_from_args`` be exercised without a parser, since the
    helper only reads these four attributes.
    """
    base = RewardConfig()
    values: dict[str, float] = {
        "enemy_hp_removed_coef": base.enemy_hp_removed,
        "damage_taken_coef": base.damage_taken,
        "floor_progress_coef": base.floor_progress,
        "boss_kill_coef": base.boss_kill,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_reward_config_from_args_maps_all_four_coefficients() -> None:
    """Each shaping flag maps to its RewardConfig field, and the un-exposed anneal
    schedule (beta_min / t_anneal) keeps the RewardConfig default."""
    cfg = reward_config_from_args(
        _reward_ns(
            enemy_hp_removed_coef=0.1,
            damage_taken_coef=-0.3,
            floor_progress_coef=0.4,
            boss_kill_coef=1.0,
        )
    )
    assert (cfg.enemy_hp_removed, cfg.damage_taken, cfg.floor_progress, cfg.boss_kill) == (
        0.1,
        -0.3,
        0.4,
        1.0,
    )
    assert cfg.beta_min == RewardConfig().beta_min
    assert cfg.t_anneal == RewardConfig().t_anneal


def test_reward_config_from_args_defaults_reproduce_reward_config() -> None:
    """With every flag at its default, the helper reproduces RewardConfig() exactly,
    so an unspecified run keeps the default shaping."""
    assert reward_config_from_args(_reward_ns()) == RewardConfig()


def test_add_reward_shaping_args_defaults_reconstruct_reward_config() -> None:
    """Registering the flags on a bare parser defaults every coefficient to its
    RewardConfig() value, so parsing an empty argv round-trips to RewardConfig()."""
    parser = argparse.ArgumentParser()
    add_reward_shaping_args(parser)
    assert reward_config_from_args(parser.parse_args([])) == RewardConfig()


def test_add_reward_shaping_args_single_flag_overrides_only_its_field() -> None:
    """Overriding one flag changes only its coefficient; the rest stay at default."""
    parser = argparse.ArgumentParser()
    add_reward_shaping_args(parser)
    overridden = reward_config_from_args(parser.parse_args(["--boss-kill-coef", "1.0"]))
    assert overridden.boss_kill == 1.0
    assert overridden.enemy_hp_removed == RewardConfig().enemy_hp_removed
    assert overridden.damage_taken == RewardConfig().damage_taken
    assert overridden.floor_progress == RewardConfig().floor_progress
