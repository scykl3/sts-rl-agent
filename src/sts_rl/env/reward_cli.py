"""Shared CLI wiring for the reward-shaping coefficients.

Both training entry points (``scripts/train_combat.py`` and
``scripts/train_run.py``) expose the four :class:`RewardConfig` shaping
coefficients as flags. Registering the flags and mapping them back to a
:class:`RewardConfig` lives here, in one place, so adding a fifth coefficient
touches this module rather than both scripts.

Engine-free: only ``argparse`` and :class:`RewardConfig`, so importing this (and
the scripts' pure helpers that use it) needs no native engine build.
"""

from __future__ import annotations

import argparse

from sts_rl.env.reward import RewardConfig


def add_reward_shaping_args(parser: argparse.ArgumentParser) -> None:
    """Add the four reward-shaping coefficient flags to ``parser``.

    Each flag defaults to the corresponding ``RewardConfig()`` field, so a run
    that sets none of them reproduces the default shaping exactly. These are the
    four potential weights of the potential-based shaping (see
    :mod:`sts_rl.env.reward`); :class:`RewardConfig` carries no other fields.

    ``floor_progress`` and ``boss_kill`` are overworld signals: they fire only in
    full-run training and stay ``0.0`` within a single combat, so on the combat
    driver they are inert (exposed there for parity with the run driver).
    """
    reward_defaults = RewardConfig()
    parser.add_argument(
        "--enemy-hp-removed-coef",
        type=float,
        default=reward_defaults.enemy_hp_removed,
        help="shaping weight for the drop in enemy HP fraction (combat)",
    )
    parser.add_argument(
        "--damage-taken-coef",
        type=float,
        default=reward_defaults.damage_taken,
        help="shaping weight for the drop in player HP fraction (negative penalizes damage)",
    )
    parser.add_argument(
        "--floor-progress-coef",
        type=float,
        default=reward_defaults.floor_progress,
        help="shaping weight per new floor descended (overworld; 0.0 in a single combat)",
    )
    parser.add_argument(
        "--boss-kill-coef",
        type=float,
        default=reward_defaults.boss_kill,
        help="shaping weight per act boss defeated (overworld; 0.0 in a single combat)",
    )


def reward_config_from_args(args: argparse.Namespace) -> RewardConfig:
    """Build the env's :class:`RewardConfig` from the reward-shaping CLI flags.

    The four potential weights are the only :class:`RewardConfig` fields, and each
    flag defaults to the corresponding ``RewardConfig()`` value, so an unspecified
    run reproduces the default shaping exactly.
    """
    return RewardConfig(
        enemy_hp_removed=args.enemy_hp_removed_coef,
        damage_taken=args.damage_taken_coef,
        floor_progress=args.floor_progress_coef,
        boss_kill=args.boss_kill_coef,
    )
