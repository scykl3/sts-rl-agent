"""Train the Ironclad agent on single Act 1 combats against the live engine.

Wires the merged environment surface (:class:`~sts_rl.env.adapter.StsEnv`) and
the agent stack (:func:`~sts_rl.agent.train.train` plus
:func:`~sts_rl.eval.evaluate`) into one runnable entry point: it trains a fresh
actor-critic with our PPO on seeded single combats, then reports the greedy win
rate over a distinct, reproducible eval seed band. No new learning logic lives
here, only the composition of already-merged pieces.

Because it drives the real C++ engine through ``StsEnv``, it runs only with the
engine built (``scripts/build_engine.sh``); that is expected for an entry point.

Usage:

    # Full single-combat run (defaults are sane, not the tuned milestone sweep):
    PYTHONPATH=src python scripts/train_combat.py

    # Override the budget and enable learning-rate annealing:
    PYTHONPATH=src python scripts/train_combat.py \\
        --num-iterations 400 --n-steps 4096 --anneal-lr

    # Short end-to-end smoke (plumbing only; win rate is near-random):
    PYTHONPATH=src python scripts/train_combat.py \\
        --num-iterations 2 --n-steps 64 --hidden-dim 32 --eval-episodes 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sts_rl.agent.encoder import HIDDEN_DIM
from sts_rl.agent.ppo import DEFAULT_GAMMA
from sts_rl.agent.train import DEFAULT_LEARNING_RATE, TrainConfig, train
from sts_rl.env.adapter import DEFAULT_MAX_EPISODE_STEPS, StsEnv
from sts_rl.eval import evaluate, make_holdout_seeds

REPO_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("train_combat")

# Loop-budget defaults: sane for a real single-combat run, not the tuned
# >=90% win-rate milestone sweep (that is a separate, longer run).
DEFAULT_NUM_ITERATIONS = 200
DEFAULT_N_STEPS = 2048
DEFAULT_ASCENSION = 0
DEFAULT_EVAL_EPISODES = 256
# Training draws episode seeds from the collector's reset stream, whose in-collect
# resets are unseeded and sample the full engine-seed range; the eval band is a
# distinct, reproducible run of seeds from --eval-base-seed. The bands are not
# guaranteed disjoint, so overlap is possible but negligible over a run.
DEFAULT_SEED = 0
DEFAULT_EVAL_BASE_SEED = 1_000_000


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; every knob has a sane default and is overridable."""
    parser = argparse.ArgumentParser(
        description="Train the Ironclad agent on single Act 1 combats (live engine).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="global + env reset seed")
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help="number of collect->update iterations",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help="transitions collected per iteration",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Adam step size",
    )
    parser.add_argument(
        "--anneal-lr",
        action="store_true",
        help="linearly decay the learning rate across the run",
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="GAE discount factor")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="encoder trunk width")
    parser.add_argument("--ascension", type=int, default=DEFAULT_ASCENSION, help="ascension level")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="per-episode step cap (episode truncates when hit)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="number of greedy holdout episodes",
    )
    parser.add_argument(
        "--eval-base-seed",
        type=int,
        default=DEFAULT_EVAL_BASE_SEED,
        help="first seed of the reproducible eval band (distinct from --seed)",
    )
    return parser


def main() -> None:
    # INFO level so train()'s per-iteration diagnostics (its module logger) surface.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()

    # One env instance serves both phases: training seeds its reset stream via
    # TrainConfig.seed (through the collector), and evaluate() resets it once per
    # holdout seed.
    env = StsEnv(ascension=args.ascension, max_episode_steps=args.max_episode_steps)

    # Provenance for reproducibility: engine_commit and interface_version are
    # always-present info keys. Read them from an initial reset; train() re-resets
    # the env through its collector, so this read has no lasting effect.
    _obs, reset_info = env.reset(seed=args.seed)
    engine_commit = reset_info["engine_commit"]
    interface_version = reset_info["interface_version"]

    config = TrainConfig(
        num_iterations=args.num_iterations,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        anneal_lr=args.anneal_lr,
        gamma=args.gamma,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
    )

    # Train, then hand the trained network straight to the greedy evaluator.
    history = train(env, config)

    # Eval uses a distinct, reproducible seed band from --eval-base-seed. Training
    # samples the full engine-seed range (its in-collect resets are unseeded), so
    # the band is not guaranteed disjoint - overlap is possible but negligible over
    # a run - giving a stable, reproducible win-rate readout.
    holdout_seeds = make_holdout_seeds(args.eval_base_seed, args.eval_episodes)
    report = evaluate(history.actor_critic, env, holdout_seeds)
    env.close()

    logger.info(
        "eval avg_floor=%.2f avg_hp=%.2f avg_ep_len=%.2f avg_return=%.3f",
        report.avg_floor,
        report.avg_hp,
        report.avg_ep_len,
        report.avg_return,
    )
    logger.info(
        "repro seed=%d engine_commit=%s interface_version=%s",
        args.seed,
        engine_commit,
        interface_version,
    )
    # Headline (final line): the greedy holdout win rate is the run's result.
    logger.info(
        "RESULT greedy win_rate=%.3f over %d holdout episodes (eval_base_seed=%d)",
        report.win_rate,
        report.n_episodes,
        args.eval_base_seed,
    )


if __name__ == "__main__":
    main()
