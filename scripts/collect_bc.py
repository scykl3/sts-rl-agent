"""CLI: Collect a behavior-cloning dataset for the card-reward-pick decision.

Loads a warm-start checkpoint as the driving policy for non-card decisions, runs
episodes in the full run-mode environment, and records teacher actions at every
card-reward step. The resulting dataset is written to disk for offline BC
training via ``train_bc.py``.

Usage:
    python scripts/collect_bc.py --warm-start <checkpoint> --output bc_data.npz \
        --n-episodes 200 --seed 42
"""

from __future__ import annotations

import argparse
import logging

DEFAULT_N_EPISODES = 200
DEFAULT_SEED = 42
DEFAULT_OUTPUT = "bc_dataset.npz"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a BC dataset for the card-reward-pick head."
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        required=True,
        help="Path to a checkpoint (best.pt) for the driving policy.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path for the BC dataset (.npz). Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULT_N_EPISODES,
        help=f"Number of episodes to collect. Default: {DEFAULT_N_EPISODES}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base seed for env resets. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use greedy (deterministic) actions for the base policy. Default: True",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic",
        help="Use stochastic (sampled) actions for the base policy.",
    )
    parser.add_argument(
        "--ascension",
        type=int,
        default=0,
        help="Ascension level for the run env. Default: 0",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()

    # Engine-dependent imports deferred to here so the CLI is parseable without
    # the engine (tests import and inspect the arg parser without running main).
    from sts_rl.agent.bc import collect_bc_dataset
    from sts_rl.agent.card_teacher import CardRewardTeacher
    from sts_rl.agent.checkpoint_migration import load_checkpoint
    from sts_rl.env.run_adapter import StsRunEnv

    # Load the driving policy
    policy = load_checkpoint(args.warm_start)
    policy.eval()

    # Build the run-mode environment
    env = StsRunEnv(ascension=args.ascension)

    # Build the teacher
    teacher = CardRewardTeacher(validate=True)

    # Collect
    dataset = collect_bc_dataset(
        env,
        policy,
        teacher,
        n_episodes=args.n_episodes,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    # Save
    dataset.save(args.output)
    logging.getLogger(__name__).info(
        "Done. %d samples saved to %s (skip_rate=%.3f)",
        len(dataset),
        args.output,
        dataset.manifest.skip_rate,
    )


if __name__ == "__main__":
    main()
