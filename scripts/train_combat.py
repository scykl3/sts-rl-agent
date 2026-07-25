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

    # Periodic eval every 10 iterations + best/last checkpoints, tuned entropy:
    PYTHONPATH=src python scripts/train_combat.py \\
        --eval-every 10 --checkpoint-dir /tmp/sts_ckpt --ent-coef 0.02
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym

from sts_rl.agent.encoder import HIDDEN_DIM
from sts_rl.agent.ppo import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA
from sts_rl.agent.ppo_update import PPOConfig
from sts_rl.agent.train import DEFAULT_LEARNING_RATE, TrainConfig, TrainHistory, train
from sts_rl.eval import EvalReport, evaluate, make_holdout_seeds

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
    # Imported here (and in main) rather than at module scope so importing this
    # module stays engine-free: the adapter pulls in the native engine, and keeping
    # it out of module scope lets the pure helpers here (e.g. _final_eval_report) be
    # unit-tested without a built engine.
    from sts_rl.env.adapter import DEFAULT_MAX_EPISODE_STEPS

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
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=DEFAULT_GAE_LAMBDA,
        help="GAE trace-decay lambda",
    )
    # PPO objective/schedule knobs. Defaults are read from PPOConfig() so they
    # track the canonical source instead of restating its literals here.
    ppo_defaults = PPOConfig()
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=ppo_defaults.clip_coef,
        help="PPO surrogate/value clip coefficient",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=ppo_defaults.vf_coef,
        help="value-loss weight in the PPO objective",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=ppo_defaults.ent_coef,
        help="entropy-bonus weight in the PPO objective",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=ppo_defaults.n_epochs,
        help="PPO epochs per collected rollout",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=ppo_defaults.minibatch_size,
        help="PPO minibatch size",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=ppo_defaults.max_grad_norm,
        help="global grad-norm clip",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=ppo_defaults.target_kl,
        help="approximate-KL early-stop threshold (unset disables the early stop)",
    )
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
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="run a greedy holdout eval every N iterations (unset disables periodic eval)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="directory for best.pt/last.pt checkpoints (unset disables checkpointing)",
    )
    return parser


def _final_eval_report(
    history: TrainHistory,
    eval_env: gym.Env,
    eval_seed_base: int,
    eval_episodes: int,
) -> EvalReport:
    """Return the final policy's holdout EvalReport for the RESULT line.

    When periodic eval ran, train() already evaluated the final iteration over
    this same holdout band on this same eval_env, so reuse that record rather
    than recomputing an identical greedy eval (saves a full holdout pass on the
    real engine). Otherwise evaluate the final net here, passing
    ``deterministic=True`` explicitly so the recompute matches train()'s greedy
    eval even if evaluate()'s default ever changes. The reused and recomputed
    bands match because main() passes the same eval_seed_base/eval_episodes into
    both TrainConfig and this call.
    """
    if history.eval_reports:
        return history.eval_reports[-1].report
    holdout_seeds = make_holdout_seeds(eval_seed_base, eval_episodes)
    return evaluate(history.actor_critic, eval_env, holdout_seeds, deterministic=True)


def main() -> None:
    # INFO level so train()'s per-iteration diagnostics (its module logger) surface.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()

    # Same lazy import as build_arg_parser, keeping this module's import engine-free.
    from sts_rl.env.adapter import StsEnv

    # Two env instances: `env` is the training env (its reset stream is seeded via
    # TrainConfig.seed through the collector), and `eval_env` is a SEPARATE
    # instance for greedy holdout eval - both the in-loop periodic eval and the
    # final eval - so evaluate() resetting per seed never disturbs the training
    # collector's rollout stream.
    env = StsEnv(ascension=args.ascension, max_episode_steps=args.max_episode_steps)
    eval_env = StsEnv(ascension=args.ascension, max_episode_steps=args.max_episode_steps)

    # Provenance for reproducibility: engine_commit and interface_version are
    # always-present info keys. Read them from an initial reset; train() re-resets
    # the env through its collector, so this read has no lasting effect.
    _obs, reset_info = env.reset(seed=args.seed)
    engine_commit = reset_info["engine_commit"]
    interface_version = reset_info["interface_version"]

    ppo_config = PPOConfig(
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )

    config = TrainConfig(
        num_iterations=args.num_iterations,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        anneal_lr=args.anneal_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        ppo=ppo_config,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_seed_base=args.eval_base_seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train (periodic eval + checkpoints run in-loop when enabled), then hand the
    # trained network to the final greedy eval on the same holdout band.
    history = train(env, config, eval_env=eval_env)

    # Eval uses a distinct, reproducible seed band from --eval-base-seed, the SAME
    # band the in-loop periodic eval uses, so the training curve and this final
    # number measure the same holdout. Training samples the full engine-seed range
    # (its in-collect resets are unseeded), so the band is not guaranteed disjoint
    # - overlap is possible but negligible over a run - giving a stable,
    # reproducible win-rate readout.
    report = _final_eval_report(history, eval_env, args.eval_base_seed, args.eval_episodes)
    env.close()
    eval_env.close()

    if args.checkpoint_dir is not None:
        # best.pt is written only when periodic eval is enabled (it needs eval win
        # rates to rank a best); last.pt is always written.
        written = "best.pt, last.pt" if args.eval_every is not None else "last.pt"
        logger.info("checkpoints written to %s (%s)", args.checkpoint_dir, written)

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
