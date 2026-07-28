"""Train the Ironclad agent on full Acts 1-3 runs against the live engine.

Wires the env owner's full-run environment (:class:`~sts_rl.env.run_adapter.StsRunEnv`,
one seeded run per episode) into the agent stack (:func:`~sts_rl.agent.train.train`
plus :func:`~sts_rl.eval.evaluate`): it trains an actor-critic with our PPO on
seeded full runs, optionally warm-started from a migrated combat checkpoint, then
reports the greedy holdout full-run win rate AND the Act 1 clear rate. It is the
run-mode sibling of ``scripts/train_combat.py``; no new learning logic lives here,
only the composition of already-merged pieces.

The learning stack is env-agnostic (observation dict + ``(ACTION_DIM,)`` mask +
scalar reward), so it drives ``StsRunEnv`` unchanged - the only run-mode-specific
additions are the warm-start hook and the Act 1 clear-rate metric. Because it
drives the real C++ engine, it runs only with the engine built
(``scripts/build_engine.sh``); that is expected for an entry point.

Run-scale defaults. A full run is hundreds of agent decisions with sparse,
delayed reward (win/lose the whole run), so the defaults are larger than the
single-combat driver's: ``--max-episode-steps`` is ``3000`` (matching the
adapter's own run-length cap, vs ``500`` for a combat), ``--n-steps`` is ``2048``
(a rollout spans several partial runs; the collector bootstraps the value at the
cutoff, so a rollout need not contain a whole episode), ``--num-iterations`` is
``1000`` (a real run budget, not the tuned milestone sweep), and
``--eval-episodes`` is ``100`` (full-run greedy eval is far costlier per episode
than combat eval, so this trades clear-rate stability against wall-clock). All are
overridable.

Vectorization. ``--num-envs`` currently accepts only ``1`` (single-env training).
The parallel path would build ``SubprocVecEnv(lambda i: StsRunEnv(...))``, but
:class:`~sts_rl.env.vec_env.SubprocVecEnv` types its ``make_env`` factory as
``Callable[[int], StsEnv]`` (the combat env), so a ``StsRunEnv`` factory is
rejected under ``mypy``. Enabling ``--num-envs > 1`` needs a one-line env-owner
widening of that hint to ``Callable[[int], gym.Env]``; until then this driver is
single-env and rejects ``--num-envs > 1`` with a clear error rather than reaching
for a cast.

Usage:

    # Full run (defaults are sane, not a tuned sweep):
    PYTHONPATH=src python scripts/train_run.py

    # Warm-start from a migrated combat checkpoint, with periodic eval:
    PYTHONPATH=src python scripts/train_run.py \\
        --warm-start /tmp/sts_ckpt/best.pt --eval-every 20 --checkpoint-dir /tmp/run_ckpt

    # Short end-to-end smoke (plumbing only; clear rate is near-zero):
    PYTHONPATH=src python scripts/train_run.py \\
        --num-iterations 2 --n-steps 64 --hidden-dim 32 \\
        --max-episode-steps 64 --eval-episodes 4
"""

from __future__ import annotations

import argparse
import logging

import gymnasium as gym

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.checkpoint_migration import load_checkpoint
from sts_rl.agent.deck_economy_wrapper import DeckEconomyShapingWrapper
from sts_rl.agent.encoder import HIDDEN_DIM
from sts_rl.agent.ppo import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA
from sts_rl.agent.ppo_update import PPOConfig
from sts_rl.agent.train import DEFAULT_LEARNING_RATE, TrainConfig, TrainHistory, train
from sts_rl.env.reward_cli import add_reward_shaping_args, reward_config_from_args
from sts_rl.eval import EvalReport, evaluate, make_holdout_seeds

logger = logging.getLogger("train_run")

# Loop-budget defaults: sane for a real full-run training run, not a tuned
# >80%-Act-1-clear milestone sweep. Larger than the single-combat driver's
# because a full run is hundreds of decisions with sparse terminal reward (see
# the module docstring for the rationale behind each).
DEFAULT_NUM_ITERATIONS = 1000
DEFAULT_N_STEPS = 2048
DEFAULT_ASCENSION = 0
# A full run can span up to this many agent decisions before the episode is
# truncated; mirrors StsRunEnv's own run-length cap. Defined here (rather than
# imported from the engine-coupled run_adapter) so this module - and its argument
# parser - imports engine-free and its pure helpers stay unit-testable without a
# native build.
DEFAULT_MAX_EPISODE_STEPS = 3000
# Single-env only for now: the vectorized StsRunEnv path is blocked on an env-owner
# type-hint widening (see the module docstring and _validate_num_envs).
DEFAULT_NUM_ENVS = 1
# Full-run greedy eval is costly per episode (each run is up to
# --max-episode-steps engine steps), so this is smaller than the combat driver's
# 256; still enough for a reasonably stable clear-rate readout.
DEFAULT_EVAL_EPISODES = 100
# Training draws episode seeds from the collector's reset stream, whose in-collect
# resets are unseeded and sample the full engine-seed range; the eval band is a
# distinct, reproducible run of seeds from --eval-base-seed. The bands are not
# guaranteed disjoint, so overlap is possible but negligible over a run.
DEFAULT_SEED = 0
DEFAULT_EVAL_BASE_SEED = 1_000_000
# best.pt / TrainHistory.best_eval rank on this EvalReport field instead of the
# default win_rate: a full-run win_rate is ~0 for a long time (every early eval
# ties at 0, making best.pt degenerate), whereas the Act 1 clear rate is the
# informative progress curve this driver trains toward. TrainConfig validates it.
RUN_BEST_METRIC = "act1_clear_rate"


def _validate_num_envs(num_envs: int) -> None:
    """Reject the not-yet-supported vectorized run-mode path.

    Only single-env (``num_envs == 1``) training is supported in this driver. The
    parallel path would build ``SubprocVecEnv(lambda i: StsRunEnv(...))``, but
    ``SubprocVecEnv.make_env`` is typed ``Callable[[int], StsEnv]`` (the combat
    env), so a ``StsRunEnv`` factory does not type-check; enabling it needs a
    one-line env-owner widening of that hint to ``Callable[[int], gym.Env]``. Fail
    fast with that pointer rather than silently degrading or reaching for a cast.
    """
    if num_envs != 1:
        raise NotImplementedError(
            f"--num-envs={num_envs} is not supported yet: run-mode training is "
            f"single-env for now. Vectorized StsRunEnv training is blocked on "
            f"widening SubprocVecEnv.make_env's type hint from "
            f"Callable[[int], StsEnv] to Callable[[int], gym.Env] (an env-owner "
            f"change); pass --num-envs 1."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; every knob has a sane default and is overridable.

    Imports engine-free: run-scale defaults live here as module constants rather
    than being pulled from the engine-coupled ``run_adapter``, so building the
    parser (and unit-testing the pure helpers) needs no native build.
    """
    parser = argparse.ArgumentParser(
        description="Train the Ironclad agent on full Acts 1-3 runs (live engine).",
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
    parser.add_argument(
        "--deck-economy-shaping",
        action="store_true",
        help="add potential-based deck/economy reward shaping (relics + gold + potions) to "
        "the TRAINING env only; policy-invariant, un-annealed (see "
        "sts_rl.agent.deck_economy_wrapper)",
    )
    parser.add_argument(
        "--deck-economy-scale",
        type=float,
        default=1.0,
        help="strength multiplier on the deck/economy potential (only used with "
        "--deck-economy-shaping)",
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
    # Reward-shaping coefficients (shared with train_combat). floor_progress and
    # boss_kill drive overworld shaping (boss_kill fires once per act boss
    # defeated); enemy_hp_removed and damage_taken drive combat shaping.
    add_reward_shaping_args(parser)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="encoder trunk width")
    parser.add_argument("--ascension", type=int, default=DEFAULT_ASCENSION, help="ascension level")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="per-episode step cap (episode truncates when hit)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help="parallel training envs (only 1 is supported for now; see module docstring)",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help=(
            "path to a checkpoint (best.pt/last.pt) to warm-start the network from; "
            "loaded and migrated to the current action layout via load_checkpoint, "
            "and its trunk width overrides --hidden-dim (unset trains from scratch)"
        ),
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
    parser.add_argument(
        "--value-warmup-iters",
        type=int,
        default=0,
        help="freeze the trunk + policy head for the first N iterations so only the "
        "value head trains (0 disables; calibrates the critic before it can corrupt "
        "the shared trunk)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="stop once this many consecutive post-warmup evals fail to improve the "
        "ranked metric (unset disables early stop)",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="minimum ranked-metric gain that counts as an improvement for early stop; "
        "the eval metric is noisy (a ~100-episode clear rate has std ~0.03-0.04), so set "
        "this above roughly one eval-std to avoid noise-driven premature or deferred stops",
    )
    return parser


def _final_eval_report(
    history: TrainHistory,
    eval_env: gym.Env,
    eval_seed_base: int,
    eval_episodes: int,
) -> EvalReport:
    """Return the final policy's holdout EvalReport for the RESULT line.

    When periodic eval ran, train() already evaluated the final iteration over a
    holdout band on this same eval_env, so reuse that last snapshot rather than
    recomputing an identical greedy eval (saves a full holdout pass on the real
    engine, which for full runs is expensive) - but ONLY when it matches the
    requested band: its recorded ``eval_seed_base`` equals ``eval_seed_base`` and
    its report's ``n_episodes`` equals ``eval_episodes``. On a mismatch (or when no
    periodic eval ran), recompute here over
    ``make_holdout_seeds(eval_seed_base, eval_episodes)``, passing
    ``deterministic=True`` explicitly so the recompute matches train()'s greedy
    eval even if evaluate()'s default ever changes.
    """
    if history.eval_reports:
        last = history.eval_reports[-1]
        if last.eval_seed_base == eval_seed_base and last.report.n_episodes == eval_episodes:
            return last.report
    holdout_seeds = make_holdout_seeds(eval_seed_base, eval_episodes)
    return evaluate(history.actor_critic, eval_env, holdout_seeds, deterministic=True)


def _load_warm_start(path: str) -> tuple[ActorCritic, int]:
    """Load (and migrate) a warm-start checkpoint; return the net and its trunk width.

    Reads the checkpoint at ``path`` via
    :func:`~sts_rl.agent.checkpoint_migration.load_checkpoint` (which migrates the
    policy head to the current action layout when the checkpoint predates it) and
    returns the loaded :class:`~sts_rl.agent.actor_critic.ActorCritic` together
    with its encoder trunk width (``encoder.output_dim``). That width becomes
    ``config.hidden_dim`` so the two always agree - train() asserts they match,
    and the width is recorded in any checkpoints this run writes. ``load_checkpoint``
    is engine-free, so this runs before the engine import in main().
    """
    net = load_checkpoint(path)
    return net, net.encoder.output_dim


def main() -> None:
    # INFO level so train()'s per-iteration diagnostics (its module logger) surface.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()

    # Single-env only for now; reject the vectorized request before touching the
    # engine so the message surfaces without a build.
    _validate_num_envs(args.num_envs)

    # Optional warm-start: load (and migrate) the checkpoint into an ActorCritic
    # via _load_warm_start, then train from it. Its trunk width defines
    # config.hidden_dim so the two always agree - train() asserts they match, and
    # the width is recorded in any checkpoints this run writes. load_checkpoint is
    # engine-free, so this runs before the engine import below.
    warm_net: ActorCritic | None = None
    hidden_dim = args.hidden_dim
    if args.warm_start is not None:
        warm_net, hidden_dim = _load_warm_start(args.warm_start)
        if hidden_dim != args.hidden_dim:
            logger.info(
                "warm-start: using checkpoint trunk width %d (overriding --hidden-dim %d)",
                hidden_dim,
                args.hidden_dim,
            )

    # Imported here (not at module scope) so this module imports engine-free: the
    # run adapter pulls in the native engine, and keeping it out of module scope
    # lets the pure helpers (build_arg_parser, _validate_num_envs, _final_eval_report)
    # be unit-tested without a built engine.
    from sts_rl.env.run_adapter import StsRunEnv

    # Same shaping config for both envs so training and holdout eval score the same
    # reward; unset flags reproduce the RewardConfig() default.
    reward_config = reward_config_from_args(args)

    # Two env instances: `env` is the training env (its reset stream is seeded via
    # TrainConfig.seed through the collector), and `eval_env` is a SEPARATE
    # instance for greedy holdout eval - both the in-loop periodic eval and the
    # final eval - so evaluate() resetting per seed never disturbs the training
    # collector's rollout stream.
    env = StsRunEnv(
        ascension=args.ascension,
        max_episode_steps=args.max_episode_steps,
        reward_config=reward_config,
    )
    eval_env = StsRunEnv(
        ascension=args.ascension,
        max_episode_steps=args.max_episode_steps,
        reward_config=reward_config,
    )

    # Optional potential-based deck/economy shaping on the TRAINING env ONLY. gamma is
    # single-sourced from args.gamma (the same discount TrainConfig/GAE use) so the term
    # telescopes against the return and stays policy-invariant. eval_env is deliberately
    # left unwrapped: the reported metric is terminal-win based, so eval reward stays clean.
    if args.deck_economy_shaping:
        env = DeckEconomyShapingWrapper(env, gamma=args.gamma, scale=args.deck_economy_scale)
        logger.info(
            "deck-economy shaping ENABLED on train env (scale=%.3f, gamma=%.3f)",
            args.deck_economy_scale,
            args.gamma,
        )

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
        hidden_dim=hidden_dim,
        ppo=ppo_config,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_seed_base=args.eval_base_seed,
        checkpoint_dir=args.checkpoint_dir,
        # Opt-in warmup + early-stop, passed explicitly at run time; defaults OFF
        # (warmup 0, patience None) leave the driver's default behavior unchanged.
        value_warmup_iters=args.value_warmup_iters,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        # Rank best.pt / best_eval on the Act 1 clear rate, not the default win_rate
        # (full-run win_rate is ~0 for a long time; see RUN_BEST_METRIC).
        best_metric=RUN_BEST_METRIC,
    )

    # Train (periodic eval + checkpoints run in-loop when enabled), warm-started
    # when --warm-start was given, then hand the trained network to the final
    # greedy eval on the same holdout band.
    history = train(env, config, eval_env=eval_env, init_actor_critic=warm_net)

    # Eval uses a distinct, reproducible seed band from --eval-base-seed, the SAME
    # band the in-loop periodic eval uses, so the training curve and this final
    # number measure the same holdout.
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
    # Headline (final line): the greedy holdout Act 1 clear rate is the run's
    # primary result (the >80% Act 1 clear-rate goal this driver trains toward),
    # reported alongside the full-run win rate (the whole-run victory fraction, a
    # strict subset of the clear rate).
    logger.info(
        "RESULT greedy act1_clear_rate=%.3f full_run_win_rate=%.3f over %d holdout "
        "episodes (eval_base_seed=%d)",
        report.act1_clear_rate,
        report.win_rate,
        report.n_episodes,
        args.eval_base_seed,
    )


if __name__ == "__main__":
    main()
