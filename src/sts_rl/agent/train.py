"""PPO outer training loop for the Slay the Spire RL agent.

Ties the already-built pieces - the :class:`~sts_rl.agent.actor_critic.ActorCritic`
network, the :class:`~sts_rl.agent.rollout_collector.RolloutCollector` (env
stepping plus GAE), and :func:`~sts_rl.agent.ppo_update.ppo_update` (the optimizer
pass) - into the standard on-policy loop: each iteration collects one
fixed-length rollout and then runs one PPO update over it. This is SB3's
``collect_rollouts`` + ``train`` split expressed as a plain loop; no new learning
math lives here, only the composition.

``global_step`` counts environment steps taken - the x-axis for a learning curve.
It advances by ``num_envs * n_steps`` per iteration (exactly ``n_steps`` in the
default single-env case).

GAE ``gamma``/``gae_lambda`` ARE configurable through this loop: ``TrainConfig``
carries both (defaulting to ``DEFAULT_GAMMA``/``DEFAULT_GAE_LAMBDA``) and forwards
them to the ``RolloutCollector``, which passes them to
``buffer.compute_advantages`` on every collect. Leaving them unset reproduces the
prior fixed-default behavior exactly.

The loop is environment-agnostic: it accepts any gymnasium env satisfying the
shared interface (an observation dict plus ``info['action_mask']``), so the same
``train`` drives the engine-free stub today and the real engine env later - it
never imports or references a concrete env. When ``config.num_envs > 1`` it
instead drives a vectorized env (a ``VecEnvProtocol`` such as ``SubprocVecEnv``)
over ``num_envs`` parallel envs, collecting ``n_steps`` per env with per-env GAE;
the caller constructs and passes that vec env. ``num_envs == 1`` (the default)
runs the original single-env path unchanged.

Opt-in add-ons: ``target_kl`` early-stop (via ``PPOConfig``), learning-rate
annealing (``anneal_lr``), periodic holdout evaluation (``eval_every`` plus a
separate ``eval_env``), and best/last checkpointing (``checkpoint_dir``).
``explained_variance`` is now logged every iteration (always on, not opt-in).
Every add-on defaults OFF, so leaving them unset reproduces the prior behavior.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.encoder import HIDDEN_DIM
from sts_rl.agent.ppo import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA
from sts_rl.agent.ppo_update import PPOConfig, PPOStats, ppo_update
from sts_rl.agent.rollout_buffer import RolloutBuffer, SupportsMinibatches, VecRolloutBuffer
from sts_rl.agent.rollout_collector import (
    CollectStats,
    RolloutCollector,
    VecEnvProtocol,
    VecRolloutCollector,
)
from sts_rl.eval import EvalReport, evaluate, make_holdout_seeds
from sts_rl.interface import INTERFACE_VERSION

logger = logging.getLogger(__name__)

# Adam step size for the update. This is a training-loop hyperparameter, not a
# shared interface constant, so the loop owns it here rather than importing one.
DEFAULT_LEARNING_RATE: float = 3e-4

# Periodic-eval defaults, applied only when eval is enabled (eval_every set). The
# seed base is deliberately distinct from the training seed so the holdout band
# does not coincide with the training reset stream.
DEFAULT_EVAL_EPISODES: int = 64
DEFAULT_EVAL_SEED_BASE: int = 2_000_000

# Checkpoint filenames and payload keys. best.pt is (over)written on each new-best
# eval win rate; last.pt holds the final trained weights. Both are written
# atomically via a sibling temp file (CHECKPOINT_TMP_SUFFIX) plus os.replace.
BEST_CHECKPOINT_NAME: str = "best.pt"
LAST_CHECKPOINT_NAME: str = "last.pt"
CHECKPOINT_TMP_SUFFIX: str = ".tmp"
CHECKPOINT_MODEL_KEY: str = "model_state_dict"
CHECKPOINT_HIDDEN_DIM_KEY: str = "hidden_dim"
CHECKPOINT_INTERFACE_VERSION_KEY: str = "interface_version"
CHECKPOINT_ITERATION_KEY: str = "iteration"
CHECKPOINT_STEP_KEY: str = "global_step"
CHECKPOINT_WIN_RATE_KEY: str = "win_rate"


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one :func:`train` run.

    Holds the loop-level knobs (iteration count, rollout length, parallel-env
    count, optimizer step, LR-anneal toggle, GAE ``gamma``/``gae_lambda``, seed,
    trunk width) plus a nested :class:`PPOConfig` forwarded verbatim to
    :func:`ppo_update`. Nesting the existing config avoids restating - and later
    drifting from - its fields and their defaults. ``hidden_dim`` defaults to the
    encoder's :data:`HIDDEN_DIM`; ``gamma``/``gae_lambda`` to the shared
    ``DEFAULT_GAMMA``/``DEFAULT_GAE_LAMBDA``; the PPO knobs to ``PPOConfig``'s.
    ``num_envs`` defaults to 1 (the single-env path, behavior-identical to the
    pre-vectorization loop); ``num_envs > 1`` opts into the vectorized path and
    requires ``env`` to be a matching vectorized env.
    ``anneal_lr`` is off by default - a constant LR is a valid first cut.
    Periodic holdout eval (``eval_every`` cadence over the
    ``eval_seed_base``/``eval_episodes`` band) and best/last ``checkpoint_dir``
    writing are all off by default (``eval_every``/``checkpoint_dir`` ``None``),
    so an unset config reproduces the prior loop exactly.
    """

    num_iterations: int
    n_steps: int
    num_envs: int = 1
    learning_rate: float = DEFAULT_LEARNING_RATE
    anneal_lr: bool = False
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    seed: int = 0
    hidden_dim: int = HIDDEN_DIM
    ppo: PPOConfig = field(default_factory=PPOConfig)
    # Periodic holdout eval + checkpointing, all default OFF: eval_every=None
    # disables periodic eval, checkpoint_dir=None disables checkpointing, so an
    # unset config reproduces the prior train() behavior exactly. When eval runs
    # it uses the fixed band make_holdout_seeds(eval_seed_base, eval_episodes).
    eval_every: int | None = None
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    eval_seed_base: int = DEFAULT_EVAL_SEED_BASE
    checkpoint_dir: str | None = None

    def __post_init__(self) -> None:
        # Fail at construction on a degenerate budget rather than silently
        # running an empty loop (num_iterations <= 0) or deferring to collect()'s
        # own mid-run rejection of n_steps.
        if self.num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {self.num_iterations}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        # A discount/trace-decay outside [0, 1] silently produces nonsense GAE
        # returns, so fail at construction like the other loop knobs. Endpoints
        # are inclusive: gamma=1.0 (undiscounted) and gae_lambda in {0, 1}
        # (TD(0) / Monte Carlo) are legitimate here.
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        # Periodic eval is opt-in (None disables); when enabled the cadence must be
        # a positive iteration count, so fail at construction like the other knobs.
        if self.eval_every is not None and self.eval_every <= 0:
            raise ValueError(f"eval_every must be positive when set, got {self.eval_every}")


@dataclass(frozen=True)
class IterationRecord:
    """One iteration's diagnostics: the step counters, the LR used, and both stat blocks.

    ``collect`` and ``ppo`` are stored whole rather than flattened, so every
    field the collector and update expose is preserved without re-listing them.
    ``learning_rate`` is the step size actually applied this iteration, so an
    annealed schedule is observable (CleanRL's ``charts/learning_rate``).
    """

    iteration: int
    global_step: int
    learning_rate: float
    collect: CollectStats
    ppo: PPOStats


@dataclass(frozen=True)
class EvalRecord:
    """One periodic-evaluation snapshot: the step counters plus the full report.

    Kept out of :class:`IterationRecord` because eval is sparse (every
    ``eval_every`` iterations, not every iteration), so folding the report into
    the per-iteration record would carry mostly-empty eval fields on the common
    path.
    """

    iteration: int
    global_step: int
    report: EvalReport


@dataclass(frozen=True)
class TrainHistory:
    """Result of a :func:`train` run: per-iteration records and the trained net.

    ``actor_critic`` is returned so a caller (evaluation, checkpointing, a sample
    playthrough) can use the trained network directly without re-threading it.
    ``eval_reports`` holds the periodic-eval snapshots (empty when ``eval_every``
    is unset); :meth:`best_eval` is the highest-win-rate snapshot, mirroring which
    run produced ``best.pt``.
    """

    records: list[IterationRecord]
    actor_critic: ActorCritic
    eval_reports: list[EvalRecord] = field(default_factory=list)

    def mean_episode_returns(self) -> list[float | None]:
        """Per-iteration mean episode return (``None`` where no episode ended)."""
        return [record.collect.mean_episode_return for record in self.records]

    def best_eval(self) -> EvalRecord | None:
        """Highest-win-rate eval snapshot, or ``None`` if no periodic eval ran.

        Ties resolve to the EARLIEST such snapshot (``max`` returns the first
        maximal element), matching the strictly-greater ``best.pt`` overwrite rule
        so this and the saved best checkpoint always agree.
        """
        if not self.eval_reports:
            return None
        return max(self.eval_reports, key=lambda record: record.report.win_rate)


def train(
    env: gym.Env | VecEnvProtocol, config: TrainConfig, *, eval_env: gym.Env | None = None
) -> TrainHistory:
    """Run ``config.num_iterations`` collect->update iterations; return the history.

    Seeds the process-global RNGs once up front (the reproducibility contract:
    ``torch.manual_seed`` drives weight init, action sampling, and minibatch
    shuffling; ``np.random.seed`` the legacy global stream), then builds the
    network, optimizer, collector, and buffer and loops. The collector separately
    seeds only the env's own reset stream via its ``seed`` argument.

    Each iteration collects ``config.n_steps`` transitions per env (the collector
    clears the buffer and runs per-env GAE internally), runs one multi-epoch PPO
    update over that buffer, records the merged diagnostics, and emits one INFO
    log line. With ``num_envs > 1``, ``env`` must be a vectorized env whose
    ``num_envs`` matches ``config.num_envs`` (else a ``ValueError``).

    Periodic evaluation and checkpointing are opt-in and default OFF. When
    ``config.eval_every`` is set, every ``eval_every`` iterations (and always on
    the final iteration, so best/last reflect the fully trained policy) the
    current policy is greedily evaluated on the fixed holdout band via ``eval_env``
    - a SEPARATE env instance, because :func:`evaluate` resets the env once per
    seed and pointing it at the training env would corrupt the collector's
    in-progress rollout stream. When ``config.checkpoint_dir`` is set, a new
    best-win-rate eval writes ``best.pt`` and the final weights write ``last.pt``
    (state_dicts, saved atomically); a ``checkpoint_dir`` without ``eval_every``
    writes only ``last.pt`` (there are no eval win rates to rank a best).
    """
    # Periodic eval needs its OWN env: evaluate() resets the env once per holdout
    # seed, which would derail the training collector mid-rollout. Fail fast rather
    # than silently sharing (and corrupting) the training env.
    if config.eval_every is not None and eval_env is None:
        raise ValueError(
            "config.eval_every is set but eval_env is None; periodic evaluation "
            "requires a separate env instance (evaluate() resets it per seed, "
            "which would corrupt the training rollout stream)"
        )

    # One-time global seeding BEFORE any RNG is consumed: weight init below must
    # be reproducible too, so this precedes ActorCritic construction. The
    # collector's own seed arg covers only the env reset stream.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    actor_critic = ActorCritic(hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.learning_rate)
    # Rollout collection runs either the single-env path (num_envs == 1: the
    # existing RolloutCollector + RolloutBuffer, behavior-identical to the
    # pre-vectorization loop) or the vectorized path (num_envs > 1), where `env`
    # must be a vectorized env (a VecEnvProtocol such as SubprocVecEnv) whose
    # own num_envs matches the config. Both are held here and only one is built.
    single_collector: RolloutCollector | None = None
    vec_collector: VecRolloutCollector | None = None
    single_buffer: RolloutBuffer | None = None
    vec_buffer: VecRolloutBuffer | None = None
    if config.num_envs == 1:
        # Single-env path: env must be a plain gym.Env. The raise rejects a caller
        # env-type/num_envs mismatch with a clear ValueError (consistent with the
        # num_envs value check below and every boundary check in train()/TrainConfig)
        # and narrows the union for mypy - a raise on the not-a-gym.Env case leaves
        # env a gym.Env in the code that follows, so no cast is needed.
        if not isinstance(env, gym.Env):
            raise ValueError(
                f"config.num_envs=1 (single-env path) requires a gym.Env, but the "
                f"passed env is not a gym.Env (got {type(env).__name__}); pass a "
                f"gym.Env or set num_envs to the vectorized env's num_envs"
            )
        single_collector = RolloutCollector(
            env,
            actor_critic,
            seed=config.seed,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        single_buffer = RolloutBuffer()
    else:
        # Vectorized path: env must be a VecEnvProtocol (e.g. SubprocVecEnv), not a
        # gym.Env. The raise rejects a caller env-type/num_envs mismatch with a clear
        # ValueError (mirroring the single-env branch and consistent with the
        # num_envs value check below) and narrows the union to VecEnvProtocol for
        # mypy - a raise on the gym.Env case leaves env a VecEnvProtocol in the code
        # that follows (so env.num_envs is valid), without a cast.
        if isinstance(env, gym.Env):
            raise ValueError(
                f"config.num_envs={config.num_envs} (vectorized path) requires a "
                f"vectorized env (a VecEnvProtocol such as SubprocVecEnv), but the "
                f"passed env is a gym.Env; pass a vectorized env or set num_envs=1"
            )
        if env.num_envs != config.num_envs:
            raise ValueError(
                f"config.num_envs={config.num_envs} but the passed vectorized env "
                f"reports num_envs={env.num_envs}; they must match"
            )
        vec_collector = VecRolloutCollector(
            env,
            actor_critic,
            seed=config.seed,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        vec_buffer = VecRolloutBuffer(config.num_envs)

    # Create the checkpoint dir once up front (not per save), so a bad path fails
    # before training rather than after the first eval.
    checkpoint_dir_path: Path | None = None
    if config.checkpoint_dir is not None:
        checkpoint_dir_path = Path(config.checkpoint_dir)
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    records: list[IterationRecord] = []
    eval_reports: list[EvalRecord] = []
    best_win_rate: float | None = None
    global_step = 0
    for iteration in range(config.num_iterations):
        # CleanRL linear LR decay: iteration 0 keeps the full learning_rate and
        # the rate decays toward ~0 across the run (the final frac is
        # 1/num_iterations, never exactly 0). Opt-in; constant LR otherwise.
        lr_now = (
            config.learning_rate * (1.0 - iteration / config.num_iterations)
            if config.anneal_lr
            else config.learning_rate
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        # collect() resets the buffer and runs compute_advantages itself, so the
        # SAME buffer is handed straight to ppo_update with GAE already computed.
        # Both collectors expose the same collect(buffer, n_steps) -> CollectStats
        # surface and fill a SupportsMinibatches buffer, so only construction
        # differs between the single-env and vectorized paths.
        if single_collector is not None:
            assert single_buffer is not None  # paired with single_collector above
            collect_stats = single_collector.collect(single_buffer, config.n_steps)
            buffer_for_update: SupportsMinibatches = single_buffer
        else:
            assert vec_collector is not None and vec_buffer is not None  # paired above
            collect_stats = vec_collector.collect(vec_buffer, config.n_steps)
            buffer_for_update = vec_buffer
        # n_steps transitions are collected PER env across num_envs envs, so the
        # step counter advances by num_envs * n_steps (exactly n_steps when
        # num_envs == 1, preserving the single-env accounting).
        global_step += config.num_envs * config.n_steps
        ppo_stats = ppo_update(actor_critic, buffer_for_update, optimizer, config.ppo)

        records.append(
            IterationRecord(
                iteration=iteration,
                global_step=global_step,
                learning_rate=lr_now,
                collect=collect_stats,
                ppo=ppo_stats,
            )
        )
        _log_iteration(iteration, global_step, lr_now, collect_stats, ppo_stats)

        # Periodic eval on the cadence, and always on the final iteration so
        # best/last reflect the fully trained policy. Gated by eval_every (None
        # keeps this whole block off, the default).
        is_final_iteration = iteration + 1 == config.num_iterations
        if config.eval_every is not None and (
            (iteration + 1) % config.eval_every == 0 or is_final_iteration
        ):
            assert eval_env is not None  # guaranteed by the eval_every/eval_env guard
            # Greedy holdout readout on the SEPARATE eval env. deterministic=True
            # (the default, made explicit) takes the masked argmax and draws no
            # samples, so periodic eval never consumes the training torch RNG -
            # training stays bit-identical with or without eval running.
            report = evaluate(
                actor_critic,
                eval_env,
                make_holdout_seeds(config.eval_seed_base, config.eval_episodes),
                deterministic=True,
            )
            eval_reports.append(
                EvalRecord(iteration=iteration, global_step=global_step, report=report)
            )
            logger.info(
                "eval iter=%d global_step=%d win_rate=%.3f",
                iteration,
                global_step,
                report.win_rate,
            )
            # New best (STRICTLY greater) overwrites best.pt; a tie keeps the
            # earlier best, matching TrainHistory.best_eval.
            if checkpoint_dir_path is not None and (
                best_win_rate is None or report.win_rate > best_win_rate
            ):
                best_win_rate = report.win_rate
                _save_checkpoint(
                    checkpoint_dir_path,
                    BEST_CHECKPOINT_NAME,
                    actor_critic,
                    hidden_dim=config.hidden_dim,
                    iteration=iteration,
                    global_step=global_step,
                    win_rate=report.win_rate,
                )

    # last.pt always captures the final trained weights. Its win_rate is the final
    # iteration's eval win rate when eval ran (the final iteration is always an
    # eval iteration above), else None - the checkpoint_dir-without-eval_every case
    # documented on train().
    if checkpoint_dir_path is not None:
        last_win_rate = eval_reports[-1].report.win_rate if eval_reports else None
        _save_checkpoint(
            checkpoint_dir_path,
            LAST_CHECKPOINT_NAME,
            actor_critic,
            hidden_dim=config.hidden_dim,
            iteration=config.num_iterations - 1,
            global_step=global_step,
            win_rate=last_win_rate,
        )

    return TrainHistory(records=records, actor_critic=actor_critic, eval_reports=eval_reports)


def _log_iteration(
    iteration: int,
    global_step: int,
    learning_rate: float,
    collect: CollectStats,
    ppo: PPOStats,
) -> None:
    """Emit one INFO line of the iteration's key diagnostics.

    The episode means are ``None`` when no episode completed this iteration;
    rendered as ``n/a`` so the line stays fixed-shape instead of crashing a
    ``%f`` format. Lazy ``%``-args keep formatting off the hot path when INFO is
    disabled.
    """
    ret = "n/a" if collect.mean_episode_return is None else f"{collect.mean_episode_return:.3f}"
    length = "n/a" if collect.mean_episode_length is None else f"{collect.mean_episode_length:.2f}"
    logger.info(
        "iter=%d global_step=%d lr=%.2e ep_return=%s ep_len=%s n_episodes=%d "
        "policy_loss=%.4f value_loss=%.4f entropy=%.4f approx_kl=%.4f "
        "clip_frac=%.3f explained_var=%.3f grad_norm=%.3f steps_per_second=%.0f",
        iteration,
        global_step,
        learning_rate,
        ret,
        length,
        collect.n_episodes,
        ppo.policy_loss,
        ppo.value_loss,
        ppo.entropy,
        ppo.approx_kl,
        ppo.clip_fraction,
        ppo.explained_variance,
        ppo.grad_norm,
        collect.steps_per_second,
    )


def _save_checkpoint(
    checkpoint_dir: Path,
    filename: str,
    actor_critic: ActorCritic,
    *,
    hidden_dim: int,
    iteration: int,
    global_step: int,
    win_rate: float | None,
) -> None:
    """Atomically write one checkpoint: a state_dict payload, not a pickled module.

    Saves ``actor_critic.state_dict()`` (portable, and loadable under
    ``weights_only=True``) rather than the module object. The state_dict is
    DEEP-COPIED at capture time because it returns live tensor references that keep
    mutating as training continues, so a shallow grab could serialize post-capture
    weights. The write is atomic: torch saves to a sibling temp file in the SAME
    directory, then ``os.replace`` renames it over ``filename`` (an atomic
    same-filesystem rename), so a crash mid-write can never leave a truncated
    checkpoint at the final path.

    The payload is self-describing: alongside the weights it records ``hidden_dim``
    (the trunk width the net was built with) and ``interface_version`` (provenance),
    so a checkpoint reloads without externally knowing the architecture - rebuild
    ``ActorCritic(hidden_dim=ckpt[CHECKPOINT_HIDDEN_DIM_KEY])`` then
    ``load_state_dict``.
    """
    payload: dict[str, object] = {
        CHECKPOINT_MODEL_KEY: copy.deepcopy(actor_critic.state_dict()),
        CHECKPOINT_HIDDEN_DIM_KEY: hidden_dim,
        CHECKPOINT_INTERFACE_VERSION_KEY: INTERFACE_VERSION,
        CHECKPOINT_ITERATION_KEY: iteration,
        CHECKPOINT_STEP_KEY: global_step,
        CHECKPOINT_WIN_RATE_KEY: win_rate,
    }
    final_path = checkpoint_dir / filename
    # Sibling temp in the SAME dir keeps os.replace a single-filesystem atomic
    # rename (a cross-device os.replace would raise instead).
    tmp_path = final_path.with_name(final_path.name + CHECKPOINT_TMP_SUFFIX)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
