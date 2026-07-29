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
separate ``eval_env``), best/last checkpointing (``checkpoint_dir``), value-head
warmup (``value_warmup_iters``: freeze the trunk + policy head so the leading
iterations train only the critic), and plateau early-stop
(``early_stop_patience`` / ``early_stop_min_delta`` on the ranked eval metric).
``explained_variance`` is now logged every iteration (always on, not opt-in).
Every add-on defaults OFF, so leaving them unset reproduces the prior behavior.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.encoder import HIDDEN_DIM
from sts_rl.agent.ppo import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA
from sts_rl.agent.ppo_update import ADV_NORM_EPS, PPOConfig, PPOStats, ppo_update
from sts_rl.agent.rollout_buffer import RolloutBuffer, SupportsMinibatches, VecRolloutBuffer
from sts_rl.agent.rollout_collector import (
    CollectStats,
    RolloutCollector,
    VecEnvProtocol,
    VecRolloutCollector,
)
from sts_rl.agent.running_moments import RunningMoments
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
# best.pt-only provenance: the metric it was ranked on (NAME) and that metric's
# selected value, so the best checkpoint is self-describing about its ranking even
# when best_metric != win_rate (win_rate above is still recorded for provenance, but
# it is not the ranked value then). Written only on best.pt; both are
# weights_only-safe (a str + a float).
CHECKPOINT_BEST_METRIC_KEY: str = "best_metric"
CHECKPOINT_BEST_METRIC_VALUE_KEY: str = "best_metric_value"

# Default ranked metric: the EvalReport field that ranks best.pt and
# TrainHistory.best_eval, and is additionally surfaced in the periodic-eval log.
# "win_rate" preserves the original combat behavior; run-mode overrides it with a
# progress metric (e.g. act1_clear_rate) because full-run win_rate is ~0 for a long
# time. Must name an EvalReport field (validated in TrainConfig.__post_init__).
DEFAULT_BEST_METRIC: str = "win_rate"

# Valid best_metric names: the float-typed metric fields of EvalReport (win_rate,
# the avg_* aggregates, and act1_clear_rate), derived from the dataclass so a new
# float metric is rankable without editing a list here. The int n_episodes is
# excluded - it is a constant episode count, not a progress metric, so ranking on it
# is degenerate. f.type is the annotation STRING here (PEP 563 postponed
# evaluation), so the filter matches "float".
EVAL_METRIC_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(EvalReport) if f.type == "float"
)


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
    # The EvalReport field used to (a) rank best.pt / TrainHistory.best_eval and
    # (b) additionally surface in the periodic-eval log line. Default "win_rate"
    # keeps combat behavior identical; run-mode overrides it (e.g. "act1_clear_rate")
    # because full-run win_rate is ~0 for a long time, so every eval ties at 0 and
    # best.pt would be degenerate. Validated in __post_init__.
    best_metric: str = DEFAULT_BEST_METRIC
    # Value-head warmup: for the first value_warmup_iters iterations, freeze the
    # encoder trunk AND the policy head so only the critic (value head) trains.
    # This calibrates the value function to full-run returns on the fixed
    # (warm-started) representation before gradients are allowed to flow into the
    # shared trunk - a miscalibrated combat value head otherwise corrupts the trunk
    # early. Default 0 disables it (behavior unchanged); validated in __post_init__.
    # With anneal_lr, the warmup iterations consume the LR schedule (it is not offset
    # to the unfreeze), so the trunk unfreezes at the already-reduced learning rate.
    value_warmup_iters: int = 0
    # Plateau early-stop on the ranked eval metric (best_metric): stop once
    # early_stop_patience consecutive POST-warmup evals fail to improve the running
    # best by more than early_stop_min_delta. Both default OFF (patience None), so an
    # unset config runs all num_iterations exactly as before. Validated in __post_init__.
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0

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
        # best_metric must name a float metric field of EvalReport (see
        # EVAL_METRIC_FIELDS); reject an unknown name at construction rather than
        # letting a typo surface as a cryptic getattr AttributeError deep in
        # best_eval/the loop.
        if self.best_metric not in EVAL_METRIC_FIELDS:
            raise ValueError(
                f"best_metric must be one of {sorted(EVAL_METRIC_FIELDS)}, "
                f"got {self.best_metric!r}"
            )
        # Value-head warmup is a non-negative iteration count (0 disables); a
        # negative count is meaningless, so reject it at construction.
        if self.value_warmup_iters < 0:
            raise ValueError(
                f"value_warmup_iters must be non-negative, got {self.value_warmup_iters}"
            )
        # A warmup spanning the whole run never lifts the freeze: the unfreeze is gated
        # on iteration == value_warmup_iters, but iteration only reaches
        # num_iterations - 1, so value_warmup_iters >= num_iterations would silently
        # train a frozen trunk + policy for every iteration. Require at least one
        # post-warmup iteration.
        if self.value_warmup_iters > 0 and self.value_warmup_iters >= self.num_iterations:
            raise ValueError(
                f"value_warmup_iters ({self.value_warmup_iters}) must be < "
                f"num_iterations ({self.num_iterations})"
            )
        # Early-stop is opt-in (patience None disables); when set, the patience must
        # be a positive number of evals and min_delta a non-negative threshold, so
        # fail at construction like the other knobs.
        if self.early_stop_patience is not None and self.early_stop_patience <= 0:
            raise ValueError(
                f"early_stop_patience must be positive when set, got {self.early_stop_patience}"
            )
        # Early-stop acts only inside the periodic-eval branch (it plateaus on eval
        # metrics), so early_stop_patience without eval_every would be a silent no-op.
        # Reject it so a requested early stop always has the evals it needs to act on.
        if self.early_stop_patience is not None and self.eval_every is None:
            raise ValueError(
                "early_stop_patience is set but eval_every is None; early stop acts on "
                "periodic-eval metrics, so eval_every must be set for it to take effect"
            )
        if self.early_stop_min_delta < 0.0:
            raise ValueError(
                f"early_stop_min_delta must be non-negative, got {self.early_stop_min_delta}"
            )


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
    """One periodic-evaluation snapshot: the step counters, the seed base, and the report.

    Kept out of :class:`IterationRecord` because eval is sparse (every
    ``eval_every`` iterations, not every iteration), so folding the report into
    the per-iteration record would carry mostly-empty eval fields on the common
    path.
    """

    iteration: int
    global_step: int
    # The holdout seed base this snapshot was evaluated over. Recorded so a later
    # consumer can confirm a reused report matches a requested band before trusting
    # it (report.n_episodes already carries the band's episode count).
    eval_seed_base: int
    report: EvalReport


@dataclass(frozen=True)
class TrainHistory:
    """Result of a :func:`train` run: per-iteration records and the trained net.

    ``actor_critic`` is returned so a caller (evaluation, checkpointing, a sample
    playthrough) can use the trained network directly without re-threading it.
    ``eval_reports`` holds the periodic-eval snapshots (empty when ``eval_every``
    is unset); :meth:`best_eval` is the highest-``best_metric`` snapshot, mirroring which
    run produced ``best.pt``. ``stopped_early`` is True when plateau early-stop ended the
    run before ``num_iterations`` (False otherwise, including a normal full run).
    """

    records: list[IterationRecord]
    actor_critic: ActorCritic
    eval_reports: list[EvalRecord] = field(default_factory=list)
    # The EvalReport field best_eval ranks on; set from TrainConfig.best_metric by
    # train(). Defaulted so a directly-constructed history ranks on win_rate.
    best_metric: str = DEFAULT_BEST_METRIC
    # True when plateau early-stop broke the loop before num_iterations; set by
    # train(). Defaulted False so a directly-constructed history (or a normal
    # full-length run) reads as not-early-stopped.
    stopped_early: bool = False

    def mean_episode_returns(self) -> list[float | None]:
        """Per-iteration mean episode return (``None`` where no episode ended)."""
        return [record.collect.mean_episode_return for record in self.records]

    def best_eval(self) -> EvalRecord | None:
        """Highest-ranked eval snapshot by ``best_metric``, or ``None`` if none ran.

        Ties resolve to the EARLIEST such snapshot (``max`` returns the first
        maximal element), matching the strictly-greater ``best.pt`` overwrite rule
        so this and the saved best checkpoint always agree.
        """
        if not self.eval_reports:
            return None
        return max(self.eval_reports, key=lambda record: getattr(record.report, self.best_metric))


def train(
    env: gym.Env | VecEnvProtocol,
    config: TrainConfig,
    *,
    eval_env: gym.Env | None = None,
    init_actor_critic: ActorCritic | None = None,
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

    ``config.value_warmup_iters`` (default 0, off) freezes the encoder trunk and
    policy head for that many leading iterations so only the value head trains,
    calibrating the critic on the fixed (warm-started) representation before
    gradients reach the shared trunk; the freeze is lifted at the boundary
    iteration. ``config.early_stop_patience`` (default None, off) stops the loop
    early once that many consecutive POST-warmup evals fail to improve the ranked
    ``best_metric`` by more than ``config.early_stop_min_delta``; the post-loop
    ``last.pt`` write and history return still run, so a stopped run is saved.

    ``init_actor_critic`` warm-starts training from a pre-built network (e.g. a
    checkpoint loaded and migrated by
    :func:`~sts_rl.agent.checkpoint_migration.load_checkpoint`) instead of a fresh
    random init: when given, it is used as the network (after the one-time global
    seeding, so sampling/shuffling stay reproducible) and its encoder trunk width
    must equal ``config.hidden_dim`` (a mismatch raises, because that width is
    recorded in saved checkpoints). ``None`` (the default) rebuilds a fresh net, so
    the default path is behaviour-identical to before.
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

    # Warm-start hook: with init_actor_critic given, train the injected (pre-built,
    # possibly checkpoint-migrated) net instead of a fresh init. The seeding above
    # still runs first, so action sampling and minibatch shuffling stay reproducible
    # for a fixed seed regardless of the net's origin. init_actor_critic=None (the
    # default) rebuilds a fresh net exactly as before, so that path is unchanged.
    if init_actor_critic is None:
        actor_critic = ActorCritic(hidden_dim=config.hidden_dim)
    else:
        # config.hidden_dim is recorded verbatim in every saved checkpoint
        # (_save_checkpoint) and must therefore describe the actual trunk width; a
        # mismatch would write a checkpoint that load_checkpoint cannot rebuild.
        # Require the caller to pass a config whose hidden_dim matches the injected
        # net (the run driver derives it from the loaded checkpoint) and fail fast
        # with a clear error otherwise.
        net_hidden_dim = init_actor_critic.encoder.output_dim
        if net_hidden_dim != config.hidden_dim:
            raise ValueError(
                f"init_actor_critic trunk width ({net_hidden_dim}) does not match "
                f"config.hidden_dim ({config.hidden_dim}); pass a TrainConfig whose "
                f"hidden_dim equals the injected network's encoder width (saved "
                f"checkpoints record config.hidden_dim, so a mismatch would be unloadable)"
            )
        actor_critic = init_actor_critic
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
    best_metric_value: float | None = None
    # Early-stop state (used only when config.early_stop_patience is set): the best
    # POST-warmup eval metric seen so far and the number of consecutive evals since
    # that have not improved it by more than early_stop_min_delta. Kept SEPARATE from
    # best_metric_value (which ranks best.pt over every eval) because early-stop skips
    # warmup-period evals and applies the min_delta threshold.
    es_best: float | None = None
    stale_evals = 0
    # Records whether plateau early-stop broke the loop before num_iterations;
    # propagated to the returned TrainHistory. Stays False on a normal full-length run.
    stopped_early = False
    # Value-head warmup: freeze the trunk + policy head up front so the first
    # value_warmup_iters iterations train only the critic (see
    # _set_trunk_and_policy_requires_grad). warmup_active tracks the frozen state so
    # the freeze is lifted exactly once at the boundary iteration below. Disabled (no
    # freeze) when value_warmup_iters == 0, the default.
    warmup_active = config.value_warmup_iters > 0
    if warmup_active:
        _set_trunk_and_policy_requires_grad(actor_critic, requires_grad=False)
        logger.info(
            "value-head warmup: freezing trunk + policy for %d iters (critic-only)",
            config.value_warmup_iters,
        )
    # One persistent advantage-normalization tracker for the whole run, so its
    # EWMA carries across iterations (see ppo_update). Checkpoint persistence of
    # these running stats is out of scope: on resume the EWMA simply re-warms
    # from the first post-resume rollout.
    adv_moments = RunningMoments(decay=config.ppo.adv_norm_decay, eps=ADV_NORM_EPS)
    global_step = 0
    for iteration in range(config.num_iterations):
        # Lift the warmup freeze exactly at the boundary iteration: from here on
        # gradients flow into the trunk + policy again, now that the critic has been
        # calibrated on the fixed representation.
        if warmup_active and iteration == config.value_warmup_iters:
            _set_trunk_and_policy_requires_grad(actor_critic, requires_grad=True)
            warmup_active = False
            logger.info(
                "value-head warmup complete at iter %d: unfreezing trunk + policy", iteration
            )
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
        ppo_stats = ppo_update(
            actor_critic, buffer_for_update, optimizer, config.ppo, adv_moments=adv_moments
        )

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
                EvalRecord(
                    iteration=iteration,
                    global_step=global_step,
                    eval_seed_base=config.eval_seed_base,
                    report=report,
                )
            )
            # This eval's ranked metric: win_rate by default (combat), a progress
            # metric (e.g. act1_clear_rate) in run-mode. Ranks best.pt below and is
            # surfaced in the log line when it differs from the always-shown win_rate.
            metric_value: float = getattr(report, config.best_metric)
            # Always log win_rate; for run-mode (best_metric != win_rate) also append
            # the ranked metric, whose curve is the informative one while full-run
            # win_rate sits at ~0. Combat logs byte-identically (no always-zero field).
            if config.best_metric == DEFAULT_BEST_METRIC:
                logger.info(
                    "eval iter=%d global_step=%d win_rate=%.3f",
                    iteration,
                    global_step,
                    report.win_rate,
                )
            else:
                logger.info(
                    "eval iter=%d global_step=%d win_rate=%.3f %s=%.3f",
                    iteration,
                    global_step,
                    report.win_rate,
                    config.best_metric,
                    metric_value,
                )
            # New best (STRICTLY greater) overwrites best.pt; a tie keeps the earlier
            # best, matching TrainHistory.best_eval. Ranks on the configured
            # best_metric (win_rate by default); the checkpoint still records
            # report.win_rate for provenance - only the ranking metric changes.
            if checkpoint_dir_path is not None and (
                best_metric_value is None or metric_value > best_metric_value
            ):
                best_metric_value = metric_value
                _save_checkpoint(
                    checkpoint_dir_path,
                    BEST_CHECKPOINT_NAME,
                    actor_critic,
                    hidden_dim=config.hidden_dim,
                    iteration=iteration,
                    global_step=global_step,
                    win_rate=report.win_rate,
                    best_metric=config.best_metric,
                    best_metric_value=metric_value,
                )

            # Plateau early-stop (opt-in via early_stop_patience). Count only
            # POST-warmup evals (iteration >= value_warmup_iters): during warmup the
            # policy is frozen, so its greedy eval metric is flat by construction and
            # must not count toward the plateau. A new best (by more than min_delta)
            # resets the stale streak; otherwise it grows, and once it reaches
            # patience we stop and BREAK - the flat back-half of a converged run is
            # wasted compute. es_best tracks the SAME metric best.pt ranks on
            # (config.best_metric), so early-stop and best.pt follow one signal.
            if config.early_stop_patience is not None and iteration >= config.value_warmup_iters:
                if es_best is None or metric_value > es_best + config.early_stop_min_delta:
                    es_best = metric_value
                    stale_evals = 0
                else:
                    stale_evals += 1
                    if stale_evals >= config.early_stop_patience:
                        logger.info(
                            "early stop at iter %d: %s did not improve by more than %g "
                            "over %d evals (best %.3f)",
                            iteration,
                            config.best_metric,
                            config.early_stop_min_delta,
                            config.early_stop_patience,
                            es_best,
                        )
                        stopped_early = True
                        break

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
            # The live loop variable, not num_iterations - 1: on a normal completion it
            # equals num_iterations - 1, but after an early-stop break it is the actual
            # stop iteration (recording num_iterations - 1 there would overstate the run).
            iteration=iteration,
            global_step=global_step,
            win_rate=last_win_rate,
        )

    return TrainHistory(
        records=records,
        actor_critic=actor_critic,
        eval_reports=eval_reports,
        best_metric=config.best_metric,
        stopped_early=stopped_early,
    )


def _set_trunk_and_policy_requires_grad(actor_critic: ActorCritic, *, requires_grad: bool) -> None:
    """Toggle ``requires_grad`` on the encoder trunk and policy head (value head untouched).

    The value-head warmup freezes the trunk + policy head for the first
    ``value_warmup_iters`` iterations so only the critic trains, calibrating the
    value function on the fixed (warm-started) representation before gradients reach
    the shared trunk. Freezing via ``requires_grad`` is sufficient: the single
    ``Adam(actor_critic.parameters())`` skips any param whose grad stays ``None``
    (``optimizer.zero_grad`` nulls grads each step and ``backward`` never populates a
    frozen leaf), and ``clip_grad_norm_`` ignores ``None`` grads too - so no separate
    optimizer or param-group surgery is needed. Restoring ``requires_grad=True`` lets
    Adam lazily initialize fresh state for the trunk on its first post-warmup grad.

    This ``requires_grad`` freeze is COMPLETE only because the encoder has no
    BatchNorm/LayerNorm/Dropout: those layers carry running statistics or eval-mode
    behavior that keep updating in training mode even with grads off, so a future norm
    or dropout layer would additionally need ``.eval()`` (a training-mode toggle) to be
    truly frozen.
    """
    for param in actor_critic.encoder.parameters():
        param.requires_grad = requires_grad
    for param in actor_critic.policy.parameters():
        param.requires_grad = requires_grad


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
    best_metric: str | None = None,
    best_metric_value: float | None = None,
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

    ``best_metric``/``best_metric_value`` are recorded only when passed - the best.pt
    call site passes the ranked metric name and its selected value, so best.pt is
    self-describing about what it was ranked on (distinct from the always-recorded
    ``win_rate`` provenance). last.pt omits both (it passes neither), since no ranked
    value applies to a final-weights checkpoint. Both are ``weights_only``-safe (a
    ``str`` and a ``float``).
    """
    payload: dict[str, object] = {
        CHECKPOINT_MODEL_KEY: copy.deepcopy(actor_critic.state_dict()),
        CHECKPOINT_HIDDEN_DIM_KEY: hidden_dim,
        CHECKPOINT_INTERFACE_VERSION_KEY: INTERFACE_VERSION,
        CHECKPOINT_ITERATION_KEY: iteration,
        CHECKPOINT_STEP_KEY: global_step,
        CHECKPOINT_WIN_RATE_KEY: win_rate,
    }
    # best.pt records what it was ranked on (metric name + selected value); recorded
    # only when a name is passed, so last.pt's payload stays unchanged.
    if best_metric is not None:
        payload[CHECKPOINT_BEST_METRIC_KEY] = best_metric
        payload[CHECKPOINT_BEST_METRIC_VALUE_KEY] = best_metric_value
    final_path = checkpoint_dir / filename
    # Sibling temp in the SAME dir keeps os.replace a single-filesystem atomic
    # rename (a cross-device os.replace would raise instead).
    tmp_path = final_path.with_name(final_path.name + CHECKPOINT_TMP_SUFFIX)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
