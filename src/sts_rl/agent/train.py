"""PPO outer training loop for the Slay the Spire RL agent.

Ties the already-built pieces - the :class:`~sts_rl.agent.actor_critic.ActorCritic`
network, the :class:`~sts_rl.agent.rollout_collector.RolloutCollector` (env
stepping plus GAE), and :func:`~sts_rl.agent.ppo_update.ppo_update` (the optimizer
pass) - into the standard on-policy loop: each iteration collects one
fixed-length rollout and then runs one PPO update over it. This is SB3's
``collect_rollouts`` + ``train`` split expressed as a plain loop; no new learning
math lives here, only the composition.

``global_step`` counts environment steps taken - the x-axis for a learning curve.
With a single environment it advances by exactly ``n_steps`` per iteration.

GAE ``gamma``/``gae_lambda`` ARE configurable through this loop: ``TrainConfig``
carries both (defaulting to ``DEFAULT_GAMMA``/``DEFAULT_GAE_LAMBDA``) and forwards
them to the ``RolloutCollector``, which passes them to
``buffer.compute_advantages`` on every collect. Leaving them unset reproduces the
prior fixed-default behavior exactly.

The loop is environment-agnostic: it accepts any gymnasium env satisfying the
shared interface (an observation dict plus ``info['action_mask']``), so the same
``train`` drives the engine-free stub today and the real engine env later - it
never imports or references a concrete env.

Opt-in add-ons: ``target_kl`` early-stop (via ``PPOConfig``) and learning-rate
annealing (``anneal_lr``). ``explained_variance`` is now logged every iteration
(always on, not opt-in). Still deferred: checkpointing and periodic evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.encoder import HIDDEN_DIM
from sts_rl.agent.ppo import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA
from sts_rl.agent.ppo_update import PPOConfig, PPOStats, ppo_update
from sts_rl.agent.rollout_buffer import RolloutBuffer
from sts_rl.agent.rollout_collector import CollectStats, RolloutCollector

logger = logging.getLogger(__name__)

# Adam step size for the update. This is a training-loop hyperparameter, not a
# shared interface constant, so the loop owns it here rather than importing one.
DEFAULT_LEARNING_RATE: float = 3e-4


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one :func:`train` run.

    Holds the loop-level knobs (iteration count, rollout length, optimizer step,
    LR-anneal toggle, GAE ``gamma``/``gae_lambda``, seed, trunk width) plus a
    nested :class:`PPOConfig` forwarded verbatim to :func:`ppo_update`. Nesting
    the existing config avoids restating - and later drifting from - its fields
    and their defaults. ``hidden_dim`` defaults to the encoder's
    :data:`HIDDEN_DIM`; ``gamma``/``gae_lambda`` to the shared
    ``DEFAULT_GAMMA``/``DEFAULT_GAE_LAMBDA``; the PPO knobs to ``PPOConfig``'s.
    ``anneal_lr`` is off by default - a constant LR is a valid first cut.
    """

    num_iterations: int
    n_steps: int
    learning_rate: float = DEFAULT_LEARNING_RATE
    anneal_lr: bool = False
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    seed: int = 0
    hidden_dim: int = HIDDEN_DIM
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        # Fail at construction on a degenerate budget rather than silently
        # running an empty loop (num_iterations <= 0) or deferring to collect()'s
        # own mid-run rejection of n_steps.
        if self.num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {self.num_iterations}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
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
class TrainHistory:
    """Result of a :func:`train` run: per-iteration records and the trained net.

    ``actor_critic`` is returned so a caller (evaluation, checkpointing, a sample
    playthrough) can use the trained network directly without re-threading it.
    """

    records: list[IterationRecord]
    actor_critic: ActorCritic

    def mean_episode_returns(self) -> list[float | None]:
        """Per-iteration mean episode return (``None`` where no episode ended)."""
        return [record.collect.mean_episode_return for record in self.records]


def train(env: gym.Env, config: TrainConfig) -> TrainHistory:
    """Run ``config.num_iterations`` collect->update iterations; return the history.

    Seeds the process-global RNGs once up front (the reproducibility contract:
    ``torch.manual_seed`` drives weight init, action sampling, and minibatch
    shuffling; ``np.random.seed`` the legacy global stream), then builds the
    network, optimizer, collector, and buffer and loops. The collector separately
    seeds only the env's own reset stream via its ``seed`` argument.

    Each iteration collects exactly ``config.n_steps`` transitions (the collector
    clears the buffer and runs GAE internally), runs one multi-epoch PPO update
    over that buffer, records the merged diagnostics, and emits one INFO log line.
    """
    # One-time global seeding BEFORE any RNG is consumed: weight init below must
    # be reproducible too, so this precedes ActorCritic construction. The
    # collector's own seed arg covers only the env reset stream.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    actor_critic = ActorCritic(hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.learning_rate)
    collector = RolloutCollector(
        env,
        actor_critic,
        seed=config.seed,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    buffer = RolloutBuffer()

    records: list[IterationRecord] = []
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
        collect_stats = collector.collect(buffer, config.n_steps)
        # Single env: one collected transition == one env step, so the step
        # counter advances by exactly n_steps each iteration.
        global_step += config.n_steps
        ppo_stats = ppo_update(actor_critic, buffer, optimizer, config.ppo)

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

    return TrainHistory(records=records, actor_critic=actor_critic)


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
