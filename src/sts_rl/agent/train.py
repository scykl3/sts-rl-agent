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

GAE ``gamma``/``gae_lambda`` are NOT configurable through this loop: ``collect``
runs ``buffer.compute_advantages`` internally with the buffer's defaults
(``DEFAULT_GAMMA``/``DEFAULT_GAE_LAMBDA``) and does not forward overrides.
Plumbing a discount through ``collect`` is a noted follow-up; until it lands, a
discount change must edit the buffer/ppo defaults.

The loop is environment-agnostic: it accepts any gymnasium env satisfying the
shared interface (an observation dict plus ``info['action_mask']``), so the same
``train`` drives the engine-free stub today and the real engine env later - it
never imports or references a concrete env.

Deferred, each an isolated add-on not required for a correct first loop:
``target_kl`` early-stop (belongs in ``ppo_update``), learning-rate annealing,
``explained_variance`` reporting (needs a new ``PPOStats`` field), checkpointing,
and periodic evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.encoder import HIDDEN_DIM
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
    seed, trunk width) plus a nested :class:`PPOConfig` forwarded verbatim to
    :func:`ppo_update`. Nesting the existing config avoids restating - and later
    drifting from - its eight fields and their defaults. ``hidden_dim`` defaults
    to the encoder's :data:`HIDDEN_DIM`; the PPO knobs default to ``PPOConfig``'s.
    """

    num_iterations: int
    n_steps: int
    learning_rate: float = DEFAULT_LEARNING_RATE
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


@dataclass(frozen=True)
class IterationRecord:
    """One iteration's diagnostics: the step counters plus both stat blocks.

    ``collect`` and ``ppo`` are stored whole rather than flattened, so every
    field the collector and update expose is preserved without re-listing them.
    """

    iteration: int
    global_step: int
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
    collector = RolloutCollector(env, actor_critic, seed=config.seed)
    buffer = RolloutBuffer()

    records: list[IterationRecord] = []
    global_step = 0
    for iteration in range(config.num_iterations):
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
                collect=collect_stats,
                ppo=ppo_stats,
            )
        )
        _log_iteration(iteration, global_step, collect_stats, ppo_stats)

    return TrainHistory(records=records, actor_critic=actor_critic)


def _log_iteration(iteration: int, global_step: int, collect: CollectStats, ppo: PPOStats) -> None:
    """Emit one INFO line of the iteration's key diagnostics.

    The episode means are ``None`` when no episode completed this iteration;
    rendered as ``n/a`` so the line stays fixed-shape instead of crashing a
    ``%f`` format. Lazy ``%``-args keep formatting off the hot path when INFO is
    disabled.
    """
    ret = "n/a" if collect.mean_episode_return is None else f"{collect.mean_episode_return:.3f}"
    length = "n/a" if collect.mean_episode_length is None else f"{collect.mean_episode_length:.2f}"
    logger.info(
        "iter=%d global_step=%d ep_return=%s ep_len=%s n_episodes=%d "
        "policy_loss=%.4f value_loss=%.4f entropy=%.4f approx_kl=%.4f "
        "clip_frac=%.3f grad_norm=%.3f steps_per_second=%.0f",
        iteration,
        global_step,
        ret,
        length,
        collect.n_episodes,
        ppo.policy_loss,
        ppo.value_loss,
        ppo.entropy,
        ppo.approx_kl,
        ppo.clip_fraction,
        ppo.grad_norm,
        collect.steps_per_second,
    )
