"""One PPO optimizer pass over a filled rollout buffer.

Ties together the pure objective math in :mod:`sts_rl.agent.ppo`, the
:class:`~sts_rl.agent.actor_critic.ActorCritic` network, and the shuffled
minibatches served by :class:`~sts_rl.agent.rollout_buffer.RolloutBuffer` into a
single multi-epoch update. This is ONLY the update pass: the caller owns env
stepping, rollout collection, GAE, and optimizer construction. It runs
``n_epochs`` passes over the buffer, and for each minibatch forms the combined
policy + value - entropy objective, backprops it, clips the global grad norm,
and steps the optimizer, accumulating diagnostics along the way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.ppo import DEFAULT_CLIP_COEF, clipped_policy_loss, clipped_value_loss
from sts_rl.agent.rollout_buffer import RolloutBuffer

# Standard PPO objective weights and update schedule. Named so a tuning change
# propagates instead of being buried as literals in PPOConfig's signature.
DEFAULT_VF_COEF: float = 0.5
DEFAULT_ENT_COEF: float = 0.01
DEFAULT_N_EPOCHS: int = 4
DEFAULT_MINIBATCH_SIZE: int = 64
DEFAULT_MAX_GRAD_NORM: float = 0.5
# Denominator floor for per-minibatch advantage normalization; avoids a blow-up
# when a minibatch has near-zero advantage spread.
ADV_NORM_EPS: float = 1e-8


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for one :func:`ppo_update` call.

    ``clip_coef`` is shared with the surrogate/value clipping in
    :mod:`sts_rl.agent.ppo`, so its default is imported from there rather than
    re-declared. The two booleans toggle the per-minibatch advantage
    normalization and the PPO2 clipped value loss respectively. ``target_kl``
    is the approximate-KL threshold above which the update stops early (``None``
    disables it): a standard PPO guard against moving the policy too far from
    the data-collection policy in one update.
    """

    clip_coef: float = DEFAULT_CLIP_COEF
    vf_coef: float = DEFAULT_VF_COEF
    ent_coef: float = DEFAULT_ENT_COEF
    n_epochs: int = DEFAULT_N_EPOCHS
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    normalize_advantages: bool = True
    clip_value_loss: bool = True
    target_kl: float | None = None


@dataclass(frozen=True)
class PPOStats:
    """Mean-over-minibatches diagnostics from one :func:`ppo_update` call.

    Every float is averaged across all ``n_updates`` minibatch steps; an empty
    buffer yields all-zero fields with ``n_updates == 0``.
    """

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    n_updates: int


def ppo_update(
    actor_critic: ActorCritic,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig = PPOConfig(),
) -> PPOStats:
    """Run ``config.n_epochs`` PPO epochs over ``buffer``; return mean diagnostics.

    Each epoch iterates freshly shuffled minibatches. Per minibatch the network
    recomputes grad-tracked log-prob/entropy/value for the stored actions, forms
    the clipped policy loss, the (optionally clipped) value loss, and the entropy
    bonus, then backprops the combined objective, clips the global grad norm, and
    steps the optimizer. Returns the mean of each scalar stat over every
    minibatch update; an empty buffer (zero minibatches) yields zeroed stats
    with ``n_updates == 0`` rather than dividing by zero.
    """
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    total_loss_sum = 0.0
    approx_kl_sum = 0.0
    clip_fraction_sum = 0.0
    grad_norm_sum = 0.0
    n_updates = 0

    for _ in range(config.n_epochs):
        epoch_approx_kl_sum = 0.0
        epoch_minibatches = 0
        for mb in buffer.iter_minibatches(config.minibatch_size, shuffle=True):
            advantages = mb.advantages
            if config.normalize_advantages and advantages.numel() > 1:
                # Normalize per minibatch: the buffer stores RAW advantages, so
                # each minibatch is standardized against its own mean/std here.
                # Skip a singleton minibatch: torch.std applies Bessel's
                # correction (N-1), which is NaN for N=1 and would silently
                # poison the update; one advantage has nothing to standardize.
                advantages = (advantages - advantages.mean()) / (advantages.std() + ADV_NORM_EPS)

            log_prob, entropy, value = actor_critic.evaluate_actions(mb.obs, mb.masks, mb.actions)

            policy_loss = clipped_policy_loss(
                log_prob, mb.old_log_probs, advantages, config.clip_coef
            )
            if config.clip_value_loss:
                value_loss = clipped_value_loss(value, mb.returns, mb.old_values, config.clip_coef)
            else:
                value_loss = clipped_value_loss(value, mb.returns)
            entropy_bonus = entropy.mean()

            # Entropy is SUBTRACTED: it is a bonus we MAXIMIZE (to keep the
            # policy exploratory), and the optimizer minimizes, so a larger
            # entropy lowers the total loss.
            total_loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_bonus

            optimizer.zero_grad()
            total_loss.backward()
            # clip_grad_norm_ returns the pre-clip global grad norm - a free
            # training-health diagnostic (a spike flags instability).
            grad_norm = torch.nn.utils.clip_grad_norm_(
                actor_critic.parameters(), config.max_grad_norm
            )
            optimizer.step()

            # Diagnostics only; no_grad so these never retain the autograd graph.
            with torch.no_grad():
                # Schulman's k3 KL estimator: non-negative and lower-variance
                # than (old - new).mean(), and the signal the target_kl
                # early-stop reads at the end of each epoch.
                logratio = log_prob - mb.old_log_probs
                ratio = logratio.exp()
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy_bonus.item()
            total_loss_sum += total_loss.item()
            approx_kl_value = approx_kl.item()
            approx_kl_sum += approx_kl_value
            epoch_approx_kl_sum += approx_kl_value
            clip_fraction_sum += clip_fraction.item()
            grad_norm_sum += grad_norm.item()
            n_updates += 1
            epoch_minibatches += 1

        # CleanRL end-of-epoch early stop: once this epoch's mean approx_kl
        # exceeds target_kl, halt before further epochs so a single update does
        # not push the policy too far from the data-collection policy.
        if (
            config.target_kl is not None
            and epoch_minibatches > 0
            and epoch_approx_kl_sum / epoch_minibatches > config.target_kl
        ):
            break

    if n_updates == 0:  # empty buffer / no minibatches: avoid divide-by-zero
        return PPOStats(
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            total_loss=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
            grad_norm=0.0,
            n_updates=0,
        )

    return PPOStats(
        policy_loss=policy_loss_sum / n_updates,
        value_loss=value_loss_sum / n_updates,
        entropy=entropy_sum / n_updates,
        total_loss=total_loss_sum / n_updates,
        approx_kl=approx_kl_sum / n_updates,
        clip_fraction=clip_fraction_sum / n_updates,
        grad_norm=grad_norm_sum / n_updates,
        n_updates=n_updates,
    )
