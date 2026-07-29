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
from sts_rl.agent.rollout_buffer import SupportsMinibatches
from sts_rl.agent.running_moments import RunningMoments

# Standard PPO objective weights and update schedule. Named so a tuning change
# propagates instead of being buried as literals in PPOConfig's signature.
DEFAULT_VF_COEF: float = 0.5
DEFAULT_ENT_COEF: float = 0.01
DEFAULT_N_EPOCHS: int = 4
DEFAULT_MINIBATCH_SIZE: int = 64
DEFAULT_MAX_GRAD_NORM: float = 0.5
# EWMA decay for advantage normalization, applied per item to the running
# moments (see RunningMoments). decay=1.0 gives retain 0, so every minibatch is
# normalized against the whole rollout's population stats; that differs from the
# prior per-minibatch normalization whenever there is more than one minibatch.
# Effective smoothing depends on rollout size via the per-item exponent,
# retain = (1 - decay) ** n, so at 5e-4 a ~2048-step rollout retains ~0.36 of
# the prior estimate.
DEFAULT_ADV_NORM_DECAY: float = 5e-4
# Denominator floor for advantage normalization; avoids a blow-up when the
# running advantage spread is near zero (all-equal or still-cold stream).
ADV_NORM_EPS: float = 1e-8
# Return-variance floor below which explained variance is ill-conditioned; we
# return a finite 0.0 rather than SB3/CleanRL's nan (see _explained_variance).
EXPLAINED_VAR_EPS: float = 1e-8


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for one :func:`ppo_update` call.

    ``clip_coef`` is shared with the surrogate/value clipping in
    :mod:`sts_rl.agent.ppo`, so its default is imported from there rather than
    re-declared. The two booleans toggle advantage normalization and the PPO2
    clipped value loss respectively. ``adv_norm_decay`` is the per-item EWMA
    decay for that advantage normalization (see :class:`RunningMoments`); 1.0
    gives retain 0, normalizing every minibatch against the whole rollout's
    population stats, which differs from the prior per-minibatch normalization
    when there is more than one minibatch. ``target_kl`` is the approximate-KL
    threshold above which the update stops early (``None`` disables it): a
    standard PPO guard against moving the policy too far from the
    data-collection policy in one update.
    """

    clip_coef: float = DEFAULT_CLIP_COEF
    vf_coef: float = DEFAULT_VF_COEF
    ent_coef: float = DEFAULT_ENT_COEF
    n_epochs: int = DEFAULT_N_EPOCHS
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    normalize_advantages: bool = True
    adv_norm_decay: float = DEFAULT_ADV_NORM_DECAY
    clip_value_loss: bool = True
    target_kl: float | None = None


@dataclass(frozen=True)
class PPOStats:
    """Mean-over-minibatches diagnostics from one :func:`ppo_update` call.

    Every float is averaged across all ``n_updates`` minibatch steps EXCEPT
    ``explained_variance``, which is computed once over the whole rollout; an
    empty buffer yields all-zero fields with ``n_updates == 0``.
    """

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    # Fraction of the return's variance the value function predicts: 1.0 is
    # perfect, 0.0 is no better than predicting the mean, and it can be negative.
    explained_variance: float
    n_updates: int
    # Running advantage-normalization scale from the shared/transient
    # RunningMoments (its std and mean) as of this update; 0.0 when advantage
    # normalization is off. Observability only. Defaulted and placed last so
    # existing PPOStats(...) call sites are unaffected.
    adv_norm_std: float = 0.0
    adv_norm_mean: float = 0.0


def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Fraction of the return variance the value function explains (SB3 idiom).

    ``1 - Var(y_true - y_pred) / Var(y_true)``. Uses POPULATION variance
    (``unbiased=False``) to match the ``np.var`` SB3/CleanRL use, not torch's
    default sample variance. On a (near-)constant-return target ``Var(y_true)``
    is ~0 and the ratio is ill-conditioned: SB3/CleanRL divide through and yield
    nan, but we clamp to a finite 0.0 so the diagnostic never poisons a
    downstream finite-value assertion. EV is only meaningful on state-dependent
    returns anyway.
    """
    var_y = torch.var(y_true, unbiased=False)
    if float(var_y) < EXPLAINED_VAR_EPS:
        return 0.0
    # Divide as tensors so the guarded-out zero-variance path matches SB3's nan.
    return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y)


def ppo_update(
    actor_critic: ActorCritic,
    buffer: SupportsMinibatches,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig = PPOConfig(),
    adv_moments: RunningMoments | None = None,
) -> PPOStats:
    """Run ``config.n_epochs`` PPO epochs over ``buffer``; return mean diagnostics.

    Each epoch iterates freshly shuffled minibatches. Per minibatch the network
    recomputes grad-tracked log-prob/entropy/value for the stored actions, forms
    the clipped policy loss, the (optionally clipped) value loss, and the entropy
    bonus, then backprops the combined objective, clips the global grad norm, and
    steps the optimizer. Returns the mean of each scalar stat over every
    minibatch update; an empty buffer (zero minibatches) yields zeroed stats
    with ``n_updates == 0`` rather than dividing by zero.

    When ``config.normalize_advantages`` is set, advantages are standardized
    against a :class:`RunningMoments` EWMA rather than each minibatch's own
    spread: the whole rollout's advantages are folded into the tracker ONCE per
    call (before the epochs), then every minibatch is normalized against those
    running stats. Pass ``adv_moments`` to share one tracker across calls so the
    EWMA persists over training; when it is ``None`` a transient tracker is used
    so a standalone call still normalizes (against just this rollout).
    """
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    entropy_sum = 0.0
    total_loss_sum = 0.0
    approx_kl_sum = 0.0
    clip_fraction_sum = 0.0
    grad_norm_sum = 0.0
    n_updates = 0

    # Advantage normalization tracker. A shared instance passed by the caller
    # makes the EWMA persist across calls; a None arg falls back to a transient
    # tracker so standalone/test calls still normalize (against just this
    # rollout). Built ONLY under normalize_advantages, so a normalization-disabled
    # call never constructs (or validates) a tracker it will not read.
    moments = adv_moments
    if config.normalize_advantages:
        if moments is None:
            moments = RunningMoments(decay=config.adv_norm_decay, eps=ADV_NORM_EPS)
        if len(buffer) > 0:
            # Fold the WHOLE rollout's advantages into the running moments ONCE, before
            # the epochs. A full-width unshuffled pass yields them in a single MiniBatch
            # (the same buffer idiom as the explained-variance pass below), so this
            # never depends on a buffer-specific advantages accessor - it works for
            # both the single-env and the flattened vec buffer.
            (adv_batch,) = buffer.iter_minibatches(len(buffer), shuffle=False)
            moments.update(adv_batch.advantages)

    for _ in range(config.n_epochs):
        epoch_approx_kl_sum = 0.0
        epoch_minibatches = 0
        for mb in buffer.iter_minibatches(config.minibatch_size, shuffle=True):
            advantages = mb.advantages
            if config.normalize_advantages and advantages.numel() > 0:
                # Standardize against the PERSISTENT running stats (folded once
                # above), not this minibatch's own mean/std, so the normalization
                # scale stays stable across updates. Every non-empty minibatch is
                # standardized against those running stats, which are N=1-safe
                # (population moments, no Bessel term), so no singleton is skipped;
                # the > 0 guard only skips a (degenerate) empty minibatch.
                assert moments is not None  # built above whenever normalize_advantages
                advantages = moments.normalize(advantages)

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

        # End-of-epoch early stop: once this epoch's mean approx_kl exceeds
        # target_kl, halt before further epochs so a single update does not push
        # the policy too far from the data-collection policy. The end-of-epoch
        # TIMING follows CleanRL; the statistic is our own choice - we break on
        # the epoch-mean approx_kl (a lower-variance stop signal), whereas CleanRL
        # reads the last minibatch's approx_kl and SB3 checks each minibatch
        # against 1.5 * target_kl.
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
            explained_variance=0.0,
            n_updates=0,
            adv_norm_std=moments.std if moments is not None else 0.0,
            adv_norm_mean=moments.mean if moments is not None else 0.0,
        )

    # Explained variance is a property of the collection-time values vs the GAE
    # returns (both fixed across the update), so compute it ONCE over the full
    # rollout, not as a per-minibatch average. One unshuffled full-width pass
    # yields a single batch carrying the whole rollout's old_values/returns.
    (full_batch,) = buffer.iter_minibatches(len(buffer), shuffle=False)
    explained_variance = _explained_variance(full_batch.old_values, full_batch.returns)

    return PPOStats(
        policy_loss=policy_loss_sum / n_updates,
        value_loss=value_loss_sum / n_updates,
        entropy=entropy_sum / n_updates,
        total_loss=total_loss_sum / n_updates,
        approx_kl=approx_kl_sum / n_updates,
        clip_fraction=clip_fraction_sum / n_updates,
        grad_norm=grad_norm_sum / n_updates,
        explained_variance=explained_variance,
        n_updates=n_updates,
        adv_norm_std=moments.std if moments is not None else 0.0,
        adv_norm_mean=moments.mean if moments is not None else 0.0,
    )
