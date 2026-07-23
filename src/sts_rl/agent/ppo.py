"""Pure PPO objective math for the Slay the Spire RL agent.

Engine-free, stateless functions over a single rollout trajectory of length
``T``. They consume the ``(B,)`` log-probs/values emitted by
:class:`~sts_rl.agent.policy_head.MaskedPolicyHead` and
:class:`~sts_rl.agent.value_head.ValueHead` and turn them into the scalar
losses a training loop minimizes. No buffer, no environment, no optimizer lives
here: this is the math only, so it is trivially unit-testable in isolation.

Sign convention: PPO *maximizes* the clipped surrogate, but optimizers
*minimize*, so :func:`clipped_policy_loss` returns the negated surrogate. A
lower returned value therefore means a better policy update.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Discount and GAE trace-decay defaults; standard PPO values. Named so a config
# bump propagates instead of being buried as literals in the signatures.
DEFAULT_GAMMA: float = 0.99
DEFAULT_GAE_LAMBDA: float = 0.95
# PPO surrogate clip range: ratio is confined to [1 - eps, 1 + eps].
DEFAULT_CLIP_COEF: float = 0.2
# The 0.5 that cancels the factor of 2 from the squared-error derivative. This
# is NOT the value-loss weighting (vf_coef) that scales the value term in the
# total objective - that weighting belongs to the training loop, not here.
HALF_MSE_FACTOR: float = 0.5


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation over one trajectory.

    Returns ``(advantages, returns)``, both shape ``(T,)``.

    ``rewards``, ``values`` and ``dones`` are each ``(T,)``; ``last_value`` is a
    scalar tensor (0-dim or shape ``(1,)``) holding the critic's value of the
    state AFTER the final step, used to bootstrap the last delta.

    ``dones[t]`` convention: ``1.0`` if step ``t`` was terminal (the episode
    ended AT ``t``), else ``0.0``. A terminal step's ``next_nonterminal`` is
    ``0``, which both zeroes the bootstrap in its own delta and blocks the
    lambda recursion from carrying advantage backward across the episode
    boundary, so a done truncates GAE exactly at ``t``.

    Advantages are intentionally NOT normalized here; per-minibatch
    normalization belongs in the training loop, so leaving raw advantages keeps
    this function a pure GAE definition. ``returns = advantages + values`` is
    the value-head regression target.

    The backward loop only reads/writes tensors and never assumes gradients are
    tracked; the caller decides whether to wrap it in ``torch.no_grad()``.
    """
    length = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    # adv_{T} = 0; carried backward through the recursion.
    next_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(length)):
        # Bootstrap from last_value at the tail, otherwise from the next stored
        # value. reshape(()) accepts either a 0-dim or (1,) last_value.
        next_value = last_value.reshape(()) if t == length - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        next_advantage = delta + gamma * gae_lambda * next_nonterminal * next_advantage
        advantages[t] = next_advantage
    returns = advantages + values
    return advantages, returns


def clipped_policy_loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    clip_coef: float = DEFAULT_CLIP_COEF,
) -> Tensor:
    """PPO clipped surrogate, returned as a scalar loss to MINIMIZE.

    All args are ``(B,)``. The probability ratio is formed in log space
    (``exp(new - old)``) for numerical stability. The surrogate is the
    elementwise min of the unclipped and clipped terms; we negate its mean
    because the surrogate is maximized but the optimizer minimizes.
    """
    ratio = (new_log_prob - old_log_prob).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages
    return -torch.min(surr1, surr2).mean()


def clipped_value_loss(
    new_values: Tensor,
    returns: Tensor,
    old_values: Tensor | None = None,
    clip_coef: float | None = None,
) -> Tensor:
    """Value-head regression loss, returned as a scalar to MINIMIZE.

    All tensor args are ``(B,)``. By default this is the plain
    ``0.5 * MSE(new_values, returns)``.

    Passing BOTH ``old_values`` and ``clip_coef`` switches on the PPO2 clipped
    value loss: the new value is confined to ``old_values +/- clip_coef`` and
    the loss is the elementwise max of the clipped and unclipped squared
    errors, which discourages the critic from moving too far in one update.
    Supplying only one of the two is a misconfiguration and raises
    ``ValueError`` rather than silently ignoring it.
    """
    if (old_values is None) != (clip_coef is None):
        raise ValueError(
            "clipped value loss needs BOTH old_values and clip_coef, or neither; "
            f"got old_values={'set' if old_values is not None else 'None'}, "
            f"clip_coef={'set' if clip_coef is not None else 'None'}"
        )
    unclipped = (new_values - returns) ** 2
    if old_values is None or clip_coef is None:  # narrow both for the type checker
        return HALF_MSE_FACTOR * unclipped.mean()
    v_clipped = old_values + (new_values - old_values).clamp(-clip_coef, clip_coef)
    clipped = (v_clipped - returns) ** 2
    return HALF_MSE_FACTOR * torch.max(unclipped, clipped).mean()
