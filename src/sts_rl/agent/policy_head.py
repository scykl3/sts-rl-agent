"""Masked policy head for the Slay the Spire RL agent.

The first consumer of the shared observation encoder's trunk. It takes the
head-agnostic feature vector (shape ``(B, input_dim)``, where ``input_dim`` is
the encoder's ``output_dim``) and projects it to one logit per action index.

Action masking lives HERE, not in the encoder: the trunk is deliberately
head-agnostic, so legality is applied only at the point where a distribution
over actions is formed. The mask is supplied by the caller (sourced from the
environment's ``action_mask`` info key) as a boolean tensor ``(B, action_dim)``
where True == legal. Illegal logits are pushed to a large finite negative value
so ``softmax`` assigns them ~zero probability.

Why a finite constant and not ``float('-inf')``: an ``-inf`` logit makes
``softmax`` exactly zero, but its gradient and the ``log_prob``/entropy of a
Categorical built on it produce ``NaN`` (``0 * -inf``) that then poisons the
whole backward pass. A large finite negative value gives ~zero probability with
clean, finite gradients. The interface's ``assert_valid_mask`` guarantees at
least one legal action per row, so no row is fully masked (which would still
NaN even the finite path once every logit is driven to the floor).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from sts_rl.agent.encoder import HIDDEN_DIM
from sts_rl.interface import ACTION_DIM, InterfaceError

# Logit assigned to illegal actions: large and negative but FINITE, so softmax
# gives ~0 probability while keeping gradients/log_prob/entropy finite. Using
# the dtype floor (float('-inf')) would NaN the backward pass; -1e9 is small
# enough that exp(masked - max_legal) underflows to 0 in float32.
MASKED_LOGIT: float = -1e9


class MaskedPolicyHead(nn.Module):
    """Project trunk features to masked action logits and a Categorical policy.

    A single ``Linear`` logit head over the shared trunk output. All masking is
    contained in this module; the encoder never sees the mask. The helpers
    (:meth:`act`, :meth:`evaluate_actions`) are shaped for a PPO consumer:
    ``act`` runs during rollout, ``evaluate_actions`` recomputes log-prob and
    entropy of stored actions during the update phase.
    """

    def __init__(self, input_dim: int = HIDDEN_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        # Logit head only: no nonlinearity here, the trunk already applied one.
        self.logits = nn.Linear(input_dim, action_dim)

    @staticmethod
    def _apply_mask(logits: Tensor, mask: Tensor) -> Tensor:
        """Overwrite illegal logits with :data:`MASKED_LOGIT`.

        ``mask`` is coerced to bool so an int/float mask indexes safely; True
        marks a legal action and is left untouched.
        """
        # Shape must match exactly: an unbatched (action_dim,) mask would
        # otherwise broadcast silently across a batched (B, action_dim) logits
        # tensor, applying the wrong legality to every row.
        if mask.shape != logits.shape:
            raise InterfaceError(
                f"mask shape {tuple(mask.shape)} does not match logits shape "
                f"{tuple(logits.shape)} (no broadcasting allowed)"
            )
        legal = mask.bool()
        # Every row needs at least one legal action; a fully-masked row would
        # drive all its logits to the floor and yield a uniform distribution
        # over illegal actions instead of failing.
        if not legal.any(dim=-1).all():
            raise InterfaceError(
                "mask has a fully-masked row with no legal action "
                "(would sample uniformly over illegal actions)"
            )
        # torch.where keeps this differentiable w.r.t. the legal logits while
        # replacing (not adding to) the illegal ones with the finite floor.
        floor = torch.full_like(logits, MASKED_LOGIT)
        return torch.where(legal, logits, floor)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        """Return masked logits of shape ``(B, action_dim)``.

        ``features`` is ``(B, input_dim)``; ``mask`` is a bool ``(B, action_dim)``
        tensor (True == legal). Illegal entries are set to the finite floor.
        """
        return self._apply_mask(self.logits(features), mask)

    def masked_distribution(self, features: Tensor, mask: Tensor) -> Categorical:
        """Build a Categorical from masked logits (``logits=``, not ``probs=``).

        Passing ``logits`` lets torch normalize in log-space, so the floored
        entries stay numerically well behaved instead of being renormalized
        from ~0 probabilities.
        """
        return Categorical(logits=self.forward(features, mask))

    def act(
        self,
        features: Tensor,
        mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select actions for a rollout step.

        Returns ``(action, log_prob, entropy)``, each ``(B,)``. With
        ``deterministic=True`` the action is the argmax over masked logits (the
        greedy/eval path); otherwise it is sampled. log-prob and entropy are
        always taken from the masked distribution so a PPO buffer stores
        consistent quantities regardless of the sampling mode.
        """
        dist = self.masked_distribution(features, mask)
        if deterministic:
            # argmax over masked logits: illegal entries sit at the floor, so
            # the greedy pick is always legal.
            action = dist.logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self,
        features: Tensor,
        mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Recompute ``(log_prob, entropy)`` of ``actions`` under the current policy.

        Used in PPO's update phase: the ratio between this log-prob and the
        rollout log-prob drives the surrogate objective. ``actions`` is ``(B,)``.
        """
        dist = self.masked_distribution(features, mask)
        return dist.log_prob(actions), dist.entropy()
