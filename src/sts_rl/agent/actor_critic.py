"""Actor-critic wrapper composing the shared trunk with both heads.

Ties the head-agnostic :class:`~sts_rl.agent.encoder.ObsFeatureEncoder` trunk to
its two consumers - :class:`~sts_rl.agent.policy_head.MaskedPolicyHead` (actor)
and :class:`~sts_rl.agent.value_head.ValueHead` (critic) - behind one module.
The point of the shared trunk is that a single encode feeds both heads, so this
wrapper encodes exactly once per call and hands the same feature vector to each.

No env interaction, rollout buffer, optimizer, or training loop lives here: this
is only the network. The PPO caller owns those and decides gradient context -
rollout collection wraps :meth:`act` in ``no_grad``, while the update path calls
:meth:`evaluate_actions` with grad enabled.
"""

from __future__ import annotations

from torch import Tensor, nn

from sts_rl.agent.encoder import HIDDEN_DIM, ObsFeatureEncoder
from sts_rl.agent.policy_head import MaskedPolicyHead
from sts_rl.agent.value_head import ValueHead


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic: one encoder feeding a policy and a value head.

    ``act`` is the rollout path (sample or greedy) and ``evaluate_actions`` is
    the PPO update path (recompute log-prob/entropy/value with gradients). Both
    encode once and share the features across the two heads.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.encoder = ObsFeatureEncoder(hidden_dim)
        # Wire each head's input width from the encoder's output_dim (not a
        # literal), so a trunk-width change propagates without a mismatch.
        self.policy = MaskedPolicyHead(input_dim=self.encoder.output_dim)
        self.value = ValueHead(input_dim=self.encoder.output_dim)

    def act(
        self,
        obs: dict[str, Tensor],
        mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select an action and estimate its state-value for a rollout step.

        Returns ``(action, log_prob, entropy, value)``, each ``(B,)``. Not
        wrapped in ``no_grad`` here: the rollout caller owns the gradient
        context (it wraps collection in ``no_grad``; the update path needs grad).
        """
        # Encode ONCE and share features across both heads - the whole point of
        # the trunk is no double-encoding per step.
        features = self.encoder(obs)
        action, log_prob, entropy = self.policy.act(features, mask, deterministic)
        value = self.value(features)
        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        obs: dict[str, Tensor],
        mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Recompute ``(log_prob, entropy, value)`` of ``actions`` for the PPO update.

        Gradients flow through the shared encoder and both heads. ``actions`` is
        ``(B,)``; each returned tensor is ``(B,)``.
        """
        # Single encode shared by both heads (same trunk as the rollout path).
        features = self.encoder(obs)
        log_prob, entropy = self.policy.evaluate_actions(features, mask, actions)
        value = self.value(features)
        return log_prob, entropy, value

    def get_value(self, obs: dict[str, Tensor]) -> Tensor:
        """Estimate state-value only, of shape ``(B,)``.

        The PPO loop calls this to bootstrap GAE from the value of the state
        after the last rollout step (``compute_gae``'s ``last_value``): no action
        is taken there, so running the policy head or building an action mask
        would be wasted work.
        """
        return self.value(self.encoder(obs))
