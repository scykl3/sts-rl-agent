"""Actor-critic wrapper composing the shared encoder with both heads.

Ties the head-agnostic :class:`~sts_rl.agent.encoder.ObsFeatureEncoder`
transformer to its two consumers - :class:`~sts_rl.agent.policy_head.PointerPolicyHead`
(actor) and :class:`~sts_rl.agent.value_head.ValueHead` (critic) - behind one
module. The point of the shared encoder is that a single encode feeds both
heads, so this wrapper encodes exactly once per call and shares that one encode
across the two heads.

The encoder returns ``(per_token, pooled_cls, key_padding_mask)``. The pointer
policy head reads the per-token embeddings and the padding mask (it scores each
per-entity action against its entity token); the value head reads the pooled
``CLS`` context. ``get_value`` runs only the value head, so it skips the pointer
head entirely.

No env interaction, rollout buffer, optimizer, or training loop lives here: this
is only the network. The PPO caller owns those and decides gradient context -
rollout collection wraps :meth:`act` in ``no_grad``, while the update path calls
:meth:`evaluate_actions` with grad enabled.
"""

from __future__ import annotations

from torch import Tensor, nn

from sts_rl.agent.encoder import HIDDEN_DIM, ObsFeatureEncoder
from sts_rl.agent.policy_head import PointerPolicyHead
from sts_rl.agent.value_head import ValueHead


class ActorCritic(nn.Module):
    """Shared-encoder actor-critic: one encoder feeding a policy and a value head.

    ``act`` is the rollout path (sample or greedy) and ``evaluate_actions`` is
    the PPO update path (recompute log-prob/entropy/value with gradients). Both
    encode once and share that single encode across the two heads.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        # hidden_dim is the transformer d_model; the constructor name is kept so
        # the training config (TrainConfig.hidden_dim) and saved checkpoints keep
        # a single width setting.
        self.encoder = ObsFeatureEncoder(d_model=hidden_dim)
        # Wire each head's input width from the encoder's output_dim (not a
        # literal), so a width change propagates without a mismatch. The pointer
        # head reads the per-token embeddings; the value head reads pooled CLS.
        self.policy = PointerPolicyHead(input_dim=self.encoder.output_dim)
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
        # Encode ONCE and share the single encode across both heads - the whole
        # point of the shared encoder is no double-encoding per step. The pointer
        # head reads the per-token embeddings + padding mask; value reads pooled CLS.
        per_token, pooled_cls, key_padding_mask = self.encoder(obs)
        action, log_prob, entropy = self.policy.act(
            per_token, key_padding_mask, mask, deterministic
        )
        value = self.value(pooled_cls)
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
        # Single encode shared by both heads (same encoder as the rollout path).
        per_token, pooled_cls, key_padding_mask = self.encoder(obs)
        log_prob, entropy = self.policy.evaluate_actions(per_token, key_padding_mask, mask, actions)
        value = self.value(pooled_cls)
        return log_prob, entropy, value

    def get_value(self, obs: dict[str, Tensor]) -> Tensor:
        """Estimate state-value only, of shape ``(B,)``.

        The PPO loop calls this to bootstrap GAE from the value of the state
        after the last rollout step (``compute_gae``'s ``last_value``): no action
        is taken there, so running the policy head or building an action mask
        would be wasted work.
        """
        _per_token, pooled_cls, _mask = self.encoder(obs)
        return self.value(pooled_cls)
