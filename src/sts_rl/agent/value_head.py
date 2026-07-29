"""State-value head (critic) for the Slay the Spire RL agent.

The critic counterpart to :class:`~sts_rl.agent.policy_head.MaskedPolicyHead`.
It takes the shared observation encoder's pooled ``CLS`` context vector (shape
``(B, input_dim)``, where ``input_dim`` is the encoder's ``output_dim`` == the
transformer ``d_model``) and projects it to a single scalar state-value estimate
``V(s)`` per batch row.

No masking lives here: a value is over states only, not actions, so the pooled
context is projected directly with no legality applied.
"""

from __future__ import annotations

from torch import Tensor, nn

from sts_rl.agent.encoder import HIDDEN_DIM


class ValueHead(nn.Module):
    """Project the pooled ``CLS`` context to a scalar state-value estimate ``V(s)``.

    A single ``Linear`` head over the encoder's pooled context, the critic
    sibling of the policy head. Shaped for a PPO consumer: the returned ``(B,)``
    value is differenced elementwise against a ``(B,)`` returns vector in the
    advantage and value-loss computations.
    """

    def __init__(self, input_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        # Value head only: no nonlinearity here, the encoder's final LayerNorm
        # already normalizes the pooled context this head reads.
        self.value = nn.Linear(input_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        """Return the state-value estimate of shape ``(B,)``.

        ``features`` is ``(B, input_dim)``. The Linear emits ``(B, 1)``; the
        ``squeeze(-1)`` collapses it to ``(B,)``. This matters: PPO computes
        ``returns - values`` elementwise, and a ``(B, 1)`` value would broadcast
        against a ``(B,)`` returns vector to ``(B, B)``. The ``(B,)`` convention
        also matches the policy head's ``(B,)``
        log_prob/entropy outputs.
        """
        return self.value(features).squeeze(-1)
