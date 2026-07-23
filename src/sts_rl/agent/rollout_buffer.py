"""Single-environment PPO rollout buffer.

Buffers one environment's fixed-length trajectory of transitions collected
during rollout, then serves shuffled minibatches for the PPO update epochs.
Between the two it computes GAE once via :func:`~sts_rl.agent.ppo.compute_gae`.

This is a passive data structure: no environment, optimizer, or training loop
lives here. Per-step tensors carry NO batch dim (a single env) - an observation
is a dict of ``(*field_shape)`` tensors, ``mask`` is ``(ACTION_DIM,)`` bool, and
``action``/``log_prob``/``value`` are scalars. Stacking to ``(T, ...)`` happens
lazily in :meth:`RolloutBuffer.iter_minibatches`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from sts_rl.agent.ppo import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA, compute_gae


@dataclass(frozen=True)
class MiniBatch:
    """One shuffled slice of the rollout, batched to ``(mb, ...)`` per field.

    ``obs`` maps each field name to a ``(mb, *field_shape)`` tensor; ``masks`` is
    ``(mb, ACTION_DIM)``; the remaining vector fields are ``(mb,)``. ``mb`` is at
    most the requested minibatch size and may be smaller for the final batch.
    """

    obs: dict[str, Tensor]
    masks: Tensor
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor


class RolloutBuffer:
    """Accumulates one env's transitions, computes GAE, and yields minibatches.

    Usage is three phases: :meth:`add` per step during rollout,
    :meth:`compute_advantages` once at the end, then :meth:`iter_minibatches`
    across the PPO update epochs. :meth:`reset` clears it for the next rollout.
    """

    def __init__(self) -> None:
        # Parallel per-field accumulators, one entry appended per transition.
        self._obs: list[dict[str, Tensor]] = []
        self._actions: list[Tensor] = []
        self._log_probs: list[Tensor] = []
        self._values: list[Tensor] = []
        self._rewards: list[float] = []
        self._dones: list[float] = []
        self._masks: list[Tensor] = []
        # Populated only by compute_advantages; None guards iter_minibatches.
        self.advantages: Tensor | None = None
        self.returns: Tensor | None = None

    def add(
        self,
        obs: dict[str, Tensor],
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: float,
        done: float,
        mask: Tensor,
    ) -> None:
        """Append one transition.

        Tensors are stored ``.detach()``ed so a buffered rollout never retains
        the autograd graph that produced its log-probs/values: the update path
        recomputes fresh, grad-tracked values from the stored obs, and keeping
        the old graph alive would leak memory and risk a double-backward.

        ``done`` must be the TERMINAL flag used for bootstrapping (``1.0`` iff
        the episode truly ended at this step), NOT a time-limit truncation:
        :func:`compute_gae` zeroes the value bootstrap on a done, so a
        truncation flagged as done would wrongly discard the tail value.
        """
        self._obs.append({key: tensor.detach() for key, tensor in obs.items()})
        self._actions.append(action.detach())
        self._log_probs.append(log_prob.detach())
        self._values.append(value.detach())
        self._rewards.append(reward)
        self._dones.append(done)
        self._masks.append(mask.detach())

    def __len__(self) -> int:
        return len(self._actions)

    def compute_advantages(
        self,
        last_value: Tensor,
        gamma: float = DEFAULT_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
    ) -> None:
        """Run GAE over the stored trajectory, filling ``advantages``/``returns``.

        ``last_value`` is the critic's value of the state AFTER the final step,
        used to bootstrap the last delta; it is detached so the stored
        advantages/returns (constant targets for the update) never retain the
        value head's autograd graph. Advantages are stored RAW: per-minibatch
        advantage normalization is the training loop's job, not the buffer's,
        matching :func:`compute_gae`'s own contract.
        """
        rewards = torch.tensor(self._rewards, dtype=torch.float32)
        dones = torch.tensor(self._dones, dtype=torch.float32)
        values = torch.stack(self._values).to(torch.float32)
        self.advantages, self.returns = compute_gae(
            rewards, values, dones, last_value.detach(), gamma=gamma, gae_lambda=gae_lambda
        )

    def iter_minibatches(self, minibatch_size: int, shuffle: bool = True) -> Iterator[MiniBatch]:
        """Yield the rollout as ``(mb, ...)`` minibatches for one update epoch.

        Requires :meth:`compute_advantages` to have run first. The stored per-
        step data is stacked once into ``(T, ...)`` tensors, then a single index
        permutation is sliced into consecutive ``minibatch_size`` chunks. Every
        index appears in exactly one minibatch, and the final chunk keeps the
        remainder when ``T`` is not a multiple of ``minibatch_size``.
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "iter_minibatches requires compute_advantages to run first (advantages are None)"
            )
        if minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be positive, got {minibatch_size}")
        length = len(self._actions)
        if length == 0:  # nothing collected; yield no minibatches rather than crash on stack
            return

        # Stack once per epoch call; obs is stacked field-by-field.
        obs: dict[str, Tensor] = {
            key: torch.stack([step[key] for step in self._obs]) for key in self._obs[0]
        }
        masks = torch.stack(self._masks)
        actions = torch.stack(self._actions)
        log_probs = torch.stack(self._log_probs)
        # Match the float32 of advantages/returns so the value loss sees one dtype.
        values = torch.stack(self._values).to(torch.float32)

        order = torch.randperm(length) if shuffle else torch.arange(length)
        for start in range(0, length, minibatch_size):
            idx = order[start : start + minibatch_size]
            yield MiniBatch(
                obs={key: tensor[idx] for key, tensor in obs.items()},
                masks=masks[idx],
                actions=actions[idx],
                old_log_probs=log_probs[idx],
                old_values=values[idx],
                advantages=self.advantages[idx],
                returns=self.returns[idx],
            )

    def reset(self) -> None:
        """Clear all accumulators and computed tensors for buffer reuse."""
        self._obs.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()
        self._masks.clear()
        self.advantages = None
        self.returns = None
