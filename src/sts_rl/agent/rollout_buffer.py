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
from typing import Protocol

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


class SupportsMinibatches(Protocol):
    """The rollout-buffer surface :func:`~sts_rl.agent.ppo_update.ppo_update` consumes.

    A structural type covering exactly what the update pass touches - the length
    and the shuffled-minibatch iterator - and nothing else. Both the single-env
    :class:`RolloutBuffer` and the multi-env :class:`VecRolloutBuffer` satisfy it,
    so ``ppo_update`` is agnostic to the buffer layout (one env vs a flattened
    ``num_envs`` batch).
    """

    def __len__(self) -> int: ...

    def iter_minibatches(
        self, minibatch_size: int, shuffle: bool = True
    ) -> Iterator[MiniBatch]: ...


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

        Tensors are stored ``.detach().clone()``d: detached so a buffered
        rollout never retains the autograd graph that produced its
        log-probs/values (the update path recomputes fresh grad-tracked values
        from the stored obs, and keeping the old graph alive would leak memory
        and risk a double-backward), and cloned so the buffer owns its copy and
        is immune to a caller mutating the passed tensors in place after ``add``.

        ``done`` must be the TERMINAL flag used for bootstrapping (``1.0`` iff
        the episode truly ended at this step), NOT a time-limit truncation:
        :func:`compute_gae` zeroes the value bootstrap on a done, so a
        truncation flagged as done would wrongly discard the tail value.
        """
        self._obs.append({key: tensor.detach().clone() for key, tensor in obs.items()})
        self._actions.append(action.detach().clone())
        self._log_probs.append(log_prob.detach().clone())
        self._values.append(value.detach().clone())
        self._rewards.append(reward)
        self._dones.append(done)
        self._masks.append(mask.detach().clone())

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
        if not self._actions:
            raise RuntimeError(
                "compute_advantages requires at least one transition; call add() first"
            )
        # Co-locate rewards/dones with values so compute_gae never mixes devices on GPU.
        values = torch.stack(self._values).to(torch.float32)
        rewards = torch.tensor(self._rewards, dtype=torch.float32, device=values.device)
        dones = torch.tensor(self._dones, dtype=torch.float32, device=values.device)
        self.advantages, self.returns = compute_gae(
            rewards,
            values,
            dones,
            last_value.detach().to(values.device),
            gamma=gamma,
            gae_lambda=gae_lambda,
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

        # Co-locate the shuffle index with the data so the gather stays on-device on GPU.
        order = (
            torch.randperm(length, device=values.device)
            if shuffle
            else torch.arange(length, device=values.device)
        )
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


def _assert_leading_dim(tensor: Tensor, expected: int, name: str) -> None:
    """Raise unless ``tensor``'s leading axis is ``expected`` (the num_envs boundary)."""
    if tensor.shape[0] != expected:
        raise ValueError(
            f"{name} must have leading dim {expected}, got shape {tuple(tensor.shape)}"
        )


class VecRolloutBuffer:
    """Multi-env rollout buffer: ``num_envs`` independent trajectories, per-env GAE.

    Holds one single-env :class:`RolloutBuffer` per environment and delegates to
    them, so GAE is computed INDEPENDENTLY per env through the same trusted
    :func:`~sts_rl.agent.ppo.compute_gae` path - a done or truncation in one env
    never leaks advantage into another. For the PPO update the per-env
    trajectories are flattened into one ``(T * num_envs, ...)`` batch (env-major:
    env 0's ``T`` steps, then env 1's, ...). The flattening order is irrelevant to
    the update because :meth:`iter_minibatches` shuffles, and the update
    normalizes advantages per minibatch over that flattened set (standard
    vectorized-PPO semantics).

    Satisfies :class:`SupportsMinibatches`, so it is a drop-in for
    :func:`~sts_rl.agent.ppo_update.ppo_update` in place of a single
    :class:`RolloutBuffer`.
    """

    def __init__(self, num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        # One trusted single-env buffer per env; all storage/GAE delegates here.
        self._buffers: list[RolloutBuffer] = [RolloutBuffer() for _ in range(num_envs)]

    def add_batch(
        self,
        obs: dict[str, Tensor],
        actions: Tensor,
        log_probs: Tensor,
        values: Tensor,
        rewards: Tensor,
        dones: Tensor,
        masks: Tensor,
    ) -> None:
        """Append one batched transition, scattering row ``i`` to env ``i``'s buffer.

        Every tensor carries a leading ``num_envs`` axis: ``obs`` maps each field
        to ``(num_envs, *field_shape)``, ``masks`` is ``(num_envs, ACTION_DIM)``,
        and ``actions``/``log_probs``/``values``/``rewards``/``dones`` are
        ``(num_envs,)``. ``dones[i]`` is env ``i``'s TERMINAL flag only (never a
        truncation), matching :meth:`RolloutBuffer.add`; the per-step
        ``.detach().clone()`` ownership is inherited from the sub-buffers' ``add``.
        """
        # Assert the num_envs leading axis on EVERY batched field (not just
        # actions): a mismatch must fail loudly here rather than silently drop or
        # misalign rows through the per-env indexing below.
        for name, tensor in obs.items():
            _assert_leading_dim(tensor, self.num_envs, f"obs[{name}]")
        _assert_leading_dim(actions, self.num_envs, "actions")
        _assert_leading_dim(log_probs, self.num_envs, "log_probs")
        _assert_leading_dim(values, self.num_envs, "values")
        _assert_leading_dim(rewards, self.num_envs, "rewards")
        _assert_leading_dim(dones, self.num_envs, "dones")
        _assert_leading_dim(masks, self.num_envs, "masks")
        for i, buffer in enumerate(self._buffers):
            buffer.add(
                obs={name: tensor[i] for name, tensor in obs.items()},
                action=actions[i],
                log_prob=log_probs[i],
                value=values[i],
                reward=float(rewards[i]),
                done=float(dones[i]),
                mask=masks[i],
            )

    def compute_advantages(
        self,
        last_values: Tensor,
        gamma: float = DEFAULT_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
    ) -> None:
        """Run GAE PER env, bootstrapping env ``i`` from ``last_values[i]``.

        ``last_values`` is ``(num_envs,)`` (the critic's value of each env's
        post-rollout cursor state). Each env's advantages/returns come from the
        single-env :func:`~sts_rl.agent.ppo.compute_gae` over that env's own
        stored trajectory, so the recursion is severed at that env's episode
        boundaries alone. Advantages stay RAW (per-minibatch normalization is the
        update's job), matching :meth:`RolloutBuffer.compute_advantages`.
        """
        _assert_leading_dim(last_values, self.num_envs, "last_values")
        for i, buffer in enumerate(self._buffers):
            buffer.compute_advantages(last_values[i], gamma=gamma, gae_lambda=gae_lambda)

    def __len__(self) -> int:
        """Total stored transitions across all envs (``T * num_envs`` after a collect)."""
        return sum(len(buffer) for buffer in self._buffers)

    def reset(self) -> None:
        """Clear every env's sub-buffer for the next rollout."""
        for buffer in self._buffers:
            buffer.reset()

    def iter_minibatches(self, minibatch_size: int, shuffle: bool = True) -> Iterator[MiniBatch]:
        """Yield the flattened ``(T * num_envs, ...)`` rollout as ``(mb, ...)`` minibatches.

        Requires :meth:`compute_advantages` to have run (per env). Each env's full
        in-order ``(T, ...)`` slice - already carrying that env's GAE - is taken
        from its sub-buffer and concatenated env-major, then a single index
        permutation is sliced into ``minibatch_size`` chunks, mirroring
        :meth:`RolloutBuffer.iter_minibatches`. Every transition lands in exactly
        one minibatch; the final chunk keeps the remainder.
        """
        if minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be positive, got {minibatch_size}")
        length = len(self)
        if length == 0:  # nothing collected; yield nothing rather than cat([]) crashing
            return
        # The collector steps every env in lockstep, so all sub-buffers share T. An
        # uneven fill is a caller bug (e.g. add_batch skipped an env); fail loudly.
        sub_lengths = {len(buffer) for buffer in self._buffers}
        if len(sub_lengths) != 1:
            raise RuntimeError(
                f"vec sub-buffers have unequal lengths {sub_lengths}; every env must be "
                "stepped the same number of times before iter_minibatches"
            )
        # One full-width, in-order MiniBatch per env (each already GAE-computed),
        # concatenated along the batch axis into the flattened rollout.
        per_env = [
            next(buffer.iter_minibatches(len(buffer), shuffle=False)) for buffer in self._buffers
        ]
        obs = {key: torch.cat([mb.obs[key] for mb in per_env]) for key in per_env[0].obs}
        masks = torch.cat([mb.masks for mb in per_env])
        actions = torch.cat([mb.actions for mb in per_env])
        old_log_probs = torch.cat([mb.old_log_probs for mb in per_env])
        old_values = torch.cat([mb.old_values for mb in per_env])
        advantages = torch.cat([mb.advantages for mb in per_env])
        returns = torch.cat([mb.returns for mb in per_env])

        # Co-locate the shuffle index with the data so the gather stays on-device.
        order = (
            torch.randperm(length, device=advantages.device)
            if shuffle
            else torch.arange(length, device=advantages.device)
        )
        for start in range(0, length, minibatch_size):
            idx = order[start : start + minibatch_size]
            yield MiniBatch(
                obs={key: tensor[idx] for key, tensor in obs.items()},
                masks=masks[idx],
                actions=actions[idx],
                old_log_probs=old_log_probs[idx],
                old_values=old_values[idx],
                advantages=advantages[idx],
                returns=returns[idx],
            )
