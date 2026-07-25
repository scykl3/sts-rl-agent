"""Single-environment PPO rollout collection.

The piece that turns a network plus an env into a filled
:class:`~sts_rl.agent.rollout_buffer.RolloutBuffer`: it steps a gymnasium-style
env with :meth:`~sts_rl.agent.actor_critic.ActorCritic.act`, stores each
transition, bootstraps the trajectory tail with
:meth:`~sts_rl.agent.actor_critic.ActorCritic.get_value`, runs GAE via the
buffer, and returns per-collect diagnostics. It owns the env-stepping side of
PPO (SB3's ``collect_rollouts``); the optimizer pass lives in
:mod:`sts_rl.agent.ppo_update`. It is engine-free testable against
:class:`~sts_rl.env.stub_env.StubEnv`.

Single env, so every tensor handed to the network is batched to ``B == 1`` and
every tensor handed to the buffer is unbatched (the buffer stacks to ``(T, ...)``
lazily). The rollout is ONE continuous stream: the env is reset once at
construction and thereafter only when an episode ends, so consecutive
:meth:`RolloutCollector.collect` calls resume the same stream rather than
restarting it.

``done`` vs truncation. ``buffer.add`` receives ``done = float(terminated)``,
the TRUE episode-end flag only, never a time-limit truncation. ``compute_gae``
zeroes the value bootstrap on a done, so flagging a truncation as done would
wrongly discard its tail value; leaving ``done = 0`` on truncation keeps the
next-state bootstrap ``V(s_{t+1})``. That leaves TWO consequences on a
truncation step, both benign only because the toy task is stationary:

(1) Value bootstrap: the buffer takes the next value from the following stored
    step (or the tail ``last_value``), which after a reset is ``V`` of the
    freshly presented observation, not the truncated state's true successor -
    :class:`StubEnv` does not expose a terminal/successor obs.
(2) GAE recursion: with ``done = 0`` the lambda recursion is NOT severed at the
    truncation step (``next_nonterminal = 1 - done = 1``), so the next episode's
    advantage leaks backward across the reset into the truncated episode's last
    step.

The real fix for both is a separate truncation/episode-start signal to
``compute_gae`` (like SB3's ``episode_starts``) that severs the recursion at the
boundary, not merely storing the successor obs - a future buffer/GAE enhancement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.ppo import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA
from sts_rl.agent.rollout_buffer import RolloutBuffer
from sts_rl.interface import (
    OBS_FIELDS,
    Obs,
    assert_valid_mask,
)

# Id (embedding-index) fields get a long tensor; every other field is float32.
# Derived from the interface registry - the single source of truth - so an
# OBS_FIELDS change propagates without a hardcoded field-name list here (this
# mirrors the encoder's own _ID_FIELDS derivation).
_ID_FIELDS = frozenset(field.name for field in OBS_FIELDS if field.bounds == "id")

# Info-dict keys read during collection. "action_mask" is on every info (it is
# INFO_KEYS_ALWAYS[0]); "episode" appears only on a terminal step and carries
# gymnasium RecordEpisodeStatistics' {"r": return, "l": length}.
MASK_INFO_KEY = "action_mask"
EPISODE_INFO_KEY = "episode"
EPISODE_RETURN_KEY = "r"
EPISODE_LENGTH_KEY = "l"


def observation_to_batched_tensors(obs: Obs, device: torch.device) -> dict[str, Tensor]:
    """Convert a single-env numpy observation into batched network input.

    Adds a leading batch dim (``B == 1``), casts id fields to ``torch.long`` and
    every other field to ``float32``, and moves each field to ``device``. Dtypes
    are set explicitly rather than left to ``torch.as_tensor``'s numpy inference,
    which would keep float64 obs as double and later mismatch the trunk's Linear.
    """
    batched: dict[str, Tensor] = {}
    for field in OBS_FIELDS:
        dtype = torch.long if field.name in _ID_FIELDS else torch.float32
        tensor = torch.as_tensor(obs[field.name], dtype=dtype, device=device)
        batched[field.name] = tensor.unsqueeze(0)  # (*shape) -> (1, *shape)
    return batched


@dataclass(frozen=True)
class CollectStats:
    """Diagnostics for one :meth:`RolloutCollector.collect` call.

    ``n_episodes`` counts episodes that ENDED during this collect (terminated or
    truncated). ``mean_episode_return``/``mean_episode_length`` are ``None`` when
    no episode ended - a degenerate-input guard, not zero, so a caller never
    divides by zero and never reads a fabricated 0 as a real mean. Both means are
    taken over ``info["episode"]`` (gymnasium RecordEpisodeStatistics' ``r``/``l``).
    ``steps_per_second`` times the env-stepping loop only.
    """

    n_steps: int
    n_episodes: int
    mean_episode_return: float | None
    mean_episode_length: float | None
    steps_per_second: float


class RolloutCollector:
    """Steps one env with a policy and fills a rollout buffer for the PPO update.

    Holds a persistent stream cursor (``self._obs``/``self._mask``) reset once at
    construction, so successive :meth:`collect` calls continue one trajectory.
    The network is not owned here; the collector only reads it (under
    ``no_grad``) to act and to bootstrap the tail value.
    """

    def __init__(
        self,
        env: gym.Env,
        actor_critic: ActorCritic,
        device: torch.device | None = None,
        seed: int | None = None,
        gamma: float = DEFAULT_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
    ) -> None:
        """Bind an env and network, resetting the env once to prime the stream.

        ``seed`` seeds ONLY the environment's reset stream (its instance-local
        Generator); it deliberately does not touch the global torch/numpy RNGs.
        Reproducible action sampling and buffer shuffling therefore require
        seeding the global torch RNG once at the training entry point
        (``torch.manual_seed``), per CleanRL/SB3 - so constructing a collector
        never clobbers another collector's RNG. ``device`` is inferred from the
        network's parameters when omitted. ``gamma``/``gae_lambda`` are the GAE
        discount and trace-decay forwarded to ``compute_advantages`` on every
        collect, defaulting to the shared ``DEFAULT_GAMMA``/``DEFAULT_GAE_LAMBDA``.
        """
        self._env = env
        self._actor_critic = actor_critic
        if device is None:
            try:
                device = next(actor_critic.parameters()).device
            except StopIteration as exc:  # a param-less module gives no device to infer
                raise ValueError(
                    "cannot infer device from an actor_critic with no parameters; "
                    "pass device explicitly"
                ) from exc
        self._device = device
        self._gamma = gamma
        self._gae_lambda = gae_lambda

        # Seed only the env, once, via its construction-time reset: the env owns
        # an instance-local Generator (gymnasium's self.np_random), so this can't
        # affect any other collector. The process-global RNGs are deliberately
        # NOT reseeded here - torch's default generator drives act()'s
        # dist.sample() and the buffer's randperm shuffle, so a torch.manual_seed
        # side effect in this constructor would let a second seeded collector
        # silently break an earlier one's reproducibility. Seeding the global
        # generators is the training entry point's job, done once at startup (per
        # CleanRL/SB3). In-collect resets are NOT reseeded - the rollout is one
        # continuous stream, and reseeding each episode would replay identical
        # episodes.
        if seed is not None:
            obs, info = env.reset(seed=seed)
        else:
            obs, info = env.reset()
        self._obs: Obs = obs
        self._mask: np.ndarray = info[MASK_INFO_KEY]

    def collect(self, buffer: RolloutBuffer, n_steps: int) -> CollectStats:
        """Fill ``buffer`` with exactly ``n_steps`` transitions, then run GAE.

        Clears ``buffer`` first (each collect fills a fresh rollout, mirroring
        SB3 resetting its buffer in ``collect_rollouts``), steps the env
        ``n_steps`` times under ``no_grad`` in eval mode, bootstraps the tail with
        ``get_value``, and calls :meth:`RolloutBuffer.compute_advantages`. The
        eval/train mode is saved and restored so a caller mid-training is
        unaffected (defensive for future Dropout/BatchNorm; SB3's
        ``set_training_mode(False)`` idiom).
        """
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        buffer.reset()
        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        was_training = self._actor_critic.training
        self._actor_critic.eval()
        try:
            loop_start = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_steps):
                    obs_batched = observation_to_batched_tensors(self._obs, self._device)
                    mask_batched = self._batched_mask(self._mask)

                    action, log_prob, _entropy, value = self._actor_critic.act(
                        obs_batched, mask_batched
                    )
                    # Single env: act returns (B=1,); a genuine scalar is required
                    # before .item() steps the env with a python int.
                    assert action.shape == (1,), f"expected (1,) action, got {tuple(action.shape)}"
                    next_obs, reward, terminated, truncated, info = self._env.step(
                        int(action.item())
                    )

                    # Store UNBATCHED per-step tensors (drop only the batch dim, not
                    # any size-1 feature dim). done is the TERMINAL flag only, never
                    # truncation - see the module docstring.
                    buffer.add(
                        obs={name: tensor.squeeze(0) for name, tensor in obs_batched.items()},
                        action=action.squeeze(0),
                        log_prob=log_prob.squeeze(0),
                        value=value.squeeze(0),
                        reward=float(reward),
                        done=float(terminated),
                        mask=mask_batched.squeeze(0),
                    )

                    if terminated or truncated:
                        episode = info[EPISODE_INFO_KEY]
                        episode_returns.append(float(episode[EPISODE_RETURN_KEY]))
                        episode_lengths.append(int(episode[EPISODE_LENGTH_KEY]))
                        # StubEnv does not autoreset; advance the stream ourselves.
                        reset_obs, reset_info = self._env.reset()
                        self._obs = reset_obs
                        self._mask = reset_info[MASK_INFO_KEY]
                    else:
                        self._obs = next_obs
                        self._mask = info[MASK_INFO_KEY]
                loop_elapsed = time.perf_counter() - loop_start

                # Tail bootstrap: value of the current cursor state (the successor
                # if the last step did not end, else the post-reset obs). On a
                # terminated final step done=1 masks this bootstrap in GAE anyway.
                last_value = self._actor_critic.get_value(
                    observation_to_batched_tensors(self._obs, self._device)
                )
                buffer.compute_advantages(
                    last_value.squeeze(0), gamma=self._gamma, gae_lambda=self._gae_lambda
                )
        finally:
            self._actor_critic.train(was_training)

        n_episodes = len(episode_returns)
        mean_episode_return: float | None = None
        mean_episode_length: float | None = None
        if n_episodes > 0:
            mean_episode_return = sum(episode_returns) / n_episodes
            mean_episode_length = sum(episode_lengths) / n_episodes
        # Guard a zero elapsed (perf_counter tie on a tiny loop) rather than divide.
        steps_per_second = n_steps / loop_elapsed if loop_elapsed > 0 else float("inf")

        return CollectStats(
            n_steps=n_steps,
            n_episodes=n_episodes,
            mean_episode_return=mean_episode_return,
            mean_episode_length=mean_episode_length,
            steps_per_second=steps_per_second,
        )

    def _batched_mask(self, mask: np.ndarray) -> Tensor:
        """Validate the env's action mask and batch it to ``(1, ACTION_DIM)`` bool.

        Delegates to :func:`~sts_rl.interface.assert_valid_mask` at the env->agent
        boundary, so a malformed mask (wrong shape/dtype, or all-illegal) raises
        :class:`~sts_rl.interface.InterfaceError` rather than a bare ``assert`` that
        vanishes under ``python -O``. A numpy bool array then becomes a
        ``torch.bool`` tensor without an implicit cast that could mask a
        wrong-dtype bug.
        """
        assert_valid_mask(mask)
        return torch.as_tensor(mask, device=self._device).unsqueeze(0)
