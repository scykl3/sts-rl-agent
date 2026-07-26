"""Shared pytest helpers importable by the test modules.

pytest inserts the ``tests/`` directory onto ``sys.path`` (it has no
``__init__.py``), so test modules import these with ``from conftest import ...``.
Both the encoder and policy-head suites sample the interface's own observation
space, so the sampler lives here to avoid drift between the two copies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from sts_rl import interface
from sts_rl.env import spaces
from sts_rl.env.stub_env import StubEnv

# Field names whose dtype is an id (embedding index) -> long tensors.
ID_FIELDS = {f.name for f in interface.OBS_FIELDS if f.bounds == "id"}

# A multi-action block (count 18 > 1) so the learnable task is non-degenerate (a
# legal mask every step, something to learn), matching the rollout-collector suite.
STUB_ACTIVE_BLOCK = "REWARD_SELECT"


def sample_observation_batch(batch: int) -> dict[str, torch.Tensor]:
    """Stack ``batch`` interface-space samples into batched torch tensors."""
    space = spaces.build_observation_space()
    space.seed(0)
    samples = [space.sample() for _ in range(batch)]
    obs: dict[str, torch.Tensor] = {}
    for field in interface.OBS_FIELDS:
        stacked = np.stack([s[field.name] for s in samples], axis=0)
        if field.name in ID_FIELDS:
            obs[field.name] = torch.as_tensor(stacked, dtype=torch.long)
        else:
            obs[field.name] = torch.as_tensor(stacked, dtype=torch.float32)
    return obs


class StubVecEnv:
    """In-process synchronous vec env mirroring SubprocVecEnv's observable contract.

    An engine-free stand-in so the vectorized collector is testable without
    spawning worker processes or building the C++ engine. It batches ``num_envs``
    :class:`StubEnv` instances and auto-resets any that finish using the SAME
    same-step idiom ``SubprocVecEnv`` uses: on a step where an env ends, its
    terminal obs/info are preserved under ``final_observation``/``final_info`` and
    the returned row is the next episode's first observation, while the batched
    mask is taken from ``legal_actions()`` AFTER the reset. It is NOT a subprocess
    (no pickling), so it is deterministic and fast for unit tests, and it
    structurally satisfies ``VecEnvProtocol``.
    """

    def __init__(self, make_env: Callable[[int], StubEnv], num_envs: int) -> None:
        self.num_envs = num_envs
        self._envs = [make_env(index) for index in range(num_envs)]
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space

    def reset(self, seeds: Sequence[int] | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
        seed_list = [None] * self.num_envs if seeds is None else list(seeds)
        obs_list = []
        masks = []
        for env, seed in zip(self._envs, seed_list):
            obs, _info = env.reset(seed=seed)
            obs_list.append(obs)
            masks.append(env.legal_actions())
        return self._stack(obs_list), np.stack(masks)

    def step(self, actions: np.ndarray) -> tuple:
        actions = np.asarray(actions)
        obs_list = []
        rewards = []
        terminated = []
        truncated = []
        masks = []
        infos = []
        for env, action in zip(self._envs, actions):
            obs, reward, term, trunc, info = env.step(int(action))
            if term or trunc:
                # Same-step auto-reset: stash the terminal obs/info, return the
                # next episode's first obs, and take the mask post-reset.
                final_obs, final_info = obs, info
                obs, reset_info = env.reset()
                info = dict(reset_info)
                info["final_observation"] = final_obs
                info["final_info"] = final_info
            obs_list.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            masks.append(env.legal_actions())
            infos.append(info)
        return (
            self._stack(obs_list),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
            np.stack(masks),
            infos,
        )

    @staticmethod
    def _stack(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {key: np.stack([obs[key] for obs in obs_list]) for key in obs_list[0]}


def make_stub_env(**overrides: object) -> StubEnv:
    """Build one non-degenerate learnable :class:`StubEnv` (REWARD_SELECT).

    Overrides (e.g. ``terminate_prob``, ``max_episode_steps``) replace the
    defaults, matching the single-env rollout-collector test factory.
    """
    kwargs: dict[str, object] = {
        "reward_mode": "learnable",
        "active_blocks": (STUB_ACTIVE_BLOCK,),
    }
    kwargs.update(overrides)
    return StubEnv(**kwargs)


def make_stub_vec_env(num_envs: int, **overrides: object) -> StubVecEnv:
    """Build a :class:`StubVecEnv` of ``num_envs`` learnable StubEnvs (same overrides)."""
    return StubVecEnv(lambda index: make_stub_env(**overrides), num_envs)
