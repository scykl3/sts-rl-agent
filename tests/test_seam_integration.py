"""Seam test: the agent driving the LIVE C++ engine end to end.

The integration point where the merged agent stack (network, rollout collector,
PPO update) first drives the real :class:`~sts_rl.env.adapter.StsEnv` rather than
the engine-free stub. It checks three seams: the interface-version handshake, a
collect that spans at least one live episode boundary (terminal -> reset) into a
GAE -> PPO-update pass producing finite losses on real observations, and that the
masked policy never emits an action the live engine rejects. The exhaustive mask
fuzz lives in the env suite; this is the light live wiring check.

Each test uses a tiny trunk and short rollouts, with the per-episode step cap set
below the rollout length so at least one episode ends (truncates) inside every
collect - exercising the live terminal/reset/GAE-done path while keeping the seam
check quick.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest
import torch

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - only without a native build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.ppo_update import PPOConfig, ppo_update
from sts_rl.agent.rollout_buffer import RolloutBuffer
from sts_rl.agent.rollout_collector import RolloutCollector, observation_to_batched_tensors
from sts_rl.env.adapter import StsEnv
from sts_rl.interface import ACTION_DIM, INTERFACE_VERSION

SEED = 0
SMALL_HIDDEN = 32  # tiny trunk; width is irrelevant to the seam wiring
N_STEPS = 48  # short rollout to bound the real-engine collect
MINIBATCH = 16  # < N_STEPS so an epoch runs multiple minibatch updates
N_EPOCHS = 2
LEARNING_RATE = 3e-4
MAX_EPISODE_STEPS = 20  # < N_STEPS/ROLLOUT_STEPS, so an episode truncates inside every collect
ROLLOUT_STEPS = 40  # manual mask-wiring probe length


def test_interface_version_handshake() -> None:
    """StsEnv advertises the interface version as an attribute and in reset info."""
    env = StsEnv()
    try:
        assert env.interface_version == INTERFACE_VERSION
        _obs, info = env.reset(seed=SEED)
        assert info["interface_version"] == INTERFACE_VERSION
        assert "action_mask" in info
    finally:
        env.close()


def test_seam_collect_update_end_to_end() -> None:
    """Collect a rollout spanning >=1 live episode, run one PPO update, get finite losses."""
    torch.manual_seed(SEED)  # weight init + action sampling + minibatch shuffle
    ac = ActorCritic(hidden_dim=SMALL_HIDDEN)
    env = StsEnv(max_episode_steps=MAX_EPISODE_STEPS)
    collector = RolloutCollector(env, ac, seed=SEED)
    buffer = RolloutBuffer()

    collect_stats = collector.collect(buffer, N_STEPS)

    # MAX_EPISODE_STEPS < N_STEPS, so at least one episode ends inside the collect:
    # this exercises the live terminal -> info["episode"] -> reset -> GAE-done path
    # on the real engine, which a collect that fits inside one episode would skip.
    assert collect_stats.n_episodes >= 1

    # collect() fills exactly n_steps transitions and runs GAE itself, so the
    # buffer is update-ready: advantages are computed and cover every transition.
    assert len(buffer) == N_STEPS
    assert buffer.advantages is not None
    minibatches = list(buffer.iter_minibatches(MINIBATCH, shuffle=False))
    assert minibatches, "expected at least one minibatch from a filled buffer"
    assert sum(mb.advantages.numel() for mb in minibatches) == N_STEPS
    assert all(torch.isfinite(mb.advantages).all().item() for mb in minibatches)

    optimizer = torch.optim.Adam(ac.parameters(), lr=LEARNING_RATE)
    stats = ppo_update(
        ac, buffer, optimizer, PPOConfig(minibatch_size=MINIBATCH, n_epochs=N_EPOCHS)
    )

    # Every scalar diagnostic must be finite on the real engine end to end; a
    # NaN/inf here means shapes or reward flow broke across the seam.
    for name, value in asdict(stats).items():
        assert math.isfinite(value), f"PPOStats.{name} is not finite: {value}"
    assert stats.entropy >= 0.0
    assert 0.0 <= stats.clip_fraction <= 1.0
    assert stats.n_updates > 0

    env.close()


def test_no_illegal_actions_on_live_engine() -> None:
    """The masked policy never emits an action the live engine rejects."""
    torch.manual_seed(SEED)
    ac = ActorCritic(hidden_dim=SMALL_HIDDEN)
    device = next(ac.parameters()).device
    env = StsEnv(max_episode_steps=MAX_EPISODE_STEPS)

    obs, info = env.reset(seed=SEED)
    try:
        with torch.no_grad():
            for _ in range(ROLLOUT_STEPS):
                mask = info["action_mask"]
                assert mask.shape == (ACTION_DIM,)
                # Reuse the collector's obs->tensor conversion so the seam matches
                # the real rollout path exactly (dtypes/device/batch dim).
                obs_batched = observation_to_batched_tensors(obs, device)
                mask_batched = torch.as_tensor(mask, device=device).unsqueeze(0)
                action, _log_prob, _entropy, _value = ac.act(obs_batched, mask_batched)
                obs, _reward, terminated, truncated, info = env.step(int(action.item()))
                # invalid_action is always present (interface INFO_KEYS_ALWAYS), so
                # index it directly: a missing key must fail loudly, not silently
                # pass. A correctly masked policy keeps it False every step.
                assert not info[
                    "invalid_action"
                ], f"masked policy emitted an illegal action: {int(action.item())}"
                if terminated or truncated:
                    obs, info = env.reset()
    finally:
        env.close()
