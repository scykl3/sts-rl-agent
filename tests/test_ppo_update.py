"""Tests for the PPO update pass.

Engine-free: a helper fills a :class:`RolloutBuffer` from one batched
``sample_observation_batch(T)`` call (per-step obs are the batch slices), a legal
prefix mask, actions sampled by a real :meth:`ActorCritic.act`, and its log-prob
/value squeezed to scalars, then runs GAE. The suite pins the minibatch-step
count, proves parameters move, checks stat sanity, both normalization paths, the
empty-buffer zero-division guard, and run-to-run determinism.
"""

from __future__ import annotations

import math

import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.ppo_update import PPOConfig, PPOStats, ppo_update
from sts_rl.agent.rollout_buffer import RolloutBuffer
from sts_rl.interface import ACTION_DIM
from conftest import sample_observation_batch

T = 16
MINIBATCH = 8
N_EPOCHS = 2
# Legal prefix width: mask[:K] True guarantees the sampled action lands in [0, K).
K = 5
EXPECTED_UPDATES = N_EPOCHS * math.ceil(T / MINIBATCH)  # 2 * 2 = 4


def _fill_buffer(actor_critic: ActorCritic, length: int = T) -> RolloutBuffer:
    """Add ``length`` transitions with real act() log-probs/values (as scalars)."""
    buffer = RolloutBuffer()
    full = sample_observation_batch(length)
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    mask[:K] = True
    for t in range(length):
        obs = {key: value[t] for key, value in full.items()}
        # act expects a batch dim; squeeze its (1,) outputs back to scalars.
        batched_obs = {key: value.unsqueeze(0) for key, value in obs.items()}
        action, log_prob, _, value = actor_critic.act(batched_obs, mask.unsqueeze(0))
        buffer.add(
            obs=obs,
            action=action.squeeze(0),
            log_prob=log_prob.squeeze(0),
            value=value.squeeze(0),
            reward=float(torch.randn(())),
            done=1.0 if t == length - 1 else 0.0,
            mask=mask,
        )
    buffer.compute_advantages(torch.zeros(()))
    return buffer


def test_n_updates_matches_epochs_times_minibatches():
    """n_updates == n_epochs * ceil(T / minibatch_size); all stats finite."""
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)

    stats = ppo_update(ac, buffer, optimizer, config)
    assert isinstance(stats, PPOStats)
    assert stats.n_updates == EXPECTED_UPDATES
    for field in (
        stats.policy_loss,
        stats.value_loss,
        stats.entropy,
        stats.total_loss,
        stats.approx_kl,
        stats.clip_fraction,
        stats.grad_norm,
    ):
        assert math.isfinite(field)


def test_parameters_change_after_update():
    """A real optimizer step must move at least one encoder and/or head weight."""
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    encoder_before = ac.encoder.card_embed.weight.detach().clone()
    head_before = ac.value.value.weight.detach().clone()

    ppo_update(ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH))

    encoder_changed = not torch.equal(encoder_before, ac.encoder.card_embed.weight)
    head_changed = not torch.equal(head_before, ac.value.value.weight)
    assert encoder_changed or head_changed


def test_stats_are_sane():
    """entropy >= 0 and clip_fraction in [0, 1]; losses finite."""
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    stats = ppo_update(
        ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)
    )
    assert stats.entropy >= 0.0
    assert 0.0 <= stats.clip_fraction <= 1.0
    assert stats.approx_kl >= 0.0  # Schulman's k3 estimator is non-negative
    assert stats.grad_norm >= 0.0
    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.value_loss)


def test_both_normalization_paths_run():
    """normalize_advantages True and False both return finite stats."""
    for normalize in (True, False):
        ac = ActorCritic()
        buffer = _fill_buffer(ac)
        optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
        config = PPOConfig(
            n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, normalize_advantages=normalize
        )
        stats = ppo_update(ac, buffer, optimizer, config)
        assert stats.n_updates == EXPECTED_UPDATES
        assert math.isfinite(stats.total_loss)


def test_empty_buffer_returns_zeroed_stats():
    """A buffer yielding zero minibatches returns zeroed stats, not a ZeroDivisionError.

    compute_advantages rejects an empty buffer, so we reproduce the smallest
    zero-minibatch case directly: an empty buffer whose advantages/returns are
    set (mirroring iter_minibatches' own length==0 early return, which yields
    nothing). ppo_update must then take the n_updates == 0 guard.
    """
    ac = ActorCritic()
    buffer = RolloutBuffer()
    buffer.advantages = torch.zeros(0)
    buffer.returns = torch.zeros(0)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    stats = ppo_update(
        ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)
    )
    assert stats.n_updates == 0
    assert stats.policy_loss == 0.0
    assert stats.value_loss == 0.0
    assert stats.total_loss == 0.0


def test_update_is_deterministic_under_fixed_seed():
    """Two identically seeded runs produce equal stats (model, buffer, and shuffle)."""

    def run() -> PPOStats:
        torch.manual_seed(1234)
        ac = ActorCritic()
        buffer = _fill_buffer(ac)
        optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
        return ppo_update(
            ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)
        )

    first = run()
    second = run()
    assert first == second


def test_singleton_tail_minibatch_does_not_nan():
    """A size-1 remainder minibatch must not NaN advantage normalization.

    With T % minibatch_size == 1 the final minibatch holds one transition;
    torch.std applies Bessel's correction (N-1) and returns NaN for N=1, which
    without a guard would poison the update. length=9, mb=4 -> sizes [4,4,1].
    """
    ac = ActorCritic()
    buffer = _fill_buffer(ac, length=9)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=1, minibatch_size=4, normalize_advantages=True)
    stats = ppo_update(ac, buffer, optimizer, config)
    assert stats.n_updates == 3  # [4, 4, 1]
    assert math.isfinite(stats.total_loss)
    assert math.isfinite(stats.policy_loss)
    # The update must leave the WHOLE network finite, not NaN-poisoned.
    assert all(torch.isfinite(p).all() for p in ac.parameters())
