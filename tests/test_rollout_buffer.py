"""Tests for the single-environment PPO rollout buffer.

Engine-free: per-step observations are sliced out of one batched
``sample_observation_batch(T)`` call (dropping the batch dim so each step is a
dict of ``(*field_shape)`` tensors), and actions/log-probs/values/rewards/dones
are random. The GAE test cross-checks the buffer against a direct
:func:`compute_gae` call to prove the wiring, and the minibatch tests pin down
coverage, ordering, and shapes.
"""

from __future__ import annotations

import pytest
import torch

from sts_rl.agent.ppo import compute_gae
from sts_rl.agent.rollout_buffer import RolloutBuffer
from sts_rl.interface import ACTION_DIM
from conftest import sample_observation_batch

T = 10
MINIBATCH = 4
# T=10, mb=4 is deliberately indivisible so the final partial batch is exercised.
EXPECTED_SIZES = [4, 4, 2]
SAMPLE_FIELD = "player_scalars"


def _fill_buffer(buffer: RolloutBuffer, length: int = T) -> dict[str, torch.Tensor]:
    """Add ``length`` transitions with deterministic actions and random rest.

    Actions are ``arange(length)`` so a covering set is trivially identifiable.
    Returns the batched obs used, for shape assertions on the sample field.
    """
    torch.manual_seed(0)
    full = sample_observation_batch(length)
    mask = torch.ones(ACTION_DIM, dtype=torch.bool)
    for t in range(length):
        obs = {key: value[t] for key, value in full.items()}
        buffer.add(
            obs=obs,
            action=torch.tensor(t),
            log_prob=torch.randn(()),
            value=torch.randn(()),
            reward=float(torch.randn(())),
            done=1.0 if t == length - 1 else 0.0,
            mask=mask,
        )
    return full


def test_len_equals_number_of_adds():
    buffer = RolloutBuffer()
    _fill_buffer(buffer)
    assert len(buffer) == T


def test_compute_advantages_matches_direct_gae():
    """Buffer GAE equals a direct compute_gae on the same stacked inputs."""
    buffer = RolloutBuffer()
    _fill_buffer(buffer)
    last_value = torch.randn(())

    rewards = torch.tensor(buffer._rewards, dtype=torch.float32)
    dones = torch.tensor(buffer._dones, dtype=torch.float32)
    values = torch.stack(buffer._values).to(torch.float32)
    expected_adv, expected_ret = compute_gae(rewards, values, dones, last_value)

    buffer.compute_advantages(last_value)
    assert buffer.advantages is not None and buffer.returns is not None
    assert torch.allclose(buffer.advantages, expected_adv)
    assert torch.allclose(buffer.returns, expected_ret)


def test_iter_minibatches_covers_every_index_once():
    """Shuffled minibatches partition the rollout: sizes [4,4,2], each index once."""
    buffer = RolloutBuffer()
    _fill_buffer(buffer)
    buffer.compute_advantages(torch.randn(()))

    batches = list(buffer.iter_minibatches(MINIBATCH, shuffle=True))
    assert [mb.actions.shape[0] for mb in batches] == EXPECTED_SIZES

    all_actions = torch.cat([mb.actions for mb in batches])
    assert sorted(all_actions.tolist()) == list(range(T))

    field_shape = sample_observation_batch(1)[SAMPLE_FIELD].shape[1:]
    for mb in batches:
        n = mb.actions.shape[0]
        assert mb.obs[SAMPLE_FIELD].shape == (n, *field_shape)
        assert mb.masks.shape == (n, ACTION_DIM)
        for vec in (mb.actions, mb.old_log_probs, mb.old_values, mb.advantages, mb.returns):
            assert vec.shape == (n,)


def test_iter_minibatches_no_shuffle_preserves_order():
    """shuffle=False yields transitions in stored order."""
    buffer = RolloutBuffer()
    _fill_buffer(buffer)
    buffer.compute_advantages(torch.randn(()))

    first = next(iter(buffer.iter_minibatches(MINIBATCH, shuffle=False)))
    assert first.actions.tolist() == list(range(MINIBATCH))


def test_iter_minibatches_before_compute_raises():
    """iter_minibatches must fail loudly if advantages were never computed."""
    buffer = RolloutBuffer()
    _fill_buffer(buffer)
    try:
        next(iter(buffer.iter_minibatches(MINIBATCH)))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError before compute_advantages")


def test_stored_tensors_are_detached():
    """add stores detached copies so the buffer never holds the autograd graph."""
    buffer = RolloutBuffer()
    full = sample_observation_batch(1)
    obs = {key: value[0] for key, value in full.items()}
    log_prob = torch.randn((), requires_grad=True)
    value = torch.randn((), requires_grad=True)
    buffer.add(
        obs=obs,
        action=torch.tensor(0),
        log_prob=log_prob,
        value=value,
        reward=0.0,
        done=1.0,
        mask=torch.ones(ACTION_DIM, dtype=torch.bool),
    )
    assert buffer._log_probs[0].requires_grad is False
    assert buffer._values[0].requires_grad is False


def test_compute_advantages_detaches_last_value():
    """A grad-enabled last_value must not leave advantages/returns on the graph.

    They are constant targets/weights in the PPO update; retaining the value
    head's graph here would risk a double-backward and a memory leak.
    """
    buffer = RolloutBuffer()
    _fill_buffer(buffer)
    last_value = torch.tensor(0.5, requires_grad=True)
    buffer.compute_advantages(last_value)
    assert buffer.advantages is not None and buffer.returns is not None
    assert not buffer.advantages.requires_grad
    assert not buffer.returns.requires_grad


def test_iter_minibatches_rejects_nonpositive_size():
    """minibatch_size <= 0 raises rather than silently yielding a zero-update epoch."""
    buffer = RolloutBuffer()
    _fill_buffer(buffer)
    buffer.compute_advantages(torch.zeros(()))
    for bad in (0, -1):
        with pytest.raises(ValueError):
            list(buffer.iter_minibatches(bad))


def test_compute_advantages_on_empty_buffer_raises():
    """Empty buffer gives a clear error, not an opaque torch.stack failure."""
    buffer = RolloutBuffer()
    with pytest.raises(RuntimeError):
        buffer.compute_advantages(torch.zeros(()))


def test_stored_tensors_do_not_alias_caller_memory():
    """Mutating a passed obs/mask in place after add must not reach the buffer.

    add stores .detach().clone() copies, so the buffer owns its data.
    """
    buffer = RolloutBuffer()
    full = sample_observation_batch(1)
    obs = {key: value[0] for key, value in full.items()}
    original_field = obs[SAMPLE_FIELD].clone()
    mask = torch.ones(ACTION_DIM, dtype=torch.bool)
    buffer.add(
        obs=obs,
        action=torch.tensor(0),
        log_prob=torch.zeros(()),
        value=torch.zeros(()),
        reward=0.0,
        done=1.0,
        mask=mask,
    )
    # Mutate the caller's tensors in place AFTER add; the buffer must not see it.
    obs[SAMPLE_FIELD].add_(99.0)
    mask[0] = False
    buffer.compute_advantages(torch.zeros(()))
    batch = next(iter(buffer.iter_minibatches(1, shuffle=False)))
    assert torch.equal(batch.obs[SAMPLE_FIELD][0], original_field)
    assert bool(batch.masks[0, 0])  # still legal despite the caller's in-place edit
