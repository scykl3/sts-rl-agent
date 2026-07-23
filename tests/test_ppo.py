"""Analytic tests for the pure PPO objective math.

Engine-free: every case uses hand-built ``torch`` tensors, no encoder or
observation space, so the GAE recursion and the two loss functions are checked
against arithmetic that is written out explicitly in the test.
"""

from __future__ import annotations

import pytest
import torch

from sts_rl.agent.ppo import (
    DEFAULT_CLIP_COEF,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_GAMMA,
    HALF_MSE_FACTOR,
    clipped_policy_loss,
    clipped_value_loss,
    compute_gae,
)


def test_gae_returns_equal_advantages_plus_values():
    """returns == advantages + values on a random trajectory (definition)."""
    gen = torch.Generator().manual_seed(0)
    length = 8
    rewards = torch.rand(length, generator=gen)
    values = torch.rand(length, generator=gen)
    dones = (torch.rand(length, generator=gen) > 0.7).float()
    last_value = torch.rand((), generator=gen)
    advantages, returns = compute_gae(rewards, values, dones, last_value)
    assert torch.allclose(returns, advantages + values)


def test_gae_all_zero_gives_zero_advantages():
    """Zero rewards/values/dones/bootstrap -> every delta and advantage is 0."""
    length = 5
    zeros = torch.zeros(length)
    advantages, returns = compute_gae(zeros, zeros, zeros, torch.zeros(()))
    assert torch.allclose(advantages, torch.zeros(length))
    assert torch.allclose(returns, torch.zeros(length))


def test_gae_single_step_no_done():
    """T=1: adv = r + gamma*last_value - v; return = adv + v (hand numbers)."""
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    dones = torch.tensor([0.0])
    last_value = torch.tensor(2.0)
    advantages, returns = compute_gae(
        rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95
    )
    # delta = 1.0 + 0.99*2.0 - 0.5 = 2.48; adv = delta (no future term).
    expected_adv = 1.0 + 0.99 * 2.0 - 0.5
    assert advantages.item() == pytest.approx(expected_adv)
    assert returns.item() == pytest.approx(expected_adv + 0.5)


def test_gae_hand_computed_multi_step():
    """T=3 trajectory with every intermediate value worked out by hand.

    gamma=0.99, lambda=0.95, dones all 0, last_value=2.0
    rewards = [1.0, 2.0, 3.0]; values = [0.5, 1.0, 1.5]
    gl = gamma*lambda = 0.99*0.95 = 0.9405

    delta_2 = 3.0 + 0.99*2.0 - 1.5 = 3.48
    delta_1 = 2.0 + 0.99*1.5 - 1.0 = 2.485
    delta_0 = 1.0 + 0.99*1.0 - 0.5 = 1.49

    adv_2 = delta_2                       = 3.48
    adv_1 = delta_1 + gl*adv_2 = 2.485 + 0.9405*3.48   = 5.75794
    adv_0 = delta_0 + gl*adv_1 = 1.49  + 0.9405*5.75794 = 6.90534257
    """
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 1.5])
    dones = torch.zeros(3)
    last_value = torch.tensor(2.0)
    advantages, returns = compute_gae(
        rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95
    )
    expected_adv = torch.tensor([6.90534257, 5.75794, 3.48])
    assert torch.allclose(advantages, expected_adv, atol=1e-5)
    assert torch.allclose(returns, expected_adv + values, atol=1e-5)


def test_gae_done_in_middle_truncates_bootstrap():
    """dones[t]=1 zeroes both the bootstrap and the lambda carry at step t."""
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 1.5])
    dones = torch.tensor([0.0, 1.0, 0.0])  # step 1 terminal
    last_value = torch.tensor(2.0)
    advantages, _ = compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95)
    # At the terminal step next_nonterminal=0: no gamma*next_value term and no
    # lambda carry from adv_2, so adv_1 collapses to reward - value.
    assert advantages[1].item() == pytest.approx(2.0 - 1.0)


def test_gae_last_step_terminal_drops_bootstrap():
    """dones[-1]=1 zeroes the tail bootstrap even with a nonzero last_value."""
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    dones = torch.tensor([0.0, 1.0])  # final step terminal
    last_value = torch.tensor(9.0)  # deliberately large; must NOT be used
    advantages, _ = compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95)
    # Terminal tail: next_nonterminal=0, so adv_1 = reward - value, last_value ignored.
    assert advantages[1].item() == pytest.approx(2.0 - 1.0)


def test_gae_output_shapes():
    """Both outputs preserve the trajectory length (T,)."""
    length = 6
    rewards = torch.zeros(length)
    advantages, returns = compute_gae(
        rewards, torch.zeros(length), torch.zeros(length), torch.zeros(())
    )
    assert advantages.shape == (length,)
    assert returns.shape == (length,)


def test_gae_accepts_one_dim_last_value():
    """last_value shaped (1,) behaves identically to a 0-dim scalar."""
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 0.25])
    dones = torch.zeros(2)
    scalar = compute_gae(rewards, values, dones, torch.tensor(1.5))[0]
    one_dim = compute_gae(rewards, values, dones, torch.tensor([1.5]))[0]
    assert torch.allclose(scalar, one_dim)


def test_policy_loss_ratio_one_equals_neg_mean_advantage():
    """new_log_prob == old_log_prob -> ratio 1 -> loss = -mean(advantages)."""
    gen = torch.Generator().manual_seed(1)
    log_prob = torch.randn(4, generator=gen)
    advantages = torch.randn(4, generator=gen)
    loss = clipped_policy_loss(log_prob, log_prob.clone(), advantages)
    assert loss.item() == pytest.approx(-advantages.mean().item())


def test_policy_loss_positive_advantage_clips_upper():
    """Positive advantage with ratio >> 1+clip -> clipped surrogate -(1+clip)*adv."""
    advantages = torch.tensor([2.0])
    # new - old = 10 -> ratio = exp(10), well above 1 + clip.
    new_log_prob = torch.tensor([10.0])
    old_log_prob = torch.tensor([0.0])
    loss = clipped_policy_loss(new_log_prob, old_log_prob, advantages)
    assert loss.item() == pytest.approx(-(1.0 + DEFAULT_CLIP_COEF) * 2.0)


def test_policy_loss_negative_advantage_clips_lower():
    """Negative advantage with ratio << 1-clip -> clipped surrogate -(1-clip)*adv."""
    advantages = torch.tensor([-2.0])
    # new - old = -10 -> ratio = exp(-10), well below 1 - clip.
    new_log_prob = torch.tensor([-10.0])
    old_log_prob = torch.tensor([0.0])
    loss = clipped_policy_loss(new_log_prob, old_log_prob, advantages)
    assert loss.item() == pytest.approx(-(1.0 - DEFAULT_CLIP_COEF) * -2.0)


def test_value_loss_unclipped_is_half_mse():
    """Default branch equals HALF_MSE_FACTOR * MSE(new_values, returns)."""
    gen = torch.Generator().manual_seed(2)
    new_values = torch.randn(5, generator=gen)
    returns = torch.randn(5, generator=gen)
    loss = clipped_value_loss(new_values, returns)
    expected = HALF_MSE_FACTOR * ((new_values - returns) ** 2).mean()
    assert loss.item() == pytest.approx(expected.item())


def test_value_loss_clipped_never_below_unclipped():
    """The elementwise-max clipped branch is >= the unclipped loss."""
    gen = torch.Generator().manual_seed(3)
    old_values = torch.randn(6, generator=gen)
    returns = torch.randn(6, generator=gen)
    # Move new_values far from old so the clip is active on every element.
    new_values = old_values + torch.ones(6)
    unclipped = clipped_value_loss(new_values, returns)
    clipped = clipped_value_loss(new_values, returns, old_values, DEFAULT_CLIP_COEF)
    assert clipped.item() >= unclipped.item()


def test_value_loss_only_old_values_raises():
    """Supplying old_values without clip_coef is a misconfiguration."""
    values = torch.zeros(3)
    with pytest.raises(ValueError):
        clipped_value_loss(values, values, old_values=values)


def test_value_loss_only_clip_coef_raises():
    """Supplying clip_coef without old_values is a misconfiguration."""
    values = torch.zeros(3)
    with pytest.raises(ValueError):
        clipped_value_loss(values, values, clip_coef=DEFAULT_CLIP_COEF)


def test_losses_are_differentiable():
    """Both losses backprop finite grads to their learnable inputs.

    The training loop optimizes these, so each must be differentiable w.r.t. the
    tensor the network produces (new_log_prob for the policy, new_values for the
    critic); old_* and advantages/returns are treated as constants.
    """
    advantages = torch.randn(4)
    new_log_prob = torch.randn(4, requires_grad=True)
    old_log_prob = torch.randn(4)
    p_loss = clipped_policy_loss(new_log_prob, old_log_prob, advantages)
    p_loss.backward()
    assert new_log_prob.grad is not None and torch.isfinite(new_log_prob.grad).all()

    returns = torch.randn(4)
    new_values = torch.randn(4, requires_grad=True)
    v_loss = clipped_value_loss(new_values, returns)
    v_loss.backward()
    assert new_values.grad is not None and torch.isfinite(new_values.grad).all()


def test_module_default_constants_are_sane():
    """Defaults land in the usual PPO ranges (guards accidental edits)."""
    assert 0.0 < DEFAULT_GAMMA <= 1.0
    assert 0.0 < DEFAULT_GAE_LAMBDA <= 1.0
    assert 0.0 < DEFAULT_CLIP_COEF < 1.0
