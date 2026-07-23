"""Tests for the masked policy head.

Driven WITHOUT the engine: features come from the shared encoder run on samples
of the interface's own observation space, and masks are synthesized to respect
the interface's ``assert_valid_mask`` invariant (at least one legal action per
row). Dimensions are recomputed from the interface/encoder constants, never
hardcoded, so an interface enum bump moves the tests and the code together.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sts_rl import interface
from sts_rl.agent.encoder import HIDDEN_DIM, ObsFeatureEncoder
from sts_rl.agent.policy_head import MASKED_LOGIT, MaskedPolicyHead
from sts_rl.interface import InterfaceError
from conftest import sample_observation_batch

BATCH = 3


def _features(batch: int = BATCH) -> torch.Tensor:
    """Trunk features for a sampled batch (detached from the encoder graph)."""
    enc = ObsFeatureEncoder()
    return enc(sample_observation_batch(batch)).detach()


def _random_valid_mask(batch: int, seed: int = 0) -> torch.Tensor:
    """Random bool mask (B, ACTION_DIM) with >=1 legal action per row.

    Mirrors ``assert_valid_mask``: each row is guaranteed at least one True by
    forcing a random legal index on, so no row is fully masked.
    """
    gen = torch.Generator().manual_seed(seed)
    mask = torch.rand(batch, interface.ACTION_DIM, generator=gen) > 0.5
    forced = torch.randint(0, interface.ACTION_DIM, (batch,), generator=gen)
    mask[torch.arange(batch), forced] = True
    # Sanity-check every row against the interface invariant.
    for row in mask:
        interface.assert_valid_mask(row.numpy())
    return mask


def _single_legal_mask(batch: int, index: int) -> torch.Tensor:
    """Mask with exactly one legal action (``index``) for every row."""
    mask = torch.zeros(batch, interface.ACTION_DIM, dtype=torch.bool)
    mask[:, index] = True
    return mask


def test_forward_shape_and_finite():
    """Output is (B, ACTION_DIM) and finite despite the large-negative floor."""
    head = MaskedPolicyHead()
    logits = head(_features(), _random_valid_mask(BATCH))
    assert logits.shape == (BATCH, interface.ACTION_DIM)
    assert torch.isfinite(logits).all()


def test_input_dim_defaults_to_encoder_output():
    """Head's default input width tracks the encoder trunk, not a literal."""
    enc = ObsFeatureEncoder()
    head = MaskedPolicyHead()
    assert head.input_dim == enc.output_dim == HIDDEN_DIM
    assert head.action_dim == interface.ACTION_DIM


def test_illegal_actions_get_zero_probability():
    """softmax mass on illegal actions ~0; legal probs sum to ~1."""
    head = MaskedPolicyHead()
    mask = _random_valid_mask(BATCH)
    probs = torch.softmax(head(_features(), mask), dim=-1)
    illegal = probs[~mask]
    legal_sum = (probs * mask).sum(dim=-1)
    assert (illegal < 1e-6).all()
    assert torch.allclose(legal_sum, torch.ones(BATCH), atol=1e-5)


def test_sampled_actions_are_always_legal():
    """Every sampled action lands on a legal index (large batch of draws)."""
    head = MaskedPolicyHead()
    big = 256
    mask = _random_valid_mask(big, seed=1)
    feats = _features(big)
    for _ in range(20):
        actions, _, _ = head.act(feats, mask)
        assert mask[torch.arange(big), actions].all()


def test_gradient_flows_to_logit_head():
    """A log-prob loss backprops finite grads into the Linear weight."""
    head = MaskedPolicyHead()
    mask = _random_valid_mask(BATCH)
    _, log_prob, _ = head.act(_features(), mask)
    loss = -log_prob.mean()
    loss.backward()
    grad = head.logits.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_evaluate_actions_matches_distribution():
    """evaluate_actions reproduces the masked distribution's log_prob/entropy."""
    head = MaskedPolicyHead()
    feats = _features()
    mask = _random_valid_mask(BATCH)
    dist = head.masked_distribution(feats, mask)
    actions = dist.sample()
    log_prob, entropy = head.evaluate_actions(feats, mask, actions)
    assert torch.allclose(log_prob, dist.log_prob(actions))
    assert torch.allclose(entropy, dist.entropy())


def test_deterministic_act_is_argmax_and_reproducible():
    """deterministic=True returns argmax over legal logits, reproducibly."""
    head = MaskedPolicyHead()
    feats = _features()
    mask = _random_valid_mask(BATCH)
    first, _, _ = head.act(feats, mask, deterministic=True)
    second, _, _ = head.act(feats, mask, deterministic=True)
    expected = head(feats, mask).argmax(dim=-1)
    assert torch.equal(first, expected)
    assert torch.equal(first, second)
    # The greedy pick must itself be legal.
    assert mask[torch.arange(BATCH), first].all()


def test_entropy_finite_and_nonnegative():
    """Entropy is finite and >= 0 for a valid mask."""
    head = MaskedPolicyHead()
    _, _, entropy = head.act(_features(), _random_valid_mask(BATCH))
    assert torch.isfinite(entropy).all()
    assert (entropy >= 0).all()


def test_single_legal_action_forces_choice_and_zero_entropy():
    """One legal action: entropy ~0 and both sampling modes pick that action."""
    head = MaskedPolicyHead()
    feats = _features()
    index = interface.ACTION_DIM // 2
    mask = _single_legal_mask(BATCH, index)
    sampled, log_prob, entropy = head.act(feats, mask)
    greedy, _, _ = head.act(feats, mask, deterministic=True)
    assert (sampled == index).all()
    assert (greedy == index).all()
    assert torch.allclose(entropy, torch.zeros(BATCH), atol=1e-5)
    # Certain action -> log_prob ~ log(1) = 0.
    assert torch.allclose(log_prob, torch.zeros(BATCH), atol=1e-5)


def test_masked_logit_is_finite_and_negative():
    """The masking constant is a large finite negative (never -inf)."""
    assert np.isfinite(MASKED_LOGIT)
    assert MASKED_LOGIT < 0


def test_fully_masked_single_row_raises():
    """An all-False mask row must raise, not sample uniformly over illegals."""
    head = MaskedPolicyHead()
    mask = torch.zeros(BATCH, interface.ACTION_DIM, dtype=torch.bool)
    with pytest.raises(InterfaceError):
        head(_features(), mask)


def test_one_fully_masked_row_in_batch_raises():
    """A single fully-masked row inside an otherwise-valid batch still raises."""
    head = MaskedPolicyHead()
    mask = _random_valid_mask(BATCH)
    mask[1] = False  # blank one row's legality entirely
    with pytest.raises(InterfaceError):
        head(_features(), mask)


def test_unbatched_mask_raises_instead_of_broadcasting():
    """A (ACTION_DIM,) mask must raise, not broadcast over (B, ACTION_DIM)."""
    head = MaskedPolicyHead()
    unbatched = _random_valid_mask(1)[0]  # shape (ACTION_DIM,)
    assert unbatched.shape == (interface.ACTION_DIM,)
    with pytest.raises(InterfaceError):
        head(_features(), unbatched)


def test_device_mismatch_raises_domain_error():
    """A mask on a different device raises InterfaceError, not a raw torch error.

    Uses the ``meta`` device (no real allocation, no GPU needed) so the guard
    is exercised on a CPU-only machine; shape matches so the device check, not
    the shape check, is what fires.
    """
    head = MaskedPolicyHead()
    meta_mask = torch.ones(BATCH, interface.ACTION_DIM, dtype=torch.bool, device="meta")
    with pytest.raises(InterfaceError):
        head(_features(), meta_mask)


def test_non_bool_mask_is_coerced():
    """An int/float 0/1 mask is accepted and matches the bool-mask result."""
    head = MaskedPolicyHead()
    feats = _features()
    bool_mask = _random_valid_mask(BATCH)
    int_mask = bool_mask.to(torch.int64)
    float_mask = bool_mask.to(torch.float32)
    expected = head(feats, bool_mask)
    assert torch.equal(head(feats, int_mask), expected)
    assert torch.equal(head(feats, float_mask), expected)
