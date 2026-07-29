"""Tests for the state-value head (critic).

Driven WITHOUT the engine: features come from the shared encoder run on samples
of the interface's own observation space. Dimensions are recomputed from the
encoder constants, never hardcoded, so a constant bump moves the tests and the
code together.
"""

from __future__ import annotations

import torch

from sts_rl.agent.encoder import HIDDEN_DIM, ObsFeatureEncoder
from sts_rl.agent.value_head import ValueHead
from conftest import sample_observation_batch

BATCH = 4


def _features(batch: int = BATCH) -> torch.Tensor:
    """Pooled CLS context for a sampled batch (detached from the encoder graph)."""
    enc = ObsFeatureEncoder()
    _per_token, pooled_cls, _mask = enc(sample_observation_batch(batch))
    return pooled_cls.detach()


def test_forward_shape_is_flat_and_finite():
    """Output is exactly (B,), NOT (B, 1), and finite.

    The (B, 1) guard is a regression against the PPO broadcasting footgun:
    a (B, 1) value differenced against a (B,) returns vector expands to (B, B).
    """
    head = ValueHead()
    values = head(_features())
    assert values.shape == (BATCH,)
    assert values.shape != (BATCH, 1)
    assert torch.isfinite(values).all()


def test_singleton_batch_stays_one_dim():
    """A B=1 batch yields shape (1,), not a 0-dim scalar.

    Locks in ``squeeze(-1)`` over a bare ``squeeze()``: the latter would collapse
    the size-1 batch dimension to a scalar, which every larger-batch test would
    still pass. This is the regression that distinguishes the two.
    """
    head = ValueHead()
    values = head(_features(1))
    assert values.shape == (1,)


def test_input_dim_defaults_to_encoder_output():
    """Head's default input width tracks the encoder output width, not a literal."""
    enc = ObsFeatureEncoder()
    head = ValueHead()
    assert head.input_dim == enc.output_dim == HIDDEN_DIM
    assert head.value.in_features == HIDDEN_DIM
    # A state value is a single scalar per row.
    assert head.value.out_features == 1


def test_gradient_flows_to_value_head():
    """A scalar loss backprops finite grads into the Linear weight."""
    head = ValueHead()
    head(_features()).sum().backward()
    grad = head.value.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_forward_is_deterministic():
    """The same features produce identical values (no dropout/randomness)."""
    head = ValueHead()
    feats = _features()
    assert torch.equal(head(feats), head(feats))
