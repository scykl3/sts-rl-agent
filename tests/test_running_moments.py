"""Tests for the EWMA running-moments tracker used by advantage normalization.

Engine-free and pure-tensor: each test drives :class:`RunningMoments` directly
with small known batches or a synthetic stationary stream. The suite pins the
EWMA formula against a hand-computed reference, the decay=1.0 per-batch recovery,
convergence on a stationary stream, that :meth:`normalize` standardizes, the
degenerate no-NaN cases (N=1, all-equal, empty no-op), and the decay-range guard.
"""

from __future__ import annotations

import math

import pytest
import torch

from sts_rl.agent.running_moments import RunningMoments


def test_update_matches_ewma_reference():
    """Two updates match a hand-computed EWMA: first seeds, second blends per-item."""
    decay = 0.2
    rm = RunningMoments(decay=decay)
    a = torch.tensor([1.0, 3.0])
    b = torch.tensor([5.0, 7.0, 9.0])
    rm.update(a)
    rm.update(b)
    # Reference computed identically: the first update seeds mean/second directly
    # from batch a, the second blends batch b in with retain = (1 - decay) ** |b|.
    m = float(a.mean())
    v = float((a * a).mean())
    retain = (1.0 - decay) ** b.numel()
    m = retain * m + (1.0 - retain) * float(b.mean())
    v = retain * v + (1.0 - retain) * float((b * b).mean())
    assert rm.mean == pytest.approx(m)
    assert rm.second_moment == pytest.approx(v)
    assert rm.std == pytest.approx(math.sqrt(max(v - m * m, 0.0)))


def test_decay_one_recovers_per_batch():
    """decay=1.0 (retain==0) makes running mean/std equal the latest batch's population."""
    rm = RunningMoments(decay=1.0)
    x = torch.tensor([2.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    rm.update(x)
    assert rm.mean == pytest.approx(float(x.mean()))
    assert rm.std == pytest.approx(float(x.std(unbiased=False)))
    # A second batch fully replaces the first (retain == 0), not blends.
    y = torch.tensor([-3.0, 3.0])
    rm.update(y)
    assert rm.mean == pytest.approx(float(y.mean()))
    assert rm.std == pytest.approx(float(y.std(unbiased=False)))


def test_converges_on_stationary_stream():
    """Repeated updates on a stationary stream track its true mean and std."""
    torch.manual_seed(0)
    true_mean, true_std = 3.0, 2.0
    rm = RunningMoments(decay=0.005)
    for _ in range(4000):
        batch = true_mean + true_std * torch.randn(64)
        rm.update(batch)
    assert rm.mean == pytest.approx(true_mean, abs=0.3)
    assert rm.std == pytest.approx(true_std, abs=0.3)


def test_normalize_standardizes_with_decay_one():
    """normalize against a decay=1.0 fit yields ~0 mean / ~1 population std."""
    torch.manual_seed(0)
    rm = RunningMoments(decay=1.0)
    x = 5.0 + 3.0 * torch.randn(256)
    rm.update(x)
    z = rm.normalize(x)
    assert float(z.mean()) == pytest.approx(0.0, abs=1e-4)
    assert float(z.std(unbiased=False)) == pytest.approx(1.0, abs=1e-3)


def test_single_value_update_is_finite():
    """N=1 uses population variance (0), so std is a finite 0.0 - never NaN."""
    rm = RunningMoments(decay=0.5)
    rm.update(torch.tensor([4.2]))
    assert rm.mean == pytest.approx(4.2)
    assert rm.std == pytest.approx(0.0)
    assert math.isfinite(rm.std)
    z = rm.normalize(torch.tensor([4.2, 4.2]))
    assert torch.isfinite(z).all()


def test_all_equal_batch_is_finite():
    """An all-equal batch drives std -> 0; normalize stays finite via the eps floor."""
    rm = RunningMoments(decay=0.5)
    rm.update(torch.full((5,), 2.0))
    assert rm.std == pytest.approx(0.0)
    assert math.isfinite(rm.std)
    assert torch.isfinite(rm.normalize(torch.full((5,), 2.0))).all()


def test_empty_update_is_noop():
    """An empty batch leaves the estimate (and initialized flag) untouched."""
    rm = RunningMoments(decay=0.5)
    rm.update(torch.empty(0))
    assert rm.mean == 0.0
    assert rm.second_moment == 0.0
    assert not rm.initialized
    # After a real update, a subsequent empty update must not perturb the stats.
    rm.update(torch.tensor([1.0, 2.0, 3.0]))
    mean_before, second_before = rm.mean, rm.second_moment
    rm.update(torch.empty(0))
    assert rm.mean == mean_before
    assert rm.second_moment == second_before


def test_invalid_decay_raises():
    """decay outside (0, 1] raises ValueError; the boundaries 1.0 and small >0 are valid."""
    for bad in (0.0, -0.1, 1.5, 2.0):
        with pytest.raises(ValueError, match="decay"):
            RunningMoments(decay=bad)
    RunningMoments(decay=1.0)
    RunningMoments(decay=1e-6)
