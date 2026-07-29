"""Persistent exponentially-weighted running mean and variance.

A small utility for advantage normalization in PPO. Standardizing advantages by
a single batch's std is noisy for a small or heavy-tailed batch, so the
normalization scale jumps between updates. Tracking the mean and second moment
``E[x^2]`` with an exponentially weighted moving average (EWMA) across updates
keeps that scale stable while still adapting as the advantage distribution
drifts over training.

The decay is applied PER ITEM so the amount of smoothing is invariant to batch
size: folding a batch of ``n`` values retains ``(1 - decay) ** n`` of the prior
estimate. A larger ``decay`` trusts the current batch more; ``decay == 1.0``
retains nothing (``retain == 0``) and recovers pure per-batch statistics - the
running mean and second moment equal the batch's own population mean/second
moment. Population moments (no Bessel correction) are used throughout, so even a
single value can never yield a NaN variance.
"""

from __future__ import annotations

import math

from torch import Tensor


class RunningMoments:
    """EWMA estimate of the mean and std of a stream of values.

    On each :meth:`update` with a batch of ``n`` values, ``retain = (1 - decay)
    ** n`` and both moments blend as ``m <- retain * m + (1 - retain) *
    batch_m``. The first non-empty update seeds the estimate directly from the
    batch, so there is no cold-start bias toward zero while the EWMA warms up.
    ``decay == 1.0`` gives ``retain == 0`` and recovers pure per-batch
    statistics.

    ``mean`` and ``second_moment`` are stored as Python floats (device-agnostic:
    batch statistics are computed on the values' own device, then read out as
    floats). ``std`` is derived as ``sqrt(max(second_moment - mean ** 2, 0.0))``,
    clamped at zero so floating-point drift on an all-equal batch never produces
    a NaN.
    """

    def __init__(self, decay: float, eps: float = 1e-8) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.decay = decay
        self.eps = eps
        self.mean: float = 0.0
        self.second_moment: float = 0.0  # running EWMA of E[x^2]
        self.initialized = False

    @property
    def std(self) -> float:
        """Population std from the running moments; clamped at 0 so it is never NaN."""
        # Deriving variance as E[x^2] - mean^2 is more cancellation-prone than the
        # Welford / parallel-variance idiom (e.g. SB3's RunningMeanStd), but the max
        # clamp keeps it NaN-safe and it is fine for advantage magnitudes.
        return math.sqrt(max(self.second_moment - self.mean**2, 0.0))

    def update(self, values: Tensor) -> None:
        """Fold ``values`` into the running moments with per-item decay.

        An empty batch (``numel() == 0``) is a documented no-op that leaves the
        estimate unchanged. Batch statistics are computed on ``values``' device
        and read out as Python floats, so the tracker stays device-agnostic. The
        exponent is ``values.numel()`` so smoothing is invariant to batch size,
        and the first non-empty update seeds the estimate directly.
        """
        n = values.numel()
        if n == 0:  # no-op: nothing to fold, leave stats (and initialized) unchanged
            return
        batch_mean = float(values.mean())
        batch_second = float((values * values).mean())
        if not self.initialized:
            # Seed from the first batch so the estimate carries no cold-start bias
            # toward zero while the EWMA warms up.
            self.mean = batch_mean
            self.second_moment = batch_second
            self.initialized = True
            return
        # Per-item decay: a batch of n items retains (1 - decay) ** n of the prior
        # estimate, so smoothing is invariant to batch size. decay == 1.0 gives
        # retain == 0 and recovers the batch's own population moments.
        retain = (1.0 - self.decay) ** n
        self.mean = retain * self.mean + (1.0 - retain) * batch_mean
        self.second_moment = retain * self.second_moment + (1.0 - retain) * batch_second

    def normalize(self, values: Tensor) -> Tensor:
        """Standardize ``values`` as ``(values - mean) / (std + eps)``.

        The ``eps`` floor keeps an all-equal or still-cold stream (``std == 0``)
        finite instead of dividing by zero. The result stays on ``values``'
        device and dtype.
        """
        return (values - self.mean) / (self.std + self.eps)
