"""Determinism and range guarantees for the seeding utilities."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from sts_rl.utils import seeding

_SEED = 12345
_MAX_SEED = 2**32


@pytest.fixture(autouse=True)
def _restore_torch_determinism():
    """Restore global torch determinism state after each test.

    ``seed_everything(torch_deterministic=True)`` flips process-global switches
    (``use_deterministic_algorithms``, cuDNN flags); leaving them on would force
    every subsequent test in the session onto deterministic kernels and make an
    op with no deterministic impl raise. Snapshot and restore so these tests do
    not leak state into the rest of the suite.
    """
    prev_algos = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_algos)
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.backends.cudnn.benchmark = prev_cudnn_bench


def _draw() -> tuple[float, float, float]:
    """One sample from each process RNG the seeder controls."""
    return (random.random(), float(np.random.random()), float(torch.rand(1).item()))


def test_torch_deterministic_flag_is_set() -> None:
    seeding.seed_everything(_SEED, torch_deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_torch_deterministic_flag_can_be_disabled() -> None:
    torch.use_deterministic_algorithms(False)
    seeding.seed_everything(_SEED, torch_deterministic=False)
    assert torch.are_deterministic_algorithms_enabled() is False


def test_seed_everything_is_reproducible() -> None:
    seeding.seed_everything(_SEED)
    first = _draw()
    seeding.seed_everything(_SEED)
    second = _draw()
    assert first == second


def test_seed_everything_returns_seed() -> None:
    assert seeding.seed_everything(_SEED) == _SEED


def test_different_seeds_diverge() -> None:
    seeding.seed_everything(_SEED)
    first = _draw()
    seeding.seed_everything(_SEED + 1)
    second = _draw()
    assert first != second


@pytest.mark.parametrize("bad_seed", [-1, _MAX_SEED, _MAX_SEED + 1])
def test_seed_everything_rejects_out_of_range(bad_seed: int) -> None:
    with pytest.raises(ValueError):
        seeding.seed_everything(bad_seed)


def test_derive_worker_seeds_is_deterministic() -> None:
    assert seeding.derive_worker_seeds(_SEED, 8) == seeding.derive_worker_seeds(_SEED, 8)


def test_derive_worker_seeds_length_and_range() -> None:
    seeds = seeding.derive_worker_seeds(_SEED, 16)
    assert len(seeds) == 16
    assert all(0 <= s < _MAX_SEED for s in seeds)


def test_derive_worker_seeds_are_distinct() -> None:
    seeds = seeding.derive_worker_seeds(_SEED, 64)
    assert len(set(seeds)) == len(seeds)


def test_derive_worker_seeds_prefix_stability() -> None:
    # A larger request extends the same stream, so the smaller list is a prefix.
    # This lets the worker count grow without reshuffling existing workers.
    assert seeding.derive_worker_seeds(_SEED, 4) == seeding.derive_worker_seeds(_SEED, 8)[:4]


def test_derive_worker_seeds_varies_with_base() -> None:
    assert seeding.derive_worker_seeds(_SEED, 8) != seeding.derive_worker_seeds(_SEED + 1, 8)


def test_derive_worker_seeds_zero_is_empty() -> None:
    assert seeding.derive_worker_seeds(_SEED, 0) == []


def test_derive_worker_seeds_rejects_negative() -> None:
    with pytest.raises(ValueError):
        seeding.derive_worker_seeds(_SEED, -1)


def test_capture_restore_round_trip() -> None:
    seeding.seed_everything(_SEED)
    state = seeding.capture_rng_state()
    after_capture = _draw()
    # Advancing the RNGs then restoring must replay the same next draw.
    _draw()
    seeding.restore_rng_state(state)
    assert _draw() == after_capture
