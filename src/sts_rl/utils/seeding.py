"""Global and per-worker seeding for reproducible runs.

Reproducibility has two layers here. :func:`seed_everything` pins the process
RNGs (Python ``random``, NumPy, and torch, CPU and CUDA) and, optionally, forces
torch onto deterministic kernels. :func:`derive_worker_seeds` turns one base
seed into a fixed list of independent per-worker seeds so a vectorized env
(which takes an explicit seed per worker at ``reset``) is reproducible from a
single knob.

The per-worker derivation uses NumPy's :class:`~numpy.random.SeedSequence`,
which is designed for exactly this: spawning many statistically-independent
streams from one entropy source. The same ``(base_seed, num_workers)`` always
yields the same seed list, and different workers get well-separated seeds rather
than the correlated ``base + i`` sequence a naive offset would produce.

Two reproducibility levers live outside this module and must be set by the
launch environment, not at runtime:

- ``PYTHONHASHSEED`` - fixes string hash randomization (set/dict iteration
  order). Assigning ``os.environ`` after interpreter start is a no-op, so it
  must be exported before the process launches.
- ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` - required for deterministic CUDA matmul;
  without it the deterministic-algorithms path raises when a matmul runs.

Seeding also assumes callers use NumPy's global ``np.random.*`` functions; a
``np.random.default_rng()`` ``Generator`` has its own state and is unaffected.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

# torch is the one heavy dependency; import lazily inside the functions that
# need it so seed derivation (pure NumPy) stays importable in environments that
# only pull in the env side.

# SeedSequence.generate_state emits uint32 words; one word per worker gives a
# seed in [0, 2**32) - the range engine/env seeders and NumPy both accept.
_SEED_BITS = 32
_MAX_SEED = 2**_SEED_BITS


def seed_everything(seed: int, *, torch_deterministic: bool = True) -> int:
    """Seed all process RNGs and return the seed for logging.

    Seeds Python ``random``, NumPy's legacy global RNG, and torch (CPU and all
    CUDA devices). When ``torch_deterministic`` is set, also forces torch onto
    deterministic algorithms (``use_deterministic_algorithms(True)``) and
    deterministic cuDNN kernels with the autotuner off, so repeated runs on the
    same hardware match, at some throughput cost. Full CUDA matmul determinism
    additionally needs ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the launch
    environment (see module docstring).

    Returning ``seed`` lets a caller both seed and record it in one expression.
    """
    if not 0 <= seed < _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}), got {seed}")

    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    # Seeds every visible CUDA device; a no-op when CUDA is unavailable.
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # cudnn flags alone only cover convolutions; use_deterministic_algorithms
        # also pins matmul/scatter and errors on ops with no deterministic impl,
        # so a silent nondeterministic fallback cannot slip through unnoticed.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def derive_worker_seeds(base_seed: int, num_workers: int) -> list[int]:
    """Derive ``num_workers`` independent seeds from ``base_seed``.

    Deterministic in ``(base_seed, num_workers)`` and suitable to hand straight
    to a vectorized env's per-worker ``reset(seeds=...)``. Each seed is a uint32
    drawn from a :class:`~numpy.random.SeedSequence`, so the streams are
    well-separated rather than the correlated ``base_seed + i`` an offset gives.
    """
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if num_workers == 0:
        return []

    sequence = np.random.SeedSequence(base_seed)
    words = sequence.generate_state(num_workers, dtype=np.uint32)
    return [int(word) for word in words]


def capture_rng_state() -> dict[str, Any]:
    """Snapshot the Python/NumPy/torch RNG states for later replay.

    Pairs with :func:`restore_rng_state` to resume a run at the exact RNG
    position it was checkpointed at, so a replay continues the same stream
    rather than merely re-seeding from scratch.
    """
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by :func:`capture_rng_state`."""
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
