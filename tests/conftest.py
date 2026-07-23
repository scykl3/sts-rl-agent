"""Shared pytest helpers importable by the test modules.

pytest inserts the ``tests/`` directory onto ``sys.path`` (it has no
``__init__.py``), so test modules import these with ``from conftest import ...``.
Both the encoder and policy-head suites sample the interface's own observation
space, so the sampler lives here to avoid drift between the two copies.
"""

from __future__ import annotations

import numpy as np
import torch

from sts_rl import interface
from sts_rl.env import spaces

# Field names whose dtype is an id (embedding index) -> long tensors.
ID_FIELDS = {f.name for f in interface.OBS_FIELDS if f.bounds == "id"}


def sample_observation_batch(batch: int) -> dict[str, torch.Tensor]:
    """Stack ``batch`` interface-space samples into batched torch tensors."""
    space = spaces.build_observation_space()
    space.seed(0)
    samples = [space.sample() for _ in range(batch)]
    obs: dict[str, torch.Tensor] = {}
    for field in interface.OBS_FIELDS:
        stacked = np.stack([s[field.name] for s in samples], axis=0)
        if field.name in ID_FIELDS:
            obs[field.name] = torch.as_tensor(stacked, dtype=torch.long)
        else:
            obs[field.name] = torch.as_tensor(stacked, dtype=torch.float32)
    return obs
