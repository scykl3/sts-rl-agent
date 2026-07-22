"""Gymnasium space builders derived from the contract registry.

The observation and action spaces are constructed directly from the constants
and field registry in :mod:`sts_rl.contract`, so the spaces can never drift
from the contract.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sts_rl import contract
from sts_rl.contract import ACTION_DIM, OBS_FIELDS, ObsField


def build_action_space() -> gym.spaces.Discrete:
    return spaces.Discrete(ACTION_DIM)


def _build_box(field: ObsField) -> gym.spaces.Box:
    if field.bounds == "unit":
        return spaces.Box(low=0.0, high=1.0, shape=field.shape, dtype=np.float32)
    if field.bounds == "real":
        return spaces.Box(low=-np.inf, high=np.inf, shape=field.shape, dtype=np.float32)
    if field.bounds == "id":
        return spaces.Box(low=0, high=field.id_high, shape=field.shape, dtype=np.int32)
    raise contract.ContractError(
        f"ObsField {field.name!r}: unknown bounds value {field.bounds!r}"
    )


def build_observation_space() -> gym.spaces.Dict:
    return spaces.Dict({field.name: _build_box(field) for field in OBS_FIELDS})


def build_spaces() -> tuple[gym.spaces.Dict, gym.spaces.Discrete]:
    return (build_observation_space(), build_action_space())
