"""Checkpoint loading for the transformer ``ActorCritic``.

The transformer encoder rewrite is a new checkpoint format. The prior
flat-concat encoder stored a first dense trunk ``Linear(feature_dim, hidden)``
and dense ``Linear`` policy / value heads; the transformer's parameter tensors
are entirely different (per-type input projections, pre-LN blocks, type and
slot-id embeddings, a relic embedding table). None of the old tensors map onto
the new architecture, so load-compat with pre-transformer checkpoints is
intentionally dropped, and the name-aware policy-head remap and trunk-input-width
widen the old format needed no longer apply.

``INTERFACE_VERSION`` did NOT bump across this agent-internal rewrite (the
observation and action interface is unchanged), so the recorded
``interface_version`` cannot distinguish an old flat-trunk checkpoint from a new
transformer one. Instead a pre-transformer checkpoint is detected by the
architecture itself - the presence of the old first-trunk weight
:data:`LEGACY_TRUNK_INPUT_WEIGHT_KEY` - and refused with a clear typed error
rather than silently mismapping weights into a differently shaped model.

New-format checkpoints load with a plain ``strict=True`` load, which requires
every current key be present and correctly shaped (a missing or misshaped tensor
is a hard error, not a silent partial load).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.train import (
    CHECKPOINT_HIDDEN_DIM_KEY,
    CHECKPOINT_INTERFACE_VERSION_KEY,
    CHECKPOINT_MODEL_KEY,
)
from sts_rl.interface import InterfaceError

# State-dict key of the pre-transformer flat encoder's first dense ``Linear``
# (``ObsFeatureEncoder.trunk[0]`` in that era). The transformer encoder has no
# such tensor, so its presence in a checkpoint identifies an incompatible
# pre-transformer format. This is the reliable discriminator because
# INTERFACE_VERSION is unchanged across the rewrite (see the module docstring).
LEGACY_TRUNK_INPUT_WEIGHT_KEY: str = "encoder.trunk.0.weight"


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> ActorCritic:
    """Load a training checkpoint into an :class:`ActorCritic`.

    Reads the :func:`~sts_rl.agent.train._save_checkpoint` payload (keys
    :data:`~sts_rl.agent.train.CHECKPOINT_MODEL_KEY`,
    :data:`~sts_rl.agent.train.CHECKPOINT_HIDDEN_DIM_KEY`,
    :data:`~sts_rl.agent.train.CHECKPOINT_INTERFACE_VERSION_KEY`), builds an
    ``ActorCritic`` at the checkpoint's own width (``hidden_dim`` == the
    transformer ``d_model``), and loads the weights ``strict=True``.

    A pre-transformer (flat-trunk) checkpoint is detected by
    :data:`LEGACY_TRUNK_INPUT_WEIGHT_KEY` and rejected with a clear
    :class:`~sts_rl.interface.InterfaceError`: its weights do not map onto the
    transformer architecture, and load-compat was intentionally dropped. A
    new-format checkpoint whose tensors do not match the current architecture
    (a missing or misshaped key) fails the strict load, surfaced as a typed
    ``InterfaceError`` rather than a bare ``RuntimeError``.

    ``map_location`` defaults to ``"cpu"`` so a checkpoint loads regardless of
    the device it was trained on; move the returned model with ``.to(device)``
    afterward if needed. ``weights_only=True`` is passed to :func:`torch.load`:
    the payload holds only tensors and plain scalars, so the safe unpickler
    suffices and an untrusted checkpoint cannot execute arbitrary code on load.
    """
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=True)
    for key in (CHECKPOINT_MODEL_KEY, CHECKPOINT_HIDDEN_DIM_KEY, CHECKPOINT_INTERFACE_VERSION_KEY):
        if key not in payload:
            raise InterfaceError(f"checkpoint at {path} is missing required key {key!r}")

    model_state: Any = payload[CHECKPOINT_MODEL_KEY]
    hidden_dim: Any = payload[CHECKPOINT_HIDDEN_DIM_KEY]

    if LEGACY_TRUNK_INPUT_WEIGHT_KEY in model_state:
        raise InterfaceError(
            f"checkpoint at {path} was trained with the pre-transformer flat-trunk "
            f"encoder (has {LEGACY_TRUNK_INPUT_WEIGHT_KEY!r}); its weights do not map "
            f"onto the transformer architecture. Old-checkpoint load-compat was "
            f"intentionally dropped in the transformer rewrite - retrain from scratch."
        )

    model = ActorCritic(hidden_dim=hidden_dim)
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise InterfaceError(
            f"checkpoint at {path} is incompatible with the current transformer "
            f"ActorCritic architecture (strict load failed): {exc}"
        ) from exc
    return model
