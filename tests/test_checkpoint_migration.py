"""Tests for transformer-ActorCritic checkpoint loading.

The transformer rewrite is a new checkpoint format: :func:`load_checkpoint`
builds an :class:`ActorCritic` at the recorded width and strict-loads a
new-format payload, and refuses a pre-transformer flat-trunk checkpoint with a
clear typed error. The old name-aware policy-head remap and trunk-input-width
widen no longer apply (the flat concat is gone) and have been removed, so these
tests cover only the load path and its guards.

Engine-free: checkpoints are built from a fresh ``ActorCritic`` state_dict.
"""

from __future__ import annotations

import pytest
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.aux_heads import AUX_TARGETS
from sts_rl.agent.checkpoint_migration import (
    AUX_HEAD_WEIGHT_KEY,
    LEGACY_TRUNK_INPUT_WEIGHT_KEY,
    load_checkpoint,
)
from sts_rl.agent.train import (
    CHECKPOINT_HIDDEN_DIM_KEY,
    CHECKPOINT_INTERFACE_VERSION_KEY,
    CHECKPOINT_MODEL_KEY,
)
from sts_rl.interface import ACTION_DIM, INTERFACE_VERSION, InterfaceError

# A small d_model keeps the built network tiny; it must be divisible by the
# attention head count (the encoder guards this), and 32 is.
HIDDEN = 32


def _write_checkpoint(path, state_dict: dict[str, torch.Tensor], version: str, hidden: int) -> None:
    """Write a minimal _save_checkpoint-shaped payload to ``path``."""
    payload = {
        CHECKPOINT_MODEL_KEY: state_dict,
        CHECKPOINT_HIDDEN_DIM_KEY: hidden,
        CHECKPOINT_INTERFACE_VERSION_KEY: version,
    }
    torch.save(payload, path)


def test_round_trip_loads_weights_byte_identical(tmp_path) -> None:
    """A current checkpoint loads into an equal ActorCritic (every tensor identical)."""
    reference = ActorCritic(hidden_dim=HIDDEN)
    path = tmp_path / "current.pt"
    _write_checkpoint(path, reference.state_dict(), INTERFACE_VERSION, HIDDEN)

    model = load_checkpoint(path)
    assert isinstance(model, ActorCritic)
    ref_state = reference.state_dict()
    got_state = model.state_dict()
    assert set(got_state) == set(ref_state)  # no keys added or dropped
    for key in ref_state:
        assert torch.equal(got_state[key], ref_state[key]), f"{key} changed on load"


def test_rebuilds_at_recorded_hidden_dim(tmp_path) -> None:
    """The model is rebuilt at the checkpoint's recorded width (d_model)."""
    reference = ActorCritic(hidden_dim=HIDDEN)
    path = tmp_path / "c.pt"
    _write_checkpoint(path, reference.state_dict(), INTERFACE_VERSION, HIDDEN)

    model = load_checkpoint(path)
    assert model.encoder.output_dim == HIDDEN
    assert model.policy.action_dim == ACTION_DIM
    assert model.policy.input_dim == HIDDEN


def test_future_version_new_format_loads_verbatim(tmp_path) -> None:
    """The recorded interface_version does not gate loading a new-format checkpoint.

    Architecture match (a clean strict load), not the version string, is what
    decides loadability, so a version numerically newer than the current
    interface still loads when its tensors match.
    """
    reference = ActorCritic(hidden_dim=HIDDEN)
    path = tmp_path / "future.pt"
    _write_checkpoint(path, reference.state_dict(), "99.0.0", HIDDEN)

    model = load_checkpoint(path)
    assert torch.equal(model.policy.learned_logit.weight, reference.policy.learned_logit.weight)
    assert torch.equal(model.encoder.card_embed.weight, reference.encoder.card_embed.weight)


def test_aux_enabled_checkpoint_round_trips(tmp_path) -> None:
    """An aux-enabled checkpoint rebuilds WITH the aux head and loads byte-identical.

    A net trained with ``aux_coef > 0`` carries ``aux_head.*`` in its state_dict.
    The loader must detect :data:`AUX_HEAD_WEIGHT_KEY` and rebuild with the aux head;
    a headless ActorCritic would reject the extra keys under the strict load. This is
    the revert guard: without the aux-aware rebuild the load raises ``InterfaceError``
    instead of round-tripping, so an aux checkpoint could not be warm-started or
    evaluated from disk.
    """
    reference = ActorCritic(hidden_dim=HIDDEN, aux_targets=AUX_TARGETS)
    assert reference.aux_head is not None
    assert AUX_HEAD_WEIGHT_KEY in reference.state_dict()
    path = tmp_path / "aux.pt"
    _write_checkpoint(path, reference.state_dict(), INTERFACE_VERSION, HIDDEN)

    model = load_checkpoint(path)
    assert model.aux_head is not None
    assert model.aux_head.out_features == len(AUX_TARGETS)
    ref_state = reference.state_dict()
    got_state = model.state_dict()
    assert set(got_state) == set(ref_state)  # aux keys present, none dropped or added
    for key in ref_state:
        assert torch.equal(got_state[key], ref_state[key]), f"{key} changed on load"


def test_aux_free_checkpoint_loads_headless(tmp_path) -> None:
    """An aux-free checkpoint still builds the default headless net (no aux head)."""
    reference = ActorCritic(hidden_dim=HIDDEN)
    assert reference.aux_head is None
    path = tmp_path / "plain.pt"
    _write_checkpoint(path, reference.state_dict(), INTERFACE_VERSION, HIDDEN)

    model = load_checkpoint(path)
    assert model.aux_head is None
    assert not any("aux" in key for key in model.state_dict())


def test_missing_required_key_raises(tmp_path) -> None:
    """A payload missing a required key is rejected with a clear error."""
    path = tmp_path / "bad.pt"
    torch.save({CHECKPOINT_MODEL_KEY: {}}, path)  # missing hidden_dim + version
    with pytest.raises(InterfaceError, match="missing required key"):
        load_checkpoint(path)


def test_legacy_flat_trunk_checkpoint_rejected(tmp_path) -> None:
    """A pre-transformer (flat-trunk) checkpoint is refused with a clear typed error.

    The legacy first-trunk key cannot map onto the transformer architecture, and
    INTERFACE_VERSION did not bump across the rewrite, so the key (not the
    version) is the discriminator. This is the revert guard: the load must fail
    loudly rather than silently mismapping weights.
    """
    state = dict(ActorCritic(hidden_dim=HIDDEN).state_dict())
    # Inject the legacy flat-trunk weight to simulate an old-format checkpoint;
    # the load must reject on the KEY, before any shape is inspected.
    state[LEGACY_TRUNK_INPUT_WEIGHT_KEY] = torch.zeros(HIDDEN, 8)
    path = tmp_path / "legacy.pt"
    _write_checkpoint(path, state, INTERFACE_VERSION, HIDDEN)

    with pytest.raises(InterfaceError, match="pre-transformer flat-trunk"):
        load_checkpoint(path)


def test_incompatible_new_format_state_dict_raises(tmp_path) -> None:
    """A new-format payload whose tensors don't match the architecture fails typed.

    Dropping a required tensor makes the strict load fail; the loader surfaces it
    as a typed InterfaceError rather than a bare RuntimeError, so a partial or
    misshaped checkpoint is a hard, legible error.
    """
    state = dict(ActorCritic(hidden_dim=HIDDEN).state_dict())
    del state["value.value.weight"]  # a required tensor -> strict load fails
    path = tmp_path / "incompatible.pt"
    _write_checkpoint(path, state, INTERFACE_VERSION, HIDDEN)

    with pytest.raises(InterfaceError, match="strict load failed"):
        load_checkpoint(path)


def test_shape_mismatch_state_dict_raises(tmp_path) -> None:
    """A tensor of the wrong shape (width mismatch) also fails the strict load, typed."""
    state = dict(ActorCritic(hidden_dim=HIDDEN).state_dict())
    # A pointer readout sized for a different d_model than the recorded HIDDEN.
    state["policy.learned_logit.weight"] = torch.zeros(1, HIDDEN * 2)
    path = tmp_path / "mismatch.pt"
    _write_checkpoint(path, state, INTERFACE_VERSION, HIDDEN)

    with pytest.raises(InterfaceError, match="strict load failed"):
        load_checkpoint(path)
