"""Tests for the shared observation encoder.

Observations are generated from the interface's own space
(:func:`sts_rl.env.spaces.build_observation_space`) so the tests exercise real
interface-shaped data and the feature width is locked to the interface constants
rather than a hardcoded literal.
"""

from __future__ import annotations

import numpy as np
import torch

from sts_rl import interface
from sts_rl.agent.encoder import (
    CARD_EMBED_DIM,
    ENEMY_EMBED_DIM,
    POTION_EMBED_DIM,
    ObsFeatureEncoder,
    _N_PILES,
    _PILE_POOLS,
)
from sts_rl.env import spaces

BATCH = 2

# Field names whose dtype is an id (embedding index) -> long tensors.
_ID_FIELDS = {f.name for f in interface.OBS_FIELDS if f.bounds == "id"}


def _expected_feature_dim() -> int:
    """Recompute the concat width straight from the interface constants.

    Mirrors the encoder's own derivation; asserting equality locks the feature
    width to the interface constants (if the interface sizes change, both this
    test and the encoder must move together).
    """
    hand = interface.HAND_MAX * (CARD_EMBED_DIM + interface.HAND_FEAT_DIM)
    enemy = interface.MAX_ENEMIES * (
        ENEMY_EMBED_DIM
        + interface.ENEMY_SCALAR_DIM
        + interface.N_INTENT
        + interface.N_POWER_IDS
        + 1
    )
    piles = _N_PILES * (_PILE_POOLS * CARD_EMBED_DIM)
    potion = interface.POTION_SLOTS * POTION_EMBED_DIM + interface.POTION_SLOTS
    passthrough = (
        interface.N_RELIC_IDS
        + interface.N_POWER_IDS
        + interface.PLAYER_SCALAR_DIM
        + interface.N_SCREENS
        + interface.MAP_CONTEXT_DIM
    )
    return hand + enemy + piles + potion + passthrough


def _sample_batch(batch: int = BATCH) -> dict[str, torch.Tensor]:
    """Stack ``batch`` interface-space samples into batched torch tensors."""
    space = spaces.build_observation_space()
    space.seed(0)
    samples = [space.sample() for _ in range(batch)]
    obs: dict[str, torch.Tensor] = {}
    for field in interface.OBS_FIELDS:
        stacked = np.stack([s[field.name] for s in samples], axis=0)
        if field.name in _ID_FIELDS:
            obs[field.name] = torch.as_tensor(stacked, dtype=torch.long)
        else:
            obs[field.name] = torch.as_tensor(stacked, dtype=torch.float32)
    return obs


def test_forward_output_shape():
    enc = ObsFeatureEncoder()
    out = enc(_sample_batch())
    assert out.shape == (BATCH, enc.output_dim)


def test_feature_dim_matches_interface():
    enc = ObsFeatureEncoder()
    expected = _expected_feature_dim()
    assert enc.feature_dim == expected
    # And the actual concat produced at runtime has that width.
    feats = enc.encode_features(_sample_batch())
    assert feats.shape == (BATCH, expected)


def test_output_is_finite():
    enc = ObsFeatureEncoder()
    out = enc(_sample_batch())
    assert torch.isfinite(out).all()


def test_accepts_float64_observations():
    """Float64 obs (the common numpy default) must not break the trunk Linear.

    Regression: without internal float32 coercion the concat promotes to double
    and ``nn.Linear`` raises "mat1 and mat2 must have the same dtype". Ids stay
    integer; every float field is recast to double to mimic
    ``torch.as_tensor(numpy_float64_array)``.
    """
    enc = ObsFeatureEncoder()
    obs64 = {name: (t if name in _ID_FIELDS else t.double()) for name, t in _sample_batch().items()}
    out = enc(obs64)
    assert out.shape == (BATCH, enc.output_dim)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_padding_rows_are_zero():
    enc = ObsFeatureEncoder()
    assert torch.equal(
        enc.card_embed.weight[interface.PAD_ID],
        torch.zeros(CARD_EMBED_DIM),
    )
    assert torch.equal(
        enc.enemy_embed.weight[interface.PAD_ID],
        torch.zeros(ENEMY_EMBED_DIM),
    )
    assert torch.equal(
        enc.potion_embed.weight[interface.PAD_ID],
        torch.zeros(POTION_EMBED_DIM),
    )


def test_all_pad_id_fields_contribute_zero_embedding():
    """All-PAD id slots must map to zero embeddings (padding_idx row 0)."""
    enc = ObsFeatureEncoder()
    zeros = torch.zeros(1, interface.PILE_MAX, dtype=torch.long)
    pooled = enc._pool_pile(zeros)  # mean+max over all-PAD pile
    assert torch.equal(pooled, torch.zeros(1, 2 * CARD_EMBED_DIM))


def test_gradient_flows_to_embeddings():
    enc = ObsFeatureEncoder()
    out = enc(_sample_batch())
    out.sum().backward()
    assert enc.card_embed.weight.grad is not None
    assert enc.enemy_embed.weight.grad is not None
    assert enc.potion_embed.weight.grad is not None


def test_deterministic_in_eval_mode():
    enc = ObsFeatureEncoder()
    enc.eval()
    obs = _sample_batch()
    with torch.no_grad():
        first = enc(obs)
        second = enc(obs)
    assert torch.equal(first, second)
