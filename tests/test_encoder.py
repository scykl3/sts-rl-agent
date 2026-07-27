"""Tests for the shared observation encoder.

Observations are generated from the interface's own space
(:func:`sts_rl.env.spaces.build_observation_space`) so the tests exercise real
interface-shaped data and the feature width is locked to the interface constants
rather than a hardcoded literal.
"""

from __future__ import annotations

import torch

from sts_rl import interface
from sts_rl.agent.encoder import (
    CARD_EMBED_DIM,
    ENEMY_EMBED_DIM,
    MOVE_EMBED_DIM,
    POTION_EMBED_DIM,
    ObsFeatureEncoder,
    _N_PILES,
    _PILE_POOLS,
)
from conftest import ID_FIELDS, sample_observation_batch

BATCH = 2


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
        + MOVE_EMBED_DIM
        + 1  # enemy_intent_hidden
        + interface.N_MONSTER_POWER_IDS
        + 1  # enemy_alive
    )
    piles = _N_PILES * (_PILE_POOLS * CARD_EMBED_DIM)
    potion = interface.POTION_SLOTS * POTION_EMBED_DIM + interface.POTION_SLOTS
    passthrough = (
        interface.N_RELIC_IDS
        + interface.N_PLAYER_POWER_IDS
        + interface.PLAYER_SCALAR_DIM
        + interface.N_SCREENS
        + interface.MAP_CONTEXT_DIM
    )
    reward = interface.MAX_REWARD_CARD_SLOTS * CARD_EMBED_DIM
    return hand + enemy + piles + potion + passthrough + reward


def test_forward_output_shape():
    enc = ObsFeatureEncoder()
    out = enc(sample_observation_batch(BATCH))
    assert out.shape == (BATCH, enc.output_dim)


def test_feature_dim_matches_interface():
    enc = ObsFeatureEncoder()
    expected = _expected_feature_dim()
    assert enc.feature_dim == expected
    # And the actual concat produced at runtime has that width.
    feats = enc.encode_features(sample_observation_batch(BATCH))
    assert feats.shape == (BATCH, expected)


def test_output_is_finite():
    enc = ObsFeatureEncoder()
    out = enc(sample_observation_batch(BATCH))
    assert torch.isfinite(out).all()


def test_accepts_float64_observations():
    """Float64 obs (the common numpy default) must not break the trunk Linear.

    Regression: without internal float32 coercion the concat promotes to double
    and ``nn.Linear`` raises "mat1 and mat2 must have the same dtype". Ids stay
    integer; every float field is recast to double to mimic
    ``torch.as_tensor(numpy_float64_array)``.
    """
    enc = ObsFeatureEncoder()
    obs64 = {
        name: (t if name in ID_FIELDS else t.double())
        for name, t in sample_observation_batch(BATCH).items()
    }
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


def test_reward_block_is_zero_when_all_pad():
    """All-PAD reward_card_ids -> the trailing reward block is exactly zero.

    Relies on card_embed's padding_idx row (no masking added): empty reward slots
    contribute a zero vector, so the appended reward columns vanish. Slicing the
    LAST reward-width columns also locks the append-at-end placement.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["reward_card_ids"] = torch.full_like(obs["reward_card_ids"], interface.PAD_ID)
    feats = enc.encode_features(obs)
    reward_width = interface.MAX_REWARD_CARD_SLOTS * CARD_EMBED_DIM
    assert torch.equal(feats[:, -reward_width:], torch.zeros(BATCH, reward_width))


def test_gradient_flows_to_embeddings():
    enc = ObsFeatureEncoder()
    out = enc(sample_observation_batch(BATCH))
    out.sum().backward()
    assert enc.card_embed.weight.grad is not None
    assert enc.enemy_embed.weight.grad is not None
    assert enc.potion_embed.weight.grad is not None


def test_gradient_flows_through_reward_path():
    """A real card in reward_card_ids (all other card fields PAD) reaches card_embed.

    Isolates the reward path: every other card-id field is PAD (padding_idx row 0
    receives no gradient), so a nonzero card_embed gradient can only come from the
    reward block, proving encode_features wires reward_card_ids into the shared
    table. The seed keeps the sparse-input ReLU liveness deterministic.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    for name in ("hand_ids", "draw_ids", "discard_ids", "exhaust_ids", "reward_card_ids"):
        obs[name] = torch.full_like(obs[name], interface.PAD_ID)
    real_id = interface.PAD_ID + 1  # any non-PAD card id
    obs["reward_card_ids"][:, 0] = real_id
    enc(obs).sum().backward()
    grad = enc.card_embed.weight.grad
    assert grad is not None
    # Only the reward slot's id can carry gradient; the PAD row must stay zero.
    assert torch.count_nonzero(grad[real_id]) > 0
    assert torch.equal(grad[interface.PAD_ID], torch.zeros(CARD_EMBED_DIM))


def test_deterministic_in_eval_mode():
    enc = ObsFeatureEncoder()
    enc.eval()
    obs = sample_observation_batch(BATCH)
    with torch.no_grad():
        first = enc(obs)
        second = enc(obs)
    assert torch.equal(first, second)
