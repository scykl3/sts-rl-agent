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
    reward_relic = interface.N_RELIC_IDS
    reward_potion = interface.MAX_REWARD_POTIONS * POTION_EMBED_DIM
    card_select = interface.CHOICE_MAX * CARD_EMBED_DIM
    return (
        hand
        + enemy
        + piles
        + potion
        + passthrough
        + reward
        + reward_relic
        + reward_potion
        + card_select
    )


def _relic_block_start(enc: ObsFeatureEncoder) -> int:
    """Start column of the reward-relic block in the pre-trunk concat.

    The relic block follows the reward-card block and precedes the reward-potion and
    card-select blocks; derived from the interface widths (no literal), mirroring the
    encoder's own append order.
    """
    card_select_width = interface.CHOICE_MAX * CARD_EMBED_DIM
    potion_width = interface.MAX_REWARD_POTIONS * POTION_EMBED_DIM
    return enc.feature_dim - card_select_width - potion_width - interface.N_RELIC_IDS


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
    """All-PAD reward_card_ids -> the reward-card block is exactly zero.

    Relies on card_embed's padding_idx row (no masking added): empty reward slots
    contribute a zero vector, so the reward-card columns vanish. Slicing the card
    block at its interface-derived offset (now followed by the relic and potion
    blocks) also locks its placement in the concat.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["reward_card_ids"] = torch.full_like(obs["reward_card_ids"], interface.PAD_ID)
    feats = enc.encode_features(obs)
    reward_width = interface.MAX_REWARD_CARD_SLOTS * CARD_EMBED_DIM
    relic_width = interface.N_RELIC_IDS
    potion_width = interface.MAX_REWARD_POTIONS * POTION_EMBED_DIM
    card_select_width = interface.CHOICE_MAX * CARD_EMBED_DIM
    # The card block sits before the appended relic, potion, and card-select blocks.
    card_start = enc.feature_dim - card_select_width - potion_width - relic_width - reward_width
    card_block = feats[:, card_start : card_start + reward_width]
    assert torch.equal(card_block, torch.zeros(BATCH, reward_width))


def test_reward_relic_and_potion_blocks_zero_when_empty():
    """Empty reward_relic_ids / reward_potion_ids -> their appended blocks are zero.

    Empty offer: the relic slots carry the INVALID sentinel (N_RELIC_IDS), which
    scatters into the encoder's dropped final column so no kept relic column is set,
    and PAD potion slots embed to the padding_idx zero vector. Both appended blocks
    vanish. Relics use INVALID (not PAD 0) for "empty" because RelicId 0 (AKABEKO) is
    a real relic whose column must stay usable. Slicing them at their interface-derived
    offsets (now before the final card-select block) locks their placement in the concat.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["reward_relic_ids"] = torch.full_like(obs["reward_relic_ids"], interface.N_RELIC_IDS)
    obs["reward_potion_ids"] = torch.full_like(obs["reward_potion_ids"], interface.PAD_ID)
    feats = enc.encode_features(obs)
    relic_width = interface.N_RELIC_IDS
    potion_width = interface.MAX_REWARD_POTIONS * POTION_EMBED_DIM
    card_select_width = interface.CHOICE_MAX * CARD_EMBED_DIM
    # card_select is the final block; the potion block sits immediately before it,
    # and the relic block immediately before the potion block.
    potion_block = feats[:, -(card_select_width + potion_width) : -card_select_width]
    assert torch.equal(potion_block, torch.zeros(BATCH, potion_width))
    relic_block = feats[
        :,
        -(card_select_width + potion_width + relic_width) : -(card_select_width + potion_width),
    ]
    assert torch.equal(relic_block, torch.zeros(BATCH, relic_width))


def test_card_select_block_is_zero_when_all_pad():
    """All-PAD card_select_ids -> the final card-select block is exactly zero.

    Relies on card_embed's padding_idx row (no masking): empty card-select slots
    contribute a zero vector, so the appended card-select columns vanish. Slicing
    the final block also locks its append-after-potions placement.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["card_select_ids"] = torch.full_like(obs["card_select_ids"], interface.PAD_ID)
    feats = enc.encode_features(obs)
    card_select_width = interface.CHOICE_MAX * CARD_EMBED_DIM
    # card_select is the final block, so it is the trailing slice of the concat.
    assert torch.equal(feats[:, -card_select_width:], torch.zeros(BATCH, card_select_width))


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


def test_gradient_flows_through_reward_potion_path():
    """A real potion in reward_potion_ids (all other potion fields PAD) reaches potion_embed.

    Isolates the reward-potion path: every other potion-id field is PAD
    (padding_idx row 0 receives no gradient), so a nonzero potion_embed gradient
    can only come from the reward-potion block, proving encode_features wires
    reward_potion_ids into the shared table. The seed keeps the sparse-input ReLU
    liveness deterministic.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    for name in ("potion_ids", "reward_potion_ids"):
        obs[name] = torch.full_like(obs[name], interface.PAD_ID)
    real_id = interface.PAD_ID + 1  # any non-PAD potion id
    obs["reward_potion_ids"][:, 0] = real_id
    enc(obs).sum().backward()
    grad = enc.potion_embed.weight.grad
    assert grad is not None
    # Only the reward-potion slot's id can carry gradient; the PAD row stays zero.
    assert torch.count_nonzero(grad[real_id]) > 0
    assert torch.equal(grad[interface.PAD_ID], torch.zeros(POTION_EMBED_DIM))


def test_gradient_flows_through_card_select_path():
    """A real card in card_select_ids (all other card fields PAD) reaches card_embed.

    Isolates the card-select path: every other card-id field is PAD (padding_idx
    row 0 receives no gradient), so a nonzero card_embed gradient can only come
    from the card-select block, proving encode_features wires card_select_ids into
    the shared table. The seed keeps the sparse-input ReLU liveness deterministic.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    for name in (
        "hand_ids",
        "draw_ids",
        "discard_ids",
        "exhaust_ids",
        "reward_card_ids",
        "card_select_ids",
    ):
        obs[name] = torch.full_like(obs[name], interface.PAD_ID)
    real_id = interface.PAD_ID + 1  # any non-PAD card id
    obs["card_select_ids"][:, 0] = real_id
    enc(obs).sum().backward()
    grad = enc.card_embed.weight.grad
    assert grad is not None
    # Only the card-select slot's id can carry gradient; the PAD row must stay zero.
    assert torch.count_nonzero(grad[real_id]) > 0
    assert torch.equal(grad[interface.PAD_ID], torch.zeros(CARD_EMBED_DIM))


def test_gradient_flows_to_trunk_relic_multihot_columns():
    """A real offered relic drives gradient into the first trunk layer's relic columns.

    The relic multihot has no embedding table, so its learning signal shows up as
    gradient on the first trunk Linear's input columns rather than on an embedding
    row. With exactly one offered relic (every other slot the INVALID empty marker),
    only that relic's column in the relic block is a nonzero input, so only its
    trunk-weight column may carry gradient; AKABEKO's column 0 (empties are INVALID,
    not 0) and any unoffered relic column must stay exactly zero. The INVALID column is
    dropped from the block, so an empty slot sets no kept column at all. The seed keeps
    the sparse-input ReLU liveness deterministic.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    # Every slot empty (INVALID sentinel), then offer one real, non-AKABEKO relic.
    obs["reward_relic_ids"] = torch.full_like(obs["reward_relic_ids"], interface.N_RELIC_IDS)
    real_id = 5  # any real relic id with 0 < real_id < N_RELIC_IDS
    obs["reward_relic_ids"][:, 0] = real_id
    enc(obs).sum().backward()

    grad = enc.trunk[0].weight.grad
    assert grad is not None
    relic_start = _relic_block_start(enc)
    # The offered relic's column carries gradient...
    assert torch.count_nonzero(grad[:, relic_start + real_id]) > 0
    # ...while AKABEKO's column 0 (unoffered here; empties are INVALID, not 0) and any
    # other unoffered relic column stay exactly zero -- an empty slot sets no kept column.
    assert torch.count_nonzero(grad[:, relic_start + 0]) == 0  # AKABEKO not offered
    unoffered_id = 2
    assert torch.count_nonzero(grad[:, relic_start + unoffered_id]) == 0


def test_offered_akabeko_sets_its_relic_output_column():
    """Offering AKABEKO (RelicId 0) sets relic-block column 0 - it is a real relic.

    Regression for the empty-sentinel fix: AKABEKO must be distinguishable from an
    empty slot. Under the earlier "clear column PAD_ID(0)" multihot, column 0 was
    forced to zero, silently erasing an offered AKABEKO; the INVALID-as-empty design
    keeps column 0 for AKABEKO and drops the INVALID column instead. FAILS if that
    clear-column-0 behavior is reverted.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    # Empty every slot (INVALID), then offer AKABEKO (id 0) in slot 0.
    obs["reward_relic_ids"] = torch.full_like(obs["reward_relic_ids"], interface.N_RELIC_IDS)
    akabeko = 0  # RelicId.AKABEKO
    obs["reward_relic_ids"][:, 0] = akabeko
    feats = enc.encode_features(obs)
    relic_start = _relic_block_start(enc)
    relic_block = feats[:, relic_start : relic_start + interface.N_RELIC_IDS]
    # AKABEKO's column 0 is set; every other relic column stays zero (the other slots
    # are INVALID empties, which land in the dropped column).
    expected = torch.zeros(BATCH, interface.N_RELIC_IDS)
    expected[:, akabeko] = 1.0
    assert torch.equal(relic_block, expected)


def test_reward_relic_block_zero_when_all_empty():
    """All-INVALID (empty) reward_relic_ids -> the relic block is exactly zero.

    Empty relic slots carry the INVALID sentinel (N_RELIC_IDS), which scatters into the
    encoder's dropped final column, so no kept column is set. The relic counterpart to
    the card / potion all-PAD-zero tests, keyed on INVALID rather than PAD 0 because
    RelicId 0 (AKABEKO) is a real relic.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["reward_relic_ids"] = torch.full_like(obs["reward_relic_ids"], interface.N_RELIC_IDS)
    feats = enc.encode_features(obs)
    relic_start = _relic_block_start(enc)
    relic_block = feats[:, relic_start : relic_start + interface.N_RELIC_IDS]
    assert torch.equal(relic_block, torch.zeros(BATCH, interface.N_RELIC_IDS))


def test_deterministic_in_eval_mode():
    enc = ObsFeatureEncoder()
    enc.eval()
    obs = sample_observation_batch(BATCH)
    with torch.no_grad():
        first = enc(obs)
        second = enc(obs)
    assert torch.equal(first, second)
