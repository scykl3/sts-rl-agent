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


def _block_layout() -> list[tuple[str, int]]:
    """The pre-trunk concat blocks in append order, ``(name, width)``, from interface
    constants.

    Mirrors ``ObsFeatureEncoder.encode_features``' concat order exactly; the tests
    slice by these spans so each block's placement is locked to the documented layout
    with no magic offsets, and a future appended block only extends this list. If the
    encoder reorders or the interface sizes change, this list and the encoder must move
    together.
    """
    card, potion = CARD_EMBED_DIM, POTION_EMBED_DIM
    return [
        ("hand", interface.HAND_MAX * (card + interface.HAND_FEAT_DIM)),
        (
            "enemy",
            interface.MAX_ENEMIES
            * (
                ENEMY_EMBED_DIM
                + interface.ENEMY_SCALAR_DIM
                + MOVE_EMBED_DIM
                + 1  # enemy_intent_hidden
                + interface.N_MONSTER_POWER_IDS
                + 1  # enemy_alive
            ),
        ),
        ("piles", _N_PILES * (_PILE_POOLS * card)),
        ("potion", interface.POTION_SLOTS * potion + interface.POTION_SLOTS),
        (
            "passthrough",
            interface.N_RELIC_IDS
            + interface.N_PLAYER_POWER_IDS
            + interface.PLAYER_SCALAR_DIM
            + interface.N_SCREENS
            + interface.MAP_CONTEXT_DIM,
        ),
        ("reward_card", interface.MAX_REWARD_CARD_SLOTS * card),
        ("reward_relic", interface.N_RELIC_IDS),
        ("reward_potion", interface.MAX_REWARD_POTIONS * potion),
        ("card_select", interface.CHOICE_MAX * card),
        ("deck", _PILE_POOLS * card),  # pooled mean+max, like one pile
        ("keys_act", interface.KEYS_ACT_DIM),
        ("shop_card", interface.SHOP_CARD_SLOTS * card),
        ("shop_relic", interface.N_RELIC_IDS),
        ("shop_potion", interface.SHOP_POTION_SLOTS * potion),
        ("shop_price", interface.SHOP_PRICE_SLOTS + 1),  # item prices + card-removal cost
        ("boss_relic", interface.N_RELIC_IDS),
        ("event", interface.N_EVENT_IDS),
        (
            "neow",
            interface.NEOW_OPTION_SLOTS * (interface.N_NEOW_BONUS + interface.N_NEOW_DRAWBACK),
        ),
    ]


def _expected_feature_dim() -> int:
    """Concat width = sum of every block, straight from the interface constants.

    Asserting equality with the encoder's own derivation locks the feature width to
    the interface constants (if the interface sizes change, both must move together).
    """
    return sum(width for _, width in _block_layout())


def _block_span(name: str) -> tuple[int, int]:
    """``(start, width)`` of a named concat block, from the ordered layout."""
    start = 0
    for block_name, width in _block_layout():
        if block_name == name:
            return start, width
        start += width
    raise KeyError(name)


def _block_slice(feats: torch.Tensor, name: str) -> torch.Tensor:
    """The columns of ``feats`` occupied by the named concat block."""
    start, width = _block_span(name)
    return feats[:, start : start + width]


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
    card_block = _block_slice(feats, "reward_card")
    assert torch.equal(card_block, torch.zeros(BATCH, card_block.shape[1]))


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
    potion_block = _block_slice(feats, "reward_potion")
    assert torch.equal(potion_block, torch.zeros(BATCH, potion_block.shape[1]))
    relic_block = _block_slice(feats, "reward_relic")
    assert torch.equal(relic_block, torch.zeros(BATCH, relic_block.shape[1]))


def test_card_select_block_is_zero_when_all_pad():
    """All-PAD card_select_ids -> the card-select block is exactly zero.

    Relies on card_embed's padding_idx row (no masking): empty card-select slots
    contribute a zero vector, so the card-select columns vanish. The block sits
    before the appended deck and keys/act blocks.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["card_select_ids"] = torch.full_like(obs["card_select_ids"], interface.PAD_ID)
    feats = enc.encode_features(obs)
    card_select_block = _block_slice(feats, "card_select")
    assert torch.equal(card_select_block, torch.zeros(BATCH, card_select_block.shape[1]))


def test_deck_block_is_zero_when_all_pad():
    """All-PAD deck_ids -> the pooled deck block is exactly zero (empty-deck safe).

    The deck is pooled mean+max through card_embed's padding_idx row, so an all-PAD
    (empty) deck contributes a zero block.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["deck_ids"] = torch.full_like(obs["deck_ids"], interface.PAD_ID)
    feats = enc.encode_features(obs)
    deck_block = _block_slice(feats, "deck")
    assert torch.equal(deck_block, torch.zeros(BATCH, deck_block.shape[1]))


def test_keys_act_is_passthrough_block():
    """keys_act is a raw passthrough occupying its KEYS_ACT_DIM columns.

    Not embedded: the encoder concatenates the coerced keys_act vector as-is, so the
    keys_act span of the concat equals the input keys_act exactly (also locking its
    placement now that the shop / boss / event-neow blocks follow it).
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    known = torch.arange(BATCH * interface.KEYS_ACT_DIM, dtype=torch.float32).reshape(
        BATCH, interface.KEYS_ACT_DIM
    )
    obs["keys_act"] = known
    feats = enc.encode_features(obs)
    assert torch.equal(_block_slice(feats, "keys_act"), known)


def test_gradient_flows_through_deck_path():
    """A real card in deck_ids (all other card fields PAD) reaches card_embed.

    Isolates the deck path: every other card-id field is PAD (padding_idx row 0
    receives no gradient), so a nonzero card_embed gradient can only come from the
    pooled deck block, proving encode_features wires deck_ids into the shared table.
    The seed keeps the sparse-input ReLU liveness deterministic.
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
        "deck_ids",
    ):
        obs[name] = torch.full_like(obs[name], interface.PAD_ID)
    real_id = interface.PAD_ID + 1  # any non-PAD card id
    obs["deck_ids"][:, 0] = real_id
    enc(obs).sum().backward()
    grad = enc.card_embed.weight.grad
    assert grad is not None
    # Only the deck slot's id can carry gradient; the PAD row must stay zero.
    assert torch.count_nonzero(grad[real_id]) > 0
    assert torch.equal(grad[interface.PAD_ID], torch.zeros(CARD_EMBED_DIM))


def test_keys_act_influences_trunk():
    """The keys/act columns feed the first trunk Linear (nonzero -> gradient there).

    keys_act has no embedding table, so its learning signal shows up as gradient on
    the first trunk Linear's final KEYS_ACT_DIM input columns; a nonzero keys_act
    input drives gradient into exactly those columns. The seed keeps the
    sparse-input ReLU liveness deterministic.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["keys_act"] = torch.ones_like(obs["keys_act"])
    enc(obs).sum().backward()
    grad = enc.trunk[0].weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[:, -interface.KEYS_ACT_DIM :]) > 0


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
    relic_start, _ = _block_span("reward_relic")
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
    relic_block = _block_slice(feats, "reward_relic")
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
    relic_block = _block_slice(feats, "reward_relic")
    assert torch.equal(relic_block, torch.zeros(BATCH, interface.N_RELIC_IDS))


def test_shop_and_boss_relic_blocks_zero_when_empty():
    """All-INVALID shop_relic_ids / boss_relic_ids -> their multihot blocks are zero.

    Both reuse the shared _relic_multihot (like reward relics): empty slots carry the
    INVALID sentinel, which scatters into the dropped final column, so no kept relic
    column is set. Slicing at the layout-derived spans also locks their placement.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["shop_relic_ids"] = torch.full_like(obs["shop_relic_ids"], interface.N_RELIC_IDS)
    obs["boss_relic_ids"] = torch.full_like(obs["boss_relic_ids"], interface.N_RELIC_IDS)
    feats = enc.encode_features(obs)
    for name in ("shop_relic", "boss_relic"):
        block = _block_slice(feats, name)
        assert torch.equal(block, torch.zeros(BATCH, block.shape[1])), name


def test_offered_akabeko_in_shop_and_boss_sets_its_column():
    """AKABEKO (RelicId 0) offered in the shop / boss slot sets that block's column 0.

    The AKABEKO-vs-empty regression, extended to the shop and boss relic offers that
    share _relic_multihot: column 0 stays usable (empties are the INVALID sentinel, not
    0), so an offered Akabeko is not silently erased.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    akabeko = 0  # RelicId.AKABEKO
    for name in ("shop_relic_ids", "boss_relic_ids"):
        obs[name] = torch.full_like(obs[name], interface.N_RELIC_IDS)
        obs[name][:, 0] = akabeko
    feats = enc.encode_features(obs)
    expected = torch.zeros(BATCH, interface.N_RELIC_IDS)
    expected[:, akabeko] = 1.0
    for name in ("shop_relic", "boss_relic"):
        assert torch.equal(_block_slice(feats, name), expected), name


def test_gradient_flows_through_shop_card_path():
    """A real card in shop_card_ids (all other card fields PAD) reaches card_embed.

    Isolates the shop-card path: every other card-id field is PAD, so a nonzero
    card_embed gradient can only come from the shop-card block, proving encode_features
    wires shop_card_ids into the shared table. The seed keeps the sparse-input ReLU
    liveness deterministic.
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
        "deck_ids",
        "shop_card_ids",
    ):
        obs[name] = torch.full_like(obs[name], interface.PAD_ID)
    real_id = interface.PAD_ID + 1  # any non-PAD card id
    obs["shop_card_ids"][:, 0] = real_id
    enc(obs).sum().backward()
    grad = enc.card_embed.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[real_id]) > 0
    assert torch.equal(grad[interface.PAD_ID], torch.zeros(CARD_EMBED_DIM))


def test_gradient_flows_through_shop_potion_path():
    """A real potion in shop_potion_ids (all other potion fields PAD) reaches potion_embed.

    Isolates the shop-potion path: every other potion-id field is PAD, so a nonzero
    potion_embed gradient can only come from the shop-potion block, proving
    encode_features wires shop_potion_ids into the shared table.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    for name in ("potion_ids", "reward_potion_ids", "shop_potion_ids"):
        obs[name] = torch.full_like(obs[name], interface.PAD_ID)
    real_id = interface.PAD_ID + 1  # any non-PAD potion id
    obs["shop_potion_ids"][:, 0] = real_id
    enc(obs).sum().backward()
    grad = enc.potion_embed.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[real_id]) > 0
    assert torch.equal(grad[interface.PAD_ID], torch.zeros(POTION_EMBED_DIM))


def test_deterministic_in_eval_mode():
    enc = ObsFeatureEncoder()
    enc.eval()
    obs = sample_observation_batch(BATCH)
    with torch.no_grad():
        first = enc(obs)
        second = enc(obs)
    assert torch.equal(first, second)


def _canonical_baseline_obs(batch: int) -> dict[str, torch.Tensor]:
    """Zero/PAD baseline: every id field all-PAD, every float field all-zero.

    A controlled constant baseline (not a random sample) so that a single-field
    perturbation is the ONLY thing that can move the encoder output, isolating each
    field's influence.
    """
    obs: dict[str, torch.Tensor] = {}
    for f in interface.OBS_FIELDS:
        if f.bounds == "id":
            obs[f.name] = torch.full((batch, *f.shape), interface.PAD_ID, dtype=torch.long)
        else:
            obs[f.name] = torch.zeros((batch, *f.shape), dtype=torch.float32)
    return obs


def _perturb_field(field: interface.ObsField, batch: int) -> torch.Tensor:
    """One field changed away from the baseline, staying in the field's declared bounds.

    id fields flip PAD -> a distinct in-range non-PAD id; float fields take a nonzero
    in-bounds constant (interior of the unit range for "unit", 1.0 for "real").
    """
    if field.bounds == "id":
        return torch.full((batch, *field.shape), interface.PAD_ID + 1, dtype=torch.long)
    value = 0.5 if field.bounds == "unit" else 1.0
    return torch.full((batch, *field.shape), value, dtype=torch.float32)


def test_every_obs_field_influences_encoder_output():
    """Every declared OBS_FIELD must move the encoder output when perturbed alone.

    The durable guard for the whole class of bug this observation-coverage work
    addressed: a field the environment declares and populates but the encoder silently
    ignores would leave the output unchanged here and fail CI. Sweeps the live
    OBS_FIELDS registry (no hardcoded field list), perturbing exactly one field against
    a zero/PAD baseline and asserting both the pre-trunk concat and the eval-mode trunk
    output change. Skips nothing: a field that cannot influence the output is surfaced
    as a failure, not skipped.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    enc.eval()
    baseline = _canonical_baseline_obs(BATCH)
    with torch.no_grad():
        base_feats = enc.encode_features(baseline)
        base_out = enc(baseline)

    ignored: list[str] = []
    for field in interface.OBS_FIELDS:
        obs = {name: t.clone() for name, t in baseline.items()}
        obs[field.name] = _perturb_field(field, BATCH)
        with torch.no_grad():
            feats = enc.encode_features(obs)
            out = enc(obs)
        # encode_features changing proves the field enters the concat (the encoder
        # consumes it); the trunk output changing confirms it reaches the actual output.
        if torch.equal(feats, base_feats) or torch.equal(out, base_out):
            ignored.append(field.name)

    assert not ignored, (
        "OBS_FIELDS declared/populated by the env but ignored by the encoder "
        f"(no output change when perturbed alone): {ignored}"
    )
