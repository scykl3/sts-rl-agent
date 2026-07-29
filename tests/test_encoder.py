"""Tests for the shared observation encoder.

Observations are generated from the interface's own space
(:func:`sts_rl.env.spaces.build_observation_space`) so the tests exercise real
interface-shaped data and the feature width is locked to the interface constants
rather than a hardcoded literal.
"""

from __future__ import annotations

import pytest
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
    deck = _PILE_POOLS * CARD_EMBED_DIM  # pooled mean+max, like one pile
    keys_act = interface.KEYS_ACT_DIM
    # Shop screen: cards / potions embedded per slot, relics a multihot, each id block
    # followed by its raw price columns, plus the scalar remove cost; then the boss
    # relic multihot. All appended last.
    shop_cards = interface.MAX_SHOP_CARDS * CARD_EMBED_DIM
    shop_card_prices = interface.MAX_SHOP_CARDS
    shop_relics = interface.N_RELIC_IDS
    shop_relic_prices = interface.MAX_SHOP_RELICS
    shop_potions = interface.MAX_SHOP_POTIONS * POTION_EMBED_DIM
    shop_potion_prices = interface.MAX_SHOP_POTIONS
    shop_remove_cost = 1
    boss_relics = interface.N_RELIC_IDS
    # Neow-event one-hots, appended last: the current event id, then per-option
    # NeowBonus / NeowDrawback one-hot spans.
    event_onehot = interface.N_EVENT_IDS
    neow_bonus = interface.MAX_NEOW_OPTIONS * interface.N_NEOW_BONUS
    neow_drawback = interface.MAX_NEOW_OPTIONS * interface.N_NEOW_DRAWBACK
    # Multi-stage event phase one-hot, appended after the Neow-event blocks.
    event_phase = interface.EVENT_PHASE_DIM
    # Map lookahead aggregates, appended last (after the event-phase block).
    map_lookahead = interface.MAP_LOOKAHEAD_DIM
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
        + deck
        + keys_act
        + shop_cards
        + shop_card_prices
        + shop_relics
        + shop_relic_prices
        + shop_potions
        + shop_potion_prices
        + shop_remove_cost
        + boss_relics
        + event_onehot
        + neow_bonus
        + neow_drawback
        + event_phase
        + map_lookahead
    )


def _shop_boss_tail_width() -> int:
    """Total width of the shop / boss-relic blocks appended after keys_act.

    The reward / card-select / deck / keys_act blocks are no longer the concat tail;
    end-relative slices below step back past this suffix to reach them. Derived from
    the interface / encoder constants, mirroring the encoder's append.
    """
    return (
        interface.MAX_SHOP_CARDS * CARD_EMBED_DIM  # shop cards embedded
        + interface.MAX_SHOP_CARDS  # shop card prices
        + interface.N_RELIC_IDS  # shop relic multihot
        + interface.MAX_SHOP_RELICS  # shop relic prices
        + interface.MAX_SHOP_POTIONS * POTION_EMBED_DIM  # shop potions embedded
        + interface.MAX_SHOP_POTIONS  # shop potion prices
        + 1  # shop remove cost
        + interface.N_RELIC_IDS  # boss relic multihot
    )


def _neow_event_tail_width() -> int:
    """Total width of the Neow-event one-hot blocks appended after the boss-relic block.

    The event-phase one-hot is appended after these, so it (not this) is the concat tail;
    every end-relative slice below steps back past the event-phase block, these Neow
    blocks, and the shop / boss-relic suffix to reach the earlier blocks. Derived from the
    interface constants, mirroring the encoder's append.
    """
    return (
        interface.N_EVENT_IDS  # event_onehot
        + interface.MAX_NEOW_OPTIONS * interface.N_NEOW_BONUS  # neow bonus one-hots
        + interface.MAX_NEOW_OPTIONS * interface.N_NEOW_DRAWBACK  # neow drawback one-hots
    )


def _event_phase_tail_width() -> int:
    """Width of the event-phase one-hot block (appended after the Neow-event blocks).

    The map-lookahead block is appended after this, so it (not this) is the concat tail;
    every end-relative slice below steps back past the map-lookahead block too. A single
    interface constant, kept as a helper so the slices read uniformly.
    """
    return interface.EVENT_PHASE_DIM


def _map_lookahead_tail_width() -> int:
    """Width of the map-lookahead block, the current concat tail (appended last, after the
    event-phase block). A single interface constant, kept as a helper so the end-relative
    slices below read uniformly alongside the other tail-width helpers.
    """
    return interface.MAP_LOOKAHEAD_DIM


def _relic_block_start(enc: ObsFeatureEncoder) -> int:
    """Start column of the reward-relic block in the pre-trunk concat.

    The relic block precedes the reward-potion, card-select, deck, keys/act, and the
    appended shop / boss-relic blocks; derived from the interface widths (no literal),
    mirroring the encoder's own append order.
    """
    card_select_width = interface.CHOICE_MAX * CARD_EMBED_DIM
    potion_width = interface.MAX_REWARD_POTIONS * POTION_EMBED_DIM
    deck_width = _PILE_POOLS * CARD_EMBED_DIM
    keys_act_width = interface.KEYS_ACT_DIM
    return (
        enc.feature_dim
        - _map_lookahead_tail_width()
        - _event_phase_tail_width()
        - _neow_event_tail_width()
        - _shop_boss_tail_width()
        - keys_act_width
        - deck_width
        - card_select_width
        - potion_width
        - interface.N_RELIC_IDS
    )


def _shop_relic_block_start(enc: ObsFeatureEncoder) -> int:
    """Start column of the shop-relic multihot block in the pre-trunk concat.

    The shop-relic block is followed by shop_relic_prices, shop_potions,
    shop_potion_prices, shop_remove_cost, and the boss-relic block; derived from the
    interface widths (no literal), mirroring the encoder's append order.
    """
    return (
        enc.feature_dim
        - _map_lookahead_tail_width()  # the map-lookahead block (the current concat tail)
        - _event_phase_tail_width()  # the event-phase block
        - _neow_event_tail_width()  # the Neow-event blocks
        - interface.N_RELIC_IDS  # boss_relic block
        - 1  # shop_remove_cost
        - interface.MAX_SHOP_POTIONS  # shop_potion_prices
        - interface.MAX_SHOP_POTIONS * POTION_EMBED_DIM  # shop_potions
        - interface.MAX_SHOP_RELICS  # shop_relic_prices
        - interface.N_RELIC_IDS  # the shop-relic block itself
    )


def _boss_relic_block_start(enc: ObsFeatureEncoder) -> int:
    """Start column of the boss-relic multihot block (followed by the Neow-event tail and
    then the event-phase block)."""
    return (
        enc.feature_dim
        - _map_lookahead_tail_width()
        - _event_phase_tail_width()
        - _neow_event_tail_width()
        - interface.N_RELIC_IDS
    )


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
    deck_width = _PILE_POOLS * CARD_EMBED_DIM
    keys_act_width = interface.KEYS_ACT_DIM
    # The card block sits before the appended relic, potion, card-select, deck,
    # keys/act, and shop / boss-relic blocks.
    card_start = (
        enc.feature_dim
        - _map_lookahead_tail_width()
        - _event_phase_tail_width()
        - _neow_event_tail_width()
        - _shop_boss_tail_width()
        - keys_act_width
        - deck_width
        - card_select_width
        - potion_width
        - relic_width
        - reward_width
    )
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
    deck_width = _PILE_POOLS * CARD_EMBED_DIM
    keys_act_width = interface.KEYS_ACT_DIM
    # Appended after the potion block, in order: card_select, deck, keys/act, then the
    # shop / boss-relic blocks (the current concat tail).
    tail = (
        _map_lookahead_tail_width()
        + _event_phase_tail_width()
        + _neow_event_tail_width()
        + _shop_boss_tail_width()
        + card_select_width
        + deck_width
        + keys_act_width
    )
    potion_block = feats[:, -(tail + potion_width) : -tail]
    assert torch.equal(potion_block, torch.zeros(BATCH, potion_width))
    relic_block = feats[:, -(tail + potion_width + relic_width) : -(tail + potion_width)]
    assert torch.equal(relic_block, torch.zeros(BATCH, relic_width))


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
    card_select_width = interface.CHOICE_MAX * CARD_EMBED_DIM
    deck_width = _PILE_POOLS * CARD_EMBED_DIM
    keys_act_width = interface.KEYS_ACT_DIM
    tail = (
        _map_lookahead_tail_width()
        + _event_phase_tail_width()
        + _neow_event_tail_width()
        + _shop_boss_tail_width()
        + deck_width
        + keys_act_width
    )
    card_select_block = feats[:, -(tail + card_select_width) : -tail]
    assert torch.equal(card_select_block, torch.zeros(BATCH, card_select_width))


def test_deck_block_is_zero_when_all_pad():
    """All-PAD deck_ids -> the pooled deck block is exactly zero (empty-deck safe).

    The deck is pooled mean+max through card_embed's padding_idx row, so an all-PAD
    (empty) deck contributes a zero block. The deck block sits before the keys/act and
    the appended shop / boss-relic blocks.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["deck_ids"] = torch.full_like(obs["deck_ids"], interface.PAD_ID)
    feats = enc.encode_features(obs)
    keys_act_width = interface.KEYS_ACT_DIM
    deck_width = _PILE_POOLS * CARD_EMBED_DIM
    tail = (
        _map_lookahead_tail_width()
        + _event_phase_tail_width()
        + _neow_event_tail_width()
        + _shop_boss_tail_width()
        + keys_act_width
    )
    deck_block = feats[:, -(tail + deck_width) : -tail]
    assert torch.equal(deck_block, torch.zeros(BATCH, deck_width))


def test_keys_act_is_passthrough_block():
    """keys_act is a raw passthrough occupying its KEYS_ACT_DIM columns unchanged.

    Not embedded: the encoder concatenates the coerced keys_act vector as-is, so the
    concat slice at keys_act's offset (just before the appended shop / boss-relic
    blocks) equals the input keys_act exactly.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    known = torch.arange(BATCH * interface.KEYS_ACT_DIM, dtype=torch.float32).reshape(
        BATCH, interface.KEYS_ACT_DIM
    )
    obs["keys_act"] = known
    feats = enc.encode_features(obs)
    tail = (
        _map_lookahead_tail_width()
        + _event_phase_tail_width()
        + _neow_event_tail_width()
        + _shop_boss_tail_width()
    )
    assert torch.equal(feats[:, -(tail + interface.KEYS_ACT_DIM) : -tail], known)


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
    the first trunk Linear's keys/act input columns (just before the appended shop /
    boss-relic columns); a nonzero keys_act input drives gradient into exactly those
    columns. The seed keeps the sparse-input ReLU liveness deterministic.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["keys_act"] = torch.ones_like(obs["keys_act"])
    enc(obs).sum().backward()
    grad = enc.trunk[0].weight.grad
    assert grad is not None
    tail = (
        _map_lookahead_tail_width()
        + _event_phase_tail_width()
        + _neow_event_tail_width()
        + _shop_boss_tail_width()
    )
    assert torch.count_nonzero(grad[:, -(tail + interface.KEYS_ACT_DIM) : -tail]) > 0


def test_event_phase_is_passthrough_block():
    """event_phase_onehot is a raw passthrough occupying the final EVENT_PHASE_DIM columns.

    Not embedded: the encoder concatenates the coerced event-phase one-hot as-is, just
    before the appended map-lookahead block, so the event-phase columns (stepping back
    past the map-lookahead tail) equal the input event_phase_onehot exactly.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    known = torch.arange(BATCH * interface.EVENT_PHASE_DIM, dtype=torch.float32).reshape(
        BATCH, interface.EVENT_PHASE_DIM
    )
    obs["event_phase_onehot"] = known
    feats = enc.encode_features(obs)
    tail = _map_lookahead_tail_width()
    assert torch.equal(feats[:, -(tail + _event_phase_tail_width()) : -tail], known)


def test_event_phase_influences_trunk():
    """The event-phase columns feed the first trunk Linear (nonzero -> gradient there).

    event_phase_onehot has no embedding table, so its learning signal shows up as gradient
    on the first trunk Linear's event-phase input columns (just before the appended
    map-lookahead columns); a nonzero event-phase input drives gradient into exactly those
    columns. The seed keeps the sparse-input ReLU liveness deterministic.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    obs["event_phase_onehot"] = torch.ones_like(obs["event_phase_onehot"])
    enc(obs).sum().backward()
    grad = enc.trunk[0].weight.grad
    assert grad is not None
    tail = _map_lookahead_tail_width()
    assert torch.count_nonzero(grad[:, -(tail + _event_phase_tail_width()) : -tail]) > 0


def test_map_lookahead_is_passthrough_tail_block():
    """map_lookahead is a raw passthrough occupying the final MAP_LOOKAHEAD_DIM columns.

    Not embedded: the encoder concatenates the coerced map-lookahead vector as-is as the
    concat tail (appended last, after the event-phase block), so the trailing
    _map_lookahead_tail_width() columns equal the input map_lookahead exactly.
    """
    enc = ObsFeatureEncoder()
    obs = sample_observation_batch(BATCH)
    known = torch.arange(BATCH * interface.MAP_LOOKAHEAD_DIM, dtype=torch.float32).reshape(
        BATCH, interface.MAP_LOOKAHEAD_DIM
    )
    obs["map_lookahead"] = known
    feats = enc.encode_features(obs)
    assert torch.equal(feats[:, -_map_lookahead_tail_width() :], known)


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


# Shop / boss relic-offer multihot fields, each mapped to its concat block start. The
# reward-relic field has its own dedicated tests above; these cover the new fields.
_SHOP_BOSS_RELIC_BLOCKS = {
    "shop_relic_ids": _shop_relic_block_start,
    "boss_relic_ids": _boss_relic_block_start,
}


@pytest.mark.parametrize("field_name", sorted(_SHOP_BOSS_RELIC_BLOCKS))
def test_offered_shop_boss_relic_distinguishable_from_empty_slot(field_name):
    """A real offered relic in a shop / boss slot sets its own output column; an empty
    slot (INVALID) sets none -- the AKABEKO-class guard for the new relic fields.

    Regression for the empty-sentinel design (mirrors the reward-relic guard): offering
    AKABEKO (RelicId 0) must set column 0, distinguishable from an empty slot. Under a
    "clear column PAD_ID(0)" multihot, column 0 would be forced to zero, silently
    erasing an offered AKABEKO; the INVALID-as-empty design keeps column 0 for AKABEKO
    and drops the INVALID column instead. FAILS if that behavior is reverted.
    """
    enc = ObsFeatureEncoder()
    block_start = _SHOP_BOSS_RELIC_BLOCKS[field_name](enc)
    obs = sample_observation_batch(BATCH)

    # Every slot empty (INVALID sentinel) -> no kept column set anywhere in the block.
    obs[field_name] = torch.full_like(obs[field_name], interface.N_RELIC_IDS)
    empty_block = enc.encode_features(obs)[:, block_start : block_start + interface.N_RELIC_IDS]
    assert torch.equal(empty_block, torch.zeros(BATCH, interface.N_RELIC_IDS))

    # Offer AKABEKO (id 0) in slot 0 -> exactly column 0 is set, distinguishable from
    # the all-empty case above (an empty slot would have left column 0 zero).
    akabeko = 0  # RelicId.AKABEKO
    obs[field_name][:, 0] = akabeko
    block = enc.encode_features(obs)[:, block_start : block_start + interface.N_RELIC_IDS]
    expected = torch.zeros(BATCH, interface.N_RELIC_IDS)
    expected[:, akabeko] = 1.0
    assert torch.equal(block, expected)


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
