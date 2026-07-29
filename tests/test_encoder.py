"""Tests for the transformer observation encoder.

Observations are generated from the interface's own space
(:func:`sts_rl.env.spaces.build_observation_space`) so the tests exercise real
interface-shaped data and every token count / width is locked to the interface
constants rather than a hardcoded literal.
"""

from __future__ import annotations

import pytest
import torch

from sts_rl import interface
from sts_rl.agent.encoder import (
    CARD_EMBED_DIM,
    ENEMY_EMBED_DIM,
    FFN_DIM,
    HIDDEN_DIM,
    MOVE_EMBED_DIM,
    N_ATTENTION_HEADS,
    N_TRANSFORMER_LAYERS,
    POTION_EMBED_DIM,
    RELIC_EMBED_DIM,
    ObsFeatureEncoder,
    _CLS_INPUT_DIM,
    _ENTITY_SPECS,
    _OFFER_INPUT_DIM,
    _PILE_POOLS,
    _PILE_SPECS,
)
from conftest import ID_FIELDS, sample_observation_batch

BATCH = 2

# The relic-backed entity id fields whose empty marker is the relic INVALID
# sentinel (N_RELIC_IDS), not PAD_ID: RelicId 0 (AKABEKO) is a real relic.
_RELIC_ID_FIELDS = frozenset(spec.id_field for spec in _ENTITY_SPECS if spec.pad_is_relic_invalid)


def _expected_seq_len() -> int:
    """Recompute the token-sequence length straight from the interface caps.

    S = every per-slot entity token + the six never-PAD context tokens (CLS, the
    four pile summaries, OFFER_CONTEXT). Derived symbolically so an interface cap
    change moves this test and the encoder together.
    """
    entity = (
        interface.HAND_MAX
        + interface.MAX_ENEMIES
        + interface.POTION_SLOTS
        + interface.CHOICE_MAX
        + interface.MAX_REWARD_CARD_SLOTS
        + interface.MAX_REWARD_POTIONS
        + interface.MAX_REWARD_RELICS
        + interface.MAX_SHOP_CARDS
        + interface.MAX_SHOP_RELICS
        + interface.MAX_SHOP_POTIONS
        + interface.MAX_BOSS_RELICS
    )
    context = 1 + len(_PILE_SPECS) + 1  # CLS + piles + OFFER
    return entity + context


def _all_pad_obs(batch: int) -> dict[str, torch.Tensor]:
    """Every entity slot empty and every float field zero.

    id fields hold their empty marker (the relic INVALID sentinel for the relic
    fields, else PAD_ID); float fields are zero. So every entity token is PAD and
    only the six context tokens are valid.
    """
    obs: dict[str, torch.Tensor] = {}
    for field in interface.OBS_FIELDS:
        if field.bounds == "id":
            pad = interface.N_RELIC_IDS if field.name in _RELIC_ID_FIELDS else interface.PAD_ID
            obs[field.name] = torch.full((batch, *field.shape), pad, dtype=torch.long)
        else:
            obs[field.name] = torch.zeros((batch, *field.shape), dtype=torch.float32)
    return obs


def _all_live_obs(batch: int) -> dict[str, torch.Tensor]:
    """Every entity slot occupied by a real id and every float field zero.

    So every entity token is non-PAD (unmasked) and each obs field feeds a live
    token. Real ids: 5 for the relic fields (a real, non-AKABEKO relic), 1 for the
    other id fields (a real, non-PAD id).
    """
    obs: dict[str, torch.Tensor] = {}
    for field in interface.OBS_FIELDS:
        if field.bounds == "id":
            live = 5 if field.name in _RELIC_ID_FIELDS else 1
            obs[field.name] = torch.full((batch, *field.shape), live, dtype=torch.long)
        else:
            obs[field.name] = torch.zeros((batch, *field.shape), dtype=torch.float32)
    return obs


def _perturb_field(field: interface.ObsField, batch: int, live: bool) -> torch.Tensor:
    """One field changed away from the baseline, staying in the field's bounds.

    Against the all-live baseline, an id field takes a distinct real id and a
    float field a nonzero in-bounds constant.
    """
    if field.bounds == "id":
        base = (5 if field.name in _RELIC_ID_FIELDS else 1) if live else interface.PAD_ID
        return torch.full((batch, *field.shape), base + 1, dtype=torch.long)
    value = 0.5 if field.bounds == "unit" else 1.0
    return torch.full((batch, *field.shape), value, dtype=torch.float32)


def test_forward_output_shapes():
    """forward -> (per_token (B,S,d), pooled (B,d), key_padding_mask (B,S) bool)."""
    enc = ObsFeatureEncoder()
    per_token, pooled, mask = enc(sample_observation_batch(BATCH))
    assert per_token.shape == (BATCH, enc.seq_len, enc.output_dim)
    assert pooled.shape == (BATCH, enc.output_dim)
    # Output lands on the model's device (CPU-safe: both sides are CPU here).
    assert pooled.device == enc.cls_mlp[0].weight.device
    assert mask.shape == (BATCH, enc.seq_len)
    assert mask.dtype == torch.bool


def test_seq_len_and_cls_index_match_interface_constants():
    """seq_len / n_entity_tokens / cls_index are the interface-derived counts."""
    enc = ObsFeatureEncoder()
    assert enc.seq_len == _expected_seq_len()
    # Entity tokens come first, so the first context token (CLS) sits at the
    # entity-token count, and that count is S minus the six context tokens.
    expected_entity = _expected_seq_len() - (1 + len(_PILE_SPECS) + 1)
    assert enc.n_entity_tokens == expected_entity
    assert enc.cls_index == expected_entity


def test_output_is_finite():
    enc = ObsFeatureEncoder()
    per_token, pooled, _ = enc(sample_observation_batch(BATCH))
    assert torch.isfinite(per_token).all()
    assert torch.isfinite(pooled).all()


def test_pooled_is_the_cls_token_row():
    """The pooled context is exactly the CLS token's per-token output row."""
    enc = ObsFeatureEncoder()
    enc.eval()
    with torch.no_grad():
        per_token, pooled, _ = enc(sample_observation_batch(BATCH))
    assert torch.equal(pooled, per_token[:, enc.cls_index])


def test_accepts_float64_observations():
    """Float64 obs (the common numpy default) must not promote the network.

    Regression: without internal float32 coercion the token features promote to
    double and ``nn.Linear`` raises "mat1 and mat2 must have the same dtype". Ids
    stay integer; every float field is recast to double to mimic
    ``torch.as_tensor(numpy_float64_array)``.
    """
    enc = ObsFeatureEncoder()
    obs64 = {
        name: (t if name in ID_FIELDS else t.double())
        for name, t in sample_observation_batch(BATCH).items()
    }
    per_token, pooled, _ = enc(obs64)
    assert pooled.shape == (BATCH, enc.output_dim)
    assert per_token.dtype == torch.float32
    assert pooled.dtype == torch.float32
    assert torch.isfinite(pooled).all()


def test_padding_rows_are_zero():
    """Each id table's empty-marker row is the permanent zero vector."""
    enc = ObsFeatureEncoder()
    assert torch.equal(enc.card_embed.weight[interface.PAD_ID], torch.zeros(CARD_EMBED_DIM))
    assert torch.equal(enc.enemy_embed.weight[interface.PAD_ID], torch.zeros(ENEMY_EMBED_DIM))
    assert torch.equal(enc.move_embed.weight[interface.PAD_ID], torch.zeros(MOVE_EMBED_DIM))
    assert torch.equal(enc.potion_embed.weight[interface.PAD_ID], torch.zeros(POTION_EMBED_DIM))
    # relic_embed pads at the INVALID sentinel (N_RELIC_IDS), not PAD_ID.
    assert enc.relic_embed.num_embeddings == interface.N_RELIC_IDS + 1
    assert enc.relic_embed.padding_idx == interface.N_RELIC_IDS
    assert torch.equal(enc.relic_embed.weight[interface.N_RELIC_IDS], torch.zeros(RELIC_EMBED_DIM))


def test_init_matches_nanogpt_reference():
    """Weights follow the nanoGPT/minGPT init the encoder docstring cites.

    A plain Linear initializes near std 0.02; every id embedding's padding row is
    exactly zero after construction (re-zeroed post-init, the load-bearing
    PAD=zero invariant); and each block's residual out_proj is scaled by
    1/sqrt(2 * N_TRANSFORMER_LAYERS), so its std is materially below 0.02.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()

    # A representative non-residual Linear sits near the 0.02 target.
    sample_std = enc.cls_mlp[0].weight.std().item()
    assert 0.005 < sample_std < 0.05

    # Every id embedding's padding row stays exactly zero after construction.
    for emb in (
        enc.card_embed,
        enc.enemy_embed,
        enc.move_embed,
        enc.potion_embed,
        enc.relic_embed,
    ):
        assert torch.equal(emb.weight[emb.padding_idx], torch.zeros(emb.embedding_dim))

    # Residual out_proj is scaled by 1/sqrt(2N), so materially below 0.02.
    residual_std = enc.blocks[0].attn.out_proj.weight.std().item()
    assert 0.0 < residual_std < 0.02 / 2


def test_key_padding_mask_true_exactly_at_pad_slots():
    """The mask is True on every entity PAD slot and False on the context tokens.

    Table-driven over the entity types: an all-PAD obs masks every entity slot;
    setting one slot per type to a real id clears exactly that slot. Context
    tokens (the sequence tail) are never masked.
    """
    enc = ObsFeatureEncoder()
    obs = _all_pad_obs(BATCH)
    _, _, mask = enc(obs)
    # Entity portion all True, context portion all False.
    assert bool(mask[:, : enc.n_entity_tokens].all())
    assert bool((~mask[:, enc.n_entity_tokens :]).all())

    for spec in _ENTITY_SPECS:
        obs_one = {k: v.clone() for k, v in obs.items()}
        real = 0 if spec.pad_is_relic_invalid else interface.PAD_ID + 1  # AKABEKO(0) for relics
        obs_one[spec.id_field][:, 0] = real
        _, _, mask_one = enc(obs_one)
        start, count = enc.token_layout[spec.name]
        block = mask_one[:, start : start + count]
        assert bool((~block[:, 0]).all()), f"{spec.name}: slot 0 should be unmasked"
        if count > 1:
            assert bool(block[:, 1:].all()), f"{spec.name}: slots 1.. should stay masked"


def test_no_all_pad_attention_row():
    """Even an all-PAD-entity obs leaves every row with the six valid context keys.

    This is the attention-side analogue of the action-side all-illegal guard: no
    attention row is all-`-inf`, so the softmax never NaNs.
    """
    enc = ObsFeatureEncoder()
    _, _, mask = enc(_all_pad_obs(BATCH))
    valid_per_row = (~mask).sum(dim=1)
    assert bool((valid_per_row >= 1).all())
    # Exactly the six context tokens are valid when every entity slot is PAD.
    assert bool((valid_per_row == (1 + len(_PILE_SPECS) + 1)).all())


def test_pad_slot_perturbation_does_not_leak_and_revert_confirms_mask():
    """Perturbing a PAD slot's raw features leaves the pooled context unchanged.

    Revert-verify: with the mask removed, CLS attends to the PAD token and the
    pooled context DOES change - proving the mask is what suppresses PAD-token
    influence, not a coincidence. The never-PAD context-token rows of per_token
    are checked the same way: masked they are unchanged, unmasked they leak.
    """
    enc = ObsFeatureEncoder()
    enc.eval()
    obs = _all_pad_obs(BATCH)  # hand slots are PAD, so hand_feats sit on PAD tokens
    perturbed = {k: v.clone() for k, v in obs.items()}
    perturbed["hand_feats"][:, 0, :] = 5.0  # a PAD hand slot's raw feature

    with torch.no_grad():
        tok_base, pooled_base, _ = enc(obs)
        tok_pert, pooled_pert, _ = enc(perturbed)
        tok_base_nomask, pooled_base_nomask, _ = enc(obs, apply_padding_mask=False)
        tok_pert_nomask, pooled_pert_nomask, _ = enc(perturbed, apply_padding_mask=False)

    assert torch.allclose(pooled_base, pooled_pert, atol=1e-6)
    assert not torch.allclose(pooled_base_nomask, pooled_pert_nomask, atol=1e-6)
    # Same guarantee for the never-PAD context-token rows (the sequence tail):
    # masked they cannot attend to the perturbed PAD slot, unmasked they do.
    ctx = slice(enc.n_entity_tokens, None)
    assert torch.allclose(tok_base[:, ctx], tok_pert[:, ctx], atol=1e-6)
    assert not torch.allclose(tok_base_nomask[:, ctx], tok_pert_nomask[:, ctx], atol=1e-6)


def test_pooled_pile_is_permutation_invariant():
    """Permuting an order-agnostic pooled pile leaves the output unchanged.

    Piles (draw/discard/exhaust/deck) are mean+max pooled, so reordering their
    slots must not move the pooled context. Slotted types are covered by the
    slot-stability test below.
    """
    enc = ObsFeatureEncoder()
    enc.eval()
    obs = _all_pad_obs(BATCH)
    obs["draw_ids"][:, 0] = 3
    obs["draw_ids"][:, 1] = 7
    permuted = {k: v.clone() for k, v in obs.items()}
    permuted["draw_ids"] = obs["draw_ids"].flip(dims=[1])
    with torch.no_grad():
        _, pooled, _ = enc(obs)
        _, pooled_perm, _ = enc(permuted)
    assert torch.allclose(pooled, pooled_perm, atol=1e-6)


def test_hand_slot_order_matters():
    """Swapping two live HAND slots' ids changes the output (slot stability).

    Unlike the pooled piles, slotted types carry a slot-id embedding, so slot
    order is significant: a later per-entity head maps action slots to token
    slots, so the encoder must not be permutation-invariant over them.
    """
    enc = ObsFeatureEncoder()
    enc.eval()
    obs = _all_pad_obs(BATCH)
    obs["hand_ids"][:, 0] = 3
    obs["hand_ids"][:, 1] = 7
    swapped = {k: v.clone() for k, v in obs.items()}
    swapped["hand_ids"][:, 0] = 7
    swapped["hand_ids"][:, 1] = 3
    with torch.no_grad():
        _, pooled, _ = enc(obs)
        _, pooled_swapped, _ = enc(swapped)
    assert not torch.allclose(pooled, pooled_swapped, atol=1e-6)


def test_gradient_flows_to_every_parameter():
    """One backward reaches every encoder parameter with a finite gradient.

    Uses an all-live obs so every token type is non-PAD, activating every
    embedding table (card/enemy/move/potion/relic), per-type projection, the type
    and slot-id embeddings, every transformer block parameter, and the CLS / pile
    / OFFER context projections.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    per_token, _, _ = enc(_all_live_obs(BATCH))
    per_token.sum().backward()
    missing = [name for name, p in enc.named_parameters() if p.grad is None]
    nonfinite = [
        name
        for name, p in enc.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not missing, f"parameters with no gradient: {missing}"
    assert not nonfinite, f"parameters with non-finite gradient: {nonfinite}"


def test_id_embedding_pad_row_gets_no_gradient():
    """A real relic drives gradient into its row; the INVALID pad row stays zero.

    Isolates the relic path: with every relic slot INVALID except one real relic,
    only that relic's embedding row may carry gradient, and relic_embed's
    padding_idx (INVALID) row must stay exactly zero.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    obs = _all_pad_obs(BATCH)
    real_relic = 5
    obs["reward_relic_ids"][:, 0] = real_relic
    per_token, _, _ = enc(obs)
    per_token.sum().backward()
    grad = enc.relic_embed.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[real_relic]) > 0
    assert torch.equal(grad[interface.N_RELIC_IDS], torch.zeros(RELIC_EMBED_DIM))


def test_deterministic_in_eval_mode():
    enc = ObsFeatureEncoder()
    enc.eval()
    obs = sample_observation_batch(BATCH)
    with torch.no_grad():
        first_tokens, first_pooled, _ = enc(obs)
        second_tokens, second_pooled, _ = enc(obs)
    assert torch.equal(first_tokens, second_tokens)
    assert torch.equal(first_pooled, second_pooled)


def test_every_obs_field_influences_output():
    """Every declared OBS_FIELD must move the pooled context when perturbed alone.

    The durable guard against a field the environment declares and populates but
    the encoder silently drops: it would leave the output unchanged here and fail.
    Sweeps the live OBS_FIELDS registry (no hardcoded field list) against an
    all-live baseline (so no perturbation is hidden behind a PAD mask), asserting
    both the per-token embeddings and the pooled context change. Skips nothing.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    enc.eval()
    baseline = _all_live_obs(BATCH)
    with torch.no_grad():
        base_tokens, base_pooled, _ = enc(baseline)

    ignored: list[str] = []
    for field in interface.OBS_FIELDS:
        obs = {name: t.clone() for name, t in baseline.items()}
        obs[field.name] = _perturb_field(field, BATCH, live=True)
        with torch.no_grad():
            tokens, pooled, _ = enc(obs)
        if torch.equal(tokens, base_tokens) or torch.equal(pooled, base_pooled):
            ignored.append(field.name)

    assert not ignored, (
        "OBS_FIELDS declared/populated by the env but not reflected in the encoder "
        f"output when perturbed alone: {ignored}"
    )


# The three per-slot relic token types, whose empty marker is the relic INVALID
# sentinel; each has its own tests here for the AKABEKO-distinguishable guard.
_RELIC_ENTITY_SPECS = [spec for spec in _ENTITY_SPECS if spec.pad_is_relic_invalid]


@pytest.mark.parametrize("spec", _RELIC_ENTITY_SPECS, ids=lambda s: s.name)
def test_offered_relic_distinguishable_from_empty_slot(spec):
    """A real offered relic (incl. AKABEKO, id 0) is unmasked; an empty slot is masked.

    Regression for the INVALID-as-empty design: RelicId 0 (AKABEKO) is a real
    relic that must be distinguishable from an empty slot. An empty relic slot is
    the INVALID sentinel (N_RELIC_IDS) -> masked; AKABEKO in slot 0 -> unmasked.
    """
    enc = ObsFeatureEncoder()
    obs = _all_pad_obs(BATCH)  # every relic slot INVALID (masked)
    start, count = enc.token_layout[spec.name]
    _, _, mask_empty = enc(obs)
    assert bool(mask_empty[:, start : start + count].all())  # all empty -> all masked

    obs[spec.id_field][:, 0] = 0  # AKABEKO, a real relic
    _, _, mask_akabeko = enc(obs)
    assert bool((~mask_akabeko[:, start]).all())  # AKABEKO slot unmasked
    if count > 1:
        assert bool(mask_akabeko[:, start + 1 : start + count].all())  # others still masked


def test_token_widths_derive_from_interface_constants():
    """Every projection input width is the interface-derived feature width, no literal."""
    enc = ObsFeatureEncoder()
    _scalar = 1
    expected_feat_dim = {
        "HAND": CARD_EMBED_DIM + interface.HAND_FEAT_DIM,
        "ENEMY": (
            ENEMY_EMBED_DIM
            + interface.ENEMY_SCALAR_DIM
            + MOVE_EMBED_DIM
            + _scalar
            + interface.N_MONSTER_POWER_IDS
            + _scalar
        ),
        "POTION": POTION_EMBED_DIM + _scalar,
        "CARD_SELECT": CARD_EMBED_DIM,
        "REWARD_CARD": CARD_EMBED_DIM,
        "REWARD_POTION": POTION_EMBED_DIM,
        "REWARD_RELIC": RELIC_EMBED_DIM,
        "SHOP_CARD": CARD_EMBED_DIM + _scalar,
        "SHOP_RELIC": RELIC_EMBED_DIM + _scalar,
        "SHOP_POTION": POTION_EMBED_DIM + _scalar,
        "BOSS_RELIC": RELIC_EMBED_DIM,
    }
    for spec in _ENTITY_SPECS:
        assert enc.entity_proj[spec.name].in_features == expected_feat_dim[spec.name]
        assert enc.entity_proj[spec.name].out_features == HIDDEN_DIM

    assert enc.pile_proj.in_features == _PILE_POOLS * CARD_EMBED_DIM
    # CLS folds the six global blocks; OFFER folds map / event / event-phase / Neow.
    expected_cls = (
        interface.PLAYER_SCALAR_DIM
        + interface.N_PLAYER_POWER_IDS
        + interface.N_RELIC_IDS
        + interface.N_SCREENS
        + interface.KEYS_ACT_DIM
        + _scalar
    )
    expected_offer = (
        interface.MAP_CONTEXT_DIM
        + interface.N_EVENT_IDS
        + interface.MAX_NEOW_OPTIONS * interface.N_NEOW_BONUS
        + interface.MAX_NEOW_OPTIONS * interface.N_NEOW_DRAWBACK
        + interface.EVENT_PHASE_DIM
        + interface.MAP_LOOKAHEAD_DIM
    )
    assert _CLS_INPUT_DIM == expected_cls
    assert _OFFER_INPUT_DIM == expected_offer
    assert enc.cls_mlp[0].in_features == expected_cls
    assert enc.offer_mlp[0].in_features == expected_offer
    # Type embedding covers every token type; slot-id table is sized to the widest.
    assert enc.type_embed.num_embeddings == len(_ENTITY_SPECS) + 1 + len(_PILE_SPECS) + 1
    assert enc.slot_id_embed.num_embeddings == max(spec.count for spec in _ENTITY_SPECS)


def test_transformer_shape_constants_match_design():
    """The transformer runs at the design's d_model / layers / heads / d_ff."""
    enc = ObsFeatureEncoder()
    assert enc.d_model == HIDDEN_DIM == 128
    assert len(enc.blocks) == N_TRANSFORMER_LAYERS == 3
    assert N_ATTENTION_HEADS == 4
    assert HIDDEN_DIM % N_ATTENTION_HEADS == 0
    assert FFN_DIM == 4 * HIDDEN_DIM


def test_non_divisible_d_model_raises():
    """A d_model not divisible by the head count fails with a clear error."""
    with pytest.raises(ValueError, match="divisible by"):
        ObsFeatureEncoder(d_model=HIDDEN_DIM + 1)
