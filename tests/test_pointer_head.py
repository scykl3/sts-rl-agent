"""Tests for the pointer-style policy head.

Driven WITHOUT the engine: per-token embeddings and the padding mask come from
the shared encoder run on samples of the interface's own observation space, and
masks are synthesized to respect ``assert_valid_mask`` (at least one legal action
per row). Every action-slice and token offset is recomputed symbolically from the
interface ``ACTION_BLOCKS`` / ``REWARD_*_OFFSET`` registry and the encoder's
``_ENTITY_SPECS`` token order, never hardcoded, so an interface bump moves the
tests and the code together.

The two mapping tests (``test_two_factor_scatter_*`` and
``test_block_slot_mapping_*``) are the action-slot-mismap safety net: a pointer
score scattered to the wrong index is silent (the logits look healthy while
describing a different action than the mask and decode). They use the interface
layout and the engine decode's ``divmod`` packing as the oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sts_rl import interface
from sts_rl.agent.encoder import HIDDEN_DIM, ObsFeatureEncoder, _ENTITY_SPECS
from sts_rl.agent.policy_head import MASKED_LOGIT, PointerPolicyHead
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    MAX_BOSS_RELICS,
    MAX_ENEMIES,
    MAX_REWARD_CARD_SLOTS,
    MAX_REWARD_POTIONS,
    MAX_REWARD_RELICS,
    MAX_SHOP_CARDS,
    MAX_SHOP_POTIONS,
    MAX_SHOP_RELICS,
    REWARD_CARD_OFFSET,
    REWARD_GOLD_OFFSET,
    REWARD_KEY_OFFSET,
    REWARD_POTION_OFFSET,
    REWARD_RELIC_OFFSET,
    REWARD_SINGING_BOWL_OFFSET,
    REWARD_SKIP_OFFSET,
    InterfaceError,
)
from conftest import sample_observation_batch

# The shop/boss cross-oracle imports env/run_actions.py, which pulls in the native
# engine; the rest of this module is engine-free and must still run without a build.
# Probe once and skip just that test when the engine is absent (matching the suite's
# ImportError -> skip idiom).
try:
    import sts_rl.env._engine  # noqa: F401

    _ENGINE_BUILT = True
except ImportError:  # pragma: no cover - exercised only without a build
    _ENGINE_BUILT = False

BATCH = 3

# Relic-backed entity id fields whose empty marker is the relic INVALID sentinel
# (N_RELIC_IDS), not PAD_ID: RelicId 0 (AKABEKO) is a real relic.
_RELIC_ID_FIELDS = frozenset(spec.id_field for spec in _ENTITY_SPECS if spec.pad_is_relic_invalid)


def _entity_span() -> dict[str, tuple[int, int]]:
    """(start, count) of each entity token group in the encoder's per-token output.

    Recomputed from ``_ENTITY_SPECS`` (the encoder's single source of layout
    truth), so it tracks the encoder without hardcoding token offsets.
    """
    span: dict[str, tuple[int, int]] = {}
    cursor = 0
    for spec in _ENTITY_SPECS:
        span[spec.name] = (cursor, spec.count)
        cursor += spec.count
    return span


def _all_live_obs(batch: int) -> dict[str, torch.Tensor]:
    """Every entity slot occupied by a real id and every float field zero.

    So every entity token is non-PAD (unmasked) and each obs field feeds a live
    token. Real ids: 5 for the relic fields (a real, non-AKABEKO relic), 1 for the
    other id fields. Mirrors the encoder suite's all-live baseline.
    """
    obs: dict[str, torch.Tensor] = {}
    for field in interface.OBS_FIELDS:
        if field.bounds == "id":
            live = 5 if field.name in _RELIC_ID_FIELDS else 1
            obs[field.name] = torch.full((batch, *field.shape), live, dtype=torch.long)
        else:
            obs[field.name] = torch.zeros((batch, *field.shape), dtype=torch.float32)
    return obs


def _encoded(
    batch: int = BATCH, *, live: bool = False, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch into ``(per_token, key_padding_mask)`` (detached, eval mode)."""
    torch.manual_seed(seed)
    enc = ObsFeatureEncoder()
    enc.eval()
    obs = _all_live_obs(batch) if live else sample_observation_batch(batch)
    with torch.no_grad():
        per_token, _pooled, mask = enc(obs)
    return per_token.detach(), mask.detach()


def _random_valid_mask(batch: int, seed: int = 0) -> torch.Tensor:
    """Random bool mask (B, ACTION_DIM) with >=1 legal action per row.

    Mirrors ``assert_valid_mask``: each row is guaranteed at least one True by
    forcing a random legal index on, so no row is fully masked.
    """
    gen = torch.Generator().manual_seed(seed)
    mask = torch.rand(batch, ACTION_DIM, generator=gen) > 0.5
    forced = torch.randint(0, ACTION_DIM, (batch,), generator=gen)
    mask[torch.arange(batch), forced] = True
    for row in mask:
        interface.assert_valid_mask(row.numpy())
    return mask


def _single_legal_mask(batch: int, index: int) -> torch.Tensor:
    """Mask with exactly one legal action (``index``) for every row."""
    mask = torch.zeros(batch, ACTION_DIM, dtype=torch.bool)
    mask[:, index] = True
    return mask


# --- Shape / dtype / device -------------------------------------------------


def test_forward_shape_and_finite():
    """Output is (B, ACTION_DIM) and finite despite the large-negative floor."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    logits = head(per_token, kpm, _random_valid_mask(BATCH))
    assert logits.shape == (BATCH, ACTION_DIM)
    assert torch.isfinite(logits).all()
    assert logits.dtype == torch.float32
    assert logits.device == per_token.device


def test_input_dim_defaults_to_encoder_output():
    """Head's default input width tracks the encoder output width, not a literal."""
    enc = ObsFeatureEncoder()
    head = PointerPolicyHead()
    assert head.input_dim == enc.output_dim == HIDDEN_DIM
    assert head.action_dim == ACTION_DIM
    assert head.d_k == HIDDEN_DIM


def test_float64_obs_does_not_promote_logits():
    """A float64 obs dict stays float32 through the encoder and head (no promotion)."""
    head = PointerPolicyHead()
    enc = ObsFeatureEncoder()
    id_fields = {f.name for f in interface.OBS_FIELDS if f.bounds == "id"}
    obs = sample_observation_batch(BATCH)
    obs64 = {name: (t if name in id_fields else t.double()) for name, t in obs.items()}
    per_token, _pooled, kpm = enc(obs64)
    logits = head(per_token, kpm, _random_valid_mask(BATCH))
    assert per_token.dtype == torch.float32
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


# --- Finite raw logits (pre-mask) -------------------------------------------


def test_raw_logits_finite_over_all_slots():
    """Every one of the 257 raw slots (incl. PAD-entity pointers and PROCEED@256) is finite.

    The raw tensor is zero-allocated, so an unscored slot is a finite 0; this test
    catches a scatter that wrote a NaN/inf (a degenerate pointer score) before it
    could reach Categorical, which masking would otherwise hide.
    """
    head = PointerPolicyHead()
    # A sampled obs (mixed live and PAD entity tokens) still yields finite raw logits: PAD
    # tokens are finite query positions, and the finite-guard covers all 257 slots.
    per_token, kpm = _encoded()
    raw = head.raw_logits(per_token, kpm)
    assert raw.shape == (BATCH, ACTION_DIM)
    assert torch.isfinite(raw).all()
    proceed = ACTION_BLOCK_BY_NAME["PROCEED"].start
    assert torch.isfinite(raw[:, proceed]).all()  # PROCEED@256 scored, always masked downstream


# --- CRITICAL: two-factor targeted scatter ----------------------------------


@pytest.mark.parametrize(
    "block_name, source_name",
    [("PLAY_CARD_TARGETED", "HAND"), ("USE_POTION_TARGETED", "POTION")],
)
def test_two_factor_scatter_is_source_major(block_name, source_name):
    """G[source, enemy] is written at block.start + source*MAX_ENEMIES + enemy.

    Cross-checked against the engine decode's ``divmod(offset, MAX_ENEMIES)``
    (env/actions.py:86 and :101): source is the slow axis, enemy the fast axis.
    Recomputes the interaction grid from the head's OWN projections and asserts the
    scattered slot equals G[i, j] at exactly index i*MAX_ENEMIES + j, for both the
    card and potion targeted blocks. Non-vacuous: an enemy-major flatten or a
    block-start off-by-one moves the score to a different slot and fails here.
    """
    head = PointerPolicyHead()
    head.eval()
    span = _entity_span()
    src_start, src_count = span[source_name]
    en_start, en_count = span["ENEMY"]
    assert en_count == MAX_ENEMIES  # the divmod modulus
    block = ACTION_BLOCK_BY_NAME[block_name]

    per_token, kpm = _encoded(live=True)
    with torch.no_grad():
        raw = head.raw_logits(per_token, kpm)
        source = per_token[:, src_start : src_start + src_count]
        enemy = per_token[:, en_start : en_start + en_count]
        q = head.targeted_query[block_name](source)
        k = head.targeted_key[block_name](enemy)
        grid = torch.matmul(q, k.transpose(1, 2)) * head.scale  # (B, src, enemy)

    assert block.count == src_count * en_count
    for i in range(src_count):
        for j in range(en_count):
            offset = i * MAX_ENEMIES + j
            # The decode oracle: this exact packing is recovered by divmod.
            assert divmod(offset, MAX_ENEMIES) == (i, j)
            scattered = raw[:, block.start + offset]
            assert torch.allclose(
                scattered, grid[:, i, j], atol=1e-6
            ), f"{block_name}: source {i} enemy {j} not at block.start+{offset}"


# --- CRITICAL: block-slot mapping coverage ----------------------------------


def _single_factor_expected() -> list[tuple[str, int, int, str]]:
    """(query_name, action_start, count, token_name) for every single-factor block.

    Built independently of the head from the interface registry so it is a true
    oracle for the head's ``_single_blocks``.
    """
    blocks = ACTION_BLOCK_BY_NAME
    reward = blocks["REWARD_SELECT"].start
    shop = blocks["SHOP_SELECT"].start
    boss = blocks["BOSS_RELIC_SELECT"].start
    return [
        ("PLAY_CARD_UNTARGETED", blocks["PLAY_CARD_UNTARGETED"].start, interface.HAND_MAX, "HAND"),
        (
            "USE_POTION_UNTARGETED",
            blocks["USE_POTION_UNTARGETED"].start,
            interface.POTION_SLOTS,
            "POTION",
        ),
        ("DISCARD_POTION", blocks["DISCARD_POTION"].start, interface.POTION_SLOTS, "POTION"),
        ("CARD_SELECT", blocks["CARD_SELECT"].start, interface.CHOICE_MAX, "CARD_SELECT"),
        ("REWARD_POTION", reward + REWARD_POTION_OFFSET, MAX_REWARD_POTIONS, "REWARD_POTION"),
        ("REWARD_RELIC", reward + REWARD_RELIC_OFFSET, MAX_REWARD_RELICS, "REWARD_RELIC"),
        ("REWARD_CARD", reward + REWARD_CARD_OFFSET, MAX_REWARD_CARD_SLOTS, "REWARD_CARD"),
        ("SHOP_CARD", shop, MAX_SHOP_CARDS, "SHOP_CARD"),
        ("SHOP_RELIC", shop + MAX_SHOP_CARDS, MAX_SHOP_RELICS, "SHOP_RELIC"),
        ("SHOP_POTION", shop + MAX_SHOP_CARDS + MAX_SHOP_RELICS, MAX_SHOP_POTIONS, "SHOP_POTION"),
        ("BOSS_RELIC", boss, MAX_BOSS_RELICS, "BOSS_RELIC"),
    ]


def test_block_slot_mapping_single_factor():
    """Each entity-addressable block's k-th token scores at its k-th action index.

    Table-driven over every single-factor pointer block (REWARD_CARD m at start+m,
    SHOP_RELIC r at start+r, and so on). Recomputes each pointer score from the
    head's shared key projection and per-action-type query, then asserts the
    scattered action slot at start+k equals the score of token k. Non-vacuous: an
    off-by-one scatter puts an adjacent token's (distinct) score at start+k and
    fails.
    """
    head = PointerPolicyHead()
    head.eval()
    span = _entity_span()
    per_token, kpm = _encoded(live=True)
    with torch.no_grad():
        raw = head.raw_logits(per_token, kpm)
        entity_keys = head.pointer_key(per_token[:, : head.n_entity_tokens])

    # The head's derived table must equal the independently built oracle table.
    assert head._single_blocks == [
        (name, start, count, span[token][0])
        for name, start, count, token in _single_factor_expected()
    ]

    for name, action_start, count, token_name in _single_factor_expected():
        token_start = span[token_name][0]
        keys = entity_keys[:, token_start : token_start + count]
        query = head.single_query[name]
        scores = torch.matmul(keys, query) * head.scale  # (B, count)
        for k in range(count):
            assert torch.allclose(
                raw[:, action_start + k], scores[:, k], atol=1e-6
            ), f"{name}: token {k} not scored at action index {action_start + k}"


def _learned_index_oracle() -> list[int]:
    """The learned-query action indices, built independently from the registry."""
    blocks = ACTION_BLOCK_BY_NAME
    reward = blocks["REWARD_SELECT"].start
    shop = blocks["SHOP_SELECT"].start
    boss = blocks["BOSS_RELIC_SELECT"].start
    shop_remove = shop + MAX_SHOP_CARDS + MAX_SHOP_RELICS + MAX_SHOP_POTIONS
    return [
        blocks["END_TURN"].start,
        blocks["CONFIRM_SELECT"].start,
        reward + REWARD_GOLD_OFFSET,
        reward + REWARD_KEY_OFFSET,
        reward + REWARD_SINGING_BOWL_OFFSET,
        reward + REWARD_SKIP_OFFSET,
        *range(blocks["MAP_SELECT"].start, blocks["MAP_SELECT"].stop),
        *range(blocks["REST_SELECT"].start, blocks["REST_SELECT"].stop),
        *range(blocks["TREASURE_SELECT"].start, blocks["TREASURE_SELECT"].stop),
        *range(blocks["EVENT_SELECT"].start, blocks["EVENT_SELECT"].stop),
        shop_remove,
        shop_remove + 1,
        boss + MAX_BOSS_RELICS,
        blocks["PROCEED"].start,
    ]


def test_block_slot_mapping_learned_queries():
    """Each learned-query bank row scores at its fixed action index.

    The head's ``learned_action_index`` must equal the registry-derived oracle,
    and the k-th bank row's logit must land at ``learned_action_index[k]``.
    Recomputes the bank cross-attention from the head's own modules and compares
    to the scattered slots. Non-vacuous: a permuted or shifted scatter fails.
    """
    head = PointerPolicyHead()
    head.eval()
    expected = _learned_index_oracle()
    assert head.learned_action_index.tolist() == expected

    per_token, kpm = _encoded(live=True)
    with torch.no_grad():
        raw = head.raw_logits(per_token, kpm)
        queries = head.learned_queries.unsqueeze(0).expand(per_token.shape[0], -1, -1)
        attended, _ = head.learned_attn(
            queries, per_token, per_token, key_padding_mask=kpm, need_weights=False
        )
        learned_logits = head.learned_logit(attended).squeeze(-1)  # (B, n_learned)

    for k, action_index in enumerate(expected):
        assert torch.allclose(
            raw[:, action_index], learned_logits[:, k], atol=1e-6
        ), f"learned row {k} not scored at action index {action_index}"


def test_scoring_sources_partition_the_action_space():
    """The three scoring sources cover every action index exactly once.

    Mirrors the head's construction-time invariant as an explicit test: the union
    of single-factor, two-factor, and learned indices equals range(ACTION_DIM)
    with no duplicate.
    """
    span = _entity_span()
    covered: list[int] = []
    for _name, start, count, _token in _single_factor_expected():
        covered.extend(range(start, start + count))
    for block_name, source_name in [
        ("PLAY_CARD_TARGETED", "HAND"),
        ("USE_POTION_TARGETED", "POTION"),
    ]:
        block = ACTION_BLOCK_BY_NAME[block_name]
        _src_start, src_count = span[source_name]
        covered.extend(range(block.start, block.start + src_count * MAX_ENEMIES))
    covered.extend(_learned_index_oracle())
    assert sorted(covered) == list(range(ACTION_DIM))


@pytest.mark.skipif(not _ENGINE_BUILT, reason="engine not built")
def test_shop_boss_offsets_match_env_decoder():
    """Cross-oracle the head's shop/boss sub-offsets against the env decoder.

    The head derives its shop offsets from MAX_SHOP_CARDS/RELICS/POTIONS and its
    boss skip from MAX_BOSS_RELICS; ``env/run_actions.py`` packs the same decode
    from its OWN independent constants. They agree today, but neither the head
    tests (interface caps on both sides) nor the env tests (env locals on both
    sides) would catch a divergence, and a shop-slot mismap is the silent failure
    the design flags as highest risk. Asserting the head's actual scatter
    destinations equal the env decoder's offsets turns "consistent by
    construction" into a gated cross-oracle check.
    """
    from sts_rl.env import run_actions as ra

    head = PointerPolicyHead()
    single_start = {name: start for name, start, _count, _token in head._single_blocks}
    learned = set(head.learned_action_index.tolist())
    shop_start = ACTION_BLOCK_BY_NAME["SHOP_SELECT"].start
    boss_start = ACTION_BLOCK_BY_NAME["BOSS_RELIC_SELECT"].start

    # Single-factor shop sub-blocks: the head's scatter start vs the env offset.
    assert single_start["SHOP_CARD"] == shop_start
    assert single_start["SHOP_RELIC"] == shop_start + ra._SHOP_RELIC_OFFSET
    assert single_start["SHOP_POTION"] == shop_start + ra._SHOP_POTION_OFFSET
    # Shop remove/skip and boss skip are learned-query slots, each scattered once,
    # so env-offset membership is an exact-position cross-check.
    assert shop_start + ra._SHOP_REMOVE_OFFSET in learned
    assert shop_start + ra._SHOP_SKIP_OFFSET in learned
    assert boss_start + ra._BOSS_RELIC_COUNT in learned


# --- Slot stability: HAND swap ----------------------------------------------


def test_hand_slot_swap_swaps_exactly_the_two_play_logits():
    """Swapping two HAND token rows swaps exactly their PLAY logits, nothing else.

    Operates at the head level (swapping per-token rows directly) to isolate the
    scatter: pointer scoring is per token, so swapping HAND tokens 0 and 1 swaps
    their PLAY_CARD_UNTARGETED logits and their PLAY_CARD_TARGETED source blocks.
    The learned-query cross-attention is order-invariant over its key set, and no
    other block reads HAND tokens, so every other logit is unchanged.
    """
    head = PointerPolicyHead()
    head.eval()
    span = _entity_span()
    hand_start, _ = span["HAND"]
    per_token, kpm = _encoded(live=True)  # both HAND slots non-PAD (mask False)

    swapped = per_token.clone()
    swapped[:, hand_start] = per_token[:, hand_start + 1]
    swapped[:, hand_start + 1] = per_token[:, hand_start]

    with torch.no_grad():
        raw = head.raw_logits(per_token, kpm)
        raw_sw = head.raw_logits(swapped, kpm)

    pu = ACTION_BLOCK_BY_NAME["PLAY_CARD_UNTARGETED"].start
    # The two untargeted PLAY logits swap.
    assert torch.allclose(raw_sw[:, pu], raw[:, pu + 1], atol=1e-5)
    assert torch.allclose(raw_sw[:, pu + 1], raw[:, pu], atol=1e-5)
    # Remaining untargeted HAND slots are unchanged.
    assert torch.allclose(
        raw_sw[:, pu + 2 : pu + interface.HAND_MAX],
        raw[:, pu + 2 : pu + interface.HAND_MAX],
        atol=1e-5,
    )

    pt = ACTION_BLOCK_BY_NAME["PLAY_CARD_TARGETED"].start
    e = MAX_ENEMIES
    # The targeted source-0 and source-1 blocks swap (each MAX_ENEMIES wide).
    assert torch.allclose(raw_sw[:, pt : pt + e], raw[:, pt + e : pt + 2 * e], atol=1e-5)
    assert torch.allclose(raw_sw[:, pt + e : pt + 2 * e], raw[:, pt : pt + e], atol=1e-5)

    # A learned-query slot (END_TURN) is unchanged: attention is order-invariant
    # over the swapped key set.
    end_turn = ACTION_BLOCK_BY_NAME["END_TURN"].start
    assert torch.allclose(raw_sw[:, end_turn], raw[:, end_turn], atol=1e-5)


# --- Obs-field influence ----------------------------------------------------


def test_every_obs_field_influences_raw_logits():
    """Perturbing each obs field alone moves the raw logits.

    The head-side face of the encoder's field-coverage guard: every field feeds a
    token, and every token influences the raw logits (entity tokens through pointer
    scoring, all tokens through the learned-query cross-attention). Explicitly
    covers shop_relic_prices and the per-slot offered-relic ids. Sweeps the live
    OBS_FIELDS registry against an all-live baseline (no perturbation hidden behind
    a PAD mask); skips nothing.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    enc.eval()
    head = PointerPolicyHead()
    head.eval()
    baseline = _all_live_obs(BATCH)
    with torch.no_grad():
        base_token, _pooled, base_mask = enc(baseline)
        base_raw = head.raw_logits(base_token, base_mask)

    ignored: list[str] = []
    for field in interface.OBS_FIELDS:
        obs = {name: t.clone() for name, t in baseline.items()}
        if field.bounds == "id":
            base = 5 if field.name in _RELIC_ID_FIELDS else 1
            obs[field.name] = torch.full((BATCH, *field.shape), base + 1, dtype=torch.long)
        else:
            value = 0.5 if field.bounds == "unit" else 1.0
            obs[field.name] = torch.full((BATCH, *field.shape), value, dtype=torch.float32)
        with torch.no_grad():
            token, _pooled, mask = enc(obs)
            raw = head.raw_logits(token, mask)
        if torch.equal(raw, base_raw):
            ignored.append(field.name)

    assert not ignored, f"obs fields not reflected in the pointer head raw logits: {ignored}"


# --- PAD leakage: learned-query cross-attention mask (revert-verify) ---------


def test_learned_query_logits_do_not_leak_pad_tokens():
    """A PAD entity token must not influence any learned-query logit.

    The head's ``learned_attn`` ``key_padding_mask`` is what enforces it: the
    learned-query bank re-attends over the FULL token set, including PAD entity
    rows whose encoder outputs still vary with their PAD raw features, so without
    the mask a PAD token leaks into the always-legal learned-query logits (END_TURN
    and the other tokenless slots). Revert-verify: perturbing a masked PAD slot
    leaves END_TURN unchanged WITH the mask, and this fails if ``key_padding_mask=``
    is dropped from the ``learned_attn`` call. The encoder's own no-leakage test
    never routes through the head, so it would stay green under that regression.
    """
    torch.manual_seed(0)
    enc = ObsFeatureEncoder()
    enc.eval()
    head = PointerPolicyHead()
    head.eval()
    obs = _all_live_obs(BATCH)
    obs["hand_ids"][:, 0] = interface.PAD_ID  # HAND slot 0 (token 0) -> PAD (masked)
    pert = {k: v.clone() for k, v in obs.items()}
    pert["hand_feats"][:, 0, :] = 9.0  # perturb only the PAD slot's raw features
    end_turn = ACTION_BLOCK_BY_NAME["END_TURN"].start
    with torch.no_grad():
        token, _pooled, mask = enc(obs)
        pert_token, _pert_pooled, pert_mask = enc(pert)
        assert bool(mask[:, 0].all())  # slot 0 really is PAD across the batch
        base = head.raw_logits(token, mask)[:, end_turn]
        perturbed = head.raw_logits(pert_token, pert_mask)[:, end_turn]
    assert torch.allclose(base, perturbed, atol=1e-6)  # fails if the mask is dropped


# --- Gradient flow ----------------------------------------------------------


def test_gradient_flows_to_every_pointer_parameter():
    """One backward reaches every pointer query/key and learned-query-bank entry.

    Uses an all-live obs so every token type is non-PAD, activating every pointer
    block (card-select, shop, reward, boss tokens) and the learned-query bank.
    Backprops the raw (pre-mask) logits so no legal-action masking zeroes a slot's
    gradient before it reaches a query/key.
    """
    torch.manual_seed(0)
    head = PointerPolicyHead()
    per_token, kpm = _encoded(live=True)
    per_token = per_token.clone().requires_grad_(True)
    head.raw_logits(per_token, kpm).sum().backward()

    missing = [name for name, p in head.named_parameters() if p.grad is None]
    nonfinite = [
        name
        for name, p in head.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not missing, f"pointer parameters with no gradient: {missing}"
    assert not nonfinite, f"pointer parameters with non-finite gradient: {nonfinite}"


# --- Masked-policy contract (reused verbatim vs the dense head) --------------


def test_illegal_actions_get_zero_probability():
    """softmax mass on illegal actions ~0; legal probs sum to ~1."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    mask = _random_valid_mask(BATCH)
    probs = torch.softmax(head(per_token, kpm, mask), dim=-1)
    assert (probs[~mask] < 1e-6).all()
    assert torch.allclose((probs * mask).sum(dim=-1), torch.ones(BATCH), atol=1e-5)


def test_sampled_actions_are_always_legal():
    """Every sampled action lands on a legal index (large batch of draws)."""
    head = PointerPolicyHead()
    big = 256
    per_token, kpm = _encoded(big, seed=1)
    mask = _random_valid_mask(big, seed=1)
    for _ in range(20):
        actions, _, _ = head.act(per_token, kpm, mask)
        assert mask[torch.arange(big), actions].all()


def test_gradient_flows_through_act():
    """A log-prob loss backprops finite grads into the pointer parameters."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    per_token = per_token.clone().requires_grad_(True)
    _, log_prob, _ = head.act(per_token, kpm, _random_valid_mask(BATCH))
    (-log_prob.mean()).backward()
    grad = head.learned_logit.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_evaluate_actions_matches_distribution():
    """evaluate_actions reproduces the masked distribution's log_prob/entropy."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    mask = _random_valid_mask(BATCH)
    dist = head.masked_distribution(per_token, kpm, mask)
    actions = dist.sample()
    log_prob, entropy = head.evaluate_actions(per_token, kpm, mask, actions)
    assert torch.allclose(log_prob, dist.log_prob(actions))
    assert torch.allclose(entropy, dist.entropy())


def test_deterministic_act_is_argmax_and_reproducible():
    """deterministic=True returns argmax over legal logits, reproducibly."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    mask = _random_valid_mask(BATCH)
    first, _, _ = head.act(per_token, kpm, mask, deterministic=True)
    second, _, _ = head.act(per_token, kpm, mask, deterministic=True)
    expected = head(per_token, kpm, mask).argmax(dim=-1)
    assert torch.equal(first, expected)
    assert torch.equal(first, second)
    assert mask[torch.arange(BATCH), first].all()


def test_entropy_finite_and_nonnegative():
    """Entropy is finite and >= 0 for a valid mask."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    _, _, entropy = head.act(per_token, kpm, _random_valid_mask(BATCH))
    assert torch.isfinite(entropy).all()
    assert (entropy >= 0).all()


def test_single_legal_action_forces_choice_and_zero_entropy():
    """One legal action: entropy ~0 and both sampling modes pick that action."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    index = ACTION_DIM // 2
    mask = _single_legal_mask(BATCH, index)
    sampled, log_prob, entropy = head.act(per_token, kpm, mask)
    greedy, _, _ = head.act(per_token, kpm, mask, deterministic=True)
    assert (sampled == index).all()
    assert (greedy == index).all()
    assert torch.allclose(entropy, torch.zeros(BATCH), atol=1e-5)
    assert torch.allclose(log_prob, torch.zeros(BATCH), atol=1e-5)


def test_illegal_logits_equal_masked_logit():
    """After masking, every illegal slot equals the finite MASKED_LOGIT floor."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    mask = _random_valid_mask(BATCH)
    logits = head(per_token, kpm, mask)
    assert (logits[~mask] == MASKED_LOGIT).all()


def test_fully_masked_single_row_raises():
    """An all-False mask row must raise, not sample uniformly over illegals."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    mask = torch.zeros(BATCH, ACTION_DIM, dtype=torch.bool)
    with pytest.raises(InterfaceError):
        head(per_token, kpm, mask)


def test_one_fully_masked_row_in_batch_raises():
    """A single fully-masked row inside an otherwise-valid batch still raises."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    mask = _random_valid_mask(BATCH)
    mask[1] = False
    with pytest.raises(InterfaceError):
        head(per_token, kpm, mask)


def test_unbatched_mask_raises_instead_of_broadcasting():
    """A (ACTION_DIM,) mask must raise, not broadcast over (B, ACTION_DIM)."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    unbatched = _random_valid_mask(1)[0]
    assert unbatched.shape == (ACTION_DIM,)
    with pytest.raises(InterfaceError):
        head(per_token, kpm, unbatched)


def test_device_mismatch_raises_domain_error():
    """A mask on a different device raises InterfaceError, not a raw torch error."""
    head = PointerPolicyHead()
    per_token, kpm = _encoded()
    meta_mask = torch.ones(BATCH, ACTION_DIM, dtype=torch.bool, device="meta")
    with pytest.raises(InterfaceError):
        head(per_token, kpm, meta_mask)


def test_non_bool_mask_is_coerced():
    """An int/float 0/1 mask is accepted and matches the bool-mask result."""
    head = PointerPolicyHead()
    head.eval()
    per_token, kpm = _encoded()
    bool_mask = _random_valid_mask(BATCH)
    expected = head(per_token, kpm, bool_mask)
    assert torch.equal(head(per_token, kpm, bool_mask.to(torch.int64)), expected)
    assert torch.equal(head(per_token, kpm, bool_mask.to(torch.float32)), expected)


def test_masked_logit_is_finite_and_negative():
    """The masking constant is a large finite negative (never -inf)."""
    assert np.isfinite(MASKED_LOGIT)
    assert MASKED_LOGIT < 0


# --- Eval-mode determinism --------------------------------------------------


def test_eval_mode_determinism():
    """eval() plus a fixed encode gives identical logits/actions across two forwards."""
    head = PointerPolicyHead()
    head.eval()
    per_token, kpm = _encoded()
    mask = _random_valid_mask(BATCH)
    with torch.no_grad():
        first = head(per_token, kpm, mask)
        second = head(per_token, kpm, mask)
    assert torch.equal(first, second)
    a1, _, _ = head.act(per_token, kpm, mask, deterministic=True)
    a2, _, _ = head.act(per_token, kpm, mask, deterministic=True)
    assert torch.equal(a1, a2)


# --- Widths recomputed symbolically -----------------------------------------


def test_n_entity_tokens_matches_interface_constants():
    """The head's entity-token count is the interface-derived sum, no literal."""
    head = PointerPolicyHead()
    expected = (
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
    assert head.n_entity_tokens == expected
    # And it agrees with the encoder it consumes.
    assert head.n_entity_tokens == ObsFeatureEncoder().n_entity_tokens
