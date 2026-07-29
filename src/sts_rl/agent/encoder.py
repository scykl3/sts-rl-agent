"""Shared observation encoder for the Slay the Spire RL agent.

A transformer over game-entity and context tokens. It consumes the observation
dict defined by the shared interface (its ``OBS_FIELDS`` registry) and produces
per-token embeddings plus a pooled context vector that the heads consume. Action
masking is deliberately NOT applied here - it belongs downstream in the policy
head. The encoder produces head-agnostic features only.

Every dimension is derived from the interface constants, so if the engine's enum
sizes change the token widths follow automatically; no observation-width literals
live here.

Token schema (fixed sequence order, so the per-token output is pointer
addressable by a later per-entity head):

Entity tokens - one per action-mapped slot, projected to ``d_model`` by a
per-type input linear, then summed with a learned type embedding (which entity
kind) and a learned slot-id embedding (which slot). A token is PAD when its id
field holds the empty marker (``PAD_ID`` for the card/potion/enemy/move-backed
tokens; the relic INVALID sentinel ``N_RELIC_IDS`` for the relic tokens, since
``RelicId`` 0 is a real relic and cannot double as empty):

    HAND          HAND_MAX             card_embed + hand_feats
    ENEMY         MAX_ENEMIES          enemy_embed + scalars + move_embed
                                       + intent_hidden + powers + alive
    POTION        POTION_SLOTS         potion_embed + usable
    CARD_SELECT   CHOICE_MAX           card_embed
    REWARD_CARD   MAX_REWARD_CARD_SLOTS card_embed
    REWARD_POTION MAX_REWARD_POTIONS   potion_embed
    REWARD_RELIC  MAX_REWARD_RELICS    relic_embed
    SHOP_CARD     MAX_SHOP_CARDS       card_embed + price
    SHOP_RELIC    MAX_SHOP_RELICS      relic_embed + price
    SHOP_POTION   MAX_SHOP_POTIONS     potion_embed + price
    BOSS_RELIC    MAX_BOSS_RELICS      relic_embed

Context tokens - never PAD, so every attention row has at least one valid key
(the attention-side analogue of the action-side all-illegal guard):

    CLS           1  MLP over player_scalars / player_powers / owned
                     relics_multihot / screen_onehot / keys_act / shop_remove_cost
    PILE (x4)     4  draw / discard / exhaust / deck, each mean+max pooled over
                     card_embed (order-agnostic, so pooled, not per-slot)
    OFFER_CONTEXT 1  MLP over map_context / event_onehot / event_phase_onehot /
                     neow_bonus / neow_drawback

No sinusoidal positional encoding: attention is permutation-equivariant (correct
for the pooled order-agnostic sets) and slotted types are disambiguated by the
slot-id embedding, BERT-style (type embedding = segment, slot-id = position).

The padding mask is a boolean ``key_padding_mask`` of shape ``(B, S)``, True ==
ignore (PyTorch ``nn.MultiheadAttention`` convention), True exactly where an
entity token's id equals its empty marker and always False for the context
tokens. Because the context tokens are always valid, no row is all-PAD, so the
attention softmax never sees an all-``-inf`` row.

``forward`` returns ``(per_token, pooled_cls, key_padding_mask)``: the per-token
embeddings ``(B, S, d_model)`` (for a later per-entity head), the pooled ``CLS``
vector ``(B, d_model)`` (read by the value and policy heads), and the mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sts_rl.interface import (
    CHOICE_MAX,
    ENEMY_SCALAR_DIM,
    EVENT_PHASE_DIM,
    HAND_FEAT_DIM,
    HAND_MAX,
    KEYS_ACT_DIM,
    MAP_CONTEXT_DIM,
    MAX_BOSS_RELICS,
    MAX_ENEMIES,
    MAX_NEOW_OPTIONS,
    MAX_REWARD_CARD_SLOTS,
    MAX_REWARD_POTIONS,
    MAX_REWARD_RELICS,
    MAX_SHOP_CARDS,
    MAX_SHOP_POTIONS,
    MAX_SHOP_RELICS,
    N_CARD_IDS,
    N_EVENT_IDS,
    N_MONSTER_IDS,
    N_MONSTER_MOVE_IDS,
    N_MONSTER_POWER_IDS,
    N_NEOW_BONUS,
    N_NEOW_DRAWBACK,
    N_PLAYER_POWER_IDS,
    N_POTION_IDS,
    N_RELIC_IDS,
    N_SCREENS,
    OBS_FIELDS,
    PAD_ID,
    PLAYER_SCALAR_DIM,
    POTION_SLOTS,
)

# --- Embedding widths ------------------------------------------------------
# Learned id embeddings, sized by how much each id space needs to carry: cards
# drive most decisions (widest), monsters / relics fewer, potions fewest.
CARD_EMBED_DIM = 32
ENEMY_EMBED_DIM = 16
MOVE_EMBED_DIM = 16  # enemy next-move (MonsterMoveId) embedding
POTION_EMBED_DIM = 8
RELIC_EMBED_DIM = 16  # offered-relic (RelicId) embedding, agent-internal

# --- Transformer shape -----------------------------------------------------
# ``d_model`` is exported as HIDDEN_DIM because it is the single model-width
# setting the training config threads through (``TrainConfig.hidden_dim`` ->
# ``ActorCritic`` -> encoder), and the value read by both heads. It replaces the
# old flat trunk's 512-wide hidden layer; the transformer runs at 128.
HIDDEN_DIM = 128  # d_model
N_TRANSFORMER_LAYERS = 3  # matches AlphaStar's 3-layer unit transformer
N_ATTENTION_HEADS = 4  # head dim = HIDDEN_DIM / N_ATTENTION_HEADS = 32
FFN_DIM = 4 * HIDDEN_DIM  # standard 4x feed-forward expansion (512)
TRANSFORMER_DROPOUT = 0.0  # on-policy RL; enable only if overfitting the value target

# nanoGPT/minGPT weight-init std. Residual-output projections are further scaled
# by 1/sqrt(2 * N_TRANSFORMER_LAYERS) (GPT-2's residual-accumulation scaling).
_INIT_STD = 0.02

# Piles are pooled with BOTH mean and max, so each pile summary is 2 * CARD_EMBED_DIM.
_PILE_POOLS = 2

# A single scalar feature (a price, a usable/alive/intent flag), named so the
# per-type feature widths below read as "embedding + one scalar", not "+ 1".
_SCALAR = 1

# Observation fields whose dtype is an id (embedding index), derived from the
# interface so it tracks the registry. _coerce_dtypes casts every non-id field
# to float32; ids keep their integer dtype and are cast to long at the lookups.
_ID_FIELDS = frozenset(f.name for f in OBS_FIELDS if f.bounds == "id")


@dataclass(frozen=True)
class _EntitySpec:
    """Fixed description of one per-slot entity token type.

    ``count`` slots come from an interface cap; ``id_field`` is the obs id that
    drives both the primary embedding lookup and the PAD mask; ``feat_dim`` is
    the raw feature width fed to the per-type input linear (an expression of the
    embedding-width constants, never a bare literal). ``pad_is_relic_invalid``
    selects the empty marker: the relic INVALID sentinel ``N_RELIC_IDS`` for the
    relic-backed tokens (``RelicId`` 0 is a real relic), else ``PAD_ID``.
    """

    name: str
    count: int
    id_field: str
    feat_dim: int
    pad_is_relic_invalid: bool

    @property
    def pad_value(self) -> int:
        return N_RELIC_IDS if self.pad_is_relic_invalid else PAD_ID


# Entity token types, in fixed sequence order. The order and per-type slot count
# are load-bearing: a later per-entity head addresses tokens by this layout, and
# the action layout maps to the hand / enemy / offered-item slot order. Do NOT
# reorder. Every width is an interface constant (plus the named embedding dims).
_ENTITY_SPECS: tuple[_EntitySpec, ...] = (
    _EntitySpec("HAND", HAND_MAX, "hand_ids", CARD_EMBED_DIM + HAND_FEAT_DIM, False),
    _EntitySpec(
        "ENEMY",
        MAX_ENEMIES,
        "enemy_ids",
        # id embed | scalars | move embed | intent-hidden flag | powers | alive flag
        ENEMY_EMBED_DIM
        + ENEMY_SCALAR_DIM
        + MOVE_EMBED_DIM
        + _SCALAR
        + N_MONSTER_POWER_IDS
        + _SCALAR,
        False,
    ),
    _EntitySpec("POTION", POTION_SLOTS, "potion_ids", POTION_EMBED_DIM + _SCALAR, False),
    _EntitySpec("CARD_SELECT", CHOICE_MAX, "card_select_ids", CARD_EMBED_DIM, False),
    _EntitySpec("REWARD_CARD", MAX_REWARD_CARD_SLOTS, "reward_card_ids", CARD_EMBED_DIM, False),
    _EntitySpec("REWARD_POTION", MAX_REWARD_POTIONS, "reward_potion_ids", POTION_EMBED_DIM, False),
    _EntitySpec("REWARD_RELIC", MAX_REWARD_RELICS, "reward_relic_ids", RELIC_EMBED_DIM, True),
    _EntitySpec("SHOP_CARD", MAX_SHOP_CARDS, "shop_card_ids", CARD_EMBED_DIM + _SCALAR, False),
    _EntitySpec("SHOP_RELIC", MAX_SHOP_RELICS, "shop_relic_ids", RELIC_EMBED_DIM + _SCALAR, True),
    _EntitySpec(
        "SHOP_POTION", MAX_SHOP_POTIONS, "shop_potion_ids", POTION_EMBED_DIM + _SCALAR, False
    ),
    _EntitySpec("BOSS_RELIC", MAX_BOSS_RELICS, "boss_relic_ids", RELIC_EMBED_DIM, True),
)

# Pile summary tokens: (context-token name, obs id field). Each is mean+max pooled
# over card_embed, order-agnostic, so pooled rather than per-slot flattened.
_PILE_SPECS: tuple[tuple[str, str], ...] = (
    ("PILE_DRAW", "draw_ids"),
    ("PILE_DISCARD", "discard_ids"),
    ("PILE_EXHAUST", "exhaust_ids"),
    ("PILE_DECK", "deck_ids"),
)

_CLS_NAME = "CLS"
_OFFER_NAME = "OFFER_CONTEXT"

# All token types, in sequence order: entity tokens first (pointer-addressable),
# then the six never-PAD context tokens. The type-embedding index of each token
# type is its position here.
_CONTEXT_TYPE_NAMES: tuple[str, ...] = (_CLS_NAME, *(name for name, _ in _PILE_SPECS), _OFFER_NAME)
_TOKEN_TYPE_NAMES: tuple[str, ...] = (
    *(spec.name for spec in _ENTITY_SPECS),
    *_CONTEXT_TYPE_NAMES,
)

# Widest slot count across entity types; the shared slot-id (position) table is
# sized to it, BERT-style, so slot i of any type indexes the same position row.
_MAX_ENTITY_SLOTS = max(spec.count for spec in _ENTITY_SPECS)

# Global scalar / multihot blocks folded into the CLS token (not tokenized, since
# they map to no action slot); widths from the interface. shop_remove_cost is a
# single scalar.
_CLS_INPUT_DIM = (
    PLAYER_SCALAR_DIM + N_PLAYER_POWER_IDS + N_RELIC_IDS + N_SCREENS + KEYS_ACT_DIM + _SCALAR
)
# Offer-screen context the tokenless map-node / event-option actions read;
# widths from the interface.
_OFFER_INPUT_DIM = (
    MAP_CONTEXT_DIM
    + N_EVENT_IDS
    + MAX_NEOW_OPTIONS * N_NEOW_BONUS
    + MAX_NEOW_OPTIONS * N_NEOW_DRAWBACK
    + EVENT_PHASE_DIM
)


class _PreLNEncoderBlock(nn.Module):
    """One pre-LN transformer block: ``x = x + MHA(LN(x))``; ``x = x + FFN(LN(x))``.

    Pre-LN (Xiong et al. 2020) has well-behaved gradients at init and trains
    stably without learning-rate warmup, matching the nanoGPT / minGPT ordering.
    The same ``key_padding_mask`` is passed to attention every layer.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        normed = self.ln_attn(x)
        # need_weights=False skips the averaged attention-weight tensor (unused).
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.ln_ffn(x))
        return x


class ObsFeatureEncoder(nn.Module):
    """Encode a batched interface observation dict into transformer tokens.

    ``forward(obs)`` expects a dict of *batched* tensors keyed by the interface
    ``OBS_FIELDS`` names: float fields as ``(B, *shape)`` and id fields as
    integer ``(B, *shape)`` (cast to long internally for the embedding lookup).
    Float fields are cast to float32 internally, so a float64 obs dict (the
    common result of sampling via numpy) does not promote the network to double.

    It returns ``(per_token, pooled_cls, key_padding_mask)``:
    ``per_token`` is ``(B, S, d_model)``, ``pooled_cls`` is ``(B, d_model)`` (the
    ``CLS`` token's output, read by the value and policy heads), and
    ``key_padding_mask`` is a bool ``(B, S)`` (True == ignore). ``output_dim`` is
    ``d_model`` so the heads size against it rather than a literal.
    """

    def __init__(self, d_model: int = HIDDEN_DIM) -> None:
        super().__init__()
        if d_model % N_ATTENTION_HEADS != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by N_ATTENTION_HEADS "
                f"({N_ATTENTION_HEADS}) for multi-head attention"
            )
        self.d_model = d_model
        self.output_dim = d_model

        # padding_idx pins the empty-slot row to a permanent zero vector, so PAD
        # slots (ids == the empty marker) contribute nothing and receive no
        # gradient. The four card/enemy/move/potion tables carry over from the
        # flat encoder unchanged; relic_embed is new and pads at the relic INVALID
        # sentinel (N_RELIC_IDS), since RelicId 0 (AKABEKO) is a real relic.
        self.card_embed = nn.Embedding(N_CARD_IDS, CARD_EMBED_DIM, padding_idx=PAD_ID)
        self.enemy_embed = nn.Embedding(N_MONSTER_IDS, ENEMY_EMBED_DIM, padding_idx=PAD_ID)
        # MonsterMoveId INVALID=0, so PAD_ID doubles as the empty-slot / hidden-intent row.
        self.move_embed = nn.Embedding(N_MONSTER_MOVE_IDS, MOVE_EMBED_DIM, padding_idx=PAD_ID)
        self.potion_embed = nn.Embedding(N_POTION_IDS, POTION_EMBED_DIM, padding_idx=PAD_ID)
        self.relic_embed = nn.Embedding(N_RELIC_IDS + 1, RELIC_EMBED_DIM, padding_idx=N_RELIC_IDS)

        # Type (segment) and shared slot-id (position) embeddings, BERT-style.
        self.type_embed = nn.Embedding(len(_TOKEN_TYPE_NAMES), d_model)
        self.slot_id_embed = nn.Embedding(_MAX_ENTITY_SLOTS, d_model)
        self._type_index: dict[str, int] = {
            name: index for index, name in enumerate(_TOKEN_TYPE_NAMES)
        }

        # Per-type input projections (one linear each, per the token schema).
        self.entity_proj = nn.ModuleDict(
            {spec.name: nn.Linear(spec.feat_dim, d_model) for spec in _ENTITY_SPECS}
        )
        # One shared projection for the four pile summaries; the type embedding
        # distinguishes draw / discard / exhaust / deck.
        self.pile_proj = nn.Linear(_PILE_POOLS * CARD_EMBED_DIM, d_model)
        # CLS and OFFER context tokens use a small MLP over their raw blocks.
        self.cls_mlp = _context_mlp(_CLS_INPUT_DIM, d_model)
        self.offer_mlp = _context_mlp(_OFFER_INPUT_DIM, d_model)

        self.blocks = nn.ModuleList(
            _PreLNEncoderBlock(d_model, N_ATTENTION_HEADS, FFN_DIM, TRANSFORMER_DROPOUT)
            for _ in range(N_TRANSFORMER_LAYERS)
        )
        # Final LayerNorm on the residual stream before readout (the pre-LN
        # transformer's ``ln_f``, per nanoGPT / GPT), so per-token embeddings and
        # the pooled CLS the heads read are normalized rather than an unbounded
        # residual sum.
        self.ln_f = nn.LayerNorm(d_model)

        # Sequence layout: entity tokens (pointer-addressable) then context tokens.
        # token_layout maps each token type to its (start, count) slice of the
        # sequence, derived from the specs (no literals). cls_index is the CLS
        # token's absolute position, read out for the pooled context.
        self.token_layout: dict[str, tuple[int, int]] = {}
        start = 0
        for spec in _ENTITY_SPECS:
            self.token_layout[spec.name] = (start, spec.count)
            start += spec.count
        self.n_entity_tokens = start
        for name in _CONTEXT_TYPE_NAMES:
            self.token_layout[name] = (start, 1)
            start += 1
        self.seq_len = start
        self.cls_index = self.token_layout[_CLS_NAME][0]

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights nanoGPT/minGPT-style (the init the docstring cites).

        Every ``nn.Linear`` weight and ``nn.Embedding`` weight is normal(0, 0.02)
        and every Linear bias is zeroed; ``nn.LayerNorm`` keeps its default
        (weight 1, bias 0) and ``MultiheadAttention``'s in-projection keeps
        PyTorch's default. Two follow-ups matter: each id-embedding's padding row
        is re-zeroed (``normal_`` overwrote it) so the PAD=zero invariant the
        no-leakage / gradient-isolation tests rely on holds, and each block's
        residual-output projections are scaled by 1/sqrt(2 * N_TRANSFORMER_LAYERS)
        (GPT-2's residual-accumulation scaling) so the residual stream's variance
        does not grow with depth.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)

        # Re-zero the id embeddings' padding rows that normal_ just overwrote;
        # PAD must stay exactly zero (load-bearing for the no-leakage tests).
        for emb in (
            self.card_embed,
            self.enemy_embed,
            self.move_embed,
            self.potion_embed,
            self.relic_embed,
        ):
            with torch.no_grad():
                emb.weight[emb.padding_idx].zero_()

        # Scale residual-output projections by 1/sqrt(2N), per GPT-2.
        residual_std = _INIT_STD / math.sqrt(2 * N_TRANSFORMER_LAYERS)
        for block in self.blocks:
            assert isinstance(block, _PreLNEncoderBlock)
            out_proj = block.attn.out_proj  # attention residual output
            ffn_out = block.ffn[-2]  # second FFN Linear (ffn[-1] is Dropout)
            assert isinstance(out_proj, nn.Linear)
            assert isinstance(ffn_out, nn.Linear)
            nn.init.normal_(out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(ffn_out.weight, mean=0.0, std=residual_std)

    def _pool_pile(self, pile_ids: Tensor) -> Tensor:
        """Mean+max pool a pile's card embeddings over the pile (slot) dim.

        Piles (draw/discard/exhaust/deck) are order-agnostic sets, so we pool
        rather than flatten per slot. mean+max is a deliberately lossy summary of
        this partially observed pile (it drops card multiplicity and identity
        detail); a richer alternative, deferred for now, is per-pile attention.
        """
        emb = self.card_embed(pile_ids)  # (B, PILE_MAX, CARD_EMBED_DIM)
        # The mean keeps PAD rows: their dilution is BY DESIGN (it implicitly
        # encodes pile density), not a masked-mean bug. The max is PAD-masked so
        # it summarizes real cards only (an all-PAD pile -> 0); PAD's zero
        # embedding would otherwise floor the max at 0 for any pile with a PAD slot.
        mean = emb.mean(dim=1)
        pad = (pile_ids == PAD_ID).unsqueeze(-1)  # (B, PILE_MAX, 1)
        masked = emb.masked_fill(pad, float("-inf"))
        mx = masked.max(dim=1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))  # all-PAD -> 0
        return torch.cat([mean, mx], dim=1)  # (B, 2*CARD_EMBED_DIM)

    @staticmethod
    def _coerce_dtypes(obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Cast non-id fields to float32; id fields stay integer for embedding.

        Observations sampled via numpy are commonly float64; left unchanged they
        promote the token features to double and the projections' Linear then
        raises "mat1 and mat2 must have the same dtype".
        """
        return {name: (t if name in _ID_FIELDS else t.float()) for name, t in obs.items()}

    def _finalize_entity(self, name: str, raw: Tensor) -> Tensor:
        """Project raw entity features and add the type and slot-id embeddings.

        ``raw`` is ``(B, count, feat_dim)`` for token type ``name``; returns
        ``(B, count, d_model)``. The type embedding (which entity kind) is shared
        across the type's slots; the slot-id embedding (which slot) is the shared
        position row for each slot index.
        """
        count = raw.shape[1]
        projected = self.entity_proj[name](raw)  # (B, count, d_model)
        type_vec = self.type_embed.weight[self._type_index[name]]  # (d_model,)
        slot_ids = torch.arange(count, device=raw.device)
        slot_vecs = self.slot_id_embed(slot_ids)  # (count, d_model)
        return projected + type_vec + slot_vecs

    def _finalize_context(self, name: str, projected: Tensor) -> Tensor:
        """Add the type embedding to a projected context token.

        ``projected`` is ``(B, d_model)``; returns ``(B, 1, d_model)`` (a single
        sequence position). Context tokens get no slot-id embedding: they are not
        slotted, and their distinct type embeddings tell them apart.
        """
        type_vec = self.type_embed.weight[self._type_index[name]]  # (d_model,)
        return (projected + type_vec).unsqueeze(1)

    def _build_entity_tokens(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Build the entity token block ``(B, n_entity, d_model)`` and its PAD mask.

        The mask is ``(B, n_entity)`` bool, True where a slot's id equals its
        empty marker (so attention ignores it).
        """
        hand_ids = obs["hand_ids"].long()
        enemy_ids = obs["enemy_ids"].long()
        potion_ids = obs["potion_ids"].long()

        raw_by_name: dict[str, Tensor] = {
            "HAND": torch.cat([self.card_embed(hand_ids), obs["hand_feats"]], dim=2),
            "ENEMY": torch.cat(
                [
                    self.enemy_embed(enemy_ids),
                    obs["enemy_scalars"],
                    self.move_embed(obs["enemy_move_ids"].long()),
                    obs["enemy_intent_hidden"].unsqueeze(-1),
                    obs["enemy_powers"],
                    obs["enemy_alive"].unsqueeze(-1),
                ],
                dim=2,
            ),
            "POTION": torch.cat(
                [self.potion_embed(potion_ids), obs["potion_usable"].unsqueeze(-1)], dim=2
            ),
            "CARD_SELECT": self.card_embed(obs["card_select_ids"].long()),
            "REWARD_CARD": self.card_embed(obs["reward_card_ids"].long()),
            "REWARD_POTION": self.potion_embed(obs["reward_potion_ids"].long()),
            "REWARD_RELIC": self.relic_embed(obs["reward_relic_ids"].long()),
            "SHOP_CARD": torch.cat(
                [
                    self.card_embed(obs["shop_card_ids"].long()),
                    obs["shop_card_prices"].unsqueeze(-1),
                ],
                dim=2,
            ),
            "SHOP_RELIC": torch.cat(
                [
                    self.relic_embed(obs["shop_relic_ids"].long()),
                    obs["shop_relic_prices"].unsqueeze(-1),
                ],
                dim=2,
            ),
            "SHOP_POTION": torch.cat(
                [
                    self.potion_embed(obs["shop_potion_ids"].long()),
                    obs["shop_potion_prices"].unsqueeze(-1),
                ],
                dim=2,
            ),
            "BOSS_RELIC": self.relic_embed(obs["boss_relic_ids"].long()),
        }

        tokens: list[Tensor] = []
        masks: list[Tensor] = []
        for spec in _ENTITY_SPECS:
            tokens.append(self._finalize_entity(spec.name, raw_by_name[spec.name]))
            ids = obs[spec.id_field].long()  # (B, count)
            masks.append(ids == spec.pad_value)
        entity_tokens = torch.cat(tokens, dim=1)  # (B, n_entity, d_model)
        entity_mask = torch.cat(masks, dim=1)  # (B, n_entity) bool
        return entity_tokens, entity_mask

    def _build_context_tokens(self, obs: dict[str, Tensor]) -> Tensor:
        """Build the six never-PAD context tokens ``(B, 6, d_model)``.

        CLS folds the global scalar / multihot blocks; the four PILE tokens are
        the pooled draw / discard / exhaust / deck summaries; OFFER folds the
        map / event / Neow blocks.
        """
        cls_input = torch.cat(
            [
                obs["player_scalars"],
                obs["player_powers"],
                obs["relics_multihot"],
                obs["screen_onehot"],
                obs["keys_act"],
                obs["shop_remove_cost"],
            ],
            dim=1,
        )
        cls_token = self._finalize_context(_CLS_NAME, self.cls_mlp(cls_input))

        pile_tokens = [
            self._finalize_context(name, self.pile_proj(self._pool_pile(obs[id_field].long())))
            for name, id_field in _PILE_SPECS
        ]

        offer_input = torch.cat(
            [
                obs["map_context"],
                obs["event_onehot"],
                obs["neow_bonus"],
                obs["neow_drawback"],
                obs["event_phase_onehot"],
            ],
            dim=1,
        )
        offer_token = self._finalize_context(_OFFER_NAME, self.offer_mlp(offer_input))

        return torch.cat([cls_token, *pile_tokens, offer_token], dim=1)

    def forward(
        self, obs: dict[str, Tensor], *, apply_padding_mask: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode ``obs`` into ``(per_token, pooled_cls, key_padding_mask)``.

        ``apply_padding_mask`` is exposed only so the padding-mask no-leakage test
        can revert-verify that the mask is what suppresses PAD-token influence;
        it defaults to True and the production path never sets it False.
        """
        obs = self._coerce_dtypes(obs)
        batch = obs["hand_ids"].shape[0]
        device = obs["hand_ids"].device

        entity_tokens, entity_mask = self._build_entity_tokens(obs)
        context_tokens = self._build_context_tokens(obs)
        # Context tokens are always valid, so they are never masked. This anchors
        # every attention row with at least len(context) valid keys, so the
        # softmax never sees an all-`-inf` row.
        context_mask = torch.zeros(batch, context_tokens.shape[1], dtype=torch.bool, device=device)

        x = torch.cat([entity_tokens, context_tokens], dim=1)  # (B, S, d_model)
        key_padding_mask = torch.cat([entity_mask, context_mask], dim=1)  # (B, S) bool

        attn_mask = key_padding_mask if apply_padding_mask else None
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_f(x)

        pooled_cls = x[:, self.cls_index]  # (B, d_model)
        return x, pooled_cls, key_padding_mask


def _context_mlp(input_dim: int, d_model: int) -> nn.Module:
    """Small MLP projecting a context block to ``d_model`` (Linear -> GELU -> Linear)."""
    return nn.Sequential(
        nn.Linear(input_dim, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    )
