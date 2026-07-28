"""Shared observation encoder for the Slay the Spire RL agent.

A shared trunk: it consumes the observation dict defined by the shared
interface (its ``OBS_FIELDS`` registry) and produces a single head-agnostic
feature vector that both a policy head and a value head consume. Action masking
is deliberately NOT applied here - it belongs downstream in the policy head. The
trunk only builds features.

Every dimension is derived from the interface constants, so if the engine's enum
sizes change the feature widths follow automatically; no observation-width
literals live here.

Layout of the pre-trunk concatenation (order is fixed: the action layout maps
to the hand/enemy slot order, so do NOT reorder):

    hand block   HAND_MAX * (CARD_EMBED_DIM + HAND_FEAT_DIM)
    enemy block  MAX_ENEMIES * (ENEMY_EMBED_DIM + ENEMY_SCALAR_DIM
                                + MOVE_EMBED_DIM + 1 + N_MONSTER_POWER_IDS + 1)
    pile blocks  3 * (2 * CARD_EMBED_DIM)   (draw, discard, exhaust; mean+max)
    potion block POTION_SLOTS * POTION_EMBED_DIM + POTION_SLOTS
    passthrough  relics_multihot + player_powers + player_scalars
                 + screen_onehot + map_context
    reward block MAX_REWARD_CARD_SLOTS * CARD_EMBED_DIM   (offered card-reward
                 slots; slot order fixed - REWARD_SELECT action layout maps to it)
    reward relic N_RELIC_IDS   (offered relics as an order-agnostic multihot,
                 mirroring the owned-relic multihot; no relic embedding table)
    reward potion MAX_REWARD_POTIONS * POTION_EMBED_DIM   (offered potions,
                 embedded per slot; reuses potion_embed)
    card select  CHOICE_MAX * CARD_EMBED_DIM   (candidate cards on the current
                 card-select screen, embedded per slot; slot order fixed - the
                 CARD_SELECT action layout maps to it; reuses card_embed)
    deck block   _PILE_POOLS * CARD_EMBED_DIM   (the overworld deck pooled mean+max
                 through card_embed; order-agnostic set, so pooled like a pile, not
                 per-slot flattened - the deck maps to no action slot)
    keys/act     KEYS_ACT_DIM   ([act, ruby, emerald, sapphire]; a raw passthrough
                 float block like player_scalars, not embedded)
    shop cards   MAX_SHOP_CARDS * CARD_EMBED_DIM   (offered shop cards embedded per
                 slot; slot order fixed - the SHOP_SELECT card sub-block maps to it;
                 reuses card_embed) then MAX_SHOP_CARDS raw price columns
    shop relics  N_RELIC_IDS   (offered shop relics as an order-agnostic multihot,
                 like the reward relics; no relic embedding table) then MAX_SHOP_RELICS
                 raw price columns
    shop potions MAX_SHOP_POTIONS * POTION_EMBED_DIM   (offered shop potions embedded
                 per slot; reuses potion_embed) then MAX_SHOP_POTIONS raw price columns
    shop remove  1   (the card-removal service cost, a single raw price column)
    boss relics  N_RELIC_IDS   (offered act-boss relics as an order-agnostic multihot,
                 like the reward / shop relics; no relic embedding table)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sts_rl.interface import (
    CHOICE_MAX,
    ENEMY_SCALAR_DIM,
    HAND_FEAT_DIM,
    HAND_MAX,
    KEYS_ACT_DIM,
    MAP_CONTEXT_DIM,
    MAX_ENEMIES,
    MAX_REWARD_CARD_SLOTS,
    MAX_REWARD_POTIONS,
    MAX_SHOP_CARDS,
    MAX_SHOP_POTIONS,
    MAX_SHOP_RELICS,
    N_CARD_IDS,
    N_MONSTER_IDS,
    N_MONSTER_MOVE_IDS,
    N_MONSTER_POWER_IDS,
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
# drive most decisions (widest), monsters fewer, potions fewest.
CARD_EMBED_DIM = 32
ENEMY_EMBED_DIM = 16
MOVE_EMBED_DIM = 16  # enemy next-move (MonsterMoveId) embedding
POTION_EMBED_DIM = 8

# Piles are pooled with BOTH mean and max, so each pile contributes 2 vectors.
_PILE_POOLS = 2
_N_PILES = 3  # draw, discard, exhaust

# Observation fields whose dtype is an id (embedding index), derived from the
# interface so it tracks the registry. _coerce_dtypes casts every non-id field
# to float32; ids keep their integer dtype and are cast to long at the lookups.
_ID_FIELDS = frozenset(f.name for f in OBS_FIELDS if f.bounds == "id")

# Default trunk width; constructor-overridable.
HIDDEN_DIM = 512


class ObsFeatureEncoder(nn.Module):
    """Encode a batched interface observation dict into a feature vector.

    ``forward(obs)`` expects a dict of *batched* tensors keyed by the interface
    ``OBS_FIELDS`` names: float fields as ``(B, *shape)`` and id fields as
    integer ``(B, *shape)`` (cast to long internally for the embedding lookup).
    Float fields are cast to float32 internally, so a float64 obs dict (the
    common result of sampling via numpy) does not promote the trunk to double.
    It returns the trunk output of shape ``(B, output_dim)``.
    :meth:`encode_features` exposes the raw pre-trunk concat of shape
    ``(B, feature_dim)``.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()

        # padding_idx=PAD_ID pins row 0 to a permanent zero vector, so empty /
        # pad slots (ids == PAD_ID) contribute nothing and receive no gradient.
        self.card_embed = nn.Embedding(N_CARD_IDS, CARD_EMBED_DIM, padding_idx=PAD_ID)
        self.enemy_embed = nn.Embedding(N_MONSTER_IDS, ENEMY_EMBED_DIM, padding_idx=PAD_ID)
        # MonsterMoveId INVALID=0, so PAD_ID doubles as the empty-slot / hidden-intent row.
        self.move_embed = nn.Embedding(N_MONSTER_MOVE_IDS, MOVE_EMBED_DIM, padding_idx=PAD_ID)
        self.potion_embed = nn.Embedding(N_POTION_IDS, POTION_EMBED_DIM, padding_idx=PAD_ID)

        self.feature_dim = self._compute_feature_dim()

        # Shared trunk: two-layer MLP over the concat -> head-agnostic features.
        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    @staticmethod
    def _compute_feature_dim() -> int:
        """Concat width, derived entirely from the interface constants (no literals)."""
        hand = HAND_MAX * (CARD_EMBED_DIM + HAND_FEAT_DIM)
        # per enemy: id embed | scalars | move embed | intent-hidden flag | powers | alive
        enemy = MAX_ENEMIES * (
            ENEMY_EMBED_DIM + ENEMY_SCALAR_DIM + MOVE_EMBED_DIM + 1 + N_MONSTER_POWER_IDS + 1
        )
        piles = _N_PILES * (_PILE_POOLS * CARD_EMBED_DIM)
        potion = POTION_SLOTS * POTION_EMBED_DIM + POTION_SLOTS
        passthrough = (
            N_RELIC_IDS + N_PLAYER_POWER_IDS + PLAYER_SCALAR_DIM + N_SCREENS + MAP_CONTEXT_DIM
        )
        # Offered reward-screen cards, embedded per slot (reuses card_embed).
        reward = MAX_REWARD_CARD_SLOTS * CARD_EMBED_DIM
        # Offered relics as a multihot over the relic id space (no relic embedding
        # table); offered potions embedded per slot (reuses potion_embed).
        reward_relic = N_RELIC_IDS
        reward_potion = MAX_REWARD_POTIONS * POTION_EMBED_DIM
        # Candidate cards on the current card-select screen, embedded per slot
        # (reuses card_embed).
        card_select = CHOICE_MAX * CARD_EMBED_DIM
        # Overworld deck pooled mean+max through card_embed (order-agnostic set,
        # like a pile), then the [act, ruby, emerald, sapphire] passthrough. Both
        # appended LAST, after card_select, so the prior layout stays a clean prefix.
        deck = _PILE_POOLS * CARD_EMBED_DIM
        keys_act = KEYS_ACT_DIM
        # Shop screen (SHOP_ROOM): cards / potions embedded per slot (reusing
        # card_embed / potion_embed), relics an order-agnostic multihot (no relic
        # table), each id block followed by its raw per-slot price columns, plus the
        # single card-removal cost. Then the act-boss relic multihot. All appended
        # LAST so the prior layout stays a clean prefix for warm-start migration.
        shop_cards = MAX_SHOP_CARDS * CARD_EMBED_DIM
        shop_card_prices = MAX_SHOP_CARDS
        shop_relics = N_RELIC_IDS
        shop_relic_prices = MAX_SHOP_RELICS
        shop_potions = MAX_SHOP_POTIONS * POTION_EMBED_DIM
        shop_potion_prices = MAX_SHOP_POTIONS
        shop_remove_cost = 1
        boss_relics = N_RELIC_IDS
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
        )

    def _pool_pile(self, pile_ids: Tensor) -> Tensor:
        """Mean+max pool a pile's card embeddings over the pile (slot) dim.

        Piles (draw/discard/exhaust) are order-agnostic sets, so we pool rather
        than flatten per slot. mean+max is a deliberately lossy summary of this
        partially observed pile (it drops card multiplicity and identity
        detail); a richer alternative, deferred for now, is per-pile attention.
        """
        emb = self.card_embed(pile_ids)  # (B, PILE_MAX, CARD_EMBED_DIM)
        # PAD rows dilute the mean, and max over an all-PAD pile returns PAD (0):
        # this dilution is BY DESIGN (it implicitly encodes pile density), not a
        # masked-mean bug.
        mean = emb.mean(dim=1)
        mx = emb.max(dim=1).values
        return torch.cat([mean, mx], dim=1)  # (B, 2*CARD_EMBED_DIM)

    @staticmethod
    def _coerce_dtypes(obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Cast non-id fields to float32; id fields stay integer for embedding.

        Observations sampled via numpy are commonly float64; left unchanged they
        promote the concat to double and the trunk's Linear then raises
        "mat1 and mat2 must have the same dtype".
        """
        return {name: (t if name in _ID_FIELDS else t.float()) for name, t in obs.items()}

    def encode_features(self, obs: dict[str, Tensor]) -> Tensor:
        """Build the pre-trunk concat of shape ``(B, feature_dim)``."""
        obs = self._coerce_dtypes(obs)
        batch = obs["hand_ids"].shape[0]

        # HAND: per-slot [card_embed | hand_feats], flattened over slots. Slot
        # order is preserved because the action layout maps to it.
        hand_emb = self.card_embed(obs["hand_ids"].long())  # (B, HAND_MAX, C)
        hand = torch.cat([hand_emb, obs["hand_feats"]], dim=2)  # (B, HAND_MAX, C+F)
        hand = hand.reshape(batch, -1)

        # ENEMY: per-slot [enemy_embed | scalars | move_embed | intent_hidden | powers | alive].
        enemy_emb = self.enemy_embed(obs["enemy_ids"].long())  # (B, MAX_ENEMIES, E)
        move_emb = self.move_embed(obs["enemy_move_ids"].long())  # (B, MAX_ENEMIES, M)
        enemy = torch.cat(
            [
                enemy_emb,
                obs["enemy_scalars"],
                move_emb,
                obs["enemy_intent_hidden"].unsqueeze(-1),
                obs["enemy_powers"],
                obs["enemy_alive"].unsqueeze(-1),
            ],
            dim=2,
        )
        enemy = enemy.reshape(batch, -1)

        # PILES: mean+max pooled card embeddings (order-agnostic).
        piles = torch.cat(
            [
                self._pool_pile(obs["draw_ids"].long()),
                self._pool_pile(obs["discard_ids"].long()),
                self._pool_pile(obs["exhaust_ids"].long()),
            ],
            dim=1,
        )

        # POTION: embed each slot, flatten, then append the usable flags.
        potion_emb = self.potion_embed(obs["potion_ids"].long())  # (B, SLOTS, P)
        potion = torch.cat([potion_emb.reshape(batch, -1), obs["potion_usable"]], dim=1)

        # Pass-through float vectors (already fixed-width per the interface).
        passthrough = torch.cat(
            [
                obs["relics_multihot"],
                obs["player_powers"],
                obs["player_scalars"],
                obs["screen_onehot"],
                obs["map_context"],
            ],
            dim=1,
        )

        # REWARD CARDS: per-slot offered-card embedding, flattened. Slot order is
        # fixed (the REWARD_SELECT action layout maps to it, like hand); reuses
        # card_embed, so PAD_ID slots contribute a zero vector.
        reward_emb = self.card_embed(obs["reward_card_ids"].long())  # (B, SLOTS, C)
        reward = reward_emb.reshape(batch, -1)

        # REWARD RELICS: scatter the offered relic ids into an (N_RELIC_IDS + 1)-wide
        # 0/1 multihot, then OUTPUT only columns [0:N_RELIC_IDS], dropping the final
        # INVALID/empty column. A multihot (not an embedding) because there is no relic
        # embedding table, offered relics are an order-agnostic set, and this mirrors how
        # owned relics are already encoded; it also keeps the checkpoint migration a pure
        # trunk-widen (a fixed-width N_RELIC_IDS input block, no new parameters). Empty
        # slots carry the relic INVALID sentinel (RelicId.INVALID == N_RELIC_IDS), which
        # scatters into the dropped final column and so contributes nothing. Unlike
        # cards / potions, RelicId 0 (AKABEKO) is a REAL relic, so its own column 0 is
        # kept -- the earlier "clear column PAD_ID(0)" approach silently erased an offered
        # AKABEKO, indistinguishable from an empty slot.
        reward_relic_ids = obs["reward_relic_ids"].long()  # (B, MAX_REWARD_RELICS)
        reward_relic_scatter = torch.zeros(
            batch, N_RELIC_IDS + 1, dtype=torch.float32, device=reward_relic_ids.device
        )
        reward_relic_scatter.scatter_(1, reward_relic_ids, 1.0)
        reward_relic = reward_relic_scatter[:, :N_RELIC_IDS]  # drop the INVALID/empty column

        # REWARD POTIONS: embed each offered-potion slot through potion_embed and
        # flatten, exactly as reward cards reuse card_embed; padding_idx makes PAD
        # slots contribute a zero vector.
        reward_potion_emb = self.potion_embed(obs["reward_potion_ids"].long())  # (B, SLOTS, P)
        reward_potion = reward_potion_emb.reshape(batch, -1)

        # CARD SELECT: per-slot candidate-card embedding, flattened. Slot order is
        # fixed (the CARD_SELECT action layout maps to it, like hand and reward
        # cards); reuses card_embed, so PAD_ID slots contribute a zero vector.
        card_select_emb = self.card_embed(obs["card_select_ids"].long())  # (B, CHOICE_MAX, C)
        card_select = card_select_emb.reshape(batch, -1)

        # DECK: the full overworld deck pooled mean+max through card_embed, exactly
        # like the draw / discard / exhaust piles (order-agnostic set, so pooled not
        # per-slot flattened -- the deck maps to no action slot). PAD slots pool to
        # the padding_idx zero vector, so an empty deck contributes a zero block.
        deck = self._pool_pile(obs["deck_ids"].long())  # (B, 2*CARD_EMBED_DIM)

        # KEYS/ACT: [act, ruby, emerald, sapphire], a raw passthrough float block
        # (already coerced to float32 above), like player_scalars -- not embedded.
        keys_act = obs["keys_act"]

        # SHOP CARDS: per-slot offered-card embedding, flattened (slot order fixed --
        # the SHOP_SELECT card sub-block maps to it, like reward cards); reuses
        # card_embed, so PAD_ID slots contribute a zero vector. Followed by the raw
        # per-slot price columns (already coerced to float32 above).
        shop_card_emb = self.card_embed(obs["shop_card_ids"].long())  # (B, MAX_SHOP_CARDS, C)
        shop_cards = shop_card_emb.reshape(batch, -1)
        shop_card_prices = obs["shop_card_prices"]

        # SHOP RELICS: scatter the offered relic ids into an (N_RELIC_IDS + 1)-wide 0/1
        # multihot and output only columns [0:N_RELIC_IDS], dropping the INVALID/empty
        # column -- identical to the reward-relic encoding (order-agnostic set, no relic
        # embedding table, keeps AKABEKO's real column 0). Then the raw price columns.
        shop_relic_ids = obs["shop_relic_ids"].long()  # (B, MAX_SHOP_RELICS)
        shop_relic_scatter = torch.zeros(
            batch, N_RELIC_IDS + 1, dtype=torch.float32, device=shop_relic_ids.device
        )
        shop_relic_scatter.scatter_(1, shop_relic_ids, 1.0)
        shop_relics = shop_relic_scatter[:, :N_RELIC_IDS]  # drop the INVALID/empty column
        shop_relic_prices = obs["shop_relic_prices"]

        # SHOP POTIONS: per-slot offered-potion embedding through potion_embed,
        # flattened (PAD slots zero). Then the raw per-slot price columns.
        shop_potion_emb = self.potion_embed(obs["shop_potion_ids"].long())  # (B, SLOTS, P)
        shop_potions = shop_potion_emb.reshape(batch, -1)
        shop_potion_prices = obs["shop_potion_prices"]

        # SHOP REMOVE COST: the single card-removal price, a raw passthrough column.
        shop_remove_cost = obs["shop_remove_cost"]

        # BOSS RELICS: the three offered act-boss relics, scattered into the same
        # INVALID-dropping multihot as the reward / shop relics (order-agnostic set, no
        # relic embedding table). Free, so no price columns.
        boss_relic_ids = obs["boss_relic_ids"].long()  # (B, MAX_BOSS_RELICS)
        boss_relic_scatter = torch.zeros(
            batch, N_RELIC_IDS + 1, dtype=torch.float32, device=boss_relic_ids.device
        )
        boss_relic_scatter.scatter_(1, boss_relic_ids, 1.0)
        boss_relics = boss_relic_scatter[:, :N_RELIC_IDS]  # drop the INVALID/empty column

        # CAUTION: the reward, card-select, deck, keys/act, and shop / boss-relic blocks
        # are appended LAST, in this fixed order (reward cards, relics, potions, then
        # card_select, the pooled deck, keys/act, then shop cards + prices, shop relics
        # + prices, shop potions + prices, shop remove cost, and finally the boss-relic
        # multihot), so an old checkpoint's trained input columns stay the leading prefix
        # and migrate_encoder_trunk_input_width widens the trunk by a clean zero-init
        # suffix. Do NOT insert a block ahead of these or reorder them, or the migration
        # would silently mismap columns.
        return torch.cat(
            [
                hand,
                enemy,
                piles,
                potion,
                passthrough,
                reward,
                reward_relic,
                reward_potion,
                card_select,
                deck,
                keys_act,
                shop_cards,
                shop_card_prices,
                shop_relics,
                shop_relic_prices,
                shop_potions,
                shop_potion_prices,
                shop_remove_cost,
                boss_relics,
            ],
            dim=1,
        )

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        """Return the shared trunk features of shape ``(B, output_dim)``."""
        features = self.encode_features(obs)
        return self.trunk(features)
