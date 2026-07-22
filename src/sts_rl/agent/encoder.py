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
                                + N_INTENT + N_POWER_IDS + 1)
    pile blocks  3 * (2 * CARD_EMBED_DIM)   (draw, discard, exhaust; mean+max)
    potion block POTION_SLOTS * POTION_EMBED_DIM + POTION_SLOTS
    passthrough  relics_multihot + player_powers + player_scalars
                 + screen_onehot + map_context
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sts_rl.interface import (
    ENEMY_SCALAR_DIM,
    HAND_FEAT_DIM,
    HAND_MAX,
    MAP_CONTEXT_DIM,
    MAX_ENEMIES,
    N_CARD_IDS,
    N_INTENT,
    N_MONSTER_IDS,
    N_POTION_IDS,
    N_POWER_IDS,
    N_RELIC_IDS,
    N_SCREENS,
    PAD_ID,
    PILE_MAX,
    PLAYER_SCALAR_DIM,
    POTION_SLOTS,
)

# --- Embedding widths ------------------------------------------------------
# Learned id embeddings, sized by how much each id space needs to carry: cards
# drive most decisions (widest), monsters fewer, potions fewest.
CARD_EMBED_DIM = 32
ENEMY_EMBED_DIM = 16
POTION_EMBED_DIM = 8

# Piles are pooled with BOTH mean and max, so each pile contributes 2 vectors.
_PILE_POOLS = 2
_N_PILES = 3  # draw, discard, exhaust

# Default trunk width; constructor-overridable.
HIDDEN_DIM = 512


class ObsFeatureEncoder(nn.Module):
    """Encode a batched interface observation dict into a feature vector.

    ``forward(obs)`` expects a dict of *batched* tensors keyed by the interface
    ``OBS_FIELDS`` names: float fields as ``(B, *shape)`` and id fields as
    integer ``(B, *shape)`` (cast to long internally for the embedding lookup).
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
        enemy = MAX_ENEMIES * (
            ENEMY_EMBED_DIM + ENEMY_SCALAR_DIM + N_INTENT + N_POWER_IDS + 1
        )
        piles = _N_PILES * (_PILE_POOLS * CARD_EMBED_DIM)
        potion = POTION_SLOTS * POTION_EMBED_DIM + POTION_SLOTS
        passthrough = (
            N_RELIC_IDS + N_POWER_IDS + PLAYER_SCALAR_DIM + N_SCREENS + MAP_CONTEXT_DIM
        )
        return hand + enemy + piles + potion + passthrough

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

    def encode_features(self, obs: dict[str, Tensor]) -> Tensor:
        """Build the pre-trunk concat of shape ``(B, feature_dim)``."""
        batch = obs["hand_ids"].shape[0]

        # HAND: per-slot [card_embed | hand_feats], flattened over slots. Slot
        # order is preserved because the action layout maps to it.
        hand_emb = self.card_embed(obs["hand_ids"].long())  # (B, HAND_MAX, C)
        hand = torch.cat([hand_emb, obs["hand_feats"]], dim=2)  # (B, HAND_MAX, C+F)
        hand = hand.reshape(batch, -1)

        # ENEMY: per-slot [enemy_embed | scalars | intent | powers | alive].
        enemy_emb = self.enemy_embed(obs["enemy_ids"].long())  # (B, MAX_ENEMIES, E)
        enemy = torch.cat(
            [
                enemy_emb,
                obs["enemy_scalars"],
                obs["enemy_intent"],
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
        potion = torch.cat(
            [potion_emb.reshape(batch, -1), obs["potion_usable"]], dim=1
        )

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

        return torch.cat([hand, enemy, piles, potion, passthrough], dim=1)

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        """Return the shared trunk features of shape ``(B, output_dim)``."""
        features = self.encode_features(obs)
        return self.trunk(features)
