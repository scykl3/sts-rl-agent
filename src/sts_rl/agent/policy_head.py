"""Masked policy head for the Slay the Spire RL agent.

The first consumer of the shared observation encoder's pooled context. It takes
the pooled ``CLS`` context vector (shape ``(B, input_dim)``, where ``input_dim``
is the encoder's ``output_dim`` == the transformer ``d_model``) and projects it
to one logit per action index.

Action masking lives HERE, not in the encoder: the encoder is deliberately
head-agnostic, so legality is applied only at the point where a distribution
over actions is formed. The mask is supplied by the caller (sourced from the
environment's ``action_mask`` info key) as a boolean tensor ``(B, action_dim)``
where True == legal. Illegal logits are pushed to a large finite negative value
so ``softmax`` assigns them ~zero probability.

Why a finite constant and not ``float('-inf')``: an ``-inf`` logit makes
``softmax`` exactly zero, but its gradient and the ``log_prob``/entropy of a
Categorical built on it produce ``NaN`` (``0 * -inf``) that then poisons the
whole backward pass. A large finite negative value gives ~zero probability with
clean, finite gradients. The interface's ``assert_valid_mask`` guarantees at
least one legal action per row, so no row is fully masked (which would still
NaN even the finite path once every logit is driven to the floor).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from sts_rl.agent.encoder import HIDDEN_DIM, N_ATTENTION_HEADS, _ENTITY_SPECS
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

# Logit assigned to illegal actions: large and negative but FINITE, so softmax
# gives ~0 probability while keeping gradients/log_prob/entropy finite. Using
# the dtype floor (float('-inf')) would NaN the backward pass; -1e9 is small
# enough that exp(masked - max_legal) underflows to 0 in float32.
MASKED_LOGIT: float = -1e9


class MaskedPolicyHead(nn.Module):
    """Project the pooled context to masked action logits and a Categorical policy.

    A single ``Linear`` logit head over the encoder's pooled context. All masking
    is contained in this module; the encoder never sees the mask. The helpers
    (:meth:`act`, :meth:`evaluate_actions`) are shaped for a PPO consumer:
    ``act`` runs during rollout, ``evaluate_actions`` recomputes log-prob and
    entropy of stored actions during the update phase.
    """

    def __init__(self, input_dim: int = HIDDEN_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        # Logit head only: no nonlinearity here, the encoder's final LayerNorm
        # already normalizes the pooled context this head reads.
        self.logits = nn.Linear(input_dim, action_dim)

    @staticmethod
    def _apply_mask(logits: Tensor, mask: Tensor) -> Tensor:
        """Overwrite illegal logits with :data:`MASKED_LOGIT`.

        ``mask`` is coerced to bool so an int/float mask indexes safely; True
        marks a legal action and is left untouched.
        """
        # Shape must match exactly: an unbatched (action_dim,) mask would
        # otherwise broadcast silently across a batched (B, action_dim) logits
        # tensor, applying the wrong legality to every row.
        if mask.shape != logits.shape:
            raise InterfaceError(
                f"mask shape {tuple(mask.shape)} does not match logits shape "
                f"{tuple(logits.shape)} (no broadcasting allowed)"
            )
        # Both tensors must be on the same device or masked_fill raises a raw
        # torch error. Surface it as a domain error rather than moving the mask
        # silently: a hidden host<->device copy on the hot rollout path would
        # mask a real device-placement bug.
        if mask.device != logits.device:
            raise InterfaceError(
                f"mask device {mask.device} does not match logits device {logits.device}"
            )
        legal = mask.bool()
        # Every row needs at least one legal action; a fully-masked row would
        # drive all its logits to the floor and yield a uniform distribution
        # over illegal actions instead of failing.
        if not legal.any(dim=-1).all():
            raise InterfaceError(
                "mask has a fully-masked row with no legal action "
                "(would sample uniformly over illegal actions)"
            )
        # masked_fill overwrites the illegal positions with the finite floor in
        # one fused op (no full_like allocation on the hot rollout path) and
        # stays differentiable w.r.t. the untouched legal logits.
        return logits.masked_fill(~legal, MASKED_LOGIT)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        """Return masked logits of shape ``(B, action_dim)``.

        ``features`` is ``(B, input_dim)``; ``mask`` is a bool ``(B, action_dim)``
        tensor (True == legal). Illegal entries are set to the finite floor.
        """
        return self._apply_mask(self.logits(features), mask)

    def masked_distribution(self, features: Tensor, mask: Tensor) -> Categorical:
        """Build a Categorical from masked logits (``logits=``, not ``probs=``).

        Passing ``logits`` lets torch normalize in log-space, so the floored
        entries stay numerically well behaved instead of being renormalized
        from ~0 probabilities.
        """
        return Categorical(logits=self.forward(features, mask))

    def act(
        self,
        features: Tensor,
        mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select actions for a rollout step.

        Returns ``(action, log_prob, entropy)``, each ``(B,)``. With
        ``deterministic=True`` the action is the argmax over masked logits (the
        greedy/eval path); otherwise it is sampled. log-prob and entropy are
        always taken from the masked distribution so a PPO buffer stores
        consistent quantities regardless of the sampling mode.
        """
        dist = self.masked_distribution(features, mask)
        if deterministic:
            # argmax over masked logits: illegal entries sit at the floor, so
            # the greedy pick is always legal.
            action = dist.logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self,
        features: Tensor,
        mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Recompute ``(log_prob, entropy)`` of ``actions`` under the current policy.

        Used in PPO's update phase: the ratio between this log-prob and the
        rollout log-prob drives the surrogate objective. ``actions`` is ``(B,)``.
        """
        dist = self.masked_distribution(features, mask)
        return dist.log_prob(actions), dist.entropy()


# Small std for the learned pointer queries and the learned-query bank, matching
# the encoder's nanoGPT-style 0.02 init so no query starts on a degenerate scale.
_POINTER_INIT_STD = 0.02


class PointerPolicyHead(nn.Module):
    """Pointer-style policy head: score per-entity actions by attention, mask, sample.

    The block-aware successor to :class:`MaskedPolicyHead`. Instead of a single
    dense ``Linear`` over the pooled context, it scores each of the 257 action
    indices from the encoder's per-token embeddings, so a per-entity action reads
    its own entity token rather than a global summary. Three scoring sources cover
    the action layout (see the design's 257-slot table):

    - Per-entity single-factor blocks (``PLAY_CARD_UNTARGETED``, the two potion
      blocks, ``CARD_SELECT``, and the offered reward/shop/boss card/relic/potion
      sub-slices): a learned per-action-type query ``q_t`` scores each entity token
      ``x`` of the matching type as ``(q_t . W_K x) / sqrt(d_k)``, scattered to that
      entity's fixed slot. ``W_K`` (``pointer_key``) is shared; the query differs
      per action type, so two action types over the same entity (potion use vs
      discard) get distinct logits.
    - Two-factor targeted blocks (``PLAY_CARD_TARGETED`` = HAND x ENEMY,
      ``USE_POTION_TARGETED`` = POTION x ENEMY): a per-source query and per-enemy
      key form an interaction grid ``G[i, j] = (W_Q src_i . W_K enemy_j) / sqrt(d_k)``,
      flattened SOURCE-MAJOR (``index = i * MAX_ENEMIES + j``) to match the engine
      decode's ``divmod(offset, MAX_ENEMIES)``. Each targeted block projects its own
      source query and enemy key; the ENEMY tokens are the shared target set.
    - Learned-query (global or tokenless-per-slot) actions (``END_TURN``,
      ``CONFIRM_SELECT``, the reward gold/key/bowl/skip, ``MAP_SELECT``,
      ``REST_SELECT``, ``TREASURE_SELECT``, ``EVENT_SELECT``, the shop remove/skip,
      the boss skip, and ``PROCEED``): a bank of learned query vectors cross-attends
      (one PAD-masked MHA layer over the full token set), then a shared linear emits
      each logit.

    Every action-slice and token offset is derived symbolically from the interface
    ``ACTION_BLOCKS`` / ``REWARD_*_OFFSET`` registry and the encoder's ``_ENTITY_SPECS``
    token order, so an interface cap or enum bump moves the head automatically; no
    index literal appears here. A construction-time assertion checks the three
    sources partition all ``ACTION_DIM`` indices exactly once.

    Masking is reused VERBATIM from :class:`MaskedPolicyHead`: :meth:`raw_logits`
    scatters finite scores into a zero-allocated ``(B, ACTION_DIM)`` tensor and
    asserts finiteness, then :meth:`forward` applies ``MaskedPolicyHead._apply_mask``
    (same ``MASKED_LOGIT``, exact-shape/device checks, all-illegal guard). The
    ``Categorical(logits=...)`` / :meth:`act` / :meth:`evaluate_actions` semantics
    are identical to the dense head; only the inputs differ (per-token embeddings
    and the padding mask, not the pooled context).
    """

    # register_buffer types as ``Tensor | Module`` under torch's stubs; pin it to
    # Tensor so the ``raw[:, learned_action_index]`` scatter index type-checks.
    learned_action_index: Tensor

    def __init__(
        self,
        input_dim: int = HIDDEN_DIM,
        action_dim: int = ACTION_DIM,
        n_heads: int = N_ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        # Pointer scoring runs at d_k = d_model; scale by 1/sqrt(d_k) (Vaswani et al.).
        self.d_k = input_dim
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Entity-token layout within the encoder's per-token output: entity tokens
        # come first, in _ENTITY_SPECS order, each contributing spec.count slots.
        # Derived from the same specs the encoder builds its layout from, so the
        # head's token offsets cannot drift from the encoder's.
        entity_span: dict[str, tuple[int, int]] = {}
        cursor = 0
        for spec in _ENTITY_SPECS:
            entity_span[spec.name] = (cursor, spec.count)
            cursor += spec.count
        self.n_entity_tokens = cursor

        blocks = ACTION_BLOCK_BY_NAME
        reward_start = blocks["REWARD_SELECT"].start
        shop_start = blocks["SHOP_SELECT"].start
        boss_start = blocks["BOSS_RELIC_SELECT"].start
        # Shop sub-layout offsets, derived from the interface caps (matching the
        # env's SHOP_SELECT decode order: cards, relics, potions, remove, skip).
        shop_relic_off = MAX_SHOP_CARDS
        shop_potion_off = MAX_SHOP_CARDS + MAX_SHOP_RELICS
        shop_remove_off = MAX_SHOP_CARDS + MAX_SHOP_RELICS + MAX_SHOP_POTIONS
        shop_skip_off = shop_remove_off + 1
        boss_skip_off = MAX_BOSS_RELICS

        # (query_name, action_start, count, token_name) for the single-factor
        # pointer blocks. query_name is unique per action type, so the two potion
        # blocks over the same POTION tokens learn separate queries.
        single_specs: tuple[tuple[str, int, int, str], ...] = (
            (
                "PLAY_CARD_UNTARGETED",
                blocks["PLAY_CARD_UNTARGETED"].start,
                blocks["PLAY_CARD_UNTARGETED"].count,
                "HAND",
            ),
            (
                "USE_POTION_UNTARGETED",
                blocks["USE_POTION_UNTARGETED"].start,
                blocks["USE_POTION_UNTARGETED"].count,
                "POTION",
            ),
            (
                "DISCARD_POTION",
                blocks["DISCARD_POTION"].start,
                blocks["DISCARD_POTION"].count,
                "POTION",
            ),
            (
                "CARD_SELECT",
                blocks["CARD_SELECT"].start,
                blocks["CARD_SELECT"].count,
                "CARD_SELECT",
            ),
            (
                "REWARD_POTION",
                reward_start + REWARD_POTION_OFFSET,
                MAX_REWARD_POTIONS,
                "REWARD_POTION",
            ),
            ("REWARD_RELIC", reward_start + REWARD_RELIC_OFFSET, MAX_REWARD_RELICS, "REWARD_RELIC"),
            (
                "REWARD_CARD",
                reward_start + REWARD_CARD_OFFSET,
                MAX_REWARD_CARD_SLOTS,
                "REWARD_CARD",
            ),
            ("SHOP_CARD", shop_start, MAX_SHOP_CARDS, "SHOP_CARD"),
            ("SHOP_RELIC", shop_start + shop_relic_off, MAX_SHOP_RELICS, "SHOP_RELIC"),
            ("SHOP_POTION", shop_start + shop_potion_off, MAX_SHOP_POTIONS, "SHOP_POTION"),
            ("BOSS_RELIC", boss_start, MAX_BOSS_RELICS, "BOSS_RELIC"),
        )
        # Resolve each to (query_name, action_start, count, token_start), asserting
        # the action-slice count matches the entity-token count it points at.
        self._single_blocks: list[tuple[str, int, int, int]] = []
        for query_name, action_start, count, token_name in single_specs:
            token_start, token_count = entity_span[token_name]
            if count != token_count:
                raise InterfaceError(
                    f"pointer block {query_name!r} spans {count} actions but its "
                    f"{token_name} token group has {token_count} slots"
                )
            self._single_blocks.append((query_name, action_start, count, token_start))
        self.single_query = nn.ParameterDict(
            {
                query_name: nn.Parameter(torch.randn(self.d_k) * _POINTER_INIT_STD)
                for query_name, _, _, _ in self._single_blocks
            }
        )
        # Shared key projection for all single-factor pointers (the query carries
        # the per-action-type distinction).
        self.pointer_key = nn.Linear(input_dim, self.d_k, bias=False)

        # Two-factor targeted blocks: (name, action_start, src_start, src_count,
        # enemy_start, enemy_count). The grid is flattened source-major, so the
        # enemy count MUST equal MAX_ENEMIES (the decode's divmod modulus).
        enemy_start, enemy_count = entity_span["ENEMY"]
        if enemy_count != MAX_ENEMIES:
            raise InterfaceError(
                f"ENEMY token group has {enemy_count} slots but the targeted "
                f"decode packs modulo MAX_ENEMIES={MAX_ENEMIES}"
            )
        targeted_specs: tuple[tuple[str, str], ...] = (
            ("PLAY_CARD_TARGETED", "HAND"),
            ("USE_POTION_TARGETED", "POTION"),
        )
        self._targeted_blocks: list[tuple[str, int, int, int, int, int]] = []
        self.targeted_query = nn.ModuleDict()
        self.targeted_key = nn.ModuleDict()
        for name, source_name in targeted_specs:
            block = blocks[name]
            src_start, src_count = entity_span[source_name]
            if block.count != src_count * enemy_count:
                raise InterfaceError(
                    f"targeted block {name!r} spans {block.count} actions but "
                    f"{source_name} x ENEMY is {src_count} x {enemy_count}"
                )
            self._targeted_blocks.append(
                (name, block.start, src_start, src_count, enemy_start, enemy_count)
            )
            self.targeted_query[name] = nn.Linear(input_dim, self.d_k, bias=False)
            self.targeted_key[name] = nn.Linear(input_dim, self.d_k, bias=False)

        # Learned-query slots: global actions and tokenless-per-slot menus, one
        # bank row per index, in ascending action-index order.
        map_block = blocks["MAP_SELECT"]
        rest_block = blocks["REST_SELECT"]
        treasure_block = blocks["TREASURE_SELECT"]
        event_block = blocks["EVENT_SELECT"]
        learned_indices: list[int] = [
            blocks["END_TURN"].start,
            blocks["CONFIRM_SELECT"].start,
            reward_start + REWARD_GOLD_OFFSET,
            reward_start + REWARD_KEY_OFFSET,
            reward_start + REWARD_SINGING_BOWL_OFFSET,
            reward_start + REWARD_SKIP_OFFSET,
            *range(map_block.start, map_block.stop),
            *range(rest_block.start, rest_block.stop),
            *range(treasure_block.start, treasure_block.stop),
            *range(event_block.start, event_block.stop),
            shop_start + shop_remove_off,
            shop_start + shop_skip_off,
            boss_start + boss_skip_off,
            blocks["PROCEED"].start,
        ]
        n_learned = len(learned_indices)
        self.learned_queries = nn.Parameter(torch.randn(n_learned, input_dim) * _POINTER_INIT_STD)
        # One PAD-masked cross-attention layer; the queries read the full token set
        # (context tokens are never PAD, so no all-`-inf` attention row).
        self.learned_attn = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.learned_logit = nn.Linear(input_dim, 1)  # shared readout per bank row
        self.register_buffer(
            "learned_action_index", torch.tensor(learned_indices, dtype=torch.long)
        )

        # The three sources must partition every action index exactly once: sorted
        # coverage equal to range(ACTION_DIM) proves both completeness (no slot
        # unscored) and disjointness (no slot double-written).
        covered: list[int] = []
        for _, action_start, count, _ in self._single_blocks:
            covered.extend(range(action_start, action_start + count))
        for _, action_start, _, src_count, _, en_count in self._targeted_blocks:
            covered.extend(range(action_start, action_start + src_count * en_count))
        covered.extend(learned_indices)
        if sorted(covered) != list(range(action_dim)):
            raise InterfaceError(
                "pointer head scoring sources do not partition the action space "
                f"exactly once (covered {len(covered)} of {action_dim} indices)"
            )

    def raw_logits(self, per_token: Tensor, key_padding_mask: Tensor) -> Tensor:
        """Score all ``ACTION_DIM`` indices from the per-token embeddings (pre-mask).

        ``per_token`` is ``(B, S, d_model)`` and ``key_padding_mask`` is ``(B, S)``
        (True == ignore), the encoder's outputs. Returns raw ``(B, ACTION_DIM)``
        logits with no legality applied - the BC path and the finite-logit test
        read this directly; :meth:`forward` masks it.

        The tensor is zero-allocated (never ``torch.empty``), so an unscored slot
        would be a finite 0 rather than uninitialized memory; every slot is in fact
        scored, and a finiteness assertion catches a NaN/inf scatter (a degenerate
        pointer score) before it can reach ``Categorical``, which masking would hide.
        """
        batch = per_token.shape[0]
        raw = torch.zeros((batch, self.action_dim), dtype=per_token.dtype, device=per_token.device)

        # Single-factor pointers: shared key projection over the entity tokens, a
        # per-action-type query dotted against each token's key.
        entity_keys = self.pointer_key(per_token[:, : self.n_entity_tokens])  # (B, n_ent, d_k)
        # A static per-type query dotted with the shared full-rank key is a per-type linear
        # readout (Linear(d_model, 1) per block), the reference multiplicative form
        # (AlphaStar/DETR static queries); only the two-factor grid is a genuine bilinear pointer.
        for query_name, action_start, count, token_start in self._single_blocks:
            keys = entity_keys[:, token_start : token_start + count]  # (B, count, d_k)
            query = self.single_query[query_name]  # (d_k,)
            scores = torch.matmul(keys, query) * self.scale  # (B, count)
            raw[:, action_start : action_start + count] = scores

        # Two-factor targeted grids, flattened source-major so index_in_block =
        # source * MAX_ENEMIES + enemy, matching divmod(offset, MAX_ENEMIES) in the
        # env decode. reshape(B, -1) on a (B, src, enemy) grid is row-major (enemy
        # the fast axis), which IS source-major.
        for name, action_start, src_start, src_count, en_start, en_count in self._targeted_blocks:
            source = per_token[:, src_start : src_start + src_count]  # (B, src, d_model)
            enemy = per_token[:, en_start : en_start + en_count]  # (B, enemy, d_model)
            q = self.targeted_query[name](source)  # (B, src, d_k)
            k = self.targeted_key[name](enemy)  # (B, enemy, d_k)
            grid = torch.matmul(q, k.transpose(1, 2)) * self.scale  # (B, src, enemy)
            raw[:, action_start : action_start + src_count * en_count] = grid.reshape(
                batch, src_count * en_count
            )

        # Learned-query bank: cross-attend over the full (PAD-masked) token set,
        # then a shared linear per row, scattered to each row's fixed action index.
        queries = self.learned_queries.unsqueeze(0).expand(batch, -1, -1)  # (B, n_learned, d_model)
        attended, _ = self.learned_attn(
            queries, per_token, per_token, key_padding_mask=key_padding_mask, need_weights=False
        )
        learned_logits = self.learned_logit(attended).squeeze(-1)  # (B, n_learned)
        raw[:, self.learned_action_index] = learned_logits

        if not torch.isfinite(raw).all():
            raise InterfaceError("pointer head produced a non-finite raw logit")
        return raw

    def forward(self, per_token: Tensor, key_padding_mask: Tensor, mask: Tensor) -> Tensor:
        """Return masked logits ``(B, ACTION_DIM)`` from the encoder's token outputs.

        ``per_token`` ``(B, S, d_model)`` and ``key_padding_mask`` ``(B, S)`` come
        from the encoder; ``mask`` is a bool ``(B, ACTION_DIM)`` (True == legal).
        Masking reuses :meth:`MaskedPolicyHead._apply_mask` verbatim, so illegal
        slots become the finite ``MASKED_LOGIT`` under the same shape/device/
        all-illegal guards as the dense head.
        """
        raw = self.raw_logits(per_token, key_padding_mask)
        return MaskedPolicyHead._apply_mask(raw, mask)

    def masked_distribution(
        self, per_token: Tensor, key_padding_mask: Tensor, mask: Tensor
    ) -> Categorical:
        """Build a Categorical from masked logits (``logits=``, not ``probs=``)."""
        return Categorical(logits=self.forward(per_token, key_padding_mask, mask))

    def act(
        self,
        per_token: Tensor,
        key_padding_mask: Tensor,
        mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select actions for a rollout step; returns ``(action, log_prob, entropy)``.

        Identical semantics to :meth:`MaskedPolicyHead.act`: greedy is the argmax
        over masked logits, otherwise a sample; log-prob and entropy always come
        from the masked distribution.
        """
        dist = self.masked_distribution(per_token, key_padding_mask, mask)
        if deterministic:
            action = dist.logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self,
        per_token: Tensor,
        key_padding_mask: Tensor,
        mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Recompute ``(log_prob, entropy)`` of ``actions`` under the current policy."""
        dist = self.masked_distribution(per_token, key_padding_mask, mask)
        return dist.log_prob(actions), dist.entropy()
