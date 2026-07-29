"""Name-aware policy-head checkpoint migration.

A trained checkpoint records only its ``interface_version`` (see
:func:`sts_rl.agent.train._save_checkpoint`), not the action-head layout it was
trained under. When the interface bumps and the action-index layout shifts, an
old checkpoint's ``policy.logits`` rows no longer line up with the current
:data:`sts_rl.interface.ACTION_BLOCKS`: block offsets move, and blocks are
added, removed, or renamed. A positional (row-for-row) copy is therefore WRONG -
under the 0.3.0 -> 0.4.0 bump it would load ``CONFIRM_SELECT`` (row 154 in
0.3.0) into ``TREASURE_SELECT`` (row 154 in 0.4.0), and shift every overworld
block off by the width delta.

This module upgrades an old policy head to the current layout by remapping
action-head rows BY BLOCK NAME: for each current action block, if a block of the
same name existed in the old layout its trained rows are copied to the new
offset; every current block without an old counterpart (new or renamed) is
zero-initialized. Non-policy-head tensors (encoder embeddings, shared trunk,
value head) do not depend on the action layout and are architecture-stable
across this bump, so they are copied through unchanged.

The old layouts are embedded here, keyed by ``interface_version``, because a
checkpoint does not carry its own layout. :data:`OLD_ACTION_LAYOUTS` records the
historical ``ACTION_BLOCKS`` of each prior shippable version, derived from that
version's ``interface.py`` in git history (0.3.0 is git ``5651b2e``, the last
0.3.0 mainline).

The remap never hardcodes an absolute offset: old offsets come from the embedded
old table and new offsets from the live :data:`~sts_rl.interface.ACTION_BLOCKS`,
so both sides track their single source of truth.

A second, independent migration widens the encoder's first trunk ``Linear`` when
the observation encoder gains input features. Those new features are appended at
the END of the pre-trunk concat (see :mod:`sts_rl.agent.encoder`), so an old
checkpoint's trained input columns are the leading prefix: the migration copies
them through and zero-inits the appended suffix columns, leaving the new pathway
neutral so the warm-started combat behavior is preserved. It composes with the
policy-head remap because the two touch disjoint tensors (``policy.logits.*`` vs
the first trunk ``Linear``), so a checkpoint needing both is migrated by both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.train import (
    CHECKPOINT_HIDDEN_DIM_KEY,
    CHECKPOINT_INTERFACE_VERSION_KEY,
    CHECKPOINT_MODEL_KEY,
)
from sts_rl.interface import ACTION_BLOCKS, ACTION_DIM, INTERFACE_VERSION, InterfaceError

logger = logging.getLogger(__name__)

# State-dict keys of the policy head's logit ``Linear`` (``MaskedPolicyHead.logits``).
# These two tensors are the ONLY ones whose row layout tracks the action-index
# layout, so they are the only ones remapped; every other key is copied as-is.
POLICY_LOGITS_WEIGHT_KEY: str = "policy.logits.weight"
POLICY_LOGITS_BIAS_KEY: str = "policy.logits.bias"

# State-dict key of the encoder's first trunk ``Linear`` weight
# (``ObsFeatureEncoder.trunk[0]``), shape ``(hidden, feature_dim)``. Its
# ``feature_dim`` columns are the ONLY tensor dimension that changes when the
# encoder gains input features, so it is the only tensor the trunk-input-width
# migration rebuilds; every other key is copied through unchanged.
TRUNK_INPUT_WEIGHT_KEY: str = "encoder.trunk.0.weight"

# Known prior trunk input widths (the ``feature_dim`` of ``encoder.trunk.0.weight``)
# of checkpoints that predate later-appended observation blocks. Like the policy
# head's ``_OLD_ACTION_DIM``, this is FIXED HISTORY - a checkpoint does not record
# its own encoder input width, so the widen migration validates the checkpoint's
# width against this recorded set before treating its columns as a clean leading
# prefix:
#   1349 - combat / pre-reward-vision width (before any offered-item reward block)
#   1605 - card-vision width (after the reward_card block; 1349 -> 1605, before the
#          offered-relic and offered-potion blocks)
#   4881 - pre-deck width (after the offered-relic / offered-potion and card-select
#          blocks, the shipped 0.6.0 width; before the pooled deck and keys/act
#          blocks were appended)
#   4949 - pre-shop/boss width (after the pooled deck and keys/act blocks, the shipped
#          0.7.0 width; before the shop and boss-relic screen blocks were appended)
#   5571 - pre-neow/event width (after the shop and boss-relic screen blocks, the
#          shipped 0.8.0 width; before the Neow-event one-hot blocks were appended)
#   5736 - pre-map-lookahead width (after the Neow-event one-hot blocks, the shipped
#          0.9.0 width; before the map-lookahead block was appended)
# When a new feature block is appended at the END of the concat, add the PRE-APPEND
# width here deliberately (the current width, just before the append) so an older
# checkpoint of that width still migrates and any other narrower width is rejected
# as a column mismap.
_KNOWN_PRIOR_FEATURE_DIMS: tuple[int, ...] = (1349, 1605, 4881, 4949, 5571, 5736)

# Historical action-block layouts, as ordered ``(name, count)`` specs copied
# verbatim from each version's ``interface.py`` in git. Counts (not offsets) are
# recorded; the contiguous ``{name: (start, count)}`` table is derived below with
# the same back-to-back accumulation ``interface._build_action_blocks`` uses, so
# a miscopied count cannot silently desync the offsets.
#
# 0.3.0 (git ``5651b2e``) differs from 0.4.0 in three ways that make a positional
# copy wrong: ``CONFIRM_SELECT`` was the LAST block (index 154, now 106); there
# was a ``CARD_REWARD_SELECT`` (5) block, since removed - it is NOT the same as
# 0.4.0's ``REWARD_SELECT`` (18), which differs in name, width, and semantics, so
# it is intentionally dropped rather than remapped; and there was no
# ``TREASURE_SELECT`` block.
_OLD_ACTION_BLOCK_SPECS: dict[str, tuple[tuple[str, int], ...]] = {
    "0.3.0": (
        ("END_TURN", 1),
        ("PLAY_CARD_TARGETED", 50),
        ("PLAY_CARD_UNTARGETED", 10),
        ("USE_POTION_TARGETED", 25),
        ("USE_POTION_UNTARGETED", 5),
        ("DISCARD_POTION", 5),
        ("CARD_SELECT", 10),
        ("CARD_REWARD_SELECT", 5),
        ("MAP_SELECT", 7),
        ("SHOP_SELECT", 15),
        ("REST_SELECT", 6),
        ("EVENT_SELECT", 10),
        ("BOSS_RELIC_SELECT", 4),
        ("PROCEED", 1),
        ("CONFIRM_SELECT", 1),
    ),
}

# Known total action width per embedded historical version, used only to validate
# the embedded specs against the shipped ``ACTION_DIM`` (a transcription guard on
# the constant table, mirroring ``interface._EXPECTED_ACTION_DIM``). It is not an
# input to the remap logic.
_OLD_ACTION_DIM: dict[str, int] = {"0.3.0": 155}


def _build_layout(specs: tuple[tuple[str, int], ...]) -> dict[str, tuple[int, int]]:
    """Derive a contiguous ``{name: (start, count)}`` table from ordered specs.

    Mirrors ``interface._build_action_blocks``: blocks lie back to back in order,
    so each block's start is the running offset.
    """
    layout: dict[str, tuple[int, int]] = {}
    start = 0
    for name, count in specs:
        layout[name] = (start, count)
        start += count
    return layout


# Public {version: {name: (start, count)}} table. Values are the historical
# ACTION_BLOCKS layouts a checkpoint of that version was trained under.
OLD_ACTION_LAYOUTS: dict[str, dict[str, tuple[int, int]]] = {
    version: _build_layout(specs) for version, specs in _OLD_ACTION_BLOCK_SPECS.items()
}

# Validate each embedded layout against its recorded ACTION_DIM at import time,
# so a bad transcription fails loudly here rather than silently mis-slicing rows.
for _version, _layout in OLD_ACTION_LAYOUTS.items():
    _total = sum(count for _, count in _layout.values())
    _expected = _OLD_ACTION_DIM[_version]
    if _total != _expected:
        raise InterfaceError(
            f"embedded action layout for interface {_version} sums to {_total} rows, "
            f"expected {_expected}; check _OLD_ACTION_BLOCK_SPECS"
        )


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse a ``"X.Y.Z"`` version into an int tuple for numeric comparison.

    Compares componentwise so ``0.10.0`` correctly sorts after ``0.9.0`` (a plain
    string compare would not).
    """
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError as exc:
        raise InterfaceError(f"unparseable interface version {version!r}") from exc


def migrate_policy_head(
    state_dict: dict[str, torch.Tensor], from_version: str
) -> dict[str, torch.Tensor]:
    """Remap an old checkpoint's policy head to the CURRENT action layout.

    Returns a NEW state_dict that shares NO tensor storage with the input: the two
    head tensors are freshly built and every carried-through non-head tensor is
    cloned, so mutating the returned dict (or a model loaded from it) cannot
    corrupt the caller's ``state_dict``. The rebuilt ``policy.logits.weight`` is
    ``(ACTION_DIM, hidden)`` and ``policy.logits.bias`` is ``(ACTION_DIM,)`` under
    the current interface. For each current action block, if a same-named block
    existed in ``from_version`` its first ``min(old_count, new_count)`` trained
    rows are copied from the old offset to the new offset; every other current
    block (new or renamed) is left zero-initialized. All non-policy-head tensors
    are carried through unchanged (cloned, not shared).

    ``hidden`` (the trunk width) is read from the checkpoint's own logit weight,
    so the rebuilt head matches the checkpoint's trunk rather than any default.

    ``from_version`` must be a key of :data:`OLD_ACTION_LAYOUTS`.

    Raises :class:`~sts_rl.interface.InterfaceError` if ``from_version`` has no
    embedded layout, if the policy-head tensors are absent (not an ``ActorCritic``
    checkpoint), if the logit weight is not 2-D, or if the checkpoint's head width
    does not match the recorded old layout (a version/layout mismatch).
    """
    if from_version not in OLD_ACTION_LAYOUTS:
        raise InterfaceError(
            f"no embedded action layout for interface version {from_version!r}; "
            f"known versions: {sorted(OLD_ACTION_LAYOUTS)}"
        )
    old_layout = OLD_ACTION_LAYOUTS[from_version]

    if POLICY_LOGITS_WEIGHT_KEY not in state_dict or POLICY_LOGITS_BIAS_KEY not in state_dict:
        raise InterfaceError(
            "state_dict is missing the policy-head logit tensors "
            f"({POLICY_LOGITS_WEIGHT_KEY!r}, {POLICY_LOGITS_BIAS_KEY!r}); "
            "not an ActorCritic checkpoint?"
        )

    old_weight = state_dict[POLICY_LOGITS_WEIGHT_KEY]
    old_bias = state_dict[POLICY_LOGITS_BIAS_KEY]

    # The logit weight must be 2-D (ACTION_DIM, hidden); guard before reading
    # shape[1] below so a malformed 1-D head fails typed here rather than with a
    # bare IndexError.
    if old_weight.ndim != 2:
        raise InterfaceError(
            f"checkpoint policy-head weight {POLICY_LOGITS_WEIGHT_KEY!r} must be "
            f"2-D (ACTION_DIM, hidden), got {old_weight.ndim}-D shape "
            f"{tuple(old_weight.shape)}"
        )

    # The checkpoint's head must be as wide as the recorded old layout; otherwise
    # the version and the weights disagree and any remap would slice the wrong
    # rows. Fail loudly instead of silently truncating.
    old_total = sum(count for _, count in old_layout.values())
    if old_weight.shape[0] != old_total or old_bias.shape[0] != old_total:
        raise InterfaceError(
            f"checkpoint policy head has weight rows={old_weight.shape[0]}, "
            f"bias rows={old_bias.shape[0]}, but interface {from_version} layout "
            f"expects {old_total}; version/layout mismatch"
        )

    hidden = old_weight.shape[1]
    # Zero-initialized target: any current block without an old counterpart
    # (REWARD_SELECT, TREASURE_SELECT under 0.4.0) stays zero by construction.
    # new_zeros preserves the checkpoint tensors' dtype and device.
    new_weight = old_weight.new_zeros((ACTION_DIM, hidden))
    new_bias = old_bias.new_zeros((ACTION_DIM,))

    # Diagnostic: an OLD block with no same-named current block is dropped, so its
    # trained rows have no destination. Warn if such a block carried nonzero
    # weights, so a full-run 0.3.0 checkpoint does not silently lose trained
    # signal. Combat-only checkpoints leave these overworld rows at random init
    # (nonzero but never trained), so nothing meaningful is lost for those.
    current_block_names = {block.name for block in ACTION_BLOCKS}
    for name, (old_start, old_count) in old_layout.items():
        if name in current_block_names:
            continue
        if bool(old_weight[old_start : old_start + old_count].any()):
            logger.warning(
                "policy-head migration from interface %s drops block %r (%d rows) "
                "with no current counterpart; its trained weights are discarded",
                from_version,
                name,
                old_count,
            )

    for block in ACTION_BLOCKS:
        old_span = old_layout.get(block.name)
        if old_span is None:
            continue  # new or renamed block: leave zero-initialized
        old_start, old_count = old_span
        # min() guards a block that grew or shrank across the bump (e.g.
        # REST_SELECT 6 -> 7): copy only the rows both layouts share.
        rows = min(old_count, block.count)
        new_weight[block.start : block.start + rows] = old_weight[old_start : old_start + rows]
        new_bias[block.start : block.start + rows] = old_bias[old_start : old_start + rows]

    # Clone every carried-through non-head tensor so the returned dict shares no
    # storage with the input: a caller mutating the migrated dict (or a model
    # loaded from it) cannot corrupt the source state_dict. The two head tensors
    # are freshly built above, so they are already independent.
    head_keys = {POLICY_LOGITS_WEIGHT_KEY, POLICY_LOGITS_BIAS_KEY}
    migrated = {key: value.clone() for key, value in state_dict.items() if key not in head_keys}
    migrated[POLICY_LOGITS_WEIGHT_KEY] = new_weight
    migrated[POLICY_LOGITS_BIAS_KEY] = new_bias
    return migrated


def migrate_encoder_trunk_input_width(
    state_dict: dict[str, torch.Tensor], target_feature_dim: int
) -> dict[str, torch.Tensor]:
    """Widen the encoder's first trunk ``Linear`` to the current input width.

    The encoder appends any new input features at the END of its pre-trunk concat
    (see :mod:`sts_rl.agent.encoder`), so an old checkpoint's trained columns are
    the leading prefix of the wider weight. This rebuilds
    :data:`TRUNK_INPUT_WEIGHT_KEY` as ``(hidden, target_feature_dim)``, copies the
    old columns into ``[:, :old_feat]``, and leaves the appended suffix columns
    zero. Zeroing (rather than random-initializing) the suffix keeps the new
    pathway neutral on the first forward, so the warm-started combat behavior the
    checkpoint already learned is preserved rather than perturbed.

    On the widen path, returns a NEW state_dict that shares NO tensor storage
    with the input: the rebuilt weight is freshly allocated and every
    carried-through tensor is cloned, matching :func:`migrate_policy_head`'s
    discipline. An equal width is a no-op returning the input dict unchanged
    (shared storage), so an already-current checkpoint loads verbatim.

    Raises :class:`~sts_rl.interface.InterfaceError` if the trunk weight is absent
    (not an ``ActorCritic`` checkpoint), if it is not 2-D, if its width exceeds
    ``target_feature_dim`` (shrinking the trunk input is unsupported - it would
    drop trained columns), or if a narrower width is not one of
    :data:`_KNOWN_PRIOR_FEATURE_DIMS` (an unrecorded layout whose columns would
    mismap under the append-at-end prefix-copy).
    """
    if TRUNK_INPUT_WEIGHT_KEY not in state_dict:
        raise InterfaceError(
            "state_dict is missing the encoder first-trunk weight "
            f"({TRUNK_INPUT_WEIGHT_KEY!r}); not an ActorCritic checkpoint?"
        )

    old_weight = state_dict[TRUNK_INPUT_WEIGHT_KEY]

    # The first-trunk weight must be 2-D (hidden, feature_dim); guard before
    # unpacking shape below so a malformed head fails typed here.
    if old_weight.ndim != 2:
        raise InterfaceError(
            f"checkpoint first-trunk weight {TRUNK_INPUT_WEIGHT_KEY!r} must be "
            f"2-D (hidden, feature_dim), got {old_weight.ndim}-D shape "
            f"{tuple(old_weight.shape)}"
        )

    hidden, old_feat = old_weight.shape[0], old_weight.shape[1]
    if old_feat == target_feature_dim:
        return state_dict  # widths already match: no migration needed
    if old_feat > target_feature_dim:
        raise InterfaceError(
            f"checkpoint encoder feature width {old_feat} exceeds the current "
            f"{target_feature_dim}; cannot shrink the trunk input"
        )

    # old_feat < target: widen. The prefix-copy is a valid remap ONLY when the
    # checkpoint's width is a real prior layout whose new features were all
    # appended at the END of the concat; a width from any other layout (a block
    # widened or inserted BEFORE the reward block) would mismap columns. Mirroring
    # the policy head's _OLD_ACTION_DIM guard, reject a width not in the recorded
    # history rather than silently corrupting the warm-started combat weights.
    if old_feat not in _KNOWN_PRIOR_FEATURE_DIMS:
        raise InterfaceError(
            f"checkpoint encoder feature width {old_feat} is not a known prior "
            f"width {_KNOWN_PRIOR_FEATURE_DIMS}; a narrower width from an unrecorded "
            f"layout would mismap trunk columns (the prefix-copy widen is safe only "
            f"when new features were appended at the END of the concat)"
        )

    # Copy trained columns to the leading prefix and leave the appended suffix
    # zero. new_zeros preserves dtype and device.
    logger.info(
        "encoder trunk-input-width migration widens first-trunk input %d -> %d, "
        "zero-initializing %d appended suffix column(s)",
        old_feat,
        target_feature_dim,
        target_feature_dim - old_feat,
    )
    # CAUTION: the prefix-copy is correct only because the encoder appends new
    # features at the END of the concat; a mid-concat change would mismap columns.
    # This invariant is now ENFORCED by the _KNOWN_PRIOR_FEATURE_DIMS guard above
    # (an unrecorded width raises), not merely documented here.
    new_weight = old_weight.new_zeros((hidden, target_feature_dim))
    new_weight[:, :old_feat] = old_weight

    # Clone every carried-through tensor so the returned dict shares no storage
    # with the input; the rebuilt weight is already independent.
    migrated = {
        key: value.clone() for key, value in state_dict.items() if key != TRUNK_INPUT_WEIGHT_KEY
    }
    migrated[TRUNK_INPUT_WEIGHT_KEY] = new_weight
    return migrated


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> ActorCritic:
    """Load a training checkpoint into an :class:`ActorCritic`, migrating if needed.

    Reads the :func:`~sts_rl.agent.train._save_checkpoint` payload (keys
    :data:`~sts_rl.agent.train.CHECKPOINT_MODEL_KEY`,
    :data:`~sts_rl.agent.train.CHECKPOINT_HIDDEN_DIM_KEY`,
    :data:`~sts_rl.agent.train.CHECKPOINT_INTERFACE_VERSION_KEY`), builds an
    ``ActorCritic`` at the checkpoint's own trunk width, and loads the weights
    ``strict=True``.

    If the checkpoint's policy head is already the current width
    (:data:`~sts_rl.interface.ACTION_DIM`) it loads verbatim regardless of the
    recorded ``interface_version`` - an already-current checkpoint, or a version
    bump that left the action layout unchanged, needs no remap and no recorded old
    layout. Only a head whose width differs from the current layout is migrated to
    the current action layout (see :func:`migrate_policy_head`); on that migration
    path a checkpoint newer than the current
    :data:`~sts_rl.interface.INTERFACE_VERSION` is a hard error (this loader cannot
    downgrade a layout it does not know) and an unrecorded old version is rejected
    by :func:`migrate_policy_head`.

    Independently, the encoder's first trunk ``Linear`` is widened when the
    checkpoint's input width is narrower than the current encoder's feature dim
    (the concat gained trailing features); see
    :func:`migrate_encoder_trunk_input_width`. Both migrations compose: a checkpoint may
    need the head remap, the trunk widening, both, or neither. Their target widths
    are read from a freshly built ``ActorCritic`` so they track the live
    architecture rather than any recorded constant.

    ``strict=True`` is deliberate: after migration every current key must be
    present and correctly shaped, so a missing or misshaped tensor is a hard
    error, not a silent partial load. ``map_location`` defaults to ``"cpu"`` so a
    checkpoint loads regardless of the device it was trained on; move the returned
    model with ``.to(device)`` afterward if needed.

    ``weights_only=True`` is passed to :func:`torch.load`: the payload holds only
    tensors and plain scalars, so the safe unpickler suffices and an untrusted
    checkpoint cannot execute arbitrary code on load.
    """
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=True)
    for key in (CHECKPOINT_MODEL_KEY, CHECKPOINT_HIDDEN_DIM_KEY, CHECKPOINT_INTERFACE_VERSION_KEY):
        if key not in payload:
            raise InterfaceError(f"checkpoint at {path} is missing required key {key!r}")

    model_state: Any = payload[CHECKPOINT_MODEL_KEY]
    hidden_dim: Any = payload[CHECKPOINT_HIDDEN_DIM_KEY]
    ckpt_version: str = payload[CHECKPOINT_INTERFACE_VERSION_KEY]

    # Build the target model first and read BOTH target dims from its own
    # state_dict, so the two migrations size against a single source of truth (the
    # live architecture) rather than any recorded constant.
    model = ActorCritic(hidden_dim=hidden_dim)
    target_state = model.state_dict()
    target_action_dim = target_state[POLICY_LOGITS_WEIGHT_KEY].shape[0]
    target_feature_dim = target_state[TRUNK_INPUT_WEIGHT_KEY].shape[1]

    # Policy-head remap: only when the recorded head width differs from the live
    # action layout. An unchanged layout across a version bump loads verbatim
    # without a recorded old layout; the reject-newer / unknown-version guards
    # apply only on this migration path.
    head_weight = model_state.get(POLICY_LOGITS_WEIGHT_KEY)
    head_width = head_weight.shape[0] if head_weight is not None else None
    if head_width != target_action_dim:
        if _version_tuple(ckpt_version) > _version_tuple(INTERFACE_VERSION):
            raise InterfaceError(
                f"checkpoint interface {ckpt_version} is newer than the current "
                f"{INTERFACE_VERSION}; cannot downgrade"
            )
        model_state = migrate_policy_head(model_state, from_version=ckpt_version)

    # Trunk-input-width migration: independent of the head remap and composes with it.
    # When the encoder gained input features the old first-trunk Linear is too
    # narrow; widen it and zero-init the appended suffix. A no-op when the widths
    # already match (verbatim load).
    model_state = migrate_encoder_trunk_input_width(model_state, target_feature_dim)

    model.load_state_dict(model_state, strict=True)
    return model
