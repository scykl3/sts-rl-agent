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

    # A head already at the current width needs no remap, whatever the recorded
    # version: an unchanged action layout across a version bump loads verbatim
    # without a recorded old layout. Only a differing width is a genuine
    # migration, and the reject-newer / unknown-version guards apply on that path.
    head_weight = model_state.get(POLICY_LOGITS_WEIGHT_KEY)
    head_width = head_weight.shape[0] if head_weight is not None else None
    if head_width != ACTION_DIM:
        if _version_tuple(ckpt_version) > _version_tuple(INTERFACE_VERSION):
            raise InterfaceError(
                f"checkpoint interface {ckpt_version} is newer than the current "
                f"{INTERFACE_VERSION}; cannot downgrade"
            )
        model_state = migrate_policy_head(model_state, from_version=ckpt_version)

    model = ActorCritic(hidden_dim=hidden_dim)
    model.load_state_dict(model_state, strict=True)
    return model
