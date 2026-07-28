"""Tests for the name-aware policy-head checkpoint migration.

Engine-free: the migration is pure tensor bookkeeping, so these build an
old-layout state_dict from a fresh :class:`ActorCritic` (its encoder / trunk /
value shapes are identical across the 0.3.0 -> 0.4.0 bump; only the action head
changed width) with the policy head shrunk to the historical 155-row layout and
seeded with per-block sentinels, then assert the migration lands each old block's
rows at its new offset by name.

Offsets are never hardcoded here: old offsets come from the embedded
:data:`OLD_ACTION_LAYOUTS` table and new offsets from the live
:data:`ACTION_BLOCK_BY_NAME`, so the test tracks the same sources as the code.
"""

from __future__ import annotations

import logging

import pytest
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.encoder import CARD_EMBED_DIM, _PILE_POOLS
from sts_rl.agent.checkpoint_migration import (
    OLD_ACTION_LAYOUTS,
    POLICY_LOGITS_BIAS_KEY,
    POLICY_LOGITS_WEIGHT_KEY,
    TRUNK_INPUT_WEIGHT_KEY,
    _KNOWN_PRIOR_FEATURE_DIMS,
    _version_tuple,
    load_checkpoint,
    migrate_encoder_trunk_input_width,
    migrate_policy_head,
)
from sts_rl.agent.train import (
    CHECKPOINT_HIDDEN_DIM_KEY,
    CHECKPOINT_INTERFACE_VERSION_KEY,
    CHECKPOINT_MODEL_KEY,
)
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    INTERFACE_VERSION,
    KEYS_ACT_DIM,
    InterfaceError,
)

# Small trunk width keeps the seeded head tiny; the migration is width-agnostic.
HIDDEN = 4
OLD_VERSION = "0.3.0"
OLD_LAYOUT = OLD_ACTION_LAYOUTS[OLD_VERSION]
OLD_ACTION_DIM = sum(count for _, count in OLD_LAYOUT.values())

# Per-block sentinel stride: block N's rows carry values around N * STRIDE, and a
# within-block row i adds i, so every old row's value is unique and identifies
# both its block and its position. STRIDE exceeds the widest block (50) so the
# +i offsets never collide across blocks.
_BLOCK_CODE_STRIDE = 1000

# Current-layout blocks that had NO same-named block in 0.3.0, so the migration
# must zero-initialize them. REWARD_SELECT is deliberately here: 0.3.0's
# CARD_REWARD_SELECT is a different block (name, width, semantics), not its source.
NEW_ONLY_BLOCKS = ("REWARD_SELECT", "TREASURE_SELECT")

# The recorded prior trunk widths the widen guard accepts (the narrower is the
# pre-reward-vision combat width; the wider is the card-vision width). The widen
# tests below pin old_feat to one of these so they track _KNOWN_PRIOR_FEATURE_DIMS
# rather than a synthetic ``target - <block widths>`` width, which each later block
# append pushes out of the guard's accepted set (the offered-relic/potion append and
# then the card-select append both did that to the old ``target - <block>`` forms).
PRE_REWARD_FEATURE_DIM = min(_KNOWN_PRIOR_FEATURE_DIMS)  # combat / pre-reward-vision
CARD_VISION_FEATURE_DIM = max(_KNOWN_PRIOR_FEATURE_DIMS)  # after the reward-card block


def _seed_old_head(hidden: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a 155-row policy head with a recognizable sentinel per old block.

    ``weight[r, c] = code(block) + i + c/100`` and ``bias[r] = -(code(block) + i)``
    for the row ``r = start + i`` of each old block, so every element is distinct
    (guarding against a transpose or off-by-one) and every row's value names its
    origin block and position.
    """
    weight = torch.zeros(OLD_ACTION_DIM, hidden)
    bias = torch.zeros(OLD_ACTION_DIM)
    column_ramp = torch.arange(hidden, dtype=torch.float32) / 100.0
    for order, (_name, (start, count)) in enumerate(OLD_LAYOUT.items()):
        code = float((order + 1) * _BLOCK_CODE_STRIDE)
        for i in range(count):
            row = start + i
            weight[row] = code + i + column_ramp
            bias[row] = -(code + i)
    return weight, bias


def _build_old_state_dict(hidden: int = HIDDEN) -> dict[str, torch.Tensor]:
    """A full ActorCritic state_dict with the head replaced by the seeded 155-row one.

    Cloned so the returned dict owns its tensors independently of the throwaway
    ``ActorCritic`` (the migration shares non-head references, so ownership here
    keeps the byte-identity assertion meaningful).
    """
    reference = ActorCritic(hidden_dim=hidden)
    state_dict = {key: value.clone() for key, value in reference.state_dict().items()}
    weight, bias = _seed_old_head(hidden)
    state_dict[POLICY_LOGITS_WEIGHT_KEY] = weight
    state_dict[POLICY_LOGITS_BIAS_KEY] = bias
    return state_dict


def _build_current_state_dict(hidden: int = HIDDEN) -> dict[str, torch.Tensor]:
    """A full current-layout ActorCritic state_dict (head and trunk at live widths).

    Cloned so it owns its tensors. The trunk-width round-trip swaps in a narrower
    first-trunk weight while leaving every other tensor at its current shape, so a
    strict load exercises only the widening.
    """
    reference = ActorCritic(hidden_dim=hidden)
    return {key: value.clone() for key, value in reference.state_dict().items()}


def _current_feature_dim(hidden: int = HIDDEN) -> int:
    """The live encoder's first-trunk in_features, read from a fresh model."""
    return int(ActorCritic(hidden_dim=hidden).state_dict()[TRUNK_INPUT_WEIGHT_KEY].shape[1])


def test_migrated_head_has_current_action_dim_and_loads_strict() -> None:
    """Migration resizes the head to ACTION_DIM and loads strict into a fresh net."""
    old_state = _build_old_state_dict()
    migrated = migrate_policy_head(old_state, OLD_VERSION)

    assert migrated[POLICY_LOGITS_WEIGHT_KEY].shape == (ACTION_DIM, HIDDEN)
    assert migrated[POLICY_LOGITS_BIAS_KEY].shape == (ACTION_DIM,)

    model = ActorCritic(hidden_dim=HIDDEN)
    # strict=True: every current key present and correctly shaped, or it raises.
    model.load_state_dict(migrated, strict=True)
    assert torch.equal(model.policy.logits.weight, migrated[POLICY_LOGITS_WEIGHT_KEY])
    assert torch.equal(model.policy.logits.bias, migrated[POLICY_LOGITS_BIAS_KEY])


def test_every_shared_block_lands_at_its_new_offset_by_name() -> None:
    """Each block present in both layouts has its old rows copied to the new offset."""
    old_state = _build_old_state_dict()
    old_weight = old_state[POLICY_LOGITS_WEIGHT_KEY]
    old_bias = old_state[POLICY_LOGITS_BIAS_KEY]
    migrated = migrate_policy_head(old_state, OLD_VERSION)
    new_weight = migrated[POLICY_LOGITS_WEIGHT_KEY]
    new_bias = migrated[POLICY_LOGITS_BIAS_KEY]

    shared = set(OLD_LAYOUT) & set(ACTION_BLOCK_BY_NAME)
    assert shared, "expected overlapping block names between the two layouts"
    for name in shared:
        old_start, old_count = OLD_LAYOUT[name]
        block = ACTION_BLOCK_BY_NAME[name]
        rows = min(old_count, block.count)
        assert torch.equal(
            new_weight[block.start : block.start + rows],
            old_weight[old_start : old_start + rows],
        ), f"{name}: weight rows not copied to new offset {block.start}"
        assert torch.equal(
            new_bias[block.start : block.start + rows],
            old_bias[old_start : old_start + rows],
        ), f"{name}: bias rows not copied to new offset {block.start}"


@pytest.mark.parametrize(
    "name",
    [
        "CONFIRM_SELECT",  # 154 -> 106, the headline remap
        "END_TURN",  # a combat block (unchanged offset, still verified by name)
        "PLAY_CARD_TARGETED",  # a wide combat block
        "MAP_SELECT",
        "SHOP_SELECT",
        "EVENT_SELECT",
        "BOSS_RELIC_SELECT",
        "PROCEED",
        "REST_SELECT",
    ],
)
def test_spot_check_named_block_remap(name: str) -> None:
    """Named spot-checks: the block's first row carries its old sentinel at the new offset."""
    old_state = _build_old_state_dict()
    old_weight = old_state[POLICY_LOGITS_WEIGHT_KEY]
    migrated = migrate_policy_head(old_state, OLD_VERSION)

    old_start, _ = OLD_LAYOUT[name]
    new_start = ACTION_BLOCK_BY_NAME[name].start
    assert torch.equal(
        migrated[POLICY_LOGITS_WEIGHT_KEY][new_start],
        old_weight[old_start],
    ), f"{name}: row from old offset {old_start} did not land at new offset {new_start}"


def test_confirm_select_remaps_to_current_offset() -> None:
    """Headline remap: old CONFIRM_SELECT@154 lands at its current interface offset,
    derived live (never a hardcoded literal) so it survives later layout shifts."""
    old_start, _ = OLD_LAYOUT["CONFIRM_SELECT"]
    assert old_start == 154  # fixed 0.3.0 history
    new_start = ACTION_BLOCK_BY_NAME["CONFIRM_SELECT"].start
    assert new_start != old_start  # CONFIRM shifted when the overworld blocks were inserted
    state = _build_old_state_dict()
    migrated = migrate_policy_head(state, OLD_VERSION)
    assert torch.equal(
        migrated[POLICY_LOGITS_WEIGHT_KEY][new_start],
        state[POLICY_LOGITS_WEIGHT_KEY][old_start],
    )


def test_new_only_blocks_are_zero_initialized() -> None:
    """Blocks with no old counterpart (REWARD_SELECT, TREASURE_SELECT) are zeroed."""
    migrated = migrate_policy_head(_build_old_state_dict(), OLD_VERSION)
    new_weight = migrated[POLICY_LOGITS_WEIGHT_KEY]
    new_bias = migrated[POLICY_LOGITS_BIAS_KEY]
    for name in NEW_ONLY_BLOCKS:
        block = ACTION_BLOCK_BY_NAME[name]
        assert (
            torch.count_nonzero(new_weight[block.start : block.stop]) == 0
        ), f"{name}: expected zero-init weight rows"
        assert (
            torch.count_nonzero(new_bias[block.start : block.stop]) == 0
        ), f"{name}: expected zero-init bias rows"


def test_reward_select_is_not_sourced_from_card_reward_select() -> None:
    """0.3.0 CARD_REWARD_SELECT must NOT be remapped into 0.4.0 REWARD_SELECT.

    They differ in name and semantics, so REWARD_SELECT is zero-init and shares no
    row with the old CARD_REWARD_SELECT weights.
    """
    old_state = _build_old_state_dict()
    old_weight = old_state[POLICY_LOGITS_WEIGHT_KEY]
    migrated = migrate_policy_head(old_state, OLD_VERSION)

    card_reward_start, card_reward_count = OLD_LAYOUT["CARD_REWARD_SELECT"]
    reward_block = ACTION_BLOCK_BY_NAME["REWARD_SELECT"]
    reward_rows = migrated[POLICY_LOGITS_WEIGHT_KEY][reward_block.start : reward_block.stop]
    assert torch.count_nonzero(reward_rows) == 0
    # The old CARD_REWARD_SELECT rows were nonzero sentinels; confirm none leaked in.
    old_card_reward = old_weight[card_reward_start : card_reward_start + card_reward_count]
    assert torch.count_nonzero(old_card_reward) > 0
    assert not torch.equal(reward_rows[:card_reward_count], old_card_reward)


def test_dropped_block_with_trained_rows_warns(caplog) -> None:
    """A dropped OLD block (CARD_REWARD_SELECT) with nonzero rows logs a warning.

    The seeded old head gives every old block nonzero sentinels, so the removed
    CARD_REWARD_SELECT block trips the signal-loss diagnostic.
    """
    old_state = _build_old_state_dict()
    with caplog.at_level(logging.WARNING, logger="sts_rl.agent.checkpoint_migration"):
        migrate_policy_head(old_state, OLD_VERSION)
    assert any(
        "CARD_REWARD_SELECT" in record.getMessage() for record in caplog.records
    ), "expected a warning naming the dropped CARD_REWARD_SELECT block"


def test_dropped_block_with_zero_rows_does_not_warn(caplog) -> None:
    """A dropped OLD block whose rows are all zero (untrained) logs nothing.

    Combat-only checkpoints leave these overworld rows at random init; zeroing
    them models the no-trained-signal case, which must not warn.
    """
    old_state = _build_old_state_dict()
    start, count = OLD_LAYOUT["CARD_REWARD_SELECT"]
    old_state[POLICY_LOGITS_WEIGHT_KEY][start : start + count] = 0.0
    with caplog.at_level(logging.WARNING, logger="sts_rl.agent.checkpoint_migration"):
        migrate_policy_head(old_state, OLD_VERSION)
    assert not any(
        "CARD_REWARD_SELECT" in record.getMessage() for record in caplog.records
    ), "no warning expected when the dropped block carried no trained signal"


def test_shrunk_or_grown_block_copies_only_shared_rows() -> None:
    """REST_SELECT grew 6 -> 7: the 6 old rows copy, the 7th stays zero."""
    old_state = _build_old_state_dict()
    old_weight = old_state[POLICY_LOGITS_WEIGHT_KEY]
    migrated = migrate_policy_head(old_state, OLD_VERSION)
    new_weight = migrated[POLICY_LOGITS_WEIGHT_KEY]

    old_start, old_count = OLD_LAYOUT["REST_SELECT"]
    block = ACTION_BLOCK_BY_NAME["REST_SELECT"]
    assert old_count < block.count  # guard the premise of this test
    copied = min(old_count, block.count)
    assert torch.equal(
        new_weight[block.start : block.start + copied],
        old_weight[old_start : old_start + copied],
    )
    # The extra new row(s) beyond the old width are zero-initialized.
    assert torch.count_nonzero(new_weight[block.start + copied : block.stop]) == 0


def test_non_head_tensors_are_byte_identical() -> None:
    """Every non-policy-head tensor is carried through unchanged."""
    old_state = _build_old_state_dict()
    migrated = migrate_policy_head(old_state, OLD_VERSION)

    head_keys = {POLICY_LOGITS_WEIGHT_KEY, POLICY_LOGITS_BIAS_KEY}
    assert set(migrated) == set(old_state)  # no keys added or dropped
    non_head = [key for key in migrated if key not in head_keys]
    assert non_head, "expected encoder/trunk/value keys to carry through"
    for key in non_head:
        assert torch.equal(migrated[key], old_state[key]), f"{key} was modified"


def test_migrated_non_head_tensors_are_isolated_copies() -> None:
    """Mutating a carried-through tensor in the output leaves the input unchanged.

    The migration clones non-head tensors, so the returned dict shares no storage
    with the caller's; this guards against a shallow copy that would alias them
    (identity and value, not merely torch.equal).
    """
    old_state = _build_old_state_dict()
    migrated = migrate_policy_head(old_state, OLD_VERSION)

    head_keys = {POLICY_LOGITS_WEIGHT_KEY, POLICY_LOGITS_BIAS_KEY}
    key = next(k for k, v in migrated.items() if k not in head_keys and v.is_floating_point())
    # Distinct objects, not the same tensor carried by reference.
    assert migrated[key] is not old_state[key]
    before = old_state[key].clone()
    migrated[key].add_(1.0)  # mutate the output tensor in place
    # The input tensor's values are unchanged: no shared storage.
    assert torch.equal(old_state[key], before)
    assert not torch.equal(old_state[key], migrated[key])


def test_input_state_dict_is_not_mutated() -> None:
    """The old head tensors in the input are untouched (migration returns a new dict)."""
    old_state = _build_old_state_dict()
    weight_before = old_state[POLICY_LOGITS_WEIGHT_KEY].clone()
    migrate_policy_head(old_state, OLD_VERSION)
    assert torch.equal(old_state[POLICY_LOGITS_WEIGHT_KEY], weight_before)
    assert old_state[POLICY_LOGITS_WEIGHT_KEY].shape == (OLD_ACTION_DIM, HIDDEN)


def test_positional_copy_would_misplace_confirm_select() -> None:
    """Revert guard: a positional (identity) copy fails the by-name CONFIRM check.

    Proves the CONFIRM_SELECT assertions above are discriminating - the name-aware
    remap puts the old row-154 sentinel at new offset 106, whereas a row-for-row
    copy (the WRONG migration) leaves offset 106 holding old row 106's value.
    """
    old_state = _build_old_state_dict()
    old_weight = old_state[POLICY_LOGITS_WEIGHT_KEY]
    migrated = migrate_policy_head(old_state, OLD_VERSION)

    confirm_old_start = OLD_LAYOUT["CONFIRM_SELECT"][0]  # 154
    confirm_new_start = ACTION_BLOCK_BY_NAME["CONFIRM_SELECT"].start  # 106

    # A positional migration: copy the shared prefix row-for-row, zero the rest.
    positional = old_weight.new_zeros((ACTION_DIM, HIDDEN))
    shared_rows = min(OLD_ACTION_DIM, ACTION_DIM)
    positional[:shared_rows] = old_weight[:shared_rows]

    # Name-aware migration lands CONFIRM's trained row at the new offset...
    assert torch.equal(
        migrated[POLICY_LOGITS_WEIGHT_KEY][confirm_new_start], old_weight[confirm_old_start]
    )
    # ...but a positional copy does NOT (it leaves a different block's row there).
    assert not torch.equal(positional[confirm_new_start], old_weight[confirm_old_start])
    # And the positional copy strands CONFIRM's real weights back at old row 154.
    assert torch.equal(positional[confirm_old_start], old_weight[confirm_old_start])


def test_migrate_rejects_unknown_version() -> None:
    """An unrecognized from_version raises rather than guessing a layout."""
    with pytest.raises(InterfaceError, match="no embedded action layout"):
        migrate_policy_head(_build_old_state_dict(), "0.2.0")


def test_migrate_rejects_missing_policy_head_keys() -> None:
    """A state_dict without the logit tensors is rejected."""
    broken = _build_old_state_dict()
    del broken[POLICY_LOGITS_WEIGHT_KEY]
    with pytest.raises(InterfaceError, match="missing the policy-head logit tensors"):
        migrate_policy_head(broken, OLD_VERSION)


def test_migrate_rejects_head_width_mismatch() -> None:
    """A head whose row count disagrees with the recorded old layout is rejected."""
    mismatched = _build_old_state_dict()
    # Truncate the head to a width that no longer matches the 0.3.0 layout total.
    mismatched[POLICY_LOGITS_WEIGHT_KEY] = mismatched[POLICY_LOGITS_WEIGHT_KEY][:-1]
    with pytest.raises(InterfaceError, match="version/layout mismatch"):
        migrate_policy_head(mismatched, OLD_VERSION)


def test_migrate_rejects_non_2d_policy_head() -> None:
    """A malformed 1-D logit weight fails typed, not with a bare IndexError."""
    broken = _build_old_state_dict()
    # Flatten the head to 1-D so reading shape[1] would otherwise IndexError.
    broken[POLICY_LOGITS_WEIGHT_KEY] = broken[POLICY_LOGITS_WEIGHT_KEY].reshape(-1)
    with pytest.raises(InterfaceError, match="must be 2-D"):
        migrate_policy_head(broken, OLD_VERSION)


def test_version_tuple_orders_numerically_not_lexically() -> None:
    """Semver compare is componentwise: 0.10.0 sorts after 0.9.0, unlike strings."""
    assert _version_tuple("0.10.0") > _version_tuple("0.9.0")
    # Counterexample: lexical string order ranks "0.10.0" before "0.9.0".
    assert "0.10.0" < "0.9.0"


def test_version_tuple_rejects_unparseable_version() -> None:
    """A non-integer version component raises a typed InterfaceError, not ValueError."""
    with pytest.raises(InterfaceError, match="unparseable interface version"):
        _version_tuple("1.0.0-rc1")


def _write_checkpoint(path, state_dict: dict[str, torch.Tensor], version: str, hidden: int) -> None:
    """Write a minimal _save_checkpoint-shaped payload to ``path``."""
    payload = {
        CHECKPOINT_MODEL_KEY: state_dict,
        CHECKPOINT_HIDDEN_DIM_KEY: hidden,
        CHECKPOINT_INTERFACE_VERSION_KEY: version,
    }
    torch.save(payload, path)


def test_load_checkpoint_migrates_old_version(tmp_path) -> None:
    """load_checkpoint reads an old-version payload, migrates, and returns a live net."""
    old_state = _build_old_state_dict()
    old_weight = old_state[POLICY_LOGITS_WEIGHT_KEY]
    path = tmp_path / "old.pt"
    _write_checkpoint(path, old_state, OLD_VERSION, HIDDEN)

    model = load_checkpoint(path)
    assert isinstance(model, ActorCritic)
    assert model.policy.logits.weight.shape == (ACTION_DIM, HIDDEN)
    # CONFIRM_SELECT's trained row landed at the new offset via the load path too.
    confirm_old_start = OLD_LAYOUT["CONFIRM_SELECT"][0]
    confirm_new_start = ACTION_BLOCK_BY_NAME["CONFIRM_SELECT"].start
    assert torch.equal(model.policy.logits.weight[confirm_new_start], old_weight[confirm_old_start])
    # A new-only block is zero.
    treasure = ACTION_BLOCK_BY_NAME["TREASURE_SELECT"]
    assert torch.count_nonzero(model.policy.logits.weight[treasure.start : treasure.stop]) == 0


def test_load_checkpoint_same_version_loads_verbatim(tmp_path) -> None:
    """A current-version checkpoint loads with no migration, weights unchanged."""
    reference = ActorCritic(hidden_dim=HIDDEN)
    path = tmp_path / "current.pt"
    _write_checkpoint(path, reference.state_dict(), INTERFACE_VERSION, HIDDEN)

    model = load_checkpoint(path)
    assert torch.equal(model.policy.logits.weight, reference.policy.logits.weight)
    assert torch.equal(model.encoder.card_embed.weight, reference.encoder.card_embed.weight)
    # The trunk no-op path protects this: an already-current first-trunk weight
    # loads byte-identical, not silently rebuilt.
    assert torch.equal(model.encoder.trunk[0].weight, reference.encoder.trunk[0].weight)


def test_load_checkpoint_equal_width_head_loads_verbatim(tmp_path) -> None:
    """A current-width head loads verbatim regardless of the recorded version.

    The verbatim fallback keys on head width, not the version string: a version
    bump that leaves the action layout unchanged (same ACTION_DIM) loads without a
    recorded old layout and never hits the reject-newer guard, even for a version
    numerically newer than the current interface.
    """
    reference = ActorCritic(hidden_dim=HIDDEN)
    path = tmp_path / "future_same_layout.pt"
    # A FUTURE version with no recorded old layout, but a current-width head.
    _write_checkpoint(path, reference.state_dict(), "99.0.0", HIDDEN)

    model = load_checkpoint(path)  # no migration, no reject-newer
    assert torch.equal(model.policy.logits.weight, reference.policy.logits.weight)
    assert torch.equal(model.policy.logits.bias, reference.policy.logits.bias)


def test_load_checkpoint_rejects_newer_version(tmp_path) -> None:
    """A newer-version checkpoint whose head needs migration cannot be downgraded.

    The reject-newer guard applies only on the migration path (head width differs
    from the current ACTION_DIM); a same-width head would instead load verbatim.
    """
    old_state = _build_old_state_dict()  # 155-row head, differs from ACTION_DIM
    path = tmp_path / "future.pt"
    _write_checkpoint(path, old_state, "99.0.0", HIDDEN)
    with pytest.raises(InterfaceError, match="newer than the current"):
        load_checkpoint(path)


def test_load_checkpoint_rejects_missing_key(tmp_path) -> None:
    """A payload missing a required key is rejected with a clear error."""
    path = tmp_path / "bad.pt"
    torch.save({CHECKPOINT_MODEL_KEY: {}}, path)
    with pytest.raises(InterfaceError, match="missing required key"):
        load_checkpoint(path)


def test_trunk_input_width_migration_preserves_prefix_and_zeros_suffix() -> None:
    """Widening the first-trunk Linear copies old columns and zero-inits the suffix.

    Synthesizes a narrower (pre-card-vision) first-trunk weight with an arange
    sentinel so every original column is identifiable, migrates to the current
    feature dim, and loads strict into a fresh ActorCritic. Asserts (a) it loads,
    (b) the leading old columns survive, and (c) the appended suffix is exactly
    zero - the append-at-end coupling with the encoder.
    """
    target = _current_feature_dim()
    old_feat = PRE_REWARD_FEATURE_DIM  # a recorded prior width the widen guard accepts
    assert old_feat < target  # guard the premise: the encoder actually widened

    state = _build_current_state_dict()
    old_weight = torch.arange(HIDDEN * old_feat, dtype=torch.float32).reshape(HIDDEN, old_feat)
    state[TRUNK_INPUT_WEIGHT_KEY] = old_weight

    migrated = migrate_encoder_trunk_input_width(state, target)
    new_weight = migrated[TRUNK_INPUT_WEIGHT_KEY]
    assert new_weight.shape == (HIDDEN, target)
    assert torch.equal(new_weight[:, :old_feat], old_weight)  # (b) leading columns preserved
    assert torch.count_nonzero(new_weight[:, old_feat:]) == 0  # (c) suffix exactly zero

    model = ActorCritic(hidden_dim=HIDDEN)
    model.load_state_dict(migrated, strict=True)  # (a) loads strict
    assert torch.equal(model.encoder.trunk[0].weight, new_weight)


def test_migrate_trunk_equal_width_is_noop() -> None:
    """An already-current width is a no-op: the input dict is returned unchanged."""
    target = _current_feature_dim()
    state = _build_current_state_dict()  # its first-trunk weight is already at target
    migrated = migrate_encoder_trunk_input_width(state, target)
    # Same object (and same tensor) returned, so an already-current checkpoint
    # loads verbatim rather than through a needless rebuild.
    assert migrated is state
    assert migrated[TRUNK_INPUT_WEIGHT_KEY] is state[TRUNK_INPUT_WEIGHT_KEY]


def test_current_feature_dim_not_recorded_as_prior_width() -> None:
    """The live feature dim must NOT be in _KNOWN_PRIOR_FEATURE_DIMS.

    Recorded priors are strictly PRE-APPEND widths; the current width is reached via
    the equal-width no-op path above, never the widen path. Listing it would let a
    genuinely-current checkpoint be mistaken for a prior to widen from.
    """
    assert _current_feature_dim() not in _KNOWN_PRIOR_FEATURE_DIMS


def test_migrate_trunk_output_is_isolated_copy() -> None:
    """After a widen, mutating a carried-through tensor leaves the input unchanged.

    The widen path clones every carried-through tensor, so the returned dict
    shares no storage with the caller's; this guards against a shallow copy that
    would alias them (identity and value, not merely torch.equal). Mirrors the
    head migration's isolation test.
    """
    target = _current_feature_dim()
    old_feat = PRE_REWARD_FEATURE_DIM  # a recorded prior width the widen guard accepts
    state = _build_current_state_dict()
    narrow = torch.arange(HIDDEN * old_feat, dtype=torch.float32).reshape(HIDDEN, old_feat)
    state[TRUNK_INPUT_WEIGHT_KEY] = narrow
    migrated = migrate_encoder_trunk_input_width(state, target)

    # A carried-through (non-trunk) float tensor: distinct object, not aliased.
    key = next(
        k for k, v in migrated.items() if k != TRUNK_INPUT_WEIGHT_KEY and v.is_floating_point()
    )
    assert migrated[key] is not state[key]
    before = state[key].clone()
    migrated[key].add_(1.0)  # mutate the output tensor in place
    assert torch.equal(state[key], before)  # input unchanged: no shared storage
    assert not torch.equal(state[key], migrated[key])


def test_migrate_trunk_rejects_missing_key() -> None:
    """A state_dict without the first-trunk weight is rejected (not an ActorCritic)."""
    broken = _build_current_state_dict()
    del broken[TRUNK_INPUT_WEIGHT_KEY]
    with pytest.raises(InterfaceError, match="missing the encoder first-trunk weight"):
        migrate_encoder_trunk_input_width(broken, _current_feature_dim())


def test_migrate_trunk_rejects_non_2d() -> None:
    """A malformed 1-D first-trunk weight fails typed, not with a bare error."""
    broken = _build_current_state_dict()
    broken[TRUNK_INPUT_WEIGHT_KEY] = broken[TRUNK_INPUT_WEIGHT_KEY].reshape(-1)
    with pytest.raises(InterfaceError, match="must be 2-D"):
        migrate_encoder_trunk_input_width(broken, _current_feature_dim())


def test_migrate_trunk_rejects_shrink() -> None:
    """A trunk wider than the target is rejected: shrinking would drop trained columns."""
    target = _current_feature_dim()
    state = _build_current_state_dict()
    # Over-wide the trunk so a migration would have to shrink (old_feat > target).
    state[TRUNK_INPUT_WEIGHT_KEY] = torch.zeros(HIDDEN, target + 1)
    with pytest.raises(InterfaceError, match="cannot shrink the trunk input"):
        migrate_encoder_trunk_input_width(state, target)


def test_migrate_trunk_rejects_unknown_prior_width() -> None:
    """A narrower width not in the recorded prior history is rejected as a mismap.

    The widen path treats the trained columns as a clean leading prefix, which is
    correct ONLY for a real prior layout (new features appended at the END). A
    width from no recorded layout - e.g. a block widened or inserted BEFORE the
    reward block - would mismap columns, so the guard raises rather than silently
    corrupting the warm-started weights, mirroring the policy head's
    _OLD_ACTION_DIM check. This is the revert guard: dropping the width check lets
    this widen succeed silently.
    """
    target = _current_feature_dim()
    # One wider than the narrowest recorded prior: guaranteed to match no recorded
    # layout, yet narrower than the target, so it reaches the widen path.
    bad_feat = min(_KNOWN_PRIOR_FEATURE_DIMS) + 1
    assert bad_feat < target  # narrower than target: reaches the widen path
    assert bad_feat not in _KNOWN_PRIOR_FEATURE_DIMS  # but not a recorded prior
    state = _build_current_state_dict()
    state[TRUNK_INPUT_WEIGHT_KEY] = torch.zeros(HIDDEN, bad_feat)
    with pytest.raises(InterfaceError, match="not a known prior width"):
        migrate_encoder_trunk_input_width(state, target)


@pytest.mark.parametrize("known_feat", _KNOWN_PRIOR_FEATURE_DIMS)
def test_migrate_trunk_accepts_known_prior_widths(known_feat: int) -> None:
    """Every recorded prior width still widens cleanly: prefix preserved, suffix zero.

    The guard rejects only UNRECORDED narrower widths; each width in
    _KNOWN_PRIOR_FEATURE_DIMS is a real prior layout and must migrate exactly as
    before, loading strict into a fresh net.
    """
    target = _current_feature_dim()
    assert known_feat < target  # each recorded prior is narrower than the current
    state = _build_current_state_dict()
    old_weight = torch.arange(HIDDEN * known_feat, dtype=torch.float32).reshape(HIDDEN, known_feat)
    state[TRUNK_INPUT_WEIGHT_KEY] = old_weight
    migrated = migrate_encoder_trunk_input_width(state, target)
    new_weight = migrated[TRUNK_INPUT_WEIGHT_KEY]
    assert new_weight.shape == (HIDDEN, target)
    assert torch.equal(new_weight[:, :known_feat], old_weight)  # prefix preserved
    assert torch.count_nonzero(new_weight[:, known_feat:]) == 0  # suffix zeroed
    model = ActorCritic(hidden_dim=HIDDEN)
    model.load_state_dict(migrated, strict=True)
    assert torch.equal(model.encoder.trunk[0].weight, new_weight)


# The shipped 0.6.0 feature width (before the deck / keys-act append) and 0.7.0 width
# (before the shop / boss / event-neow append), now recorded priors that must widen to
# the current layout. Fixed history, like the widths in _KNOWN_PRIOR_FEATURE_DIMS. The
# exact current width is locked in test_encoder (_expected_feature_dim); here we only
# assert each prior is recorded, is narrower than current, and widens cleanly.
SHIPPED_0_6_0_FEATURE_DIM = 4881
SHIPPED_0_7_0_FEATURE_DIM = 4949


def test_shipped_4881_prior_widens_to_current_layout() -> None:
    """The 0.6.0 width (4881, pre deck / keys-act) is a recorded prior and widens cleanly.

    A warm-start checkpoint trained at the shipped 4881-wide encoder must widen to the
    current layout: its trained columns preserved as the leading prefix, every appended
    column (the deck, keys/act, and later shop / boss / event-neow blocks) zero-initialized.
    """
    assert SHIPPED_0_6_0_FEATURE_DIM in _KNOWN_PRIOR_FEATURE_DIMS
    target = _current_feature_dim()
    old_feat = SHIPPED_0_6_0_FEATURE_DIM
    # The deck + keys/act blocks were the first append after 4881, so the current width
    # is at least that much wider (later appends only add more).
    assert target >= old_feat + _PILE_POOLS * CARD_EMBED_DIM + KEYS_ACT_DIM

    state = _build_current_state_dict()
    old_weight = torch.arange(HIDDEN * old_feat, dtype=torch.float32).reshape(HIDDEN, old_feat)
    state[TRUNK_INPUT_WEIGHT_KEY] = old_weight

    migrated = migrate_encoder_trunk_input_width(state, target)
    new_weight = migrated[TRUNK_INPUT_WEIGHT_KEY]
    assert new_weight.shape == (HIDDEN, target)
    assert torch.equal(new_weight[:, :old_feat], old_weight)  # prefix preserved
    assert torch.count_nonzero(new_weight[:, old_feat:]) == 0  # appended suffix zero
    model = ActorCritic(hidden_dim=HIDDEN)
    model.load_state_dict(migrated, strict=True)
    assert torch.equal(model.encoder.trunk[0].weight, new_weight)


def test_shipped_4949_prior_widens_to_shop_layout() -> None:
    """The 0.7.0 width (4949, pre shop / boss / event-neow) is a recorded prior and widens.

    A warm-start checkpoint trained at the shipped 4949-wide encoder (deck + keys/act, no
    shop / boss / event-neow blocks yet) must widen to the current layout: trained columns
    preserved as the leading prefix, the appended shop / boss / event-neow columns zeroed.
    """
    assert SHIPPED_0_7_0_FEATURE_DIM in _KNOWN_PRIOR_FEATURE_DIMS
    target = _current_feature_dim()
    old_feat = SHIPPED_0_7_0_FEATURE_DIM
    assert old_feat < target

    state = _build_current_state_dict()
    old_weight = torch.arange(HIDDEN * old_feat, dtype=torch.float32).reshape(HIDDEN, old_feat)
    state[TRUNK_INPUT_WEIGHT_KEY] = old_weight

    migrated = migrate_encoder_trunk_input_width(state, target)
    new_weight = migrated[TRUNK_INPUT_WEIGHT_KEY]
    assert new_weight.shape == (HIDDEN, target)
    assert torch.equal(new_weight[:, :old_feat], old_weight)  # prefix preserved
    assert torch.count_nonzero(new_weight[:, old_feat:]) == 0  # appended suffix zero
    model = ActorCritic(hidden_dim=HIDDEN)
    model.load_state_dict(migrated, strict=True)
    assert torch.equal(model.encoder.trunk[0].weight, new_weight)


def test_load_checkpoint_widens_narrow_trunk(tmp_path) -> None:
    """A warm-start checkpoint with the pre-reward-vision (narrower) trunk widens on load.

    The end-to-end deliverable: an existing current-layout checkpoint whose
    encoder predates every reward block loads into the now-wider trunk via
    load_checkpoint, its trained columns preserved and the appended reward-vision
    columns zero-initialized (so combat competence starts unperturbed).
    """
    target = _current_feature_dim()
    old_feat = PRE_REWARD_FEATURE_DIM  # a recorded prior width the widen guard accepts
    state = _build_current_state_dict()
    old_weight = torch.arange(HIDDEN * old_feat, dtype=torch.float32).reshape(HIDDEN, old_feat)
    state[TRUNK_INPUT_WEIGHT_KEY] = old_weight
    path = tmp_path / "warm.pt"
    _write_checkpoint(path, state, INTERFACE_VERSION, HIDDEN)

    model = load_checkpoint(path)
    loaded = model.encoder.trunk[0].weight.detach()
    assert loaded.shape == (HIDDEN, target)
    assert torch.equal(loaded[:, :old_feat], old_weight)
    assert torch.count_nonzero(loaded[:, old_feat:]) == 0


def test_load_checkpoint_widens_trunk_from_card_vision_prior(tmp_path) -> None:
    """A card-vision (recorded-prior) checkpoint widens its trunk on load.

    A current-layout checkpoint whose encoder is at the card-vision width (has the
    reward-card block but none of the later appended blocks) loads into the
    now-wider trunk: its trained columns are preserved as the leading prefix and
    every appended column (offered relic, offered potion, and card-select) is
    zero-initialized. Pinned to a recorded prior so the widen guard accepts it.
    """
    target = _current_feature_dim()
    old_feat = CARD_VISION_FEATURE_DIM  # a recorded prior width the widen guard accepts
    assert old_feat < target  # guard the premise: the encoder actually widened
    state = _build_current_state_dict()
    old_weight = torch.arange(HIDDEN * old_feat, dtype=torch.float32).reshape(HIDDEN, old_feat)
    state[TRUNK_INPUT_WEIGHT_KEY] = old_weight
    path = tmp_path / "warm_card_vision.pt"
    _write_checkpoint(path, state, INTERFACE_VERSION, HIDDEN)

    model = load_checkpoint(path)
    loaded = model.encoder.trunk[0].weight.detach()
    assert loaded.shape == (HIDDEN, target)
    assert torch.equal(loaded[:, :old_feat], old_weight)
    assert torch.count_nonzero(loaded[:, old_feat:]) == 0


def test_load_checkpoint_composes_head_remap_and_trunk_widen(tmp_path) -> None:
    """One checkpoint needing BOTH migrations: old-layout head AND a narrow trunk.

    The real warm-start scenario - a 0.3.0-layout checkpoint whose encoder also
    predates every appended reward block (cards, relics, potions, and card-select).
    load_checkpoint must compose the name-aware head remap with the trunk widening
    on the single payload: CONFIRM_SELECT lands at its live offset, and the trunk
    widens with its sentinel prefix preserved and the appended suffix zero.
    """
    old_state = _build_old_state_dict()
    old_head_weight = old_state[POLICY_LOGITS_WEIGHT_KEY]

    target = _current_feature_dim()
    # A 0.3.0-era encoder predates ALL appended vision, so its trunk is the
    # combat-only recorded prior (before cards, relics, potions, and card-select).
    old_feat = PRE_REWARD_FEATURE_DIM
    assert old_feat < target  # guard the premise: the encoder actually widened
    old_trunk = torch.arange(HIDDEN * old_feat, dtype=torch.float32).reshape(HIDDEN, old_feat)
    old_state[TRUNK_INPUT_WEIGHT_KEY] = old_trunk

    path = tmp_path / "compose.pt"
    _write_checkpoint(path, old_state, OLD_VERSION, HIDDEN)

    model = load_checkpoint(path)

    # (a) Head remapped to the current layout; CONFIRM_SELECT at its LIVE offset.
    assert model.policy.logits.weight.shape == (ACTION_DIM, HIDDEN)
    confirm_old_start = OLD_LAYOUT["CONFIRM_SELECT"][0]
    confirm_new_start = ACTION_BLOCK_BY_NAME["CONFIRM_SELECT"].start
    assert torch.equal(
        model.policy.logits.weight[confirm_new_start], old_head_weight[confirm_old_start]
    )

    # (b) Trunk widened to the current feature dim; sentinel prefix preserved,
    # appended suffix exactly zero.
    trunk = model.encoder.trunk[0].weight.detach()
    assert trunk.shape == (HIDDEN, target)
    assert torch.equal(trunk[:, :old_feat], old_trunk)
    assert torch.count_nonzero(trunk[:, old_feat:]) == 0
