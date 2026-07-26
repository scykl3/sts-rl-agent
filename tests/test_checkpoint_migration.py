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
from sts_rl.agent.checkpoint_migration import (
    OLD_ACTION_LAYOUTS,
    POLICY_LOGITS_BIAS_KEY,
    POLICY_LOGITS_WEIGHT_KEY,
    _version_tuple,
    load_checkpoint,
    migrate_policy_head,
)
from sts_rl.agent.train import (
    CHECKPOINT_HIDDEN_DIM_KEY,
    CHECKPOINT_INTERFACE_VERSION_KEY,
    CHECKPOINT_MODEL_KEY,
)
from sts_rl.interface import ACTION_BLOCK_BY_NAME, ACTION_DIM, INTERFACE_VERSION, InterfaceError

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
