"""Tests for the behavior-cloning pipeline (dataset, collection, training).

All tests are engine-free: uses StubEnv and stub card-name resolution.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from conftest import make_stub_env
from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.bc import (
    BCDataset,
    BCManifest,
    _CARD_SUBSLICE_INDICES,
    _SUBSLICE_SIZE,
    _action_to_subslice_idx,
    _has_legal_card_slot,
    bc_pretrain,
    collect_bc_dataset,
    save_bc_checkpoint,
)
from sts_rl.agent.card_teacher import (
    CARD_PICK_START,
    CARD_SKIP_IDX,
    CardRewardTeacher,
)
from sts_rl.agent.checkpoint_migration import load_checkpoint
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    INTERFACE_VERSION,
    MAX_REWARD_CARD_SLOTS,
    OBS_FIELDS,
    PAD_ID,
)

# --- Stub helpers -----------------------------------------------------------


def _stub_card_id_to_name(self: CardRewardTeacher, card_id: int) -> str | None:
    """Stub card resolution for tests: map id -> UPPER_SNAKE name."""
    if card_id == PAD_ID:
        return None
    # Map any non-zero id to a name that appears in the tier list
    # Use OFFERING (S-tier) for id=1, HEADBUTT (B-tier) for others
    if card_id == 1:
        return "OFFERING"
    return "HEADBUTT"


@pytest.fixture()
def _patch_card_resolution():
    """Patch card resolution for BC tests."""
    with patch.object(CardRewardTeacher, "_card_id_to_name", _stub_card_id_to_name):
        yield


# --- Mixed-step collection helpers ------------------------------------------
# The StubEnv keeps REWARD_SELECT legal every step, so it cannot exercise the
# "log only on card steps" contract. These engine-free helpers drive an env
# whose steps alternate card-reward and non-card so the routing is testable.

_PROCEED_IDX: int = ACTION_BLOCK_BY_NAME["PROCEED"].start


def _full_zero_obs() -> dict[str, np.ndarray]:
    """A full, in-shape obs dict (all fields zero / PAD) covering every OBS_FIELD."""
    return {f.name: np.zeros(f.shape, dtype=f.dtype) for f in OBS_FIELDS}


class _ScriptedEnv:
    """Engine-free env driven by a scripted list of per-step kinds.

    ``kinds[i]`` is ``"card"`` (a card-pick slot + skip legal, one card offered)
    or ``"noncard"`` (only PROCEED legal). The env presents that mask/obs
    sequence one entry per step, then truncates, and records the action passed
    to each ``step`` alongside the kind of that step.
    """

    def __init__(self, kinds: list[str]) -> None:
        self._kinds = kinds
        self._i = 0
        self.received_actions: list[int] = []
        self.step_kinds: list[str] = []

    def _kind_at(self, i: int) -> str:
        # The post-final obs is unused by the collector; clamp to stay in range.
        return self._kinds[i] if i < len(self._kinds) else self._kinds[-1]

    def _obs_and_mask(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        obs = _full_zero_obs()
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        if self._kind_at(self._i) == "card":
            mask[CARD_PICK_START] = True  # one card-pick slot
            mask[CARD_SKIP_IDX] = True
            obs["reward_card_ids"][0] = 1  # -> OFFERING (S-tier) via stub resolution
        else:
            mask[_PROCEED_IDX] = True
        return obs, mask

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        self._i = 0
        obs, mask = self._obs_and_mask()
        return obs, {"action_mask": mask}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        self.received_actions.append(int(action))
        self.step_kinds.append(self._kinds[self._i])
        self._i += 1
        truncated = self._i >= len(self._kinds)
        obs, mask = self._obs_and_mask()
        return obs, 0.0, False, truncated, {"action_mask": mask}


class _RecordingPolicy:
    """Base-policy stub: returns a fixed action and counts how often it is asked."""

    def __init__(self, fixed_action: int) -> None:
        self._fixed_action = fixed_action
        self.n_calls = 0
        self._device_param = torch.zeros(1)  # only to report a device to the caller

    def parameters(self):
        return iter([self._device_param])

    def act(self, obs_batched, mask_batched, deterministic: bool = True):
        self.n_calls += 1
        return torch.tensor(self._fixed_action), None, None, None


class _RecordingTeacher(CardRewardTeacher):
    """CardRewardTeacher that records each action it returns (for assertions)."""

    def __init__(self, *, validate: bool = False) -> None:
        super().__init__(validate=validate)
        self.returned: list[int] = []

    def select_action(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        action = super().select_action(obs, mask)
        self.returned.append(action)
        return action


# --- Dataset save/load round-trip -------------------------------------------


class TestBCDatasetRoundTrip:
    """BCDataset save/load preserves obs, actions, masks, and manifest."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        n_samples = 10
        obs_arrays: dict[str, np.ndarray] = {}
        for f in OBS_FIELDS:
            obs_arrays[f.name] = (
                np.random.default_rng(0)
                .standard_normal((n_samples, *f.shape))
                .astype(np.float32 if f.dtype == np.float32 else np.int32)
            )

        actions = (
            np.random.default_rng(1).integers(0, _SUBSLICE_SIZE, size=n_samples).astype(np.int64)
        )
        masks = (
            np.random.default_rng(2)
            .integers(0, 2, size=(n_samples, _SUBSLICE_SIZE))
            .astype(np.bool_)
        )

        manifest = BCManifest(
            n_samples=n_samples,
            interface_version=INTERFACE_VERSION,
            seed=42,
            teacher_config={"min_keep_tier": 2},
            tier_histogram={"5": 3, "4": 7},
            skip_rate=0.25,
        )

        dataset = BCDataset(obs_arrays, actions, masks, manifest)
        path = tmp_path / "test_bc.npz"
        dataset.save(path)

        loaded = BCDataset.load(path)
        assert len(loaded) == n_samples
        for f in OBS_FIELDS:
            np.testing.assert_array_equal(loaded.obs_arrays[f.name], obs_arrays[f.name])
        np.testing.assert_array_equal(loaded.actions, actions)
        np.testing.assert_array_equal(loaded.masks, masks)
        assert loaded.manifest.n_samples == n_samples
        assert loaded.manifest.interface_version == INTERFACE_VERSION
        assert loaded.manifest.seed == 42
        assert loaded.manifest.skip_rate == pytest.approx(0.25)
        # teacher_config and tier_histogram must survive (not reset to {}).
        assert loaded.manifest.teacher_config == {"min_keep_tier": 2}
        assert loaded.manifest.tier_histogram == {"5": 3, "4": 7}


# --- Collection with StubEnv -----------------------------------------------


class TestCollectBCDataset:
    """collect_bc_dataset logs samples only at card-reward steps."""

    @pytest.fixture()
    def stub_env(self):
        """StubEnv with REWARD_SELECT active (emits masks with card slots legal)."""
        return make_stub_env(
            active_blocks=("REWARD_SELECT",),
            max_episode_steps=8,
            terminate_prob=0.1,
        )

    @pytest.fixture()
    def policy(self):
        """A freshly initialized ActorCritic (no training needed for collection)."""
        return ActorCritic()

    @pytest.fixture()
    def teacher(self):
        """Teacher with validation disabled."""
        return CardRewardTeacher(validate=False)

    def test_collects_samples_at_card_steps(
        self, stub_env, policy, teacher, _patch_card_resolution
    ) -> None:
        dataset = collect_bc_dataset(
            stub_env, policy, teacher, n_episodes=3, seed=0, deterministic=True
        )
        # StubEnv with REWARD_SELECT active means every step has the full
        # REWARD_SELECT block legal. The card sub-slots are a subset of that
        # block, so _has_legal_card_slot should be True each step.
        assert len(dataset) > 0
        # All actions are valid sub-slice indices
        assert dataset.actions.min() >= 0
        assert dataset.actions.max() < _SUBSLICE_SIZE
        # All masks have correct shape
        assert dataset.masks.shape == (len(dataset), _SUBSLICE_SIZE)

    def test_teacher_action_is_executed(
        self, stub_env, policy, teacher, _patch_card_resolution
    ) -> None:
        """The action taken at card-reward steps equals the teacher's choice."""
        # We verify this by checking that the logged action is always legal
        # (the teacher asserts legality internally)
        dataset = collect_bc_dataset(stub_env, policy, teacher, n_episodes=2, seed=0)
        for i in range(len(dataset)):
            action_local = int(dataset.actions[i])
            assert dataset.masks[
                i, action_local
            ], f"sample {i}: logged action {action_local} is not legal in sub-slice mask"


class TestCollectBCMixedSteps:
    """collect_bc_dataset logs ONLY on card steps and routes each action correctly.

    The StubEnv suite above cannot assert this: it keeps REWARD_SELECT legal
    every step, so every step is a card step. This drives an env that alternates
    card and non-card steps, exercising the headline contract directly.
    """

    def test_logs_only_on_card_steps_and_routes_actions(self, _patch_card_resolution) -> None:
        kinds = ["card", "noncard", "card", "noncard", "card", "noncard"]
        env = _ScriptedEnv(kinds)
        base_policy = _RecordingPolicy(_PROCEED_IDX)
        teacher = _RecordingTeacher(validate=False)

        dataset = collect_bc_dataset(
            env, base_policy, teacher, n_episodes=1, seed=0, deterministic=True
        )

        n_card = kinds.count("card")
        n_noncard = kinds.count("noncard")
        card_actions = [a for a, k in zip(env.received_actions, env.step_kinds) if k == "card"]
        noncard_actions = [
            a for a, k in zip(env.received_actions, env.step_kinds) if k == "noncard"
        ]

        # (a) a sample is logged ONLY on card steps -> strictly fewer than total.
        assert len(dataset) == n_card
        assert len(dataset) < len(kinds)
        # (b) on each card step, env.step received exactly the teacher's action.
        assert card_actions == teacher.returned
        assert len(teacher.returned) == n_card
        # (c) on each non-card step, env.step received the base policy's action,
        # and the base policy was consulted ONLY on the non-card steps.
        assert noncard_actions == [_PROCEED_IDX] * n_noncard
        assert base_policy.n_calls == n_noncard


class TestCollectBCDegenerateGuard:
    """collect_bc_dataset fails loudly instead of stacking an empty buffer."""

    def test_collect_raises_when_no_card_steps(self, _patch_card_resolution) -> None:
        # No card step ever -> zero samples -> a clear ValueError, not the cryptic
        # np.stack([]) failure.
        env = _ScriptedEnv(["noncard", "noncard", "noncard"])
        base_policy = _RecordingPolicy(_PROCEED_IDX)
        teacher = CardRewardTeacher(validate=False)
        with pytest.raises(ValueError, match="no card-reward decisions"):
            collect_bc_dataset(env, base_policy, teacher, n_episodes=1, seed=0)


# --- BC Training on synthetic data ------------------------------------------


class TestBCPretrain:
    """bc_pretrain drives NLL down and matches teacher on a tiny overfit dataset."""

    @pytest.fixture()
    def synthetic_dataset(self) -> BCDataset:
        """A tiny dataset with a consistent teacher label for overfitting."""
        n_samples = 32
        rng = np.random.default_rng(42)
        obs_arrays: dict[str, np.ndarray] = {}
        for f in OBS_FIELDS:
            if f.bounds == "id":
                assert f.id_high is not None
                obs_arrays[f.name] = rng.integers(
                    0, f.id_high + 1, size=(n_samples, *f.shape), dtype=np.int32
                )
            elif f.bounds == "unit":
                obs_arrays[f.name] = rng.integers(0, 2, size=(n_samples, *f.shape)).astype(
                    np.float32
                )
            else:
                obs_arrays[f.name] = rng.standard_normal((n_samples, *f.shape)).astype(np.float32)

        # All samples: teacher picks slot 0 (sub-slice idx 0)
        teacher_action = 0
        actions = np.full(n_samples, teacher_action, dtype=np.int64)
        # Mask: slot 0 and skip are legal
        masks = np.zeros((n_samples, _SUBSLICE_SIZE), dtype=np.bool_)
        masks[:, 0] = True  # card slot 0 legal
        masks[:, MAX_REWARD_CARD_SLOTS] = True  # skip legal

        manifest = BCManifest(
            n_samples=n_samples,
            interface_version=INTERFACE_VERSION,
            seed=42,
            teacher_config={},
            tier_histogram={},
            skip_rate=0.0,
        )
        return BCDataset(obs_arrays, actions, masks, manifest)

    def test_overfit_one_batch(self, synthetic_dataset: BCDataset) -> None:
        """NLL goes down and argmax over sub-slice matches teacher."""
        model = ActorCritic()
        stats = bc_pretrain(
            model,
            synthetic_dataset,
            epochs=50,
            lr=1e-3,
            batch_size=32,
            val_frac=0.0,  # minimal holdout: one val sample is always kept, overfit on the rest
            seed=0,
            device="cpu",
            patience=100,  # don't early stop
        )
        # Loss should decrease
        assert stats.train_loss_history[-1] < stats.train_loss_history[0]

        # Check that argmax over sub-slice matches teacher for all samples
        model.eval()
        with torch.no_grad():
            obs_batch: dict[str, torch.Tensor] = {}
            id_fields = {f.name for f in OBS_FIELDS if f.bounds == "id"}
            for f in OBS_FIELDS:
                arr = synthetic_dataset.obs_arrays[f.name]
                dtype = torch.long if f.name in id_fields else torch.float32
                obs_batch[f.name] = torch.as_tensor(arr, dtype=dtype)
            features = model.encoder(obs_batch)
            logits = model.policy.logits(features)
            subslice_logits = logits[:, _CARD_SUBSLICE_INDICES]
            mask_t = torch.as_tensor(synthetic_dataset.masks, dtype=torch.bool)
            subslice_logits = subslice_logits.masked_fill(~mask_t, -1e9)
            preds = subslice_logits.argmax(dim=-1)
            # Allow some slack: at least 90% should match after overfitting
            accuracy = (preds == 0).float().mean().item()
            assert accuracy >= 0.9, f"overfit accuracy {accuracy:.3f} < 0.9"


class TestBCPretrainDegenerateGuard:
    """bc_pretrain refuses a dataset too small to form a train/val split."""

    def test_raises_on_single_sample_dataset(self) -> None:
        n_samples = 1
        obs_arrays: dict[str, np.ndarray] = {
            f.name: np.zeros((n_samples, *f.shape), dtype=f.dtype) for f in OBS_FIELDS
        }
        actions = np.zeros(n_samples, dtype=np.int64)
        masks = np.zeros((n_samples, _SUBSLICE_SIZE), dtype=np.bool_)
        masks[:, 0] = True
        manifest = BCManifest(
            n_samples=n_samples,
            interface_version=INTERFACE_VERSION,
            seed=0,
            teacher_config={},
            tier_histogram={},
            skip_rate=0.0,
        )
        dataset = BCDataset(obs_arrays, actions, masks, manifest)
        model = ActorCritic()
        with pytest.raises(ValueError, match="too small for a train/val split"):
            bc_pretrain(model, dataset, epochs=1, val_frac=0.15, device="cpu")


# --- Checkpoint round-trip via load_checkpoint ------------------------------


class TestBCCheckpointRoundTrip:
    """A BC-produced checkpoint loads through load_checkpoint (warm-start path)."""

    def test_checkpoint_loads(self, tmp_path: Path) -> None:
        model = ActorCritic()
        hidden_dim = model.encoder.output_dim
        ckpt_path = tmp_path / "bc_test.pt"
        save_bc_checkpoint(model, ckpt_path, hidden_dim)

        # load_checkpoint should succeed with strict=True
        loaded = load_checkpoint(ckpt_path)
        assert isinstance(loaded, ActorCritic)
        # Verify the weights match
        for key in model.state_dict():
            torch.testing.assert_close(
                loaded.state_dict()[key],
                model.state_dict()[key],
                msg=f"mismatch at key {key}",
            )


# --- Utility tests ----------------------------------------------------------


class TestUtilities:
    """Test internal utility functions."""

    def test_action_to_subslice_idx_card_slots(self) -> None:
        for slot in range(MAX_REWARD_CARD_SLOTS):
            idx = _action_to_subslice_idx(CARD_PICK_START + slot)
            assert idx == slot

    def test_action_to_subslice_idx_skip(self) -> None:
        idx = _action_to_subslice_idx(CARD_SKIP_IDX)
        assert idx == MAX_REWARD_CARD_SLOTS

    def test_action_to_subslice_idx_invalid(self) -> None:
        with pytest.raises(ValueError, match="outside the card-pick"):
            _action_to_subslice_idx(0)  # END_TURN

    def test_has_legal_card_slot(self) -> None:
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        assert not _has_legal_card_slot(mask)
        mask[CARD_PICK_START] = True
        assert _has_legal_card_slot(mask)
