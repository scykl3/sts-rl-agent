"""Tests for the behavior-cloning pipeline (dataset, collection, training).

All tests are engine-free: uses StubEnv / scripted envs and stub card-name
resolution. Samples are full-space (an ``ACTION_DIM`` action index + mask), so the
pipeline is exercised across the decision types the strategic teacher covers.
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
    DT_CAMPFIRE,
    DT_CARD_PICK,
    DT_MAP,
    BCDataset,
    BCManifest,
    _decision_type_of,
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
from sts_rl.agent.strategic_teacher import (
    _MAP,
    _PS_HP_CUR,
    _PS_HP_MAX,
    _REST,
    REST_REST_OFFSET,
    REST_SMITH_OFFSET,
    StrategicTeacher,
)
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    INTERFACE_VERSION,
    OBS_FIELDS,
    PAD_ID,
)

# --- Stub helpers -----------------------------------------------------------


def _stub_card_id_to_name(self: CardRewardTeacher, card_id: int) -> str | None:
    """Stub card resolution for tests: map id -> UPPER_SNAKE name."""
    if card_id == PAD_ID:
        return None
    # id 1 -> OFFERING (S-tier); any other non-zero -> HEADBUTT (B-tier).
    if card_id == 1:
        return "OFFERING"
    return "HEADBUTT"


@pytest.fixture()
def _patch_card_resolution():
    """Patch card resolution for BC tests (covers the strategic teacher's sub-teacher)."""
    with patch.object(CardRewardTeacher, "_card_id_to_name", _stub_card_id_to_name):
        yield


# --- Scripted engine-free envs ----------------------------------------------
# The StubEnv keeps REWARD_SELECT legal every step, so it cannot exercise the
# "log only when the teacher acts" contract. These engine-free helpers drive envs
# whose steps mix teacher-owned and deferred decisions so routing is testable.

_PROCEED_IDX: int = ACTION_BLOCK_BY_NAME["PROCEED"].start


def _full_zero_obs() -> dict[str, np.ndarray]:
    """A full, in-shape obs dict (all fields zero / PAD) covering every OBS_FIELD."""
    return {f.name: np.zeros(f.shape, dtype=f.dtype) for f in OBS_FIELDS}


class _ScriptedEnv:
    """Engine-free env driven by a scripted list of per-step kinds.

    ``kinds[i]`` is ``"card"`` (a card-pick slot + skip legal, one card offered) or
    ``"noncard"`` (only PROCEED legal, which the teacher defers on). The env
    presents that mask/obs sequence one entry per step, then truncates, and records
    the action passed to each ``step`` alongside the kind of that step.
    """

    def __init__(self, kinds: list[str]) -> None:
        self._kinds = kinds
        self._i = 0
        self.received_actions: list[int] = []
        self.step_kinds: list[str] = []

    def _kind_at(self, i: int) -> str:
        return self._kinds[i] if i < len(self._kinds) else self._kinds[-1]

    def _obs_and_mask(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        obs = _full_zero_obs()
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        if self._kind_at(self._i) == "card":
            mask[CARD_PICK_START] = True
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


class _MultiScreenEnv:
    """Engine-free env cycling through card / campfire / map decisions, then truncates.

    Every presented step is a screen the strategic teacher owns, so a collected
    dataset spans multiple decision types with no deferral.
    """

    def __init__(self, cycles: int) -> None:
        self._script = ["card", "campfire", "map"] * cycles
        self._i = 0

    def _obs_and_mask(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        obs = _full_zero_obs()
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        kind = self._script[min(self._i, len(self._script) - 1)]
        if kind == "card":
            mask[CARD_PICK_START] = True
            mask[CARD_SKIP_IDX] = True
            obs["reward_card_ids"][0] = 1  # OFFERING via stub
        elif kind == "campfire":
            mask[_REST.start + REST_REST_OFFSET] = True
            mask[_REST.start + REST_SMITH_OFFSET] = True
            obs["player_scalars"][_PS_HP_MAX] = 80.0
            obs["player_scalars"][_PS_HP_CUR] = 80.0  # healthy -> smith
        else:  # map
            mask[_MAP.start + 0] = True
            mask[_MAP.start + 1] = True
        return obs, mask

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        self._i = 0
        obs, mask = self._obs_and_mask()
        return obs, {"action_mask": mask}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        self._i += 1
        truncated = self._i >= len(self._script)
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


class _RecordingTeacher(StrategicTeacher):
    """StrategicTeacher that records each action it returns (for assertions)."""

    def __init__(self, *, validate: bool = False) -> None:
        super().__init__(validate=validate)
        self.returned: list[int] = []

    def select_action(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int | None:
        action = super().select_action(obs, mask)
        if action is not None:
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

        actions = np.random.default_rng(1).integers(0, ACTION_DIM, size=n_samples).astype(np.int64)
        masks = (
            np.random.default_rng(2).integers(0, 2, size=(n_samples, ACTION_DIM)).astype(np.bool_)
        )

        manifest = BCManifest(
            n_samples=n_samples,
            interface_version=INTERFACE_VERSION,
            seed=42,
            teacher_config={"min_keep_tier": 2},
            tier_histogram={"5": 3, "4": 7},
            decision_type_histogram={"card_pick": 8, "campfire": 2},
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
        assert loaded.masks.shape == (n_samples, ACTION_DIM)
        assert loaded.manifest.n_samples == n_samples
        assert loaded.manifest.interface_version == INTERFACE_VERSION
        assert loaded.manifest.seed == 42
        assert loaded.manifest.skip_rate == pytest.approx(0.25)
        # variable-size manifest dicts must survive (not reset to {}).
        assert loaded.manifest.teacher_config == {"min_keep_tier": 2}
        assert loaded.manifest.tier_histogram == {"5": 3, "4": 7}
        assert loaded.manifest.decision_type_histogram == {"card_pick": 8, "campfire": 2}

    def test_load_tolerates_missing_decision_type_histogram(self, tmp_path: Path) -> None:
        """An older-format .npz without the decision-type histogram loads with {}."""
        n_samples = 3
        arrays: dict[str, np.ndarray] = {}
        for f in OBS_FIELDS:
            arrays[f"obs_{f.name}"] = np.zeros((n_samples, *f.shape), dtype=f.dtype)
        arrays["actions"] = np.zeros(n_samples, dtype=np.int64)
        arrays["masks"] = np.zeros((n_samples, ACTION_DIM), dtype=np.bool_)
        arrays["manifest_n_samples"] = np.array([n_samples], dtype=np.int64)
        arrays["manifest_seed"] = np.array([-1], dtype=np.int64)
        arrays["manifest_skip_rate"] = np.array([0.0], dtype=np.float64)
        arrays["manifest_interface_version"] = np.array([INTERFACE_VERSION], dtype="U32")
        arrays["manifest_teacher_config"] = np.array("{}")
        arrays["manifest_tier_histogram"] = np.array("{}")
        # Deliberately omit manifest_decision_type_histogram (the pre-broadening format).
        path = tmp_path / "legacy.npz"
        np.savez(str(path), **arrays)

        loaded = BCDataset.load(path)
        assert loaded.manifest.decision_type_histogram == {}
        assert loaded.manifest.seed is None


# --- Collection with StubEnv -----------------------------------------------


class TestCollectBCDataset:
    """collect_bc_dataset logs samples whenever the teacher acts."""

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
        """Strategic teacher with validation disabled."""
        return StrategicTeacher(validate=False)

    def test_collects_samples_at_card_steps(
        self, stub_env, policy, teacher, _patch_card_resolution
    ) -> None:
        dataset = collect_bc_dataset(
            stub_env, policy, teacher, n_episodes=3, seed=0, deterministic=True
        )
        # REWARD_SELECT is legal every step, so the teacher owns every step (card).
        assert len(dataset) > 0
        # Actions are full-space indices.
        assert dataset.actions.min() >= 0
        assert dataset.actions.max() < ACTION_DIM
        # Masks are the full legality mask.
        assert dataset.masks.shape == (len(dataset), ACTION_DIM)

    def test_logged_action_is_legal_under_full_mask(
        self, stub_env, policy, teacher, _patch_card_resolution
    ) -> None:
        dataset = collect_bc_dataset(stub_env, policy, teacher, n_episodes=2, seed=0)
        for i in range(len(dataset)):
            action = int(dataset.actions[i])
            assert dataset.masks[i, action], f"sample {i}: logged action {action} is illegal"


class TestCollectBCMixedSteps:
    """collect_bc_dataset logs ONLY when the teacher acts and routes each action.

    Drives an env alternating card (teacher-owned) and non-card (deferred) steps,
    exercising the headline contract directly.
    """

    def test_logs_only_when_teacher_acts_and_routes_actions(self, _patch_card_resolution) -> None:
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

        # (a) a sample is logged ONLY when the teacher acts -> only card steps here.
        assert len(dataset) == n_card
        assert len(dataset) < len(kinds)
        # (b) on each card step, env.step received exactly the teacher's action.
        assert card_actions == teacher.returned
        assert len(teacher.returned) == n_card
        # (c) on each deferred step, env.step received the base policy's action, and
        # the base policy was consulted ONLY on the deferred steps.
        assert noncard_actions == [_PROCEED_IDX] * n_noncard
        assert base_policy.n_calls == n_noncard


class TestCollectBCSpansDecisionTypes:
    """The broadened collection yields a dataset spanning multiple decision types."""

    def test_spans_card_campfire_map_and_bc_pretrain_runs(self, _patch_card_resolution) -> None:
        env = _MultiScreenEnv(cycles=4)  # 4 x (card, campfire, map) = 12 teacher decisions
        base_policy = _RecordingPolicy(_PROCEED_IDX)
        teacher = StrategicTeacher(validate=False)

        dataset = collect_bc_dataset(env, base_policy, teacher, n_episodes=1, seed=0)

        # The teacher owns every presented screen, so the base policy is never asked.
        assert base_policy.n_calls == 0
        # The dataset spans at least three distinct decision types.
        hist = dataset.manifest.decision_type_histogram
        assert {DT_CARD_PICK, DT_CAMPFIRE, DT_MAP} <= set(hist)
        assert len(dataset) == sum(hist.values())

        # Every logged action is legal under its stored full mask.
        for i in range(len(dataset)):
            assert dataset.masks[i, int(dataset.actions[i])]

        # bc_pretrain runs end-to-end on the multi-type dataset.
        model = ActorCritic()
        stats = bc_pretrain(model, dataset, epochs=2, lr=1e-3, batch_size=4, device="cpu")
        assert stats.total_epochs >= 1
        assert len(stats.train_loss_history) == stats.total_epochs


class TestCollectBCDegenerateGuard:
    """collect_bc_dataset fails loudly instead of stacking an empty buffer."""

    def test_collect_raises_when_teacher_never_acts(self, _patch_card_resolution) -> None:
        # The teacher defers every step -> zero samples -> a clear ValueError, not
        # the cryptic np.stack([]) failure.
        env = _ScriptedEnv(["noncard", "noncard", "noncard"])
        base_policy = _RecordingPolicy(_PROCEED_IDX)
        teacher = StrategicTeacher(validate=False)
        with pytest.raises(ValueError, match="no teacher decisions"):
            collect_bc_dataset(env, base_policy, teacher, n_episodes=1, seed=0)


# --- BC Training on synthetic data ------------------------------------------


class TestBCPretrain:
    """bc_pretrain drives NLL down and matches the teacher on a tiny overfit dataset."""

    @pytest.fixture()
    def synthetic_dataset(self) -> BCDataset:
        """A tiny full-space dataset with a consistent teacher label for overfitting."""
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

        # All samples: teacher picks card slot 0 (full-space index CARD_PICK_START).
        actions = np.full(n_samples, CARD_PICK_START, dtype=np.int64)
        # Mask: card slot 0 and skip legal.
        masks = np.zeros((n_samples, ACTION_DIM), dtype=np.bool_)
        masks[:, CARD_PICK_START] = True
        masks[:, CARD_SKIP_IDX] = True

        manifest = BCManifest(
            n_samples=n_samples,
            interface_version=INTERFACE_VERSION,
            seed=42,
            teacher_config={},
            tier_histogram={},
            decision_type_histogram={"card_pick": n_samples},
            skip_rate=0.0,
        )
        return BCDataset(obs_arrays, actions, masks, manifest)

    def test_overfit_one_batch(self, synthetic_dataset: BCDataset) -> None:
        """NLL goes down and argmax over the full masked space matches the teacher."""
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
        assert stats.train_loss_history[-1] < stats.train_loss_history[0]

        # argmax over the full masked action space should match the teacher.
        model.eval()
        with torch.no_grad():
            obs_batch: dict[str, torch.Tensor] = {}
            id_fields = {f.name for f in OBS_FIELDS if f.bounds == "id"}
            for f in OBS_FIELDS:
                arr = synthetic_dataset.obs_arrays[f.name]
                dtype = torch.long if f.name in id_fields else torch.float32
                obs_batch[f.name] = torch.as_tensor(arr, dtype=dtype)
            per_token, _pooled_cls, kpm = model.encoder(obs_batch)
            mask_t = torch.as_tensor(synthetic_dataset.masks, dtype=torch.bool)
            logits = model.policy(per_token, kpm, mask_t)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == CARD_PICK_START).float().mean().item()
            assert accuracy >= 0.9, f"overfit accuracy {accuracy:.3f} < 0.9"


class TestBCPretrainDegenerateGuard:
    """bc_pretrain refuses a dataset too small to form a train/val split."""

    def test_raises_on_single_sample_dataset(self) -> None:
        n_samples = 1
        obs_arrays: dict[str, np.ndarray] = {
            f.name: np.zeros((n_samples, *f.shape), dtype=f.dtype) for f in OBS_FIELDS
        }
        actions = np.full(n_samples, CARD_PICK_START, dtype=np.int64)
        masks = np.zeros((n_samples, ACTION_DIM), dtype=np.bool_)
        masks[:, CARD_PICK_START] = True
        manifest = BCManifest(
            n_samples=n_samples,
            interface_version=INTERFACE_VERSION,
            seed=0,
            teacher_config={},
            tier_histogram={},
            decision_type_histogram={"card_pick": 1},
            skip_rate=0.0,
        )
        dataset = BCDataset(obs_arrays, actions, masks, manifest)
        model = ActorCritic()
        with pytest.raises(ValueError, match="too small for a train/val split"):
            bc_pretrain(model, dataset, epochs=1, val_frac=0.15, device="cpu")


class TestBCPretrainRejectsSubSliceMasks:
    """bc_pretrain rejects a legacy sub-slice-width mask dataset with a clear error."""

    def test_raises_on_non_full_space_masks(self) -> None:
        n_samples = 4
        obs_arrays = {f.name: np.zeros((n_samples, *f.shape), dtype=f.dtype) for f in OBS_FIELDS}
        actions = np.zeros(n_samples, dtype=np.int64)
        # A legacy card-only mask width (MAX_REWARD_CARD_SLOTS + 1), not ACTION_DIM.
        masks = np.ones((n_samples, 9), dtype=np.bool_)
        manifest = BCManifest(
            n_samples=n_samples,
            interface_version=INTERFACE_VERSION,
            seed=0,
            teacher_config={},
            tier_histogram={},
            decision_type_histogram={},
            skip_rate=0.0,
        )
        dataset = BCDataset(obs_arrays, actions, masks, manifest)
        model = ActorCritic()
        with pytest.raises(ValueError, match="must be full-space"):
            bc_pretrain(model, dataset, epochs=1, device="cpu")


# --- Checkpoint round-trip via load_checkpoint ------------------------------


class TestBCCheckpointRoundTrip:
    """A BC-produced checkpoint loads through load_checkpoint (warm-start path)."""

    def test_checkpoint_loads(self, tmp_path: Path) -> None:
        model = ActorCritic()
        hidden_dim = model.encoder.output_dim
        ckpt_path = tmp_path / "bc_test.pt"
        save_bc_checkpoint(model, ckpt_path, hidden_dim)

        loaded = load_checkpoint(ckpt_path)
        assert isinstance(loaded, ActorCritic)
        for key in model.state_dict():
            torch.testing.assert_close(
                loaded.state_dict()[key],
                model.state_dict()[key],
                msg=f"mismatch at key {key}",
            )


# --- Utility tests ----------------------------------------------------------


class TestUtilities:
    """Test internal utility functions."""

    def test_decision_type_of_card_pick(self) -> None:
        assert _decision_type_of(CARD_PICK_START) == DT_CARD_PICK
        assert _decision_type_of(CARD_SKIP_IDX) != DT_CARD_PICK

    def test_decision_type_of_campfire_and_map(self) -> None:
        assert _decision_type_of(_REST.start + REST_SMITH_OFFSET) == DT_CAMPFIRE
        assert _decision_type_of(_MAP.start) == DT_MAP

    def test_decision_type_of_other(self) -> None:
        assert _decision_type_of(ACTION_BLOCK_BY_NAME["END_TURN"].start) == "other"
