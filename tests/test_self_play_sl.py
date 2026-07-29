"""Tests for the self-play outcome-regression SL pipeline (dataset, collection, training).

All tests are engine-free: a scripted run-mode env stands in for the native
engine (StubEnv cannot vary the terminal ``act``, so it cannot produce outcome
variance), and synthetic datasets drive the trainer without collection.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.checkpoint_migration import load_checkpoint
from sts_rl.agent.rollout_collector import observation_to_batched_tensors
from sts_rl.agent.self_play_sl import (
    SelfPlaySLDataset,
    SelfPlaySLManifest,
    collect_self_play_dataset,
    outcome_regression_pretrain,
    save_bc_checkpoint,
)
from sts_rl.eval.evaluate import ACT2_INDEX
from sts_rl.interface import ACTION_BLOCK_BY_NAME, ACTION_DIM, INTERFACE_VERSION, OBS_FIELDS

# scripts/ is not an importable package, so put it on sys.path to import the
# train_sl entry-point module (for its _seed_global_rngs helper, exercised by the
# collection-determinism test), matching test_train_run / test_train_combat.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train_sl  # noqa: E402  (imported after the sys.path insert above)

# A genuinely legal action index used by the scripted env's mask and the stub
# policy: the first MAP_SELECT slot (choosing the next map node) is a real, legal
# overworld action. PROCEED is deliberately NOT used: interface.py reserves it as
# the tail slot no engine move maps to, so it is never set legal in a mask - the
# opposite of "always legal".
_LEGAL_IDX: int = ACTION_BLOCK_BY_NAME["MAP_SELECT"].start
# Two co-legal overworld actions (adjacent MAP_SELECT slots: a map screen offers a
# choice among the next nodes) for the multi-legal collection-determinism test.
_TWO_LEGAL: tuple[int, int] = (_LEGAL_IDX, _LEGAL_IDX + 1)
# Terminal-act values for scripted episodes, derived from ACT2_INDEX so the test
# tracks the clear threshold rather than a hardcoded 2: a run that stayed in Act 1
# (did not clear) vs one that reached Act 2 (cleared).
_ACT_NOT_CLEARED: int = ACT2_INDEX - 1
_ACT_CLEARED: int = ACT2_INDEX
# Non-terminal steps stay in Act 1 (below the clear threshold).
_NONTERMINAL_ACT: int = ACT2_INDEX - 1


# --- Engine-free fixtures ---------------------------------------------------


def _full_zero_obs() -> dict[str, np.ndarray]:
    """A full, in-shape obs dict (all fields zero / PAD) covering every OBS_FIELD."""
    return {f.name: np.zeros(f.shape, dtype=f.dtype) for f in OBS_FIELDS}


class _ScriptedRunEnv:
    """Engine-free run-mode env with deterministic step counts and terminal acts.

    ``episode_specs[k]`` is ``(n_steps, terminal_act)`` for the k-th episode
    (consumed one per ``reset``). Each step emits a one-legal-action mask and a
    full zero obs; the terminal step's info carries the preset ``act`` that the
    collector reads to label the episode. This is what lets a unit test exercise
    the outcome backfill without the native engine.
    """

    def __init__(
        self,
        episode_specs: list[tuple[int, int]],
        legal_indices: tuple[int, ...] = (_LEGAL_IDX,),
    ) -> None:
        self._specs = list(episode_specs)
        self._legal = tuple(legal_indices)
        self._ep = -1
        self._step = 0

    def _mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        for idx in self._legal:
            mask[idx] = True
        return mask

    def _info(self, terminal_act: int | None) -> dict[str, object]:
        act = _NONTERMINAL_ACT if terminal_act is None else terminal_act
        info: dict[str, object] = {"action_mask": self._mask(), "act": act}
        if terminal_act is not None:
            info["won"] = bool(terminal_act >= 4)
            info["episode"] = {"r": 0.0, "l": int(self._step)}
        return info

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        self._ep += 1
        self._step = 0
        return _full_zero_obs(), self._info(terminal_act=None)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        self._step += 1
        n_steps, terminal_act = self._specs[self._ep]
        terminated = self._step >= n_steps
        info = self._info(terminal_act=terminal_act if terminated else None)
        return _full_zero_obs(), 0.0, terminated, False, info


class _FixedActionPolicy:
    """Policy stub: returns a fixed legal action and counts how often it is asked."""

    def __init__(self, action: int = _LEGAL_IDX) -> None:
        self._action = action
        self.n_calls = 0
        self._device_param = torch.zeros(1)  # only to report a device to the caller

    def parameters(self):
        return iter([self._device_param])

    def act(self, obs_batched, mask_batched, deterministic: bool = False):
        self.n_calls += 1
        # (1,)-shaped action so int(action.item()) works, matching ActorCritic.act.
        return torch.tensor([self._action]), None, None, None


# --- Synthetic dataset builders ---------------------------------------------


def _synthetic_dataset(
    *,
    n_samples: int,
    actions: np.ndarray | None = None,
    outcomes: np.ndarray | None = None,
    seed: int = 0,
) -> SelfPlaySLDataset:
    """Build a SelfPlaySLDataset with random obs and a valid full-space mask per row."""
    rng = np.random.default_rng(seed)
    obs_arrays: dict[str, np.ndarray] = {}
    for f in OBS_FIELDS:
        if f.bounds == "id":
            assert f.id_high is not None
            obs_arrays[f.name] = rng.integers(
                0, f.id_high + 1, size=(n_samples, *f.shape), dtype=np.int32
            )
        elif f.bounds == "unit":
            obs_arrays[f.name] = rng.integers(0, 2, size=(n_samples, *f.shape)).astype(np.float32)
        else:
            obs_arrays[f.name] = rng.standard_normal((n_samples, *f.shape)).astype(np.float32)

    if actions is None:
        actions = rng.integers(0, ACTION_DIM, size=n_samples).astype(np.int64)
    if outcomes is None:
        outcomes = rng.integers(0, 2, size=n_samples).astype(np.float32)

    masks = np.zeros((n_samples, ACTION_DIM), dtype=np.bool_)
    for i, a in enumerate(actions):
        masks[i, int(a)] = True  # the chosen action is legal in its row

    manifest = SelfPlaySLManifest(
        n_episodes=n_samples,
        n_samples=n_samples,
        act1_clear_rate=float(outcomes.mean()),
        interface_version=INTERFACE_VERSION,
        seed=seed,
    )
    return SelfPlaySLDataset(
        obs_arrays, np.asarray(actions, dtype=np.int64), masks, outcomes, manifest
    )


def _class_obs(*, cleared: bool) -> dict[str, np.ndarray]:
    """One obs whose float context fields encode its class, so the state is predictive.

    A cleared-class state fills every non-id field with 1.0; a not-cleared state
    leaves them at 0.0 (id fields are PAD/0 in both). The two states therefore
    feed the CLS token a distinct input, so the value head over the pooled CLS
    context can separate them after training.
    """
    fill = 1.0 if cleared else 0.0
    obs: dict[str, np.ndarray] = {}
    for f in OBS_FIELDS:
        arr = np.zeros(f.shape, dtype=f.dtype)
        if f.bounds != "id":
            arr[...] = fill
        obs[f.name] = arr
    return obs


def _state_predicts_outcome_dataset(n_per_class: int = 16) -> SelfPlaySLDataset:
    """A dataset where the STATE predicts the outcome: a cleared obs pattern always
    labels outcome 1 and a distinct not-cleared pattern always labels 0.

    The two classes differ only in their float context fields, so the value head
    over the pooled CLS context can separate them after training. Actions are a
    single legal index across all rows (the aux-head loss ignores the action; it
    is kept only for dataset provenance).
    """
    n = 2 * n_per_class
    cleared = _class_obs(cleared=True)
    not_cleared = _class_obs(cleared=False)
    obs_arrays: dict[str, np.ndarray] = {}
    for f in OBS_FIELDS:
        rows = [cleared[f.name]] * n_per_class + [not_cleared[f.name]] * n_per_class
        obs_arrays[f.name] = np.stack(rows, axis=0)
    actions = np.full(n, _LEGAL_IDX, dtype=np.int64)
    outcomes = np.array([1.0] * n_per_class + [0.0] * n_per_class, dtype=np.float32)
    masks = np.zeros((n, ACTION_DIM), dtype=np.bool_)
    masks[:, _LEGAL_IDX] = True  # the single chosen action is legal in every row
    manifest = SelfPlaySLManifest(
        n_episodes=n,
        n_samples=n,
        act1_clear_rate=0.5,
        interface_version=INTERFACE_VERSION,
        seed=0,
    )
    return SelfPlaySLDataset(obs_arrays, actions, masks, outcomes, manifest)


# --- Dataset save/load round-trip -------------------------------------------


class TestSelfPlaySLDatasetRoundTrip:
    """SelfPlaySLDataset save/load preserves obs, actions, masks, outcomes, manifest."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        ds = _synthetic_dataset(n_samples=10, seed=0)
        path = tmp_path / "sp.npz"
        ds.save(path)

        loaded = SelfPlaySLDataset.load(path)
        assert len(loaded) == 10
        for f in OBS_FIELDS:
            np.testing.assert_array_equal(loaded.obs_arrays[f.name], ds.obs_arrays[f.name])
        np.testing.assert_array_equal(loaded.actions, ds.actions)
        np.testing.assert_array_equal(loaded.masks, ds.masks)
        np.testing.assert_array_equal(loaded.outcomes, ds.outcomes)
        # Full-space mask width is preserved.
        assert loaded.masks.shape == (10, ACTION_DIM)
        # Manifest survives.
        assert loaded.manifest.n_episodes == ds.manifest.n_episodes
        assert loaded.manifest.n_samples == 10
        assert loaded.manifest.interface_version == INTERFACE_VERSION
        assert loaded.manifest.act1_clear_rate == pytest.approx(ds.manifest.act1_clear_rate)
        assert loaded.manifest.seed == 0

    def test_manifest_seed_none_roundtrips(self, tmp_path: Path) -> None:
        """A None seed round-trips as None (via the -1 sentinel encoding)."""
        ds = _synthetic_dataset(n_samples=4, seed=1)
        ds.manifest.seed = None
        path = tmp_path / "sp_none.npz"
        ds.save(path)
        loaded = SelfPlaySLDataset.load(path)
        assert loaded.manifest.seed is None

    def test_dataset_rejects_wrong_mask_width(self) -> None:
        """The constructor guards the full ACTION_DIM mask-width invariant."""
        n = 3
        obs_arrays: dict[str, np.ndarray] = {
            f.name: np.zeros((n, *f.shape), dtype=f.dtype) for f in OBS_FIELDS
        }
        actions = np.zeros(n, dtype=np.int64)
        bad_masks = np.zeros((n, ACTION_DIM - 1), dtype=np.bool_)  # too narrow
        outcomes = np.zeros(n, dtype=np.float32)
        manifest = SelfPlaySLManifest(n, n, 0.0, INTERFACE_VERSION, 0)
        with pytest.raises(ValueError, match="masks must be"):
            SelfPlaySLDataset(obs_arrays, actions, bad_masks, outcomes, manifest)


# --- Collection with a scripted run env -------------------------------------


class TestCollectSelfPlayDataset:
    """collect_self_play_dataset logs every step and backfills the episode outcome."""

    def test_backfill_shapes_and_rate(self) -> None:
        # ep0: 3 steps, terminal act stays in Act 1 (not cleared -> 0); ep1: 2 steps,
        # terminal act reaches Act 2 (cleared -> 1).
        env = _ScriptedRunEnv([(3, _ACT_NOT_CLEARED), (2, _ACT_CLEARED)])
        policy = _FixedActionPolicy()
        ds = collect_self_play_dataset(policy, env, n_episodes=2, seed=0)

        assert len(ds) == 5
        assert ds.masks.shape == (5, ACTION_DIM)  # full-space mask per step
        # Outcomes are exactly 0/1.
        assert set(np.unique(ds.outcomes).tolist()).issubset({0.0, 1.0})
        # Backfill: every step of an episode shares that episode's label.
        np.testing.assert_array_equal(ds.outcomes[:3], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(ds.outcomes[3:], np.ones(2, dtype=np.float32))
        # Episode-level clear rate: 1 of 2 episodes cleared.
        assert ds.manifest.act1_clear_rate == pytest.approx(0.5)
        assert ds.manifest.n_episodes == 2
        assert ds.manifest.n_samples == 5
        # The policy was consulted exactly once per decision step.
        assert policy.n_calls == 5
        assert set(ds.actions.tolist()) == {_LEGAL_IDX}

    def test_real_actor_critic_stochastic(self) -> None:
        """The real ActorCritic.act surface drives collection engine-free (sampled)."""
        env = _ScriptedRunEnv([(2, _ACT_CLEARED), (2, _ACT_NOT_CLEARED)])
        policy = ActorCritic()
        ds = collect_self_play_dataset(policy, env, n_episodes=2, seed=0, deterministic=False)
        assert len(ds) == 4
        assert ds.masks.shape == (4, ACTION_DIM)
        assert set(np.unique(ds.outcomes).tolist()).issubset({0.0, 1.0})
        # With a single legal action, every sampled action is that index.
        assert set(ds.actions.tolist()) == {_LEGAL_IDX}

    def test_collect_raises_on_zero_episodes(self) -> None:
        env = _ScriptedRunEnv([(1, _ACT_NOT_CLEARED)])
        policy = _FixedActionPolicy()
        with pytest.raises(ValueError, match="n_episodes must be >= 1"):
            collect_self_play_dataset(policy, env, n_episodes=0, seed=0)

    def test_collection_deterministic_under_global_seed(self) -> None:
        """Same global seed -> identical sampled actions across two collections.

        collect_self_play_dataset drives policy.act with deterministic=False, so
        Categorical.sample() draws from the GLOBAL torch RNG; the collector seeds
        only env resets, leaving global seeding to the entry point (see
        train_sl._seed_global_rngs / sts_rl.agent.train.train). With a
        >=2-legal-action env the sampled actions genuinely vary, so two runs under
        the SAME global seed producing identical actions proves the seeding makes
        collection reproducible. Emptying train_sl._seed_global_rngs makes the two
        streams diverge (this test then fails) - the regression it guards.
        """
        # Fixed policy weights (seed before build) so the ONLY per-collection
        # variability is the sampling RNG the reseed below controls, and so the
        # test is reproducible run to run.
        train_sl._seed_global_rngs(0)
        policy = ActorCritic()
        specs = [(16, _ACT_CLEARED), (16, _ACT_NOT_CLEARED)]

        def _collect_actions() -> list[int]:
            train_sl._seed_global_rngs(12345)
            env = _ScriptedRunEnv(specs, legal_indices=_TWO_LEGAL)
            ds = collect_self_play_dataset(policy, env, n_episodes=2, seed=0, deterministic=False)
            return ds.actions.tolist()

        actions1 = _collect_actions()
        actions2 = _collect_actions()
        assert actions1 == actions2
        # The 2-legal-action env genuinely samples BOTH actions (else determinism
        # would be trivial and the seeding untested).
        assert set(actions1) == set(_TWO_LEGAL)


# --- Outcome-regression training --------------------------------------------


class TestOutcomeRegressionPretrain:
    """outcome_regression_pretrain drives loss down, separates predictive states, and
    trains only the encoder + value head (the policy head is frozen)."""

    def test_loss_decreases_and_value_head_separates_states(self) -> None:
        """The value head's win-prob output separates cleared from not-cleared states.

        On a dataset where the STATE predicts the outcome, the auxiliary win-prob
        head (the value head over the pooled CLS context) must, after training,
        score a cleared-class state above a not-cleared one, and the train loss
        must fall. Replaces the old chosen-logit separation test: the aux head
        predicts the outcome from the state, not from the chosen action.
        """
        torch.manual_seed(0)
        model = ActorCritic()
        ds = _state_predicts_outcome_dataset(n_per_class=16)
        _model, stats = outcome_regression_pretrain(
            model,
            ds,
            epochs=60,
            lr=1e-2,
            batch_size=32,
            val_frac=0.25,
            seed=0,
            patience=100,  # do not early stop
        )
        # Loss decreases over training.
        assert stats.train_loss_history[-1] < stats.train_loss_history[0]

        # The value head's win-prob logit is higher on a cleared-class state than
        # on a not-cleared one (forward the encoder's pooled CLS through the value
        # head exactly as the loss does).
        model.eval()
        with torch.no_grad():
            cleared = observation_to_batched_tensors(_class_obs(cleared=True), torch.device("cpu"))
            not_cleared = observation_to_batched_tensors(
                _class_obs(cleared=False), torch.device("cpu")
            )
            _pt_c, pooled_c, _m_c = model.encoder(cleared)
            _pt_n, pooled_n, _m_n = model.encoder(not_cleared)
            win_cleared = model.value(pooled_c)
            win_not_cleared = model.value(pooled_n)
        assert win_cleared.item() > win_not_cleared.item()

    def test_returns_same_model_object(self) -> None:
        """The returned model is the SAME net, fine-tuned in place."""
        model = ActorCritic()
        ds = _state_predicts_outcome_dataset(n_per_class=8)
        returned, _stats = outcome_regression_pretrain(
            model, ds, epochs=2, lr=1e-3, batch_size=8, val_frac=0.25, seed=0
        )
        assert returned is model

    def test_determinism_under_fixed_seed(self) -> None:
        torch.manual_seed(0)
        base = ActorCritic()
        ds = _state_predicts_outcome_dataset(n_per_class=8)
        m1 = copy.deepcopy(base)
        m2 = copy.deepcopy(base)
        _r1, s1 = outcome_regression_pretrain(
            m1, ds, epochs=10, lr=1e-3, batch_size=8, val_frac=0.25, seed=123, patience=100
        )
        _r2, s2 = outcome_regression_pretrain(
            m2, ds, epochs=10, lr=1e-3, batch_size=8, val_frac=0.25, seed=123, patience=100
        )
        assert s1.train_loss_history == s2.train_loss_history
        assert s1.val_loss_history == s2.val_loss_history
        for key in m1.state_dict():
            torch.testing.assert_close(
                m1.state_dict()[key], m2.state_dict()[key], msg=f"mismatch at key {key}"
            )

    def test_freezes_policy_head_and_trains_encoder_value(self) -> None:
        """One aux-head fine-tune grads the encoder + value head but NOT the policy.

        The auxiliary win-prob loss is BCEWithLogits over the value head's output
        on the pooled CLS context; the pointer (policy) head is frozen
        (requires_grad=False) before training and is absent from that loss graph,
        so no gradient reaches it and its action-logit weights are left bitwise
        intact - the property that prevents the policy collapse the earlier
        per-chosen-logit objective caused. The shared encoder and the value head DO
        receive finite gradient and their weights change. Replaces the old
        per-chosen-logit-row gradient test (the pointer head has no per-action
        logit row to isolate). The freeze is scoped: requires_grad is restored on
        return.
        """
        torch.manual_seed(0)
        model = ActorCritic()
        # Snapshot each param group before the fine-tune, to prove the policy is
        # left intact while the encoder + value head change.
        policy_before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        value_before = {k: v.detach().clone() for k, v in model.value.state_dict().items()}
        encoder_before = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}

        # n_per_class=6 -> n=12, val_frac=0.25 -> n_train=9 <= batch_size, so exactly
        # one train batch and one backward run: the post-call .grad state is
        # deterministic (encoder + value populated; policy None).
        ds = _state_predicts_outcome_dataset(n_per_class=6)
        outcome_regression_pretrain(
            model, ds, epochs=1, lr=1e-2, batch_size=64, val_frac=0.25, seed=0, patience=100
        )

        # No gradient reached ANY policy (pointer head) param: it is both frozen and
        # graph-disjoint from the value-head loss.
        for name, p in model.policy.named_parameters():
            assert p.grad is None, f"policy param {name} received a gradient"
        # ...and every policy weight is bitwise unchanged (action logits intact).
        for k, v in model.policy.state_dict().items():
            torch.testing.assert_close(
                v, policy_before[k], rtol=0, atol=0, msg=f"policy param {k} changed"
            )
        # The freeze is scoped to the call: requires_grad is restored on return.
        assert all(p.requires_grad for p in model.policy.parameters())

        # The value head received finite, nonzero gradient and its weights changed.
        value_grad = 0.0
        for p in model.value.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            value_grad += float(p.grad.abs().sum())
        assert value_grad > 0.0
        assert any(
            not torch.equal(model.value.state_dict()[k], value_before[k]) for k in value_before
        )
        # The shared encoder also received finite gradient and its weights changed
        # (its features are shaped toward outcome-awareness).
        encoder_grad = 0.0
        for p in model.encoder.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
                encoder_grad += float(p.grad.abs().sum())
        assert encoder_grad > 0.0
        assert any(
            not torch.equal(model.encoder.state_dict()[k], encoder_before[k])
            for k in encoder_before
        )


class TestOutcomeRegressionDegenerateGuards:
    """outcome_regression_pretrain fails loudly on degenerate inputs."""

    def test_raises_on_all_same_outcome(self) -> None:
        model = ActorCritic()
        ds = _state_predicts_outcome_dataset(n_per_class=8)
        ds.outcomes[:] = 1.0  # every episode "cleared" -> no discriminative signal
        with pytest.raises(ValueError, match="both cleared"):
            outcome_regression_pretrain(model, ds, epochs=5, val_frac=0.25)

    def test_raises_on_too_small_dataset(self) -> None:
        # A single-sample dataset cannot form a train/val split (checked before the
        # outcome-variance guard, so a one-row dataset fails on size).
        model = ActorCritic()
        ds = _synthetic_dataset(
            n_samples=1,
            actions=np.array([0], dtype=np.int64),
            outcomes=np.array([1.0], dtype=np.float32),
        )
        with pytest.raises(ValueError, match="too small for a train/val split"):
            outcome_regression_pretrain(model, ds, epochs=1, val_frac=0.15)


# --- Checkpoint round-trip via load_checkpoint ------------------------------


class TestSelfPlaySLCheckpointRoundTrip:
    """An SL-produced checkpoint loads through load_checkpoint (warm-start path)."""

    def test_checkpoint_loads(self, tmp_path: Path) -> None:
        model = ActorCritic()
        hidden_dim = model.encoder.output_dim
        ckpt_path = tmp_path / "sl_test.pt"
        save_bc_checkpoint(model, ckpt_path, hidden_dim)

        loaded = load_checkpoint(ckpt_path)  # strict=True inside
        assert isinstance(loaded, ActorCritic)
        for key in model.state_dict():
            torch.testing.assert_close(
                loaded.state_dict()[key],
                model.state_dict()[key],
                msg=f"mismatch at key {key}",
            )
