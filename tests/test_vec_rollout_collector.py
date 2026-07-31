"""Tests for the vectorized PPO rollout collector and buffer.

Engine-free: the vec collector drives a real :class:`ActorCritic` over a
:class:`~conftest.StubVecEnv` (an in-process synchronous stand-in that mirrors
``SubprocVecEnv``'s observable contract - batched obs, same-step auto-reset with
``final_info``), so no C++ engine and no worker processes are needed. StubEnv's
``terminate_prob``/``max_episode_steps`` knobs force the two GAE tail cases per
env deterministically. Dimensions come from the interface registry, never
hardcoded literals.

The headline correctness checks are per-env GAE isolation (a done in one env
never leaks advantage into another) and num_envs=1 parity (the vec path
reproduces the single-env :class:`RolloutCollector` byte-for-byte).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from conftest import make_stub_env, make_stub_vec_env

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.aux_heads import PLAYER_CUR_HP_INDEX
from sts_rl.agent.ppo import compute_gae
from sts_rl.agent.rollout_buffer import RolloutBuffer, VecRolloutBuffer
from sts_rl.agent.rollout_collector import (
    RolloutCollector,
    VecRolloutCollector,
    vec_observation_to_tensors,
)
from sts_rl.interface import ACTION_DIM, OBS_FIELDS, InterfaceError

# Small trunk keeps the network tests fast; width is irrelevant to collection.
SMALL_HIDDEN = 32
NUM_ENVS = 3
N_STEPS = 8
MINIBATCH = 5
# GAE knobs distinct from the ppo defaults (0.99 / 0.95) so a forwarded override
# visibly changes the advantages/returns from the buffer-default result.
DISTINCT_GAMMA = 0.5
DISTINCT_GAE_LAMBDA = 0.5


def _make_ac() -> ActorCritic:
    return ActorCritic(hidden_dim=SMALL_HIDDEN)


def _legal_bool_mask(num_envs: int) -> torch.Tensor:
    """A (num_envs, ACTION_DIM) bool mask with one legal action per row."""
    mask = torch.zeros((num_envs, ACTION_DIM), dtype=torch.bool)
    mask[:, 0] = True
    return mask


def _add_transition(sub: RolloutBuffer) -> None:
    """Append one interface-valid transition to a single-env sub-buffer.

    Used by the caller-bug guard tests, which build a VecRolloutBuffer's
    sub-buffers directly to reach paths add_batch's lockstep fill cannot produce
    (e.g. unequal per-env lengths).
    """
    obs = {
        field.name: torch.zeros(
            field.shape, dtype=torch.long if field.bounds == "id" else torch.float32
        )
        for field in OBS_FIELDS
    }
    sub.add(
        obs=obs,
        action=torch.zeros((), dtype=torch.long),
        log_prob=torch.zeros(()),
        value=torch.zeros(()),
        reward=0.0,
        done=0.0,
        mask=torch.zeros(ACTION_DIM, dtype=torch.bool),
    )


# -- Buffer layout / shapes --------------------------------------------------


def test_collect_fills_buffer_with_total_transitions() -> None:
    """collect stores n_steps PER env (n_steps*num_envs total) and leaves GAE computed."""
    collector = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=0)
    buffer = VecRolloutBuffer(NUM_ENVS)

    stats = collector.collect(buffer, N_STEPS)

    total = N_STEPS * NUM_ENVS
    assert stats.n_steps == total
    assert len(buffer) == total
    # Every env's sub-buffer holds exactly n_steps (lockstep stepping).
    assert all(len(sub) == N_STEPS for sub in buffer._buffers)
    # A full partition proves per-env GAE ran (iter_minibatches requires it).
    batches = list(buffer.iter_minibatches(MINIBATCH))
    assert sum(mb.actions.shape[0] for mb in batches) == total


def test_minibatch_shapes_flatten_env_and_time() -> None:
    """Flattened minibatches carry (mb, *shape) obs, (mb, ACTION_DIM) masks, (mb,) vectors."""
    collector = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=0)
    buffer = VecRolloutBuffer(NUM_ENVS)
    collector.collect(buffer, N_STEPS)

    total = N_STEPS * NUM_ENVS
    seen = 0
    for mb in buffer.iter_minibatches(MINIBATCH):
        mb_size = mb.actions.shape[0]
        seen += mb_size
        assert mb.masks.shape == (mb_size, ACTION_DIM)
        assert mb.masks.dtype == torch.bool
        for vector in (mb.old_log_probs, mb.old_values, mb.advantages, mb.returns):
            assert vector.shape == (mb_size,)
        for field in OBS_FIELDS:
            assert mb.obs[field.name].shape == (mb_size, *field.shape), field.name
    assert seen == total
    # A full-width request yields ONE batch spanning the whole flattened rollout -
    # the shape ppo_update's explained-variance pass relies on.
    (full,) = list(buffer.iter_minibatches(total, shuffle=False))
    assert full.actions.shape == (total,)


# -- Per-env GAE correctness -------------------------------------------------


def test_per_env_gae_matches_independent_compute_gae() -> None:
    """Each env's advantages/returns equal an independent compute_gae over that env alone.

    Fills a VecRolloutBuffer directly with DISTINCT per-env done patterns (env 0
    terminates mid-rollout, env 1 never does), so a cross-env leak - using another
    env's rewards/values/dones or bootstrap - would change the result. Reusing the
    trusted compute_gae per env is not circular: it gates that the vec buffer routed
    each env's own trajectory plus its own tail through the single-env GAE and did
    not mix envs.
    """
    num_envs = 2
    length = 4
    torch.manual_seed(0)
    buffer = VecRolloutBuffer(num_envs)

    # env 0 terminates at t=1 (done=1); env 1 never terminates.
    dones_by_step = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    ]
    rewards_log: list[list[float]] = [[] for _ in range(num_envs)]
    values_log: list[list[float]] = [[] for _ in range(num_envs)]
    dones_log: list[list[float]] = [[] for _ in range(num_envs)]
    mask = _legal_bool_mask(num_envs)
    for step in range(length):
        obs = {field.name: torch.zeros((num_envs, *field.shape)) for field in OBS_FIELDS}
        values = torch.randn(num_envs)
        rewards = torch.randn(num_envs)
        dones = dones_by_step[step]
        buffer.add_batch(
            obs=obs,
            actions=torch.zeros(num_envs, dtype=torch.long),
            log_probs=torch.zeros(num_envs),
            values=values,
            rewards=rewards,
            dones=dones,
            masks=mask,
        )
        for i in range(num_envs):
            rewards_log[i].append(float(rewards[i]))
            values_log[i].append(float(values[i]))
            dones_log[i].append(float(dones[i]))

    last_values = torch.randn(num_envs)
    buffer.compute_advantages(last_values, gamma=DISTINCT_GAMMA, gae_lambda=DISTINCT_GAE_LAMBDA)

    for i in range(num_envs):
        expected_adv, expected_ret = compute_gae(
            torch.tensor(rewards_log[i]),
            torch.tensor(values_log[i]),
            torch.tensor(dones_log[i]),
            last_values[i],
            gamma=DISTINCT_GAMMA,
            gae_lambda=DISTINCT_GAE_LAMBDA,
        )
        (batch_i,) = list(buffer._buffers[i].iter_minibatches(length, shuffle=False))
        assert torch.allclose(batch_i.advantages, expected_adv), i
        assert torch.allclose(batch_i.returns, expected_ret), i


def test_forced_termination_flags_done_per_env() -> None:
    """terminate_prob=1.0 ends every env every step: all sub-buffer dones are 1.0."""
    vec = make_stub_vec_env(NUM_ENVS, terminate_prob=1.0, max_episode_steps=10_000)
    collector = VecRolloutCollector(vec, _make_ac(), seed=0)
    buffer = VecRolloutBuffer(NUM_ENVS)

    stats = collector.collect(buffer, N_STEPS)

    for sub in buffer._buffers:
        assert all(done == 1.0 for done in sub._dones)
    # Every env terminates every step, pooled across envs.
    assert stats.n_episodes == N_STEPS * NUM_ENVS


def test_forced_truncation_keeps_done_zero_per_env() -> None:
    """A time-limit truncation is NOT flagged done in any env's sub-buffer."""
    max_steps = 4
    vec = make_stub_vec_env(NUM_ENVS, terminate_prob=0.0, max_episode_steps=max_steps)
    collector = VecRolloutCollector(vec, _make_ac(), seed=0)
    buffer = VecRolloutBuffer(NUM_ENVS)

    stats = collector.collect(buffer, N_STEPS)

    for sub in buffer._buffers:
        assert all(done == 0.0 for done in sub._dones)
    # Each env truncates every max_steps steps; N_STEPS // max_steps per env.
    assert stats.n_episodes == (N_STEPS // max_steps) * NUM_ENVS


# -- num_envs=1 parity with the single-env path ------------------------------


def test_num_envs_one_matches_single_env_path() -> None:
    """VecRolloutCollector(num_envs=1) reproduces RolloutCollector byte-for-byte.

    Same seed and identical initial weights: per-step env RNG (step, then
    reset-on-done) and global torch RNG (one act sample of shape (1,)) are consumed
    identically, so the stored action/reward/done streams, the per-env GAE, and the
    pooled episode stats all match the single-env path. This is the regression gate
    for the additive design's "num_envs=1 is behavior-identical" contract.
    """
    seed = 7
    weight_seed = 123
    steps = 12
    # A mix of terminations (terminate_prob) and truncations (small cap) so both
    # done branches are exercised under parity.
    env_kwargs = {"terminate_prob": 0.3, "max_episode_steps": 4}

    torch.manual_seed(weight_seed)
    ac_single = _make_ac()
    single = RolloutCollector(
        make_stub_env(**env_kwargs),
        ac_single,
        seed=seed,
        gamma=DISTINCT_GAMMA,
        gae_lambda=DISTINCT_GAE_LAMBDA,
    )
    single_buffer = RolloutBuffer()
    single_stats = single.collect(single_buffer, steps)

    # Re-seed: identical weight init AND resets the torch RNG to the same point, so
    # both collects sample actions from the same generator state.
    torch.manual_seed(weight_seed)
    ac_vec = _make_ac()
    vec = VecRolloutCollector(
        make_stub_vec_env(1, **env_kwargs),
        ac_vec,
        seed=seed,
        gamma=DISTINCT_GAMMA,
        gae_lambda=DISTINCT_GAE_LAMBDA,
    )
    vec_buffer = VecRolloutBuffer(1)
    vec_stats = vec.collect(vec_buffer, steps)

    sub = vec_buffer._buffers[0]
    assert torch.equal(torch.stack(single_buffer._actions), torch.stack(sub._actions))
    assert single_buffer._rewards == sub._rewards
    assert single_buffer._dones == sub._dones
    # The per-env combat-span backfill reproduces the single-env path byte-for-byte:
    # same obs stream (same RNG) -> same enemy_alive boundaries -> same end_combat_hp.
    # The single path sources a terminal combat's end HP from step()'s next_obs; the
    # vec path from final_observation - which is that same obs under num_envs=1.
    assert single_buffer._end_combat_hp_valid == sub._end_combat_hp_valid
    for single_hp, vec_hp in zip(single_buffer._end_combat_hp, sub._end_combat_hp):
        assert (math.isnan(single_hp) and math.isnan(vec_hp)) or single_hp == vec_hp

    (single_batch,) = list(single_buffer.iter_minibatches(steps, shuffle=False))
    (vec_batch,) = list(vec_buffer.iter_minibatches(steps, shuffle=False))
    assert torch.allclose(single_batch.advantages, vec_batch.advantages)
    assert torch.allclose(single_batch.returns, vec_batch.returns)

    # num_envs=1 -> total transitions == steps; episode stats pool identically.
    assert vec_stats.n_steps == single_stats.n_steps
    assert vec_stats.n_episodes == single_stats.n_episodes
    assert vec_stats.mean_episode_return == single_stats.mean_episode_return
    assert vec_stats.mean_episode_length == single_stats.mean_episode_length


# -- Seeding, boundaries, and diagnostics ------------------------------------


def test_per_env_seed_derivation_is_base_plus_index() -> None:
    """The collector seeds env i with base+i, so env i's initial obs == StubEnv(seed=base+i)."""
    base = 4
    collector = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=base)
    for i in range(NUM_ENVS):
        ref_obs, _info = make_stub_env().reset(seed=base + i)
        for field in OBS_FIELDS:
            assert np.array_equal(collector._obs[field.name][i], ref_obs[field.name]), (
                field.name,
                i,
            )


def test_vec_observation_to_tensors_dtype_no_batch_dim() -> None:
    """Vec converter casts id->long, others->float32, keeping the num_envs leading dim."""
    obs, _masks = make_stub_vec_env(NUM_ENVS).reset(seeds=[0, 1, 2])
    tensors = vec_observation_to_tensors(obs, torch.device("cpu"))

    id_fields = {field.name for field in OBS_FIELDS if field.bounds == "id"}
    assert set(tensors) == {field.name for field in OBS_FIELDS}
    for field in OBS_FIELDS:
        tensor = tensors[field.name]
        # Leading num_envs axis preserved, no extra batch dim added.
        assert tensor.shape == (NUM_ENVS, *field.shape), field.name
        expected = torch.long if field.name in id_fields else torch.float32
        assert tensor.dtype == expected, field.name


def test_stored_per_env_obs_dtypes_and_mask_shape() -> None:
    """Each sub-buffer stores unbatched obs with the interface dtype split; masks bool."""
    collector = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=0)
    buffer = VecRolloutBuffer(NUM_ENVS)
    collector.collect(buffer, N_STEPS)

    id_fields = {field.name for field in OBS_FIELDS if field.bounds == "id"}
    stored = buffer._buffers[0]._obs[0]
    for field in OBS_FIELDS:
        expected = torch.long if field.name in id_fields else torch.float32
        assert stored[field.name].dtype == expected, field.name
        assert stored[field.name].shape == field.shape, field.name  # no batch dim
    assert buffer._buffers[0]._masks[0].dtype == torch.bool
    assert buffer._buffers[0]._masks[0].shape == (ACTION_DIM,)


def test_malformed_mask_raises_interface_error() -> None:
    """A malformed batched mask is rejected at the vec collector boundary."""
    collector = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=0)
    # Non-bool dtype but with a legal action per row: the policy head would accept
    # this (via .bool()), so assert_valid_mask at the boundary is the only rejecter.
    int_mask = np.zeros((NUM_ENVS, ACTION_DIM), dtype=np.int32)
    int_mask[:, 0] = 1
    collector._masks = int_mask
    with pytest.raises(InterfaceError):
        collector.collect(VecRolloutBuffer(NUM_ENVS), 1)

    # A wrong leading (num_envs) shape is rejected too.
    fresh = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=0)
    fresh._masks = np.zeros((NUM_ENVS + 1, ACTION_DIM), dtype=np.bool_)
    with pytest.raises(InterfaceError):
        fresh.collect(VecRolloutBuffer(NUM_ENVS), 1)


def test_no_completed_episode_gives_none_means() -> None:
    """Zero completed episodes -> None means (no divide-by-zero), not 0.0."""
    vec = make_stub_vec_env(NUM_ENVS, terminate_prob=0.0, max_episode_steps=10_000)
    collector = VecRolloutCollector(vec, _make_ac(), seed=0)

    stats = collector.collect(VecRolloutBuffer(NUM_ENVS), 5)

    assert stats.n_episodes == 0
    assert stats.mean_episode_return is None
    assert stats.mean_episode_length is None


def test_stored_tensors_have_no_grad_history() -> None:
    """act runs under no_grad and the sub-buffers clone detached, so nothing tracks grad."""
    collector = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=0)
    buffer = VecRolloutBuffer(NUM_ENVS)
    collector.collect(buffer, N_STEPS)

    sub = buffer._buffers[0]
    for tensor in (sub._values[0], sub._log_probs[0], sub._actions[0]):
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None


def test_training_mode_restored_after_collect() -> None:
    """collect saves and restores the network's train/eval mode."""
    collector = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=0)

    collector._actor_critic.train()
    collector.collect(VecRolloutBuffer(NUM_ENVS), N_STEPS)
    assert collector._actor_critic.training is True

    collector._actor_critic.eval()
    collector.collect(VecRolloutBuffer(NUM_ENVS), N_STEPS)
    assert collector._actor_critic.training is False


def test_nonpositive_n_steps_raises() -> None:
    """n_steps <= 0 is a degenerate request and fails loudly."""
    collector = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=0)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            collector.collect(VecRolloutBuffer(NUM_ENVS), bad)


# -- VecRolloutBuffer guards -------------------------------------------------


def test_vec_buffer_rejects_non_positive_num_envs() -> None:
    """num_envs < 1 is degenerate and fails at construction."""
    with pytest.raises(ValueError):
        VecRolloutBuffer(0)


def test_vec_buffer_add_batch_leading_dim_guard() -> None:
    """add_batch asserts the leading num_envs axis at the boundary."""
    buffer = VecRolloutBuffer(2)
    obs = {field.name: torch.zeros((2, *field.shape)) for field in OBS_FIELDS}
    with pytest.raises(ValueError):
        buffer.add_batch(
            obs=obs,
            actions=torch.zeros(3, dtype=torch.long),  # wrong leading dim (3 != 2)
            log_probs=torch.zeros(2),
            values=torch.zeros(2),
            rewards=torch.zeros(2),
            dones=torch.zeros(2),
            masks=_legal_bool_mask(2),
        )


def test_vec_buffer_reset_clears() -> None:
    """reset clears every env's sub-buffer for reuse."""
    collector = VecRolloutCollector(make_stub_vec_env(NUM_ENVS), _make_ac(), seed=0)
    buffer = VecRolloutBuffer(NUM_ENVS)
    collector.collect(buffer, N_STEPS)
    assert len(buffer) == N_STEPS * NUM_ENVS

    buffer.reset()
    assert len(buffer) == 0


@pytest.mark.parametrize("bad_field", ["obs", "log_probs", "values", "rewards", "dones", "masks"])
def test_add_batch_guards_every_batched_field_leading_dim(bad_field: str) -> None:
    """add_batch asserts the num_envs leading dim on EVERY batched field.

    The existing guard covers only actions; a wrong leading dim on any obs tensor,
    mask, reward, done, log_prob, or value must fail loudly too, rather than
    silently drop or misalign rows through the per-env indexing.
    """
    num_envs = 2
    bad = num_envs + 1
    obs = {field.name: torch.zeros((num_envs, *field.shape)) for field in OBS_FIELDS}
    actions = torch.zeros(num_envs, dtype=torch.long)
    log_probs = torch.zeros(num_envs)
    values = torch.zeros(num_envs)
    rewards = torch.zeros(num_envs)
    dones = torch.zeros(num_envs)
    masks = _legal_bool_mask(num_envs)

    # Corrupt exactly the field under test to a wrong leading dim; the raised
    # message must name it (obs fields report as "obs[<name>]").
    if bad_field == "obs":
        corrupted = OBS_FIELDS[0]
        obs[corrupted.name] = torch.zeros((bad, *corrupted.shape))
        expected_match = corrupted.name
    elif bad_field == "log_probs":
        log_probs = torch.zeros(bad)
        expected_match = "log_probs"
    elif bad_field == "values":
        values = torch.zeros(bad)
        expected_match = "values"
    elif bad_field == "rewards":
        rewards = torch.zeros(bad)
        expected_match = "rewards"
    elif bad_field == "dones":
        dones = torch.zeros(bad)
        expected_match = "dones"
    else:  # masks
        masks = torch.zeros((bad, ACTION_DIM), dtype=torch.bool)
        expected_match = "masks"

    with pytest.raises(ValueError, match=expected_match):
        VecRolloutBuffer(num_envs).add_batch(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            dones=dones,
            masks=masks,
        )


def test_iter_minibatches_rejects_unequal_sub_buffer_lengths() -> None:
    """iter_minibatches fails loudly when envs were stepped an unequal number of times.

    A caller bug (e.g. add_batch skipped an env) leaves sub-buffers of different
    lengths; the flattened cat would otherwise silently weight envs unequally.
    Built directly on the sub-buffers because add_batch's lockstep fill cannot
    produce this. Advantages ARE computed first, so the ONLY thing left to raise
    is the length guard - reverting it makes the mismatched cat succeed silently.
    """
    buffer = VecRolloutBuffer(2)
    _add_transition(buffer._buffers[0])
    _add_transition(buffer._buffers[0])  # env 0: 2 steps
    _add_transition(buffer._buffers[1])  # env 1: 1 step
    buffer.compute_advantages(torch.zeros(2))

    with pytest.raises(RuntimeError, match="unequal"):
        list(buffer.iter_minibatches(MINIBATCH))


def test_iter_minibatches_empty_buffer_yields_nothing() -> None:
    """A never-filled buffer yields no minibatches instead of crashing on the empty cat."""
    buffer = VecRolloutBuffer(NUM_ENVS)
    assert list(buffer.iter_minibatches(MINIBATCH)) == []


def test_compute_advantages_rejects_wrong_last_values_leading_dim() -> None:
    """compute_advantages asserts last_values carries the num_envs leading dim."""
    buffer = VecRolloutBuffer(2)
    with pytest.raises(ValueError, match="last_values"):
        buffer.compute_advantages(torch.zeros(3))  # 3 != num_envs (2)


# -- end_combat_hp per-env tracking + buffer delegation ----------------------


class _ScriptedVecEnv:
    """In-process vec env emitting a scripted per-env enemy_alive / player-HP sequence.

    ``alive[i][k]`` / ``hp[i][k]`` define env ``i``'s obs at step index ``k`` (``k == 0``
    is the reset obs). No terminations/truncations, so the collector never resets
    mid-collect and the per-env combat-span tracking is exercised purely on the
    enemy_alive True->False signal and the rollout cutoff. Structurally satisfies
    ``VecEnvProtocol``.
    """

    def __init__(self, alive: list[list[bool]], hp: list[list[float]]) -> None:
        self.num_envs = len(alive)
        self._alive = alive
        self._hp = hp
        self._base, _ = make_stub_env().reset(seed=0)  # a full valid obs to overlay onto
        self._k = 0

    def reset(self, seeds=None):
        self._k = 0
        return self._batched_obs(0), self._masks()

    def step(self, actions):
        self._k += 1
        obs = self._batched_obs(self._k)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        flags = np.zeros(self.num_envs, dtype=np.bool_)
        infos = [{"action_mask": self._legal_row()} for _ in range(self.num_envs)]
        return obs, rewards, flags, flags.copy(), self._masks(), infos

    def _batched_obs(self, k: int) -> dict:
        obs = {
            field.name: np.stack(
                [np.asarray(self._base[field.name]).copy() for _ in range(self.num_envs)], axis=0
            )
            for field in OBS_FIELDS
        }
        enemy_alive = np.zeros_like(np.asarray(obs["enemy_alive"], dtype=np.float32))
        player_scalars = np.asarray(obs["player_scalars"], dtype=np.float32).copy()
        for i in range(self.num_envs):
            if self._alive[i][k]:
                enemy_alive[i, 0] = 1.0
            player_scalars[i, PLAYER_CUR_HP_INDEX] = self._hp[i][k]
        obs["enemy_alive"] = enemy_alive
        obs["player_scalars"] = player_scalars
        return obs

    def _legal_row(self) -> np.ndarray:
        row = np.zeros(ACTION_DIM, dtype=np.bool_)
        row[0] = True
        return row

    def _masks(self) -> np.ndarray:
        masks = np.zeros((self.num_envs, ACTION_DIM), dtype=np.bool_)
        masks[:, 0] = True
        return masks


def test_vec_backfill_end_combat_hp_delegates_to_one_env() -> None:
    """backfill_end_combat_hp routes to exactly one env's sub-buffer; others untouched."""
    num_envs = 2
    buffer = VecRolloutBuffer(num_envs)
    mask = _legal_bool_mask(num_envs)
    for _ in range(3):
        obs = {field.name: torch.zeros((num_envs, *field.shape)) for field in OBS_FIELDS}
        buffer.add_batch(
            obs=obs,
            actions=torch.zeros(num_envs, dtype=torch.long),
            log_probs=torch.zeros(num_envs),
            values=torch.zeros(num_envs),
            rewards=torch.zeros(num_envs),
            dones=torch.zeros(num_envs),
            masks=mask,
        )
    buffer.backfill_end_combat_hp(1, 0, 2, 33.0)  # env 1, steps [0, 2)

    env0, env1 = buffer._buffers
    assert all(valid is False for valid in env0._end_combat_hp_valid)  # env 0 untouched
    assert env1._end_combat_hp_valid == [True, True, False]
    assert env1._end_combat_hp[0] == 33.0
    assert env1._end_combat_hp[1] == 33.0


def test_vec_backfill_rejects_bad_env_index() -> None:
    """An env_index outside [0, num_envs) fails loudly at the vec boundary."""
    buffer = VecRolloutBuffer(2)
    with pytest.raises(ValueError, match="env_index"):
        buffer.backfill_end_combat_hp(2, 0, 1, 5.0)


def test_add_batch_defaults_end_combat_hp_to_invalid_placeholder() -> None:
    """add_batch without the aux fields leaves every env's steps invalid placeholders."""
    num_envs = 2
    buffer = VecRolloutBuffer(num_envs)
    obs = {field.name: torch.zeros((num_envs, *field.shape)) for field in OBS_FIELDS}
    buffer.add_batch(
        obs=obs,
        actions=torch.zeros(num_envs, dtype=torch.long),
        log_probs=torch.zeros(num_envs),
        values=torch.zeros(num_envs),
        rewards=torch.zeros(num_envs),
        dones=torch.zeros(num_envs),
        masks=_legal_bool_mask(num_envs),
    )
    for sub in buffer._buffers:
        assert sub._end_combat_hp_valid == [False]
        assert math.isnan(sub._end_combat_hp[0])


def test_vec_collector_tracks_end_combat_hp_per_env() -> None:
    """Per-env combat spans are tracked independently: each env backfills its own end HP.

    env 0: combat over steps 0,1 ends into the overworld at obs_2 (HP A0); steps 2-4
    overworld. env 1: overworld at step 0; combat over steps 1,2 ends at obs_3 (HP A1);
    step 3 overworld; a combat reopens at step 4 and is unfinished at the cutoff. The
    distinct A0/A1 prove env 1 does not read env 0's end HP (or vice versa).
    """
    a0, a1 = 41.0, 57.0
    alive = [
        [True, True, False, False, False, False],  # env 0
        [False, True, True, False, True, True],  # env 1
    ]
    hp = [
        [10.0, 11.0, a0, 13.0, 14.0, 15.0],  # env 0: obs_2 HP == a0
        [20.0, 21.0, 22.0, a1, 24.0, 25.0],  # env 1: obs_3 HP == a1
    ]
    collector = VecRolloutCollector(_ScriptedVecEnv(alive, hp), _make_ac(), seed=0)
    buffer = VecRolloutBuffer(2)

    collector.collect(buffer, 5)

    env0, env1 = buffer._buffers
    # env 0: steps 0,1 valid with a0; steps 2,3,4 invalid.
    assert env0._end_combat_hp_valid == [True, True, False, False, False]
    assert env0._end_combat_hp[0] == a0
    assert env0._end_combat_hp[1] == a0
    # env 1: steps 1,2 valid with a1; step 0 overworld, step 4 unfinished -> invalid.
    assert env1._end_combat_hp_valid == [False, True, True, False, False]
    assert env1._end_combat_hp[1] == a1
    assert env1._end_combat_hp[2] == a1


# -- Deterministic vec terminal-in-combat -> final_observation ---------------

# Distinct HP sentinels for the terminating-vec test so a cross-env leak, or a read
# of the post-reset row instead of final_observation, changes an asserted value.
TERMINAL_COMBAT_HP = 38.0  # env 0's end-of-combat HP, exposed ONLY via final_observation
POST_RESET_HP = 99.0  # env 0's post-reset top-level HP (must never be backfilled)
OVERWORLD_END_HP = 57.0  # env 1's end-of-combat HP (its combat ends into the overworld)


class _TerminatingScriptedVecEnv:
    """Scripted vec env where one env terminates IN COMBAT on a fixed step.

    A deterministic (RNG-free) sibling of :class:`_ScriptedVecEnv` that exercises the
    vec collector's terminal-in-combat path. ``alive[i][k]`` / ``hp[i][k]`` define env
    ``i``'s obs at step index ``k`` (``k == 0`` is the reset obs); ``term_step[i]`` is
    the collector step on which env ``i`` reports ``terminated``. On that step, exactly
    as :class:`~conftest.StubVecEnv` does, the terminal obs (in combat, HP
    ``term_hp[i]``) is preserved under ``infos[i]['final_observation']`` and its
    ``episode`` stats under ``infos[i]['final_info']`` while the returned row ``i`` is a
    post-reset overworld obs (HP :data:`POST_RESET_HP`). Structurally satisfies
    ``VecEnvProtocol``.
    """

    def __init__(
        self,
        alive: list[list[bool]],
        hp: list[list[float]],
        term_step: dict[int, int],
        term_hp: dict[int, float],
    ) -> None:
        self.num_envs = len(alive)
        self._alive = alive
        self._hp = hp
        self._term_step = term_step
        self._term_hp = term_hp
        self._base, _ = make_stub_env().reset(seed=0)  # a full valid obs to overlay onto
        self._calls = 0
        self._done: set[int] = set()

    def reset(self, seeds=None):
        self._calls = 0
        self._done = set()
        rows = [self._single_obs(self._alive[i][0], self._hp[i][0]) for i in range(self.num_envs)]
        return self._batched(rows), self._masks()

    def step(self, actions):
        self._calls += 1
        k = self._calls  # obs index this step transitions into
        step_idx = self._calls - 1  # collector step index (0-based)
        rows: list[dict] = []
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        infos: list[dict] = []
        for i in range(self.num_envs):
            if i not in self._done and self._term_step.get(i) == step_idx:
                # Terminal-in-combat step: stash the terminal obs (HP term_hp[i]) and
                # episode stats, return a post-reset overworld row (same-step idiom).
                terminated[i] = True
                self._done.add(i)
                rows.append(self._single_obs(False, POST_RESET_HP))
                infos.append(
                    {
                        "action_mask": self._legal_row(),
                        "final_observation": self._single_obs(True, self._term_hp[i]),
                        "final_info": {
                            "episode": {"r": 0.0, "l": step_idx + 1},
                            "action_mask": self._legal_row(),
                        },
                    }
                )
            elif i in self._done:
                # Already reset in a prior step: stays in a fresh overworld episode.
                rows.append(self._single_obs(False, POST_RESET_HP))
                infos.append({"action_mask": self._legal_row()})
            else:
                rows.append(self._single_obs(self._alive[i][k], self._hp[i][k]))
                infos.append({"action_mask": self._legal_row()})
        return (
            self._batched(rows),
            np.zeros(self.num_envs, dtype=np.float32),
            terminated,
            truncated,
            self._masks(),
            infos,
        )

    def _single_obs(self, alive_flag: bool, hp_value: float) -> dict:
        obs = {name: np.asarray(value).copy() for name, value in self._base.items()}
        enemy_alive = np.zeros_like(np.asarray(obs["enemy_alive"], dtype=np.float32))
        if alive_flag:
            enemy_alive[0] = 1.0
        obs["enemy_alive"] = enemy_alive
        player = np.asarray(obs["player_scalars"], dtype=np.float32).copy()
        player[PLAYER_CUR_HP_INDEX] = hp_value
        obs["player_scalars"] = player
        return obs

    @staticmethod
    def _batched(rows: list[dict]) -> dict:
        return {name: np.stack([row[name] for row in rows], axis=0) for name in rows[0]}

    def _legal_row(self) -> np.ndarray:
        row = np.zeros(ACTION_DIM, dtype=np.bool_)
        row[0] = True
        return row

    def _masks(self) -> np.ndarray:
        masks = np.zeros((self.num_envs, ACTION_DIM), dtype=np.bool_)
        masks[:, 0] = True
        return masks


def test_vec_terminal_in_combat_reads_own_final_observation() -> None:
    """A terminal-in-combat env backfills its OWN final_observation HP; peers unaffected.

    num_envs=2 so the per-env routing is real. env 0 is in combat over steps 0-2 and
    TERMINATES in combat at step 2, so its end-of-combat HP is available only under
    infos[0]['final_observation'] (the top-level row 0 is already the post-reset obs,
    HP POST_RESET_HP). env 1 is elsewhere: overworld at step 0, then its own combat over
    steps 1-2 ending INTO THE OVERWORLD at obs_3. The distinct sentinels prove env 0
    reads final_observation (not the post-reset row) and that neither env reads the
    other's HP. This locks the vec terminal-in-combat -> final_observation path that the
    num_envs=1 parity test covers only probabilistically.
    """
    alive = [
        [True, True, True, False],  # env 0: in combat, terminates in combat at step 2
        [False, True, True, False],  # env 1: overworld, then combat, ends into overworld
    ]
    hp = [
        [50.0, 48.0, 46.0, 44.0],  # env 0 stored HPs (backfill uses final_observation, not these)
        [60.0, 61.0, 62.0, OVERWORLD_END_HP],  # env 1: obs_3 HP is the post-combat overworld HP
    ]
    collector = VecRolloutCollector(
        _TerminatingScriptedVecEnv(alive, hp, term_step={0: 2}, term_hp={0: TERMINAL_COMBAT_HP}),
        _make_ac(),
        seed=0,
    )
    buffer = VecRolloutBuffer(2)

    collector.collect(buffer, 3)

    env0, env1 = buffer._buffers
    # env 0: all three in-combat steps backfilled from ITS OWN terminal HP, sourced from
    # final_observation - NOT the post-reset row (POST_RESET_HP) and NOT env 1's HP.
    assert env0._end_combat_hp_valid == [True, True, True]
    assert all(hp_v == TERMINAL_COMBAT_HP for hp_v in env0._end_combat_hp)
    # env 1 unaffected: its own combat (steps 1,2) ends into the overworld at obs_3;
    # step 0 was overworld (invalid). Distinct from env 0's terminal HP.
    assert env1._end_combat_hp_valid == [False, True, True]
    assert env1._end_combat_hp[1] == OVERWORLD_END_HP
    assert env1._end_combat_hp[2] == OVERWORLD_END_HP
