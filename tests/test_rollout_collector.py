"""Tests for the single-environment PPO rollout collector.

Engine-free: the collector drives a real :class:`ActorCritic` (small trunk) over
:class:`StubEnv`, so observation shapes, masks, and the act/get_value wiring are
genuine while no C++ engine is needed. StubEnv's ``terminate_prob``/
``max_episode_steps`` knobs force the two GAE tail cases (terminated vs
truncated) deterministically, so the done-flag and diagnostics behavior is
pinned without relying on sampled RNG. Dimensions come from the interface
registry, never hardcoded literals.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.ppo import compute_gae
from sts_rl.agent.rollout_buffer import RolloutBuffer
from sts_rl.agent.rollout_collector import (
    CollectStats,
    RolloutCollector,
    observation_to_batched_tensors,
)
from sts_rl.env.stub_env import CORRECT_ACTION_REWARD, WRONG_ACTION_REWARD, StubEnv
from sts_rl.interface import ACTION_DIM, OBS_FIELDS, InterfaceError

# Small trunk keeps the network tests fast; width is irrelevant to collection.
SMALL_HIDDEN = 32
N_STEPS = 8
MINIBATCH = 4
# A single multi-action block: a legal mask every step, non-degenerate learnable
# task (count 18 > 1, so no degenerate-task warning).
ACTIVE_BLOCK = "REWARD_SELECT"
# GAE knobs distinct from the ppo defaults (0.99 / 0.95) so a forwarded override
# visibly changes the advantages/returns from the buffer-default result.
DISTINCT_GAMMA = 0.5
DISTINCT_GAE_LAMBDA = 0.5


def _make_ac() -> ActorCritic:
    return ActorCritic(hidden_dim=SMALL_HIDDEN)


def _make_env(**overrides: object) -> StubEnv:
    kwargs: dict[str, object] = {
        "reward_mode": "learnable",
        "active_blocks": (ACTIVE_BLOCK,),
    }
    kwargs.update(overrides)
    return StubEnv(**kwargs)


class _CountingEnv:
    """Delegates to a StubEnv while counting reset()/step() calls.

    Lets a test assert stream continuity: that the collector resets the env only
    at construction (not at the start of each collect) so consecutive collects
    resume one trajectory.
    """

    def __init__(self, inner: StubEnv) -> None:
        self._inner = inner
        self.n_resets = 0
        self.n_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.n_resets += 1
        return self._inner.reset(seed=seed, options=options)

    def step(self, action: int):
        self.n_steps += 1
        return self._inner.step(action)


def test_collect_fills_buffer_exactly_and_advantages_ready() -> None:
    """collect stores exactly n_steps transitions and leaves GAE computed."""
    collector = RolloutCollector(_make_env(), _make_ac(), seed=0)
    buffer = RolloutBuffer()

    stats = collector.collect(buffer, N_STEPS)

    assert isinstance(stats, CollectStats)
    assert stats.n_steps == N_STEPS
    assert len(buffer) == N_STEPS
    # iter_minibatches raises unless compute_advantages ran; a full partition of
    # size N_STEPS proves advantages are ready.
    batches = list(buffer.iter_minibatches(MINIBATCH))
    assert sum(mb.actions.shape[0] for mb in batches) == N_STEPS


def test_forced_termination_flags_done_and_continues_stream() -> None:
    """terminate_prob=1.0 ends every step: dones are 1.0 and the stream continues."""
    # Large step cap so truncation never fires; every step terminates instead.
    collector = RolloutCollector(
        _make_env(terminate_prob=1.0, max_episode_steps=10_000), _make_ac(), seed=0
    )
    buffer = RolloutBuffer()

    stats = collector.collect(buffer, N_STEPS)

    assert len(buffer) == N_STEPS  # stream continued past each terminal reset
    assert any(done == 1.0 for done in buffer._dones)
    # Every episode is exactly one step here, so all steps are terminal.
    assert all(done == 1.0 for done in buffer._dones)
    assert stats.n_episodes == N_STEPS


def test_forced_truncation_keeps_done_zero() -> None:
    """A time-limit truncation is NOT flagged done; collection spans episodes."""
    max_steps = 4
    n_steps = 10
    collector = RolloutCollector(
        _make_env(terminate_prob=0.0, max_episode_steps=max_steps), _make_ac(), seed=0
    )
    buffer = RolloutBuffer()

    stats = collector.collect(buffer, n_steps)

    # No termination is possible (terminate_prob=0), so no step may be flagged done.
    assert all(done == 0.0 for done in buffer._dones)
    # The truncation boundaries land on the last step of each capped episode; those
    # transitions in particular must carry done=0 (the tail value is bootstrapped,
    # not discarded).
    for boundary in range(max_steps - 1, n_steps, max_steps):
        assert buffer._dones[boundary] == 0.0
    assert stats.n_episodes == n_steps // max_steps  # 10 // 4 -> 2 completed


def test_final_step_truncation_bootstraps_tail() -> None:
    """A truncation as the FINAL collected step still bootstraps a finite tail.

    Picking n_steps an exact multiple of max_steps lands the last iteration on a
    truncation boundary, so it takes the collector's RESET branch - unlike
    test_forced_truncation_keeps_done_zero, whose final step lands mid-episode on
    the advance branch. The tail then bootstraps get_value off the post-reset
    cursor obs, so this pins that the reset-on-final-iteration -> tail path yields
    finite GAE targets rather than erroring.
    """
    max_steps = 4
    # Exact multiple -> the final iteration truncates and resets, so the tail
    # bootstrap runs off the post-reset cursor (the branch the sibling misses).
    n_steps = 2 * max_steps
    assert n_steps % max_steps == 0
    collector = RolloutCollector(
        _make_env(terminate_prob=0.0, max_episode_steps=max_steps), _make_ac(), seed=0
    )
    buffer = RolloutBuffer()

    stats = collector.collect(buffer, n_steps)

    assert len(buffer) == n_steps
    # terminate_prob=0 -> no step may be flagged done, and the final step is a
    # truncation (done=0), not a terminal.
    assert all(done == 0.0 for done in buffer._dones)
    assert buffer._dones[-1] == 0.0
    # Each episode is a full cap, so completed episodes tile n_steps exactly.
    assert stats.n_episodes == n_steps // max_steps

    # The tail bootstrap fed compute_advantages without error: one in-order batch
    # exposes finite advantages and returns off the post-reset cursor value.
    (batch,) = buffer.iter_minibatches(n_steps, shuffle=False)
    assert torch.isfinite(batch.advantages).all()
    assert torch.isfinite(batch.returns).all()


def test_tail_bootstrap_matches_independent_gae() -> None:
    """The tail last_value is V(cursor obs) and the reward/value/done wiring is exact.

    With termination impossible (all dones 0), the tail bootstrap materially
    affects every return, so recomputing GAE independently over the buffer's own
    stored fields - with last_value = V(collector._obs) - and matching the
    buffer's advantages/returns pins the collector's end-to-end wiring. Reusing
    the trusted compute_gae is not circular: it gates that the COLLECTOR fed the
    cursor obs and stored the right reward/value/done fields.
    """
    ac = _make_ac()
    device = next(ac.parameters()).device
    collector = RolloutCollector(
        _make_env(terminate_prob=0.0, max_episode_steps=10_000), ac, seed=0
    )
    buffer = RolloutBuffer()
    collector.collect(buffer, N_STEPS)

    # Recompute exactly as compute_advantages does, bootstrapping from V of the
    # collector's post-collect cursor obs. The collector collects under eval()
    # (fused-attention path), so recompute in eval() too: get_value is otherwise
    # deterministic, but eval vs train take different attention kernels that
    # differ at the float32 epsilon, which GAE would amplify past allclose.
    rewards = torch.tensor(buffer._rewards, dtype=torch.float32)
    dones = torch.tensor(buffer._dones, dtype=torch.float32)
    values = torch.stack(buffer._values).to(torch.float32)
    ac.eval()
    with torch.no_grad():
        last_value = ac.get_value(observation_to_batched_tensors(collector._obs, device)).squeeze(0)
    expected_adv, expected_ret = compute_gae(rewards, values, dones, last_value)

    # One in-order batch exposes the buffer's computed advantages/returns.
    (batch,) = buffer.iter_minibatches(N_STEPS, shuffle=False)
    assert torch.allclose(batch.advantages, expected_adv)
    assert torch.allclose(batch.returns, expected_ret)


def test_collect_forwards_gamma_and_gae_lambda() -> None:
    """collect passes the collector's gamma/gae_lambda to compute_advantages.

    Recomputing GAE over the buffer's own stored fields with the SAME distinct
    knobs reproduces the buffer's advantages/returns, while recomputing with the
    ppo defaults does not - so the collector forwarded its overrides instead of
    silently using the buffer defaults. terminate_prob=0 keeps every done 0, so
    gamma/gae_lambda affect every step (no boundary zeroing masks the effect).

    Revert-verify: drop the gamma/gae_lambda kwargs from collect's
    compute_advantages call and the buffer reverts to the ppo defaults - the
    distinct-knob match fails and the default-mismatch assertions fail too.
    """
    ac = _make_ac()
    device = next(ac.parameters()).device
    collector = RolloutCollector(
        _make_env(terminate_prob=0.0, max_episode_steps=10_000),
        ac,
        seed=0,
        gamma=DISTINCT_GAMMA,
        gae_lambda=DISTINCT_GAE_LAMBDA,
    )
    buffer = RolloutBuffer()
    collector.collect(buffer, N_STEPS)

    # Rebuild compute_gae's inputs from the buffer's stored fields, bootstrapping
    # from V of the post-collect cursor obs. Recompute under eval() to match the
    # collector's collect-time mode (the fused vs unfused attention kernels differ
    # at the float32 epsilon, which GAE would amplify past allclose).
    rewards = torch.tensor(buffer._rewards, dtype=torch.float32)
    dones = torch.tensor(buffer._dones, dtype=torch.float32)
    values = torch.stack(buffer._values).to(torch.float32)
    ac.eval()
    with torch.no_grad():
        last_value = ac.get_value(observation_to_batched_tensors(collector._obs, device)).squeeze(0)

    expected_adv, expected_ret = compute_gae(
        rewards, values, dones, last_value, gamma=DISTINCT_GAMMA, gae_lambda=DISTINCT_GAE_LAMBDA
    )
    default_adv, default_ret = compute_gae(rewards, values, dones, last_value)

    (batch,) = buffer.iter_minibatches(N_STEPS, shuffle=False)
    # The forwarded knobs reproduce the buffer's GAE exactly...
    assert torch.allclose(batch.advantages, expected_adv)
    assert torch.allclose(batch.returns, expected_ret)
    # ...and the ppo defaults do NOT, so the override was not ignored.
    assert not torch.allclose(batch.advantages, default_adv)
    assert not torch.allclose(batch.returns, default_ret)


def test_episode_diagnostics_present_when_episodes_complete() -> None:
    """Completed episodes populate n_episodes and the mean return/length."""
    collector = RolloutCollector(
        _make_env(terminate_prob=1.0, max_episode_steps=10_000), _make_ac(), seed=0
    )
    stats = collector.collect(RolloutBuffer(), N_STEPS)

    assert stats.n_episodes >= 1
    assert stats.mean_episode_return is not None
    # Single-step learnable episodes: each return is one step's reward in
    # {WRONG_ACTION_REWARD, CORRECT_ACTION_REWARD}, so the mean is bounded by them.
    assert WRONG_ACTION_REWARD <= stats.mean_episode_return <= CORRECT_ACTION_REWARD
    assert stats.mean_episode_length is not None
    # Every episode is a single terminal step, so the mean length is exactly 1.
    assert stats.mean_episode_length == 1.0


def test_no_completed_episode_gives_none_means() -> None:
    """Zero completed episodes -> None means (no divide-by-zero), not 0.0."""
    # No termination and a step cap far above n_steps: no episode can end.
    collector = RolloutCollector(
        _make_env(terminate_prob=0.0, max_episode_steps=10_000), _make_ac(), seed=0
    )
    stats = collector.collect(RolloutBuffer(), 5)

    assert stats.n_episodes == 0
    assert stats.mean_episode_return is None
    assert stats.mean_episode_length is None


def test_observation_to_batched_tensors_dtype_device_batch() -> None:
    """Id fields -> long, others -> float32, all on device, batch dim added.

    The id/float split is recomputed from OBS_FIELDS here, matching the
    converter's own derivation - no hardcoded field-name list.
    """
    ac = _make_ac()
    device = next(ac.parameters()).device
    obs, _ = _make_env().reset(seed=0)

    batched = observation_to_batched_tensors(obs, device)

    id_fields = {field.name for field in OBS_FIELDS if field.bounds == "id"}
    assert set(batched) == {field.name for field in OBS_FIELDS}
    for field in OBS_FIELDS:
        tensor = batched[field.name]
        assert tensor.shape == (1, *field.shape), field.name  # leading batch dim
        assert tensor.device == device, field.name
        expected = torch.long if field.name in id_fields else torch.float32
        assert tensor.dtype == expected, field.name


def test_determinism_same_seed_same_actions() -> None:
    """Same seed + same-weight network -> identical stored action stream."""

    def run() -> torch.Tensor:
        torch.manual_seed(123)
        ac = _make_ac()  # identical weights across runs (same manual_seed)
        env = _make_env(terminate_prob=0.1, max_episode_steps=8)
        collector = RolloutCollector(env, ac, seed=123)  # reseeds torch + env stream
        buffer = RolloutBuffer()
        collector.collect(buffer, N_STEPS)
        return torch.stack(buffer._actions)

    first = run()
    second = run()
    assert torch.equal(first, second)
    assert first[0] == second[0]  # identical first action specifically


def test_constructor_does_not_mutate_global_rng_state() -> None:
    """A seeded constructor must not reseed the process-global RNGs.

    torch's default generator drives act()'s dist.sample() and the buffer's
    randperm shuffle, so calling torch.manual_seed() as a construction side
    effect would let a second seeded collector silently break an earlier one's
    reproducibility - the cross-instance clobber the single-collector
    determinism test cannot reach. The env is seeded through its own
    instance-local Generator, so a seeded construction must leave both the global
    torch and the legacy global numpy generators untouched.
    """
    env = _make_env()
    actor_critic = _make_ac()  # weight init consumes global torch RNG; snapshot AFTER

    torch_before = torch.random.get_rng_state()
    numpy_before = np.random.get_state()

    RolloutCollector(env, actor_critic, seed=12345)

    assert torch.equal(
        torch.random.get_rng_state(), torch_before
    ), "constructor reseeded the global torch RNG (torch.manual_seed side effect)"
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_stored_tensors_have_no_grad_history() -> None:
    """act runs under no_grad and the buffer clones detached, so nothing tracks grad."""
    collector = RolloutCollector(_make_env(), _make_ac(), seed=0)
    buffer = RolloutBuffer()
    collector.collect(buffer, N_STEPS)

    for tensor in (buffer._values[0], buffer._log_probs[0], buffer._actions[0]):
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None
    sample_field = OBS_FIELDS[0].name
    assert buffer._obs[0][sample_field].requires_grad is False
    assert buffer._masks[0].requires_grad is False


def test_act_and_get_value_run_under_no_grad() -> None:
    """collect wraps every network call in no_grad - the context itself, not detach.

    The sibling detach test passes even without the collector's ``no_grad``
    (RolloutBuffer.add always ``.detach().clone()``s its inputs), so it cannot
    catch the context being dropped. This gates it directly: a probe subclass
    records ``torch.is_grad_enabled()`` inside each act/get_value call, and every
    recorded value must be False.
    """
    grad_flags: list[bool] = []

    class _GradProbeActorCritic(ActorCritic):
        """Records torch.is_grad_enabled() on each network call, then delegates."""

        def act(self, obs, mask, deterministic=False):
            grad_flags.append(torch.is_grad_enabled())
            return super().act(obs, mask, deterministic)

        def get_value(self, obs):
            grad_flags.append(torch.is_grad_enabled())
            return super().get_value(obs)

    collector = RolloutCollector(
        _make_env(), _GradProbeActorCritic(hidden_dim=SMALL_HIDDEN), seed=0
    )
    collector.collect(RolloutBuffer(), N_STEPS)

    assert grad_flags  # act (per step) + the tail get_value were actually invoked
    assert all(flag is False for flag in grad_flags)


def test_stream_continuity_across_two_collects() -> None:
    """The env is reset once at construction, never at the start of a collect."""
    steps = 6
    # No episode can end (no termination, cap far above the total), so the only
    # reset is the construction one.
    env = _CountingEnv(_make_env(terminate_prob=0.0, max_episode_steps=10_000))
    collector = RolloutCollector(env, _make_ac(), seed=0)
    assert env.n_resets == 1  # construction reset only

    collector.collect(RolloutBuffer(), steps)
    collector.collect(RolloutBuffer(), steps)

    assert env.n_resets == 1  # no re-reset between collects
    assert env.n_steps == 2 * steps  # the second collect resumed the same stream


def test_nonpositive_n_steps_raises() -> None:
    """n_steps <= 0 is a degenerate request and fails loudly."""
    collector = RolloutCollector(_make_env(), _make_ac(), seed=0)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            collector.collect(RolloutBuffer(), bad)


def test_training_mode_restored_after_collect() -> None:
    """collect saves and restores the network's train/eval mode."""
    collector = RolloutCollector(_make_env(), _make_ac(), seed=0)

    collector._actor_critic.train()
    collector.collect(RolloutBuffer(), N_STEPS)
    assert collector._actor_critic.training is True

    collector._actor_critic.eval()
    collector.collect(RolloutBuffer(), N_STEPS)
    assert collector._actor_critic.training is False


def test_infer_device_falls_back_to_model_parameters() -> None:
    """With device=None the collector adopts the actor-critic's parameter device."""
    ac = _make_ac()
    collector = RolloutCollector(_make_env(), ac, seed=0)
    assert collector._device == next(ac.parameters()).device


def test_stored_obs_ids_are_long_and_floats_are_float32() -> None:
    """Stored per-step obs keep the interface dtype split the encoder expects."""
    collector = RolloutCollector(_make_env(), _make_ac(), seed=0)
    buffer = RolloutBuffer()
    collector.collect(buffer, N_STEPS)

    id_fields = {field.name for field in OBS_FIELDS if field.bounds == "id"}
    stored = buffer._obs[0]
    for field in OBS_FIELDS:
        expected = torch.long if field.name in id_fields else torch.float32
        assert stored[field.name].dtype == expected, field.name
        assert stored[field.name].shape == field.shape, field.name  # no batch dim
    assert buffer._masks[0].dtype == torch.bool
    assert buffer._masks[0].shape == (ACTION_DIM,)


def test_malformed_mask_raises_interface_error() -> None:
    """A malformed env mask is an interface violation raised at the collector boundary.

    _batched_mask delegates to interface.assert_valid_mask, so a malformed mask
    raises InterfaceError rather than a bare assert that ``python -O`` strips.
    The non-bool case genuinely gates THIS boundary: the policy head coerces the
    mask with ``.bool()`` and would silently accept a non-bool mask that still
    has a legal action, so only assert_valid_mask rejects it. Injected onto the
    stream cursor so collect hits it on the first step.
    """
    collector = RolloutCollector(_make_env(), _make_ac(), seed=0)

    # Non-bool dtype but WITH a legal action: the downstream policy head accepts
    # this (via .bool()), so the collector's boundary guard is the only rejecter.
    int_mask = np.zeros(ACTION_DIM, dtype=np.int32)
    int_mask[0] = 1
    collector._mask = int_mask
    with pytest.raises(InterfaceError):
        collector.collect(RolloutBuffer(), 1)

    # Wrong shape is likewise rejected at the boundary.
    collector._mask = np.zeros(ACTION_DIM + 1, dtype=np.bool_)
    with pytest.raises(InterfaceError):
        collector.collect(RolloutBuffer(), 1)
