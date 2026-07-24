"""Tests for the vectorized environment (SubprocVecEnv).

Requires the built engine and spawns worker processes; skips cleanly otherwise.
Factories are module-level so they pickle across the spawn boundary.
"""

from __future__ import annotations

import os
import time
from functools import partial

import numpy as np
import pytest

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env.adapter import StsEnv
from sts_rl.env.reward import RewardConfig
from sts_rl.env.vec_env import SubprocVecEnv
from sts_rl.interface import ACTION_DIM, ACTION_BLOCK_BY_NAME, InterfaceError, OBS_FIELDS

REGRESSION_SEED = 42
_END_TURN = ACTION_BLOCK_BY_NAME["END_TURN"].start


def _make_env(
    index: int, *, max_episode_steps: int = 500, reward_config: RewardConfig | None = None
) -> StsEnv:
    return StsEnv(max_episode_steps=max_episode_steps, reward_config=reward_config)


def _end_turn_actions(num_envs: int) -> np.ndarray:
    return np.full(num_envs, _END_TURN, dtype=np.int64)


# -- Failure-path fixtures (module-level so they pickle across spawn) --------


def _raising_make_env(index: int) -> StsEnv:
    raise ValueError("construction boom")


class _StepRaisesEnv(StsEnv):
    """Env whose step always raises, to exercise worker error propagation."""

    def step(self, action: int):  # type: ignore[override]
        raise RuntimeError("step boom")


class _StepExitsEnv(StsEnv):
    """Env whose step hard-exits the worker, to exercise worker-death handling."""

    def step(self, action: int):  # type: ignore[override]
        os._exit(1)


def _make_step_raises(index: int) -> StsEnv:
    return _StepRaisesEnv()


def _make_step_exits(index: int) -> StsEnv:
    return _StepExitsEnv()


def test_reset_batches_obs_and_masks() -> None:
    num_envs = 3
    with SubprocVecEnv(_make_env, num_envs) as vec:
        assert vec.num_envs == num_envs
        assert vec.action_space.n == ACTION_DIM
        obs, masks = vec.reset(seeds=[REGRESSION_SEED + i for i in range(num_envs)])
        assert masks.shape == (num_envs, ACTION_DIM)
        assert masks.dtype == np.bool_
        for field in OBS_FIELDS:
            assert obs[field.name].shape == (num_envs, *field.shape)
            assert obs[field.name].dtype == field.dtype
        # Every env must have at least one legal action after reset.
        assert masks.any(axis=1).all()


def test_step_batches_results() -> None:
    num_envs = 2
    with SubprocVecEnv(_make_env, num_envs) as vec:
        vec.reset(seeds=[REGRESSION_SEED, REGRESSION_SEED + 1])
        obs, rewards, terminated, truncated, masks, infos = vec.step(_end_turn_actions(num_envs))
        assert rewards.shape == (num_envs,) and rewards.dtype == np.float32
        assert terminated.shape == (num_envs,) and terminated.dtype == np.bool_
        assert truncated.shape == (num_envs,) and truncated.dtype == np.bool_
        assert masks.shape == (num_envs, ACTION_DIM)
        assert len(infos) == num_envs
        for field in OBS_FIELDS:
            assert obs[field.name].shape == (num_envs, *field.shape)
        assert all(np.isfinite(rewards))


def test_seeded_determinism_across_two_vec_envs() -> None:
    # Observations are engine-derived; identical seeds must yield identical
    # engine-legal masks and combat readouts across independent vec envs.
    num_envs = 2
    seeds = [REGRESSION_SEED, REGRESSION_SEED + 100]
    with SubprocVecEnv(_make_env, num_envs) as a, SubprocVecEnv(_make_env, num_envs) as b:
        _, masks_a = a.reset(seeds=seeds)
        _, masks_b = b.reset(seeds=seeds)
        assert np.array_equal(masks_a, masks_b)
        _, _, _, _, step_masks_a, infos_a = a.step(_end_turn_actions(num_envs))
        _, _, _, _, step_masks_b, infos_b = b.step(_end_turn_actions(num_envs))
        assert np.array_equal(step_masks_a, step_masks_b)
        assert [i["combat"] for i in infos_a] == [i["combat"] for i in infos_b]


def test_autoreset_exposes_final_observation_and_info() -> None:
    # A tiny step cap forces truncation, which must auto-reset and stash the
    # terminal obs/info under final_observation / final_info.
    num_envs = 2
    make = partial(_make_env, max_episode_steps=2)
    with SubprocVecEnv(make, num_envs) as vec:
        vec.reset(seeds=[REGRESSION_SEED, REGRESSION_SEED + 1])
        vec.step(_end_turn_actions(num_envs))  # step 1
        obs, _, terminated, truncated, masks, infos = vec.step(
            _end_turn_actions(num_envs)
        )  # cap hit
        assert (terminated | truncated).all()  # every env ended and auto-reset
        for i, info in enumerate(infos):
            assert "final_observation" in info
            assert "final_info" in info
            # The terminal stats live in final_info; the returned top-level info
            # belongs to the fresh episode, so it must NOT carry the terminal
            # "episode" key. This distinguishes terminal info from reset info even
            # when observations are placeholders.
            assert "episode" in info["final_info"]
            assert "episode" not in info
            for field in OBS_FIELDS:
                assert obs[field.name][i].shape == field.shape
        # Post-reset envs are live again: a legal action exists.
        assert masks.any(axis=1).all()


def test_set_global_step_broadcasts_and_zeros_annealed_shaping() -> None:
    # With a short anneal horizon, pushing the shared clock far past it must make
    # every worker's per-step shaping vanish (non-terminal reward == 0).
    num_envs = 2
    cfg = RewardConfig(beta_min=0.0, t_anneal=4.0)
    make = partial(_make_env, reward_config=cfg)
    with SubprocVecEnv(make, num_envs) as vec:
        vec.reset(seeds=[REGRESSION_SEED, REGRESSION_SEED + 1])
        vec.set_global_step(10_000)  # well past t_anneal on every worker
        _, rewards, terminated, truncated, _, _ = vec.step(_end_turn_actions(num_envs))
        for r, term, trunc in zip(rewards, terminated, truncated):
            if not (term or trunc):
                assert r == pytest.approx(0.0)  # beta == 0 wipes shaping


def test_step_rejects_wrong_action_shape() -> None:
    with SubprocVecEnv(_make_env, 2) as vec:
        vec.reset()
        with pytest.raises(InterfaceError):
            vec.step(np.array([_END_TURN]))  # only 1 action for 2 envs


def test_zero_num_envs_rejected() -> None:
    with pytest.raises(InterfaceError):
        SubprocVecEnv(_make_env, 0)


def test_close_terminates_workers_and_is_idempotent() -> None:
    vec = SubprocVecEnv(_make_env, 2)
    vec.reset()
    processes = list(vec.processes)
    vec.close()
    vec.close()  # idempotent
    for process in processes:
        process.join(timeout=5)
        assert not process.is_alive()
    with pytest.raises(InterfaceError):
        vec.reset()


def test_throughput_smoke_parallel_runs_all_envs() -> None:
    # Not a strict scaling assertion (per-step IPC vs engine cost makes exact
    # linear speedup workload-dependent); this confirms K envs step in parallel
    # and sustain a positive throughput.
    num_envs = 4
    steps = 20
    with SubprocVecEnv(_make_env, num_envs) as vec:
        vec.reset(seeds=[REGRESSION_SEED + i for i in range(num_envs)])
        start = time.perf_counter()
        for _ in range(steps):
            _, rewards, _, _, _, _ = vec.step(_end_turn_actions(num_envs))
            assert rewards.shape == (num_envs,)
        elapsed = time.perf_counter() - start
        assert elapsed > 0
        env_steps_per_s = (steps * num_envs) / elapsed
        assert env_steps_per_s > 0


# -- Error propagation and worker death --------------------------------------


def test_make_env_construction_failure_surfaces_at_init() -> None:
    # A make_env that raises must surface as RuntimeError at construction (all
    # workers are drained for spaces), not silently on a later call.
    with pytest.raises(RuntimeError):
        SubprocVecEnv(_raising_make_env, 2)


def test_worker_exception_propagates_as_runtimeerror() -> None:
    with SubprocVecEnv(_make_step_raises, 2) as vec:
        vec.reset(seeds=[REGRESSION_SEED, REGRESSION_SEED + 1])
        with pytest.raises(RuntimeError, match="worker .* failed"):
            vec.step(_end_turn_actions(2))


def test_worker_death_raises_instead_of_hanging() -> None:
    # A worker that hard-exits mid-step must make the parent raise (via EOF on
    # the pipe), never block forever.
    with SubprocVecEnv(_make_step_exits, 2) as vec:
        vec.reset(seeds=[REGRESSION_SEED, REGRESSION_SEED + 1])
        with pytest.raises(RuntimeError, match="died without sending a reply"):
            vec.step(_end_turn_actions(2))
