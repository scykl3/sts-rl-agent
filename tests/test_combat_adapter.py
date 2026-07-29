"""Tests for the combat environment adapter (StsEnv).

Requires the built engine; skips cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.adapter import StsEnv
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    INFO_KEYS_ALWAYS,
    INTERFACE_VERSION,
    SHAPING_TERMS,
    TERMINAL_LOSS_REWARD,
    TERMINAL_WIN_REWARD,
)

REGRESSION_SEED = 42
MAX_SCRIPTED_STEPS = 400
_END_TURN = ACTION_BLOCK_BY_NAME["END_TURN"].start
_PROCEED = ACTION_BLOCK_BY_NAME["PROCEED"].start  # a non-combat index, always illegal here

# Chosen-encounter fixtures: a single elite and the sampled elite pool the
# headroom training run uses. GREMLIN_NOB is the single-monster elite asserted on.
GREMLIN_NOB = sts.MonsterEncounter.GREMLIN_NOB
ELITE_ENCOUNTERS = (
    GREMLIN_NOB,
    sts.MonsterEncounter.LAGAVULIN,
    sts.MonsterEncounter.THREE_SENTRIES,
)
ELITE_ROLLOUT_STEPS = 8
# Seeds to sample the elite pool across when locking multi-encounter variety.
ELITE_SAMPLE_SEEDS = 40


def _greedy_to_terminal(env: StsEnv, obs_info):
    """Play any legal card each step, else end turn, until the episode ends."""
    _, info = obs_info
    for _ in range(MAX_SCRIPTED_STEPS):
        mask = info["action_mask"]
        legal = np.flatnonzero(mask)
        # Prefer a card play (anything that is not end turn) to make progress; fall
        # back to end turn.
        play = [i for i in legal if i != _END_TURN]
        action = int(play[0]) if play else _END_TURN
        _, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return reward, terminated, truncated, info
    raise AssertionError("combat did not end within the step cap")


def test_reset_returns_contract_obs_and_info() -> None:
    env = StsEnv()
    obs, info = env.reset(seed=REGRESSION_SEED)
    assert env.observation_space.contains(obs)
    assert env.interface_version == INTERFACE_VERSION
    assert info["action_mask"].shape == (env.action_space.n,)
    for key in INFO_KEYS_ALWAYS:
        assert key in info
    assert info["combat"].player_hp > 0
    env.close()


def test_scripted_combat_reaches_terminal_with_win_or_loss() -> None:
    env = StsEnv()
    obs_info = env.reset(seed=REGRESSION_SEED)
    reward, terminated, truncated, info = _greedy_to_terminal(env, obs_info)
    assert terminated and not truncated
    # The terminal step also carries a bounded shaping delta (|shaping| < 1), so
    # the terminal +1/-1 dominates the sign.
    assert info["won"] == (reward > 0)
    assert info["episode"]["l"] > 0
    env.close()


def test_reward_equals_terminal_plus_potential_shaping() -> None:
    # Ties the adapter's per-step reward to the reward module: on every step,
    # reward == terminal + sum(info['shaping_terms']) (the shaping is the
    # potential delta, un-annealed).
    env = StsEnv(encounters=(GREMLIN_NOB,))
    _, info = env.reset(seed=REGRESSION_SEED)
    assert set(info["shaping_terms"]) == set(SHAPING_TERMS)
    assert all(v == 0.0 for v in info["shaping_terms"].values())  # no delta at reset

    saw_shaping = False
    for _ in range(MAX_SCRIPTED_STEPS):
        legal = np.flatnonzero(info["action_mask"])
        play = [i for i in legal if i != _END_TURN]
        action = int(play[0]) if play else _END_TURN
        _, reward, terminated, truncated, info = env.step(action)
        terms = info["shaping_terms"]
        assert set(terms) == set(SHAPING_TERMS)
        terminal = 0.0
        if terminated:
            terminal = TERMINAL_WIN_REWARD if info["won"] else TERMINAL_LOSS_REWARD
        assert reward == pytest.approx(terminal + sum(terms.values()))
        saw_shaping = saw_shaping or any(v != 0.0 for v in terms.values())
        if terminated or truncated:
            break
    assert saw_shaping  # a real combat moves HP, so shaping must fire at least once
    env.close()


def test_potential_shaping_sums_to_zero_over_a_full_hp_combat() -> None:
    # Policy invariance in the live env: with gamma = 1 (the default) the
    # undiscounted sum of the per-step shaping telescopes to Phi(terminal) -
    # Phi(s_0). A combat starts at full player HP against full-HP enemies, so
    # Phi(s_0) = 0, and the terminal cashes Phi to 0 - hence the whole episode's
    # shaping nets to ~0, redistributed across steps rather than biasing the return.
    env = StsEnv(encounters=(GREMLIN_NOB,))  # gamma defaults to 1.0
    _, info = env.reset(seed=REGRESSION_SEED)

    total_shaping = 0.0
    saw_shaping = False
    terminated = truncated = False
    for _ in range(MAX_SCRIPTED_STEPS):
        action = _first_playable_action(info)
        _, _reward, terminated, truncated, info = env.step(action)
        step_shaping = sum(info["shaping_terms"].values())
        total_shaping += step_shaping
        saw_shaping = saw_shaping or step_shaping != 0.0
        if terminated or truncated:
            break

    assert terminated  # GREMLIN_NOB resolves well within the step cap
    assert saw_shaping  # intermediate steps carried nonzero shaping
    assert total_shaping == pytest.approx(0.0, abs=1e-6)
    env.close()


def _first_playable_action(info: dict) -> int:
    legal = np.flatnonzero(info["action_mask"])
    play = [i for i in legal if i != _END_TURN]
    return int(play[0]) if play else _END_TURN


def test_invalid_action_yields_zero_reward_then_normal_shaping() -> None:
    # An illegal step takes no engine action: reward is 0 and Phi is unchanged, so
    # the following legal step's shaping is still the plain potential delta.
    env = StsEnv(encounters=(GREMLIN_NOB,))
    _, info = env.reset(seed=REGRESSION_SEED)
    _, reward, _, _, info = env.step(_PROCEED)  # illegal in combat: state unchanged
    assert info["invalid_action"] is True
    assert reward == 0.0

    action = _first_playable_action(info)
    _, reward, terminated, _, info = env.step(action)
    terminal = 0.0
    if terminated:
        terminal = TERMINAL_WIN_REWARD if info["won"] else TERMINAL_LOSS_REWARD
    assert reward == pytest.approx(terminal + sum(info["shaping_terms"].values()))
    env.close()


def test_set_global_step_does_not_change_reward() -> None:
    # Potential-based shaping is un-annealed, so the diagnostic step counter has no
    # effect on reward; a step still equals terminal + sum(shaping_terms) after it.
    env = StsEnv(encounters=(GREMLIN_NOB,))
    _, info = env.reset(seed=REGRESSION_SEED)
    env.set_global_step(1000)
    action = _first_playable_action(info)
    _, reward, terminated, _, info = env.step(action)
    terminal = 0.0
    if terminated:
        terminal = TERMINAL_WIN_REWARD if info["won"] else TERMINAL_LOSS_REWARD
    assert reward == pytest.approx(terminal + sum(info["shaping_terms"].values()))
    env.close()


def test_truncation_cashes_out_potential_not_true_successor() -> None:
    # Regression: on a truncated step Phi(s') := 0, so the per-term shaping is
    # exactly -Phi(s_t) (the accumulated potential cashed out), NOT the true
    # gamma*Phi(s'_true) - Phi(s_t). Reverting the adapter's `terminated or
    # truncated` back to `terminated` would emit the latter; this test then fails.
    from sts_rl.env.reward import RewardConfig, state_potentials

    cfg = RewardConfig()
    # A small step cap so the fight truncates before it ends: playing a few cards
    # dents the enemy (nonzero enemy potential) but cannot kill it in this many steps.
    env = StsEnv(encounters=(GREMLIN_NOB,), max_episode_steps=5, reward_config=cfg)
    _, info = env.reset(seed=REGRESSION_SEED)
    prev_snapshot = info["combat"]  # the state before the eventual truncating step
    terminated = truncated = False
    for _ in range(MAX_SCRIPTED_STEPS):
        _, reward, terminated, truncated, info = env.step(_first_playable_action(info))
        if terminated or truncated:
            break
        prev_snapshot = info["combat"]

    assert truncated and not terminated
    prev_phi = state_potentials(cfg, combat=prev_snapshot)
    # Non-vacuous: the pre-truncation potential is not all-zero (the enemy took
    # damage), so -Phi(s_t) genuinely differs from gamma*Phi(s'_true) - Phi(s_t).
    assert any(v != 0.0 for v in prev_phi.values())
    for name in SHAPING_TERMS:
        assert info["shaping_terms"][name] == pytest.approx(-prev_phi[name])
    # A truncation carries no terminal component, so the whole reward is that shaping.
    assert reward == pytest.approx(sum(info["shaping_terms"].values()))
    env.close()


def test_set_global_step_rejects_negative() -> None:
    from sts_rl.interface import InterfaceError

    env = StsEnv()
    env.reset(seed=REGRESSION_SEED)
    with pytest.raises(InterfaceError):
        env.set_global_step(-1)
    env.close()


def test_illegal_action_is_flagged_and_not_executed() -> None:
    env = StsEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    turn_before = info["combat"].turn
    hp_before = info["combat"].player_hp
    obs, reward, terminated, truncated, info = env.step(_PROCEED)  # illegal in combat
    assert info["invalid_action"] is True
    assert reward == 0.0
    assert not terminated
    # Engine state is untouched by an illegal action.
    assert info["combat"].turn == turn_before
    assert info["combat"].player_hp == hp_before
    env.close()


def test_strict_mode_raises_on_illegal_action() -> None:
    from sts_rl.interface import InterfaceError

    env = StsEnv(strict=True)
    env.reset(seed=REGRESSION_SEED)
    with pytest.raises(InterfaceError):
        env.step(_PROCEED)
    env.close()


def test_same_seed_is_deterministic() -> None:
    env_a, env_b = StsEnv(), StsEnv()
    _, info_a = env_a.reset(seed=REGRESSION_SEED)
    _, info_b = env_b.reset(seed=REGRESSION_SEED)
    assert info_a["combat"] == info_b["combat"]
    assert np.array_equal(info_a["action_mask"], info_b["action_mask"])
    env_a.close()
    env_b.close()


def test_truncation_when_step_cap_hit() -> None:
    # A tiny cap and a no-op-ish policy (repeated end turn) forces truncation before
    # a win: end turn each step until the cap trips.
    env = StsEnv(max_episode_steps=2)
    env.reset(seed=REGRESSION_SEED)
    env.step(_END_TURN)
    _, reward, terminated, truncated, info = env.step(_END_TURN)
    assert truncated and not terminated
    assert "episode" in info
    env.close()


def _encounter_ids(info: dict) -> tuple[str, ...]:
    """The monster-id tuple of the current battle, for encounter-identity checks."""
    return tuple(m.monster_id for m in info["combat"].monsters)


def _assert_obs_finite(env: StsEnv, obs) -> None:
    """The observation is in-space and every channel is finite."""
    assert env.observation_space.contains(obs)
    for key, value in obs.items():
        arr = np.asarray(value, dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"non-finite values in obs[{key!r}]"


def test_chosen_encounter_builds_that_battle() -> None:
    # A single-encounter pool builds exactly that elite (no navigation): the
    # battle is one monster, the Gremlin Nob.
    env = StsEnv(encounters=[GREMLIN_NOB])
    _, info = env.reset(seed=REGRESSION_SEED)
    snap = info["combat"]
    assert snap.monster_count == 1
    assert snap.monsters[0].monster_id == GREMLIN_NOB.name  # engine ids "GREMLIN_NOB"
    env.close()


def test_sampled_encounter_is_deterministic_given_seed() -> None:
    # The pool is sampled from self.np_random, seeded by reset(seed=...), so two
    # resets on the same seed pick the same encounter.
    env = StsEnv(encounters=ELITE_ENCOUNTERS)
    _, info_a = env.reset(seed=REGRESSION_SEED)
    _, info_b = env.reset(seed=REGRESSION_SEED)
    assert _encounter_ids(info_a) == _encounter_ids(info_b)
    env.close()


def test_sampled_encounter_pool_yields_more_than_one_encounter() -> None:
    # Variety lock: across many seeds the pool must build more than one distinct
    # encounter, guarding against a regression that always returns the first pool
    # element. env._bc.encounter is the engine's built-encounter enum. These 40
    # seeds sample all three elites in practice, but we assert only ">1 distinct"
    # (not "== 3") so the test stays robust to any RNG-stream shift while still
    # failing hard on a first-element-only regression.
    env = StsEnv(encounters=ELITE_ENCOUNTERS)
    sampled = set()
    for seed in range(ELITE_SAMPLE_SEEDS):
        env.reset(seed=seed)
        sampled.add(env._bc.encounter)
    env.close()
    assert sampled <= set(ELITE_ENCOUNTERS)  # only pool members are ever built
    assert len(sampled) > 1


def test_empty_encounters_rejected() -> None:
    from sts_rl.interface import InterfaceError

    with pytest.raises(InterfaceError):
        StsEnv(encounters=[])


def test_elite_rollout_runs_with_finite_observations() -> None:
    # A short random-legal rollout on the sampled elite pool runs without error
    # and every observation stays finite (exercises encode_observation on the
    # directly-built elite battle).
    env = StsEnv(encounters=ELITE_ENCOUNTERS)
    obs, info = env.reset(seed=REGRESSION_SEED)
    _assert_obs_finite(env, obs)
    rng = np.random.default_rng(0)
    for _ in range(ELITE_ROLLOUT_STEPS):
        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
        obs, reward, terminated, truncated, info = env.step(action)
        _assert_obs_finite(env, obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()
