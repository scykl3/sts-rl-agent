from __future__ import annotations

import numpy as np
import pytest

import sts_rl.interface as interface
from sts_rl.env import spaces
from sts_rl.env.stub_env import (
    CORRECT_ACTION_REWARD,
    STUB_ENGINE_COMMIT,
    StubEnv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _active_block_of(mask: np.ndarray) -> interface.ActionBlock:
    """Return the single ACTION_BLOCK whose indices are exactly the True mask."""
    legal = set(np.flatnonzero(mask).tolist())
    for block in interface.ACTION_BLOCKS:
        if set(range(block.start, block.stop)) == legal:
            return block
    raise AssertionError(f"mask does not equal any single action block: {sorted(legal)}")


# ---------------------------------------------------------------------------
# Construction and spaces
# ---------------------------------------------------------------------------


def test_spaces_match_shared_builders():
    env = StubEnv()
    obs_space, act_space = spaces.build_spaces()
    assert env.observation_space == obs_space
    assert env.action_space == act_space
    assert env.interface_version == interface.INTERFACE_VERSION


def test_invalid_reward_mode_rejected():
    with pytest.raises(interface.InterfaceError):
        StubEnv(reward_mode="nonsense")


def test_invalid_active_block_rejected():
    with pytest.raises(interface.InterfaceError):
        StubEnv(active_blocks=["NOT_A_BLOCK"])


def test_invalid_render_mode_rejected():
    with pytest.raises(interface.InterfaceError):
        StubEnv(render_mode="rgb_array")


def test_bad_max_episode_steps_rejected():
    with pytest.raises(interface.InterfaceError):
        StubEnv(max_episode_steps=0)


def test_out_of_range_terminate_prob_rejected():
    with pytest.raises(interface.InterfaceError):
        StubEnv(terminate_prob=1.5)


def test_empty_active_blocks_rejected():
    with pytest.raises(interface.InterfaceError):
        StubEnv(active_blocks=[])


def test_learnable_mode_with_only_count1_blocks_warns():
    with pytest.warns(UserWarning):
        StubEnv(active_blocks=["END_TURN"], reward_mode="learnable")


def test_seed_method_reseeds_and_reports_seed():
    # seed() then reset() with no seed must reproduce a rollout deterministically.
    def rollout():
        env = StubEnv(terminate_prob=0.0)
        returned = env.seed(2024)
        assert returned == [2024]
        obs, _ = env.reset()
        hands = [obs["hand_ids"].copy()]
        for _ in range(10):
            obs, _, _, _, _ = env.step(env.optimal_action())
            hands.append(obs["hand_ids"].copy())
        return hands

    for ha, hb in zip(rollout(), rollout()):
        np.testing.assert_array_equal(ha, hb)


# ---------------------------------------------------------------------------
# reset / step contract shapes
# ---------------------------------------------------------------------------


def test_reset_returns_obs_in_space_and_full_info():
    env = StubEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for key in interface.INFO_KEYS_ALWAYS:
        assert key in info
    # Terminal-only keys must not appear on the reset step.
    for key in interface.INFO_KEYS_TERMINAL:
        assert key not in info
    assert info["interface_version"] == interface.INTERFACE_VERSION
    assert info["engine_commit"] == STUB_ENGINE_COMMIT
    assert info["seed"] == 0


def test_step_returns_five_tuple_with_correct_types():
    env = StubEnv()
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(env.optimal_action())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info["action_mask"], np.ndarray)
    np.testing.assert_array_equal(info["action_mask"], env.legal_actions())


def test_step_info_carries_all_always_keys():
    env = StubEnv()
    env.reset(seed=1)
    _, _, _, _, info = env.step(env.optimal_action())
    for key in interface.INFO_KEYS_ALWAYS:
        assert key in info
    assert info["interface_version"] == interface.INTERFACE_VERSION
    assert info["rng_state"] is not None


def test_shaping_terms_keys_match_contract():
    env = StubEnv()
    _, info = env.reset(seed=1)
    assert set(info["shaping_terms"]) == set(interface.SHAPING_TERMS)


def test_id_fields_stay_within_bounds_over_many_steps():
    # Embedding-overflow guard: every id field must stay in [0, id_high] on every
    # step, or agent-side embedding lookups would go out of range. Assert against
    # the production id_high, not literals.
    env = StubEnv(terminate_prob=0.0)
    obs, _ = env.reset(seed=99)
    id_fields = [f for f in interface.OBS_FIELDS if f.bounds == "id"]
    for _ in range(300):
        for field in id_fields:
            values = obs[field.name]
            assert values.dtype == np.int32
            assert values.min() >= 0
            assert values.max() <= field.id_high
        obs, _, _, truncated, _ = env.step(env.optimal_action())
        if truncated:
            obs, _ = env.reset()


def test_terminal_obs_stays_in_space():
    # Force an immediate terminated step and a pure-truncation env; both terminal
    # observations must still be valid members of the observation space.
    term_env = StubEnv(max_episode_steps=1000, terminate_prob=1.0)
    term_env.reset(seed=1)
    obs, _, terminated, _, _ = term_env.step(term_env.optimal_action())
    assert terminated
    assert term_env.observation_space.contains(obs)

    trunc_env = StubEnv(max_episode_steps=1, terminate_prob=0.0)
    trunc_env.reset(seed=1)
    obs, _, _, truncated, _ = trunc_env.step(trunc_env.optimal_action())
    assert truncated
    assert trunc_env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# Mask validity and semantics
# ---------------------------------------------------------------------------


def test_legal_actions_is_valid_and_single_block():
    env = StubEnv()
    env.reset(seed=2)
    for _ in range(200):
        mask = env.legal_actions()
        interface.assert_valid_mask(mask)  # shape, dtype, at least one legal
        # Exactly one action block is unmasked (the contract's screen rule).
        _active_block_of(mask)
        _, _, terminated, truncated, _ = env.step(env.optimal_action())
        if terminated or truncated:
            env.reset()


def test_action_masks_alias_matches_legal_actions():
    env = StubEnv()
    env.reset(seed=3)
    np.testing.assert_array_equal(env.action_masks(), env.legal_actions())


def test_legal_actions_returns_a_copy():
    env = StubEnv()
    env.reset(seed=4)
    mask = env.legal_actions()
    mask[:] = True
    # Mutating the returned array must not corrupt internal state.
    interface.assert_valid_mask(env.legal_actions())
    assert not env.legal_actions().all()


# ---------------------------------------------------------------------------
# Invalid-action detection
# ---------------------------------------------------------------------------


def test_illegal_action_sets_invalid_flag_but_does_not_raise():
    env = StubEnv()
    env.reset(seed=5)
    mask = env.legal_actions()
    illegal = int(np.flatnonzero(~mask)[0])
    _, _, _, _, info = env.step(illegal)
    assert info["invalid_action"] is True


def test_legal_action_clears_invalid_flag():
    env = StubEnv()
    env.reset(seed=6)
    _, _, _, _, info = env.step(env.optimal_action())
    assert info["invalid_action"] is False


def test_strict_mode_raises_on_illegal_action():
    env = StubEnv(strict=True)
    env.reset(seed=7)
    mask = env.legal_actions()
    illegal = int(np.flatnonzero(~mask)[0])
    with pytest.raises(interface.InterfaceError):
        env.step(illegal)


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


def test_episode_truncates_at_step_cap_when_termination_disabled():
    cap = 8
    env = StubEnv(max_episode_steps=cap, terminate_prob=0.0)
    env.reset(seed=8)
    for step in range(1, cap + 1):
        _, _, terminated, truncated, info = env.step(env.optimal_action())
        if step < cap:
            assert not terminated and not truncated
        else:
            assert truncated and not terminated
    assert info["won"] is False
    assert info["episode"]["l"] == cap


def test_terminated_episode_reports_won_and_episode():
    # terminate_prob=1.0 forces a terminated step immediately.
    env = StubEnv(max_episode_steps=1000, terminate_prob=1.0)
    env.reset(seed=9)
    _, _, terminated, truncated, info = env.step(env.optimal_action())
    assert terminated and not truncated
    assert isinstance(info["won"], bool)
    assert info["episode"]["l"] == 1


def test_episode_return_accumulates_reward():
    cap = 10
    env = StubEnv(max_episode_steps=cap, terminate_prob=0.0)
    env.reset(seed=10)
    total = 0.0
    info = {}
    for _ in range(cap):
        _, reward, _, truncated, info = env.step(env.optimal_action())
        total += reward
        if truncated:
            break
    assert info["episode"]["r"] == pytest.approx(total)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_observations_and_masks():
    def rollout():
        env = StubEnv(terminate_prob=0.0)
        obs, _ = env.reset(seed=123)
        masks = [env.legal_actions()]
        obs_hands = [obs["hand_ids"].copy()]
        for _ in range(15):
            obs, _, _, _, _ = env.step(env.optimal_action())
            masks.append(env.legal_actions())
            obs_hands.append(obs["hand_ids"].copy())
        return masks, obs_hands

    masks_a, hands_a = rollout()
    masks_b, hands_b = rollout()
    for ma, mb in zip(masks_a, masks_b):
        np.testing.assert_array_equal(ma, mb)
    for ha, hb in zip(hands_a, hands_b):
        np.testing.assert_array_equal(ha, hb)


# ---------------------------------------------------------------------------
# Learnable signal
# ---------------------------------------------------------------------------


def test_optimal_action_is_always_legal():
    env = StubEnv(terminate_prob=0.0)
    env.reset(seed=11)
    for _ in range(200):
        a = env.optimal_action()
        assert env.legal_actions()[a]
        env.step(a)


def test_optimal_policy_earns_max_toy_return():
    cap = 50
    env = StubEnv(max_episode_steps=cap, terminate_prob=0.0, reward_mode="learnable")
    env.reset(seed=12)
    total = 0.0
    for _ in range(cap):
        _, reward, _, truncated, _ = env.step(env.optimal_action())
        total += reward
        if truncated:
            break
    # Every step's action is correct, so the toy return is maximal.
    assert total == pytest.approx(cap * CORRECT_ACTION_REWARD)


def test_random_policy_underperforms_optimal():
    # The learnable signal must be separable: a random legal policy scores
    # strictly below the optimal policy, otherwise there is nothing to learn.
    cap = 200
    env = StubEnv(max_episode_steps=cap, terminate_prob=0.0, reward_mode="learnable")
    env.reset(seed=13)
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(cap):
        legal = np.flatnonzero(env.legal_actions())
        action = int(rng.choice(legal))
        _, reward, _, truncated, _ = env.step(action)
        total += reward
        if truncated:
            break
    assert total < cap * CORRECT_ACTION_REWARD


def test_random_mode_reward_is_zero_until_terminal():
    env = StubEnv(max_episode_steps=1000, terminate_prob=0.0, reward_mode="random")
    env.reset(seed=14)
    for _ in range(50):
        _, reward, _, _, _ = env.step(env.optimal_action())
        assert reward == 0.0


def test_random_mode_terminal_reward_is_plus_or_minus_one():
    env = StubEnv(max_episode_steps=1000, terminate_prob=1.0, reward_mode="random")
    env.reset(seed=15)
    _, reward, terminated, _, info = env.step(env.optimal_action())
    assert terminated
    expected = interface.TERMINAL_WIN_REWARD if info["won"] else interface.TERMINAL_LOSS_REWARD
    assert reward == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Restricting active blocks
# ---------------------------------------------------------------------------


def test_active_blocks_restriction_limits_legal_block():
    env = StubEnv(active_blocks=["REWARD_SELECT"], terminate_prob=0.0)
    env.reset(seed=16)
    expected = interface.ACTION_BLOCK_BY_NAME["REWARD_SELECT"]
    for _ in range(50):
        assert _active_block_of(env.legal_actions()) is expected
        env.step(env.optimal_action())


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_optimal_action_is_legal_in_random_mode():
    env = StubEnv(reward_mode="random", terminate_prob=0.0)
    env.reset(seed=19)
    for _ in range(50):
        a = env.optimal_action()
        assert env.legal_actions()[a]
        env.step(a)


def test_render_ansi_returns_string_and_none_mode_returns_none():
    env = StubEnv(render_mode="ansi")
    env.reset(seed=17)
    assert isinstance(env.render(), str)

    env_none = StubEnv(render_mode=None)
    env_none.reset(seed=18)
    assert env_none.render() is None


def test_render_human_prints_and_returns_none(capsys):
    env = StubEnv(render_mode="human")
    env.reset(seed=20)
    assert env.render() is None
    assert capsys.readouterr().out.strip() != ""
