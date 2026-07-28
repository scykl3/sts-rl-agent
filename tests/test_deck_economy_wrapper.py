"""Tests for the potential-based deck/economy shaping wrapper.

Fully engine-free: the wrapper imports only :mod:`sts_rl.interface`, so these drive
``_potential`` off ``build_observation_space().sample()`` (with the three economy fields
overwritten to known values) and isolate ``reset``/``step`` behind a scripted ``gym.Env``
stub. No native engine build is required.

The PBRS contract under test (Ng, Harada and Russell 1999):
``F(s, s') = gamma * Phi(s') - Phi(s)`` with ``Phi(terminal) := 0`` applied on BOTH
``terminated`` and ``truncated``, a pure state potential, single-sourced ``gamma``, an
additive (non-double-counting) term, and a constant (un-annealed) ``scale``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from sts_rl.agent.deck_economy_wrapper import (
    C_GOLD,
    C_POTION,
    C_RELIC,
    GOLD_NORM,
    GOLD_SCALAR_INDEX,
    DeckEconomyShapingWrapper,
)
from sts_rl.env.spaces import build_action_space, build_observation_space
from sts_rl.interface import N_RELIC_IDS, PAD_ID, POTION_SLOTS

# Any non-PAD potion id counts as one real potion (PAD_ID 0 == INVALID == empty slot).
_REAL_POTION_ID = 1
# A distinguishable sentinel mask the forwarder test checks identity against.
_STUB_MASK = np.array([True, False, True], dtype=bool)


def _economy_obs(*, num_relics: int, gold: float, num_potions: int) -> dict[str, np.ndarray]:
    """Full-shaped observation with the three economy fields set to known values.

    Samples the interface observation space for correct shapes/dtypes, then overwrites
    exactly the fields Phi reads: ``relics_multihot`` (``num_relics`` owned bits),
    the ``player_scalars`` gold slot, and ``potion_ids`` (``num_potions`` real ids, rest
    PAD). Other fields keep their sampled values and are ignored by Phi.
    """
    space = build_observation_space()
    space.seed(0)
    obs = space.sample()
    relics = np.zeros(N_RELIC_IDS, dtype=np.float32)
    relics[:num_relics] = 1.0
    obs["relics_multihot"] = relics
    obs["player_scalars"][GOLD_SCALAR_INDEX] = np.float32(gold)
    potions = np.full(POTION_SLOTS, PAD_ID, dtype=np.int32)
    potions[:num_potions] = _REAL_POTION_ID
    obs["potion_ids"] = potions
    return obs


class _ScriptedEnv(gym.Env):
    """Engine-free ``gym.Env`` stub with scripted obs/reward/flags for isolating the wrapper.

    ``reset`` returns ``reset_obs``; ``step`` returns ``next_obs``, ``base_reward``, the
    scripted ``terminated``/``truncated``, and a fresh copy of ``step_info``. It also exposes
    ``StsRunEnv``'s masking / anneal-clock API so the wrapper's explicit forwarders can be
    exercised without the engine.
    """

    def __init__(self) -> None:
        self.observation_space = build_observation_space()
        self.action_space = build_action_space()
        self.reset_obs = _economy_obs(num_relics=0, gold=0.0, num_potions=0)
        self.next_obs = _economy_obs(num_relics=0, gold=0.0, num_potions=0)
        self.base_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.step_info: dict[str, object] = {}
        self.mask = _STUB_MASK
        self.global_step = -1

    def reset(self, *, seed=None, options=None):
        return self.reset_obs, {"reset": True}

    def step(self, action):
        return (
            self.next_obs,
            self.base_reward,
            self.terminated,
            self.truncated,
            dict(self.step_info),
        )

    def legal_actions(self) -> np.ndarray:
        return self.mask

    def action_masks(self) -> np.ndarray:
        return self.legal_actions()

    def set_global_step(self, t: int) -> None:
        self.global_step = t


# --- Phi: pure, closed-form, monotone --------------------------------------


def test_potential_matches_closed_form_weights() -> None:
    w = DeckEconomyShapingWrapper(_ScriptedEnv(), gamma=0.99, scale=1.0)
    obs = _economy_obs(num_relics=3, gold=250.0, num_potions=2)
    expected = C_RELIC * 3 + C_GOLD * (250.0 / GOLD_NORM) + C_POTION * 2
    assert w._potential(obs) == pytest.approx(expected)


def test_potential_is_pure_function_of_obs() -> None:
    w = DeckEconomyShapingWrapper(_ScriptedEnv(), gamma=0.99)
    obs = _economy_obs(num_relics=3, gold=120.0, num_potions=2)
    other = _economy_obs(num_relics=1, gold=10.0, num_potions=0)
    first = w._potential(obs)
    # Same obs -> same Phi, and no path dependence (an intervening call cannot shift it).
    assert w._potential(obs) == first
    _ = w._potential(other)
    assert w._potential(obs) == first


def test_potential_monotone_in_each_economy_dimension() -> None:
    w = DeckEconomyShapingWrapper(_ScriptedEnv(), gamma=0.99)
    base = w._potential(_economy_obs(num_relics=2, gold=100.0, num_potions=1))
    assert w._potential(_economy_obs(num_relics=3, gold=100.0, num_potions=1)) > base
    assert w._potential(_economy_obs(num_relics=2, gold=150.0, num_potions=1)) > base
    assert w._potential(_economy_obs(num_relics=2, gold=100.0, num_potions=2)) > base


def test_deck_ids_excluded_from_potential() -> None:
    # deck_ids is overworld-only (PAD in combat); scoring it would make Phi jump at the
    # combat/overworld boundary, so it must not enter Phi.
    w = DeckEconomyShapingWrapper(_ScriptedEnv(), gamma=0.99)
    obs = _economy_obs(num_relics=2, gold=100.0, num_potions=1)
    phi_before = w._potential(obs)
    obs["deck_ids"] = np.arange(1, obs["deck_ids"].shape[0] + 1, dtype=obs["deck_ids"].dtype)
    assert w._potential(obs) == phi_before


def test_scale_multiplies_potential() -> None:
    env = _ScriptedEnv()
    obs = _economy_obs(num_relics=2, gold=100.0, num_potions=1)
    w1 = DeckEconomyShapingWrapper(env, gamma=0.99, scale=1.0)
    w2 = DeckEconomyShapingWrapper(env, gamma=0.99, scale=2.0)
    assert w2._potential(obs) == pytest.approx(2.0 * w1._potential(obs))


def test_zero_economy_potential_is_finite_zero() -> None:
    env = _ScriptedEnv()
    zero = _economy_obs(num_relics=0, gold=0.0, num_potions=0)
    env.reset_obs = zero
    env.next_obs = zero
    w = DeckEconomyShapingWrapper(env, gamma=0.99, scale=1.0)
    assert w._potential(zero) == 0.0
    w.reset()
    assert w._prev_phi == 0.0
    _obs, reward, _term, _trunc, info = w.step(0)
    assert reward == pytest.approx(0.0)  # gamma*0 - 0
    assert np.isfinite(reward)
    assert np.isfinite(info["deck_economy_shaping"])


# --- reset / step shaping formula ------------------------------------------


def test_reset_initializes_prev_phi_from_first_obs() -> None:
    env = _ScriptedEnv()
    env.reset_obs = _economy_obs(num_relics=5, gold=250.0, num_potions=2)
    w = DeckEconomyShapingWrapper(env, gamma=0.99, scale=1.0)
    obs, _info = w.reset()
    assert w._prev_phi == pytest.approx(w._potential(env.reset_obs))
    assert obs is env.reset_obs  # obs passes through unchanged


def test_shaping_formula_matches_gamma_phi_prime_minus_phi() -> None:
    env = _ScriptedEnv()
    s0 = _economy_obs(num_relics=1, gold=50.0, num_potions=0)
    s1 = _economy_obs(num_relics=2, gold=75.0, num_potions=1)
    env.reset_obs = s0
    w = DeckEconomyShapingWrapper(env, gamma=0.99, scale=1.0)
    w.reset()
    env.next_obs = s1
    env.base_reward = 0.3
    _obs, reward, term, trunc, info = w.step(0)
    phi0, phi1 = w._potential(s0), w._potential(s1)
    assert not term and not trunc
    assert reward == pytest.approx(0.3 + 0.99 * phi1 - phi0)
    assert info["deck_economy_shaping"] == pytest.approx(0.99 * phi1 - phi0)


def test_prev_phi_advances_and_has_no_reward_memory() -> None:
    env = _ScriptedEnv()
    s0 = _economy_obs(num_relics=0, gold=0.0, num_potions=0)
    s1 = _economy_obs(num_relics=1, gold=0.0, num_potions=0)
    s2 = _economy_obs(num_relics=1, gold=200.0, num_potions=0)
    env.reset_obs = s0
    w = DeckEconomyShapingWrapper(env, gamma=0.9, scale=1.0)
    w.reset()
    env.base_reward = 1.0
    env.next_obs = s1
    _o, r1, *_ = w.step(0)
    env.next_obs = s2
    _o, r2, *_ = w.step(0)
    phi0, phi1, phi2 = w._potential(s0), w._potential(s1), w._potential(s2)
    assert r1 == pytest.approx(1.0 + 0.9 * phi1 - phi0)
    # Second step's baseline is Phi(s1) (a pure state value), not any accumulation of r1.
    assert r2 == pytest.approx(1.0 + 0.9 * phi2 - phi1)


# --- terminated vs truncated (the load-bearing PBRS contract) --------------


def test_terminated_zeroes_successor_potential() -> None:
    env = _ScriptedEnv()
    env.reset_obs = _economy_obs(num_relics=3, gold=100.0, num_potions=1)
    w = DeckEconomyShapingWrapper(env, gamma=0.99, scale=1.0)
    w.reset()
    prev_phi = w._prev_phi
    # A large-potential successor must be treated as Phi(s') = 0 on a true terminal.
    env.next_obs = _economy_obs(num_relics=9, gold=900.0, num_potions=5)
    env.base_reward = 1.0
    env.terminated = True
    _obs, reward, term, _trunc, info = w.step(0)
    assert term
    assert info["deck_economy_shaping"] == pytest.approx(-prev_phi)
    assert reward == pytest.approx(1.0 - prev_phi)


def test_truncated_zeroes_potential_like_terminated() -> None:
    """On truncation, Phi(s') is zeroed exactly like a true terminal: F == -Phi(s).

    This repo's single-env collector bootstraps ``V`` of the post-reset observation on a
    truncation (not the truncated state's true successor) and does not sever GAE there, so
    emitting the real ``gamma * Phi(s')`` would leave an economy-correlated residual that does
    not telescope; zeroing yields the constant, policy-invariant ``-gamma * Phi(reset)``.
    This test FAILS if the step guard drops ``truncated`` (reverts to ``terminated`` only).
    """
    env = _ScriptedEnv()
    env.reset_obs = _economy_obs(num_relics=3, gold=100.0, num_potions=1)
    gamma = 0.99
    w = DeckEconomyShapingWrapper(env, gamma=gamma, scale=1.0)
    w.reset()
    prev_phi = w._prev_phi
    # A large-potential successor must be treated as Phi(s') = 0 on truncation, as on terminal.
    s_next = _economy_obs(num_relics=6, gold=400.0, num_potions=3)
    env.next_obs = s_next
    env.base_reward = 0.0
    env.truncated = True  # time-limit cutoff, treated like a terminal for the potential
    _obs, reward, term, trunc, info = w.step(0)
    phi_next = w._potential(s_next)
    assert trunc and not term
    assert info["deck_economy_shaping"] == pytest.approx(-prev_phi)
    assert reward == pytest.approx(-prev_phi)
    # A `terminated`-only guard would emit the real gamma*Phi(s') - Phi(s); phi_next > 0, so
    # the zeroed value is strictly different from that un-zeroed one.
    assert phi_next > 0
    assert info["deck_economy_shaping"] != pytest.approx(gamma * phi_next - prev_phi)


# --- gamma single-source, additivity, forwarders, config validation --------


def test_gamma_single_source_uses_constructor_value() -> None:
    env = _ScriptedEnv()
    env.reset_obs = _economy_obs(num_relics=0, gold=0.0, num_potions=0)
    w = DeckEconomyShapingWrapper(env, gamma=0.5, scale=1.0)
    assert w.gamma == 0.5
    w.reset()
    prev_phi = w._prev_phi  # 0 (empty economy at reset)
    s_next = _economy_obs(num_relics=4, gold=0.0, num_potions=0)
    env.next_obs = s_next
    _obs, reward, _term, _trunc, _info = w.step(0)
    phi_next = w._potential(s_next)
    assert reward == pytest.approx(0.5 * phi_next - prev_phi)
    # A different gamma (e.g. training's 0.99) would produce a different shaped value.
    assert reward != pytest.approx(0.99 * phi_next - prev_phi)


def test_additive_preserves_base_reward_and_info() -> None:
    env = _ScriptedEnv()
    env.reset_obs = _economy_obs(num_relics=1, gold=20.0, num_potions=0)
    w = DeckEconomyShapingWrapper(env, gamma=0.99, scale=1.0)
    w.reset()
    prev_phi = w._prev_phi
    s_next = _economy_obs(num_relics=2, gold=45.0, num_potions=1)
    env.next_obs = s_next
    env.base_reward = 0.37
    env.step_info = {"action_mask": np.ones(3, dtype=bool), "custom": 123}
    _obs, reward, _term, _trunc, info = w.step(0)
    shaped = 0.99 * w._potential(s_next) - prev_phi
    # The wrapper only ADDS the shaping term; the base reward is recoverable exactly.
    assert reward == pytest.approx(0.37 + shaped)
    assert reward - shaped == pytest.approx(0.37)
    # Existing info keys preserved; the diagnostic key is added, not clobbered.
    assert info["action_mask"] is env.step_info["action_mask"]
    assert info["custom"] == 123
    assert info["deck_economy_shaping"] == pytest.approx(shaped)


def test_forwards_env_masking_and_anneal_clock() -> None:
    # Gymnasium 1.x Wrapper does not auto-forward non-gym-API methods, so the wrapper
    # re-exposes them explicitly; confirm they reach the wrapped env.
    env = _ScriptedEnv()
    w = DeckEconomyShapingWrapper(env, gamma=0.99)
    assert np.array_equal(w.legal_actions(), _STUB_MASK)
    assert np.array_equal(w.action_masks(), _STUB_MASK)
    w.set_global_step(123)
    assert env.global_step == 123


@pytest.mark.parametrize("bad_gamma", [-0.1, 1.1, 2.0])
def test_rejects_out_of_range_gamma(bad_gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        DeckEconomyShapingWrapper(_ScriptedEnv(), gamma=bad_gamma)


def test_rejects_negative_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        DeckEconomyShapingWrapper(_ScriptedEnv(), gamma=0.99, scale=-0.5)
