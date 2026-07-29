"""Gymnasium adapter over one engine instance, combat mode.

``StsEnv`` starts a seeded Ironclad combat, applies decoded actions, and reports
the Gymnasium 5-tuple. It is the combat backbone the rest of the environment
builds on.

Observations are encoded from the live engine state by
:func:`sts_rl.env.observation.encode_observation`; the raw readout also stays in
``info['combat']`` for debugging. Reward is the terminal win/loss signal plus
potential-based per-step shaping (see :mod:`sts_rl.env.reward`):
``reward = terminal + (gamma * Phi(s') - Phi(s))``, where ``Phi`` is the shaping
potential and ``gamma`` is the trainer's discount. In this single-combat env the
potential is the enemy-HP and player-HP terms; floor / act terms are ``0``.

Action legality is enforced through :func:`sts_rl.env.actions.build_mask`: an
action is decoded and executed only after it is confirmed legal, so an invalid
move is never handed to the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.actions import auto_resolve, build_mask, decode_action
from sts_rl.env.engine import CombatSnapshot, engine_commit, read_combat, start_combat
from sts_rl.env.observation import encode_observation
from sts_rl.env.reward import (
    DEFAULT_SHAPING_GAMMA,
    RewardConfig,
    shaping_delta,
    state_potentials,
    zero_shaping_terms,
)
from sts_rl.env.spaces import build_spaces
from sts_rl.interface import (
    ACTION_DIM,
    INTERFACE_VERSION,
    TERMINAL_LOSS_REWARD,
    TERMINAL_WIN_REWARD,
    InterfaceError,
    Info,
    Obs,
    assert_valid_mask,
)

DEFAULT_MAX_EPISODE_STEPS = 500
_MAX_SEED = 2**31 - 1


class StsEnv(gym.Env):
    """Gymnasium environment over one engine instance, one combat per episode.

    Args:
        ascension: ascension level for the run the combat is drawn from.
        max_episode_steps: step cap; hitting it ends the episode with
            ``truncated=True``.
        strict: if ``True``, an illegal action passed to :meth:`step` raises
            :class:`~sts_rl.interface.InterfaceError` instead of only flagging
            ``invalid_action`` in ``info`` and leaving the state unchanged.
        reward_config: potential weights for the shaping terms; defaults to
            :class:`~sts_rl.env.reward.RewardConfig`.
        gamma: discount for the potential-based shaping term
            ``gamma * Phi(s') - Phi(s)``. Pass the trainer's GAE discount so the
            shaping telescopes against the same return; defaults to
            :data:`~sts_rl.env.reward.DEFAULT_SHAPING_GAMMA`.
        render_mode: one of ``None``, ``"ansi"``, ``"human"``.
        encounters: optional non-empty pool of engine ``MonsterEncounter`` values.
            When given, each :meth:`reset` samples one (deterministically from the
            reset seed) and builds that combat directly instead of navigating to
            the run's first monster room; ``None`` keeps the navigated behavior.
            The chosen battle is built on a fresh run state (floor 0): starting
            deck, full HP, no act-earned relics, i.e. a single-combat training
            setup, not a mid-act state.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        *,
        ascension: int = 0,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        strict: bool = False,
        reward_config: RewardConfig | None = None,
        gamma: float = DEFAULT_SHAPING_GAMMA,
        render_mode: str | None = None,
        encounters: Sequence[Any] | None = None,  # sts.MonsterEncounter values | None
    ) -> None:
        if max_episode_steps < 1:
            raise InterfaceError(f"max_episode_steps must be >= 1, got {max_episode_steps}")
        valid_render_modes = (None, *self.metadata["render_modes"])
        if render_mode not in valid_render_modes:
            raise InterfaceError(
                f"render_mode {render_mode!r} invalid; expected one of {valid_render_modes}"
            )
        if encounters is not None and len(encounters) == 0:
            raise InterfaceError("encounters, when provided, must be non-empty")

        self.observation_space, self.action_space = build_spaces()
        self.interface_version = INTERFACE_VERSION
        self.render_mode = render_mode

        self._ascension = ascension
        self._max_episode_steps = max_episode_steps
        self._strict = strict
        self._reward_config = reward_config if reward_config is not None else RewardConfig()
        self._gamma = float(gamma)
        # Fixed pool of chosen encounters sampled once per reset; None keeps the
        # default navigate-to-first-combat behavior.
        self._encounters = tuple(encounters) if encounters is not None else None
        # Cached once; the pinned commit does not change during a run.
        self._engine_commit = engine_commit()

        # Cumulative env-step counter, exposed via set_global_step for diagnostics
        # and back-compat (the vectorized runner and shaping wrappers broadcast it).
        # Potential-based shaping is un-annealed, so this no longer affects reward.
        self._global_step = 0

        # Per-episode state, populated by reset() before use.
        self._gc: Any = None
        self._bc: Any = None
        self._steps = 0
        self._ep_return = 0.0
        self._episode_seed: int | None = None
        self._mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        # Shaping potential Phi(s) of the state before the current step, per term,
        # for the potential-based delta gamma * Phi(s') - Phi(s).
        self._prev_potentials: dict[str, float] = zero_shaping_terms()

    # -- Gymnasium API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Obs, Info]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, _MAX_SEED))
        self._episode_seed = seed
        if self._encounters is not None:
            # Sample from self.np_random (seeded by super().reset above) so the
            # chosen encounter is deterministic given the reset seed.
            chosen = self._encounters[int(self.np_random.integers(len(self._encounters)))]
            self._gc, self._bc = start_combat(seed, ascension=self._ascension, encounter=chosen)
        else:
            self._gc, self._bc = start_combat(seed, ascension=self._ascension)
        self._steps = 0
        self._ep_return = 0.0
        self._refresh_mask()
        snapshot = read_combat(self._bc)
        self._prev_potentials = state_potentials(self._reward_config, combat=snapshot)
        return self._observation(), self._build_info(
            invalid_action=False, shaping_terms=zero_shaping_terms(), snapshot=snapshot
        )

    def step(self, action: int) -> tuple[Obs, float, bool, bool, Info]:
        action = int(action)
        legal = 0 <= action < ACTION_DIM and bool(self._mask[action])
        engine_action = decode_action(action, self._bc) if legal else None
        # decode_action returns None for indices this adapter does not map; combined
        # with the mask check this can only happen for an out-of-space or unmapped
        # index, which is treated as illegal.
        if engine_action is None or not engine_action.is_valid_action(self._bc):
            legal = False

        if not legal:
            if self._strict:
                raise InterfaceError(f"illegal action {action} for the current mask")
            # Do not touch the engine: executing an invalid action is unsafe. The
            # state is unchanged, so Phi is unchanged and no shaping or reward
            # accrues (the previous potential is left as the baseline); the step
            # still counts as one env interaction.
            self._steps += 1
            self._global_step += 1
            truncated = self._steps >= self._max_episode_steps
            info = self._build_info(
                invalid_action=True,
                shaping_terms=zero_shaping_terms(),
                snapshot=read_combat(self._bc),
            )
            if truncated:
                self._finalize_terminal_info(info, won=False)
            return self._observation(), 0.0, False, truncated, info

        assert engine_action is not None  # guaranteed: legal implies a mapped, valid action
        engine_action.execute(self._bc)
        self._steps += 1

        # Advance past any engine-driven states the agent cannot represent (e.g. a
        # multi-select confirm the played card triggered), then read the outcome.
        self._refresh_mask()
        snapshot = read_combat(self._bc)

        terminated = self._bc.outcome != sts.BattleOutcome.UNDECIDED
        won = self._bc.outcome == sts.BattleOutcome.PLAYER_VICTORY
        terminal = 0.0
        if terminated:
            terminal = TERMINAL_WIN_REWARD if won else TERMINAL_LOSS_REWARD
        truncated = (not terminated) and self._steps >= self._max_episode_steps

        # Potential-based shaping F = gamma * Phi(s') - Phi(s). Phi(s') := 0 on BOTH
        # a terminated and a truncated step, so every episode's shaping telescopes to
        # -Phi(s_0) regardless of where it ends - policy-invariant. Zeroing on
        # truncation (rather than emitting the real gamma * Phi(s')) is deliberate: on
        # truncation the collector bootstraps V of the post-reset observation, not the
        # truncated state's true successor (see rollout_collector), and this matches
        # the DeckEconomyShapingWrapper convention. Un-annealed.
        curr_potentials = (
            zero_shaping_terms()
            if terminated or truncated
            else state_potentials(self._reward_config, combat=snapshot)
        )
        shaping_terms = shaping_delta(self._prev_potentials, curr_potentials, self._gamma)
        reward = terminal + sum(shaping_terms.values())
        self._prev_potentials = curr_potentials
        self._global_step += 1

        self._ep_return += reward
        info = self._build_info(
            invalid_action=False, shaping_terms=shaping_terms, snapshot=snapshot
        )
        if terminated or truncated:
            self._finalize_terminal_info(info, won=won and terminated)
        return self._observation(), reward, terminated, truncated, info

    def legal_actions(self) -> np.ndarray:
        """Return the current ``(ACTION_DIM,)`` bool mask (a defensive copy)."""
        return self._mask.copy()

    def action_masks(self) -> np.ndarray:
        """Alias of :meth:`legal_actions` for MaskablePPO-style callers."""
        return self.legal_actions()

    def seed(self, seed: int | None = None) -> list[int]:
        """Reseed the env RNG; ``reset(seed=...)`` is the canonical seeding path."""
        self.np_random, actual_seed = seeding.np_random(seed)
        return [actual_seed]

    def set_global_step(self, t: int) -> None:
        """Set the cumulative env-step counter to the shared global step.

        Potential-based shaping is un-annealed, so this no longer affects reward.
        The counter is retained for diagnostics and back-compat: the vectorized
        runner and shaping wrappers broadcast a shared step count through this
        method, so it stays a supported no-op-on-reward setter. Subsequent steps
        advance from the value set here; negative values are rejected.
        """
        if t < 0:
            raise InterfaceError(f"global_step must be >= 0, got {t}")
        self._global_step = int(t)

    def render(self) -> str | None:
        if self.render_mode is None or self._bc is None:
            return None
        text = str(self._bc)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        self._gc = None
        self._bc = None

    # -- Internals ----------------------------------------------------------

    def _refresh_mask(self) -> None:
        """Advance past unrepresentable states and store the current legal mask.

        Guarantees the stored mask is non-empty on any non-terminal state, so the
        agent always has at least one legal action; a terminal battle gets an
        all-False mask (unused, the episode is over).
        """
        auto_resolve(self._bc)
        if self._bc.outcome != sts.BattleOutcome.UNDECIDED:
            self._mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        else:
            self._mask = build_mask(self._bc)
            assert_valid_mask(self._mask)

    def _observation(self) -> Obs:
        """Encode the live engine state into the interface observation dict."""
        return encode_observation(self._gc, self._bc)

    def _build_info(
        self,
        *,
        invalid_action: bool,
        shaping_terms: dict[str, float],
        snapshot: CombatSnapshot,
    ) -> Info:
        return {
            "action_mask": self._mask.copy(),
            "screen": "combat",
            "turn": snapshot.turn,
            "floor": int(self._gc.floor_num),
            "act": int(self._gc.act),
            "hp": snapshot.player_hp,
            "ascension": self._ascension,
            "shaping_terms": shaping_terms,
            "seed": self._episode_seed,
            "interface_version": self.interface_version,
            "rng_state": None,
            "engine_commit": self._engine_commit,
            "invalid_action": invalid_action,
            "combat": snapshot,
        }

    def _finalize_terminal_info(self, info: Info, *, won: bool) -> None:
        info["won"] = bool(won)
        info["episode"] = {"r": float(self._ep_return), "l": int(self._steps)}
