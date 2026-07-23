"""Gymnasium adapter over one engine instance, combat mode.

``StsEnv`` starts a seeded Ironclad combat, applies decoded actions, and reports
the Gymnasium 5-tuple. It is the combat backbone the rest of the environment
builds on. Two parts are intentionally still stubbed and will be filled by later
work:

- Observation: ``reset``/``step`` return a contract-shaped but all-zero
  placeholder observation. The real observation encoder replaces
  :meth:`StsEnv._observation`; the raw engine readout is available now in
  ``info['combat']`` for debugging.
- Reward: only the terminal win/loss signal is emitted; per-step shaping is not
  applied yet.

Action legality is enforced through :func:`sts_rl.env.actions.build_mask`: an
action is decoded and executed only after it is confirmed legal, so an invalid
move is never handed to the engine.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.actions import auto_resolve, build_mask, decode_action
from sts_rl.env.engine import engine_commit, read_combat, start_combat
from sts_rl.env.spaces import build_spaces
from sts_rl.interface import (
    ACTION_DIM,
    INTERFACE_VERSION,
    OBS_FIELDS,
    SHAPING_TERMS,
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
        render_mode: one of ``None``, ``"ansi"``, ``"human"``.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        *,
        ascension: int = 0,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        strict: bool = False,
        render_mode: str | None = None,
    ) -> None:
        if max_episode_steps < 1:
            raise InterfaceError(f"max_episode_steps must be >= 1, got {max_episode_steps}")
        valid_render_modes = (None, *self.metadata["render_modes"])
        if render_mode not in valid_render_modes:
            raise InterfaceError(
                f"render_mode {render_mode!r} invalid; expected one of {valid_render_modes}"
            )

        self.observation_space, self.action_space = build_spaces()
        self.interface_version = INTERFACE_VERSION
        self.render_mode = render_mode

        self._ascension = ascension
        self._max_episode_steps = max_episode_steps
        self._strict = strict
        # Cached once; the pinned commit does not change during a run.
        self._engine_commit = engine_commit()

        # Per-episode state, populated by reset() before use.
        self._gc: Any = None
        self._bc: Any = None
        self._steps = 0
        self._ep_return = 0.0
        self._episode_seed: int | None = None
        self._mask = np.zeros(ACTION_DIM, dtype=np.bool_)

    # -- Gymnasium API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Obs, Info]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, _MAX_SEED))
        self._episode_seed = seed
        self._gc, self._bc = start_combat(seed, ascension=self._ascension)
        self._steps = 0
        self._ep_return = 0.0
        self._refresh_mask()
        return self._observation(), self._build_info(invalid_action=False)

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
            # Do not touch the engine: executing an invalid action is unsafe.
            self._steps += 1
            truncated = self._steps >= self._max_episode_steps
            info = self._build_info(invalid_action=True)
            if truncated:
                self._finalize_terminal_info(info, won=False)
            return self._observation(), 0.0, False, truncated, info

        assert engine_action is not None  # guaranteed: legal implies a mapped, valid action
        engine_action.execute(self._bc)
        self._steps += 1

        # Advance past any engine-driven states the agent cannot represent (e.g. a
        # multi-select confirm the played card triggered), then read the outcome.
        self._refresh_mask()

        terminated = self._bc.outcome != sts.BattleOutcome.UNDECIDED
        won = self._bc.outcome == sts.BattleOutcome.PLAYER_VICTORY
        reward = 0.0
        if terminated:
            reward = TERMINAL_WIN_REWARD if won else TERMINAL_LOSS_REWARD
        truncated = (not terminated) and self._steps >= self._max_episode_steps

        self._ep_return += reward
        info = self._build_info(invalid_action=False)
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
        """Return a contract-shaped placeholder observation (all zeros).

        Replaced by the observation encoder; the raw readout is in
        ``info['combat']`` meanwhile. Zero is in-bounds for every field (id 0 is
        the PAD id), so this is a valid member of the observation space.
        """
        return {field.name: np.zeros(field.shape, dtype=field.dtype) for field in OBS_FIELDS}

    def _build_info(self, *, invalid_action: bool) -> Info:
        snapshot = read_combat(self._bc)
        return {
            "action_mask": self._mask.copy(),
            "screen": "combat",
            "turn": snapshot.turn,
            "floor": int(self._gc.floor_num),
            "act": int(self._gc.act),
            "hp": snapshot.player_hp,
            "ascension": self._ascension,
            "shaping_terms": {term: 0.0 for term in SHAPING_TERMS},
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
