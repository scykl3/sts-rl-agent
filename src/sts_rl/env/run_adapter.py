"""Gymnasium adapter over one engine instance, full-run mode.

``StsRunEnv`` plays a complete seeded Ironclad run as one episode: it drives the
overworld state machine (map, events, shops, campfires, reward screens) and the
combats those screens lead into, transitioning between the two until the run ends
in victory or death. It is the run-mode sibling of :class:`sts_rl.env.adapter.StsEnv`,
which plays a single combat.

Two engine views back one episode. The overworld is a ``GameContext`` driven by
``GameAction`` moves (see :mod:`sts_rl.env.run` / :mod:`sts_rl.env.run_actions`);
a combat is a ``BattleContext`` driven by ``search::Action`` moves (see
:mod:`sts_rl.env.engine` / :mod:`sts_rl.env.actions`). Entering a monster / elite /
boss room flips ``gc.screen_state`` to ``BATTLE``; the adapter then builds a
``BattleContext`` (``gc.create_battle_context()``), plays it out through the combat
action space, and on combat end syncs the result back
(``gc.sync_from_battle_context(bc)``) so the run continues on the overworld. One
observation encoder serves both views: combat state when a battle is live,
``map_context`` otherwise (see :func:`sts_rl.env.observation.encode_observation`).

Reward is the terminal run signal plus annealed per-step shaping:
``reward = terminal + beta(t) * sum(shaping_terms)``, where ``terminal`` is
``+1`` on run victory / ``-1`` on death and ``t`` is the cumulative step count.
A combat step contributes combat shaping (enemy HP removed, damage taken); an
overworld step contributes run shaping (floor progress, boss kill). See
:mod:`sts_rl.env.reward`.

Two auto-advance behaviors keep the agent on genuine decisions and off states it
cannot represent:

- Forced continues: an overworld screen with exactly one legal engine action
  carries no decision, so the adapter advances it itself rather than emitting an
  agent step. A screen with two or more legal actions is always a real choice and
  is handed to the agent untouched. Combat is never advanced by action count (a
  turn may legitimately offer a single legal move).
- Unrepresentable states (a match-and-keep grid, a card-select beyond the
  interface cap) are advanced by the underlying auto-resolvers so the agent is
  never handed an all-False mask.

Safety: a move is decoded and executed only after the engine confirms it legal
(combat via ``is_valid_action``, overworld via ``isValidAction``), so an invalid
action is never handed to the engine, whose ``execute`` on an invalid move is
undefined.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.actions import auto_resolve, build_mask, decode_action
from sts_rl.env.engine import CombatSnapshot, engine_commit, read_combat
from sts_rl.env.observation import encode_observation
from sts_rl.env.reward import (
    RewardConfig,
    combat_shaping_terms,
    run_shaping_terms,
    shaping_reward,
    zero_shaping_terms,
)
from sts_rl.env.run import (
    RunSnapshot,
    execute_overworld_action,
    is_run_over,
    overworld_actions,
    read_run,
    run_won,
    start_run,
)
from sts_rl.env.run_actions import (
    auto_resolve_overworld,
    build_overworld_mask,
    decode_overworld_action,
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

# A full Acts 1-3 run is on the order of hundreds of agent decisions once forced
# continues are auto-advanced; this cap only ends a stalled episode with
# ``truncated=True``. Training config may override it.
DEFAULT_MAX_EPISODE_STEPS = 3000
_MAX_SEED = 2**31 - 1

_MODE_OVERWORLD = "overworld"
_MODE_COMBAT = "combat"

# Backstop on engine transitions between two agent decisions (forced continues,
# a combat entry, or a combat exit and the screens it lands on). Far above any
# real gap, so it only trips on a state-machine loop bug. The per-call resolvers
# (:func:`auto_resolve` / :func:`auto_resolve_overworld`) bound their own loops.
_SETTLE_CAP = 512


class StsRunEnv(gym.Env):
    """Gymnasium environment over one engine instance, one full run per episode.

    Args:
        ascension: ascension level for the run.
        max_episode_steps: step cap; hitting it ends the episode with
            ``truncated=True``.
        strict: if ``True``, an illegal action passed to :meth:`step` raises
            :class:`~sts_rl.interface.InterfaceError` instead of only flagging
            ``invalid_action`` in ``info`` and leaving the state unchanged.
        reward_config: shaping coefficients and anneal schedule; defaults to
            :class:`~sts_rl.env.reward.RewardConfig`.
        render_mode: one of ``None``, ``"ansi"``, ``"human"``.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        *,
        ascension: int = 0,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        strict: bool = False,
        reward_config: RewardConfig | None = None,
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
        self._reward_config = reward_config if reward_config is not None else RewardConfig()
        # Cached once; the pinned commit does not change during a run.
        self._engine_commit = engine_commit()

        # Cumulative env steps; drives the shaping anneal beta(t). Not reset
        # between episodes (t is training progress, not within-episode time). See
        # StsEnv.set_global_step for the parallel-worker rationale.
        self._global_step = 0

        # Per-episode state, populated by reset() before use.
        self._gc: Any = None  # GameContext, always present after reset
        self._bc: Any = None  # BattleContext while a combat is live, else None
        self._mode = _MODE_OVERWORLD
        self._steps = 0
        self._ep_return = 0.0
        self._episode_seed: int | None = None
        self._mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        # Baselines for per-step shaping deltas. The run snapshot persists across
        # the whole episode and updates only on overworld steps, so run shaping
        # spans any intervening combat; the combat snapshot is (re)set on each
        # combat entry and updates on combat steps.
        self._prev_run: RunSnapshot | None = None
        self._prev_combat: CombatSnapshot | None = None

    # -- Gymnasium API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Obs, Info]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, _MAX_SEED))
        self._episode_seed = seed
        self._gc = start_run(seed, ascension=self._ascension)
        self._bc = None
        self._mode = _MODE_OVERWORLD
        self._steps = 0
        self._ep_return = 0.0
        # Combat baseline is (re)set on each combat entry; clear it so a fresh
        # episode carries no stale snapshot from the previous run.
        self._prev_combat = None
        # Advance from the run's first raw state to the first genuine decision.
        self._settle()
        self._prev_run = read_run(self._gc)
        return self._observation(), self._build_info(
            invalid_action=False, shaping_terms=zero_shaping_terms()
        )

    def step(self, action: int) -> tuple[Obs, float, bool, bool, Info]:
        action = int(action)
        in_combat = self._mode == _MODE_COMBAT
        legal = 0 <= action < ACTION_DIM and bool(self._mask[action])
        # Decode against the active view and re-check engine legality: a set mask
        # bit already maps to an engine-legal move, but this also rejects an
        # out-of-space or unmapped index the mask never set.
        if in_combat:
            engine_action = decode_action(action, self._bc) if legal else None
            if engine_action is None or not engine_action.is_valid_action(self._bc):
                legal = False
        else:
            engine_action = decode_overworld_action(action, self._gc) if legal else None
            if engine_action is None or not engine_action.isValidAction(self._gc):
                legal = False

        if not legal and self._strict:
            # Raise before counting the step: the episode is abandoned (matches StsEnv).
            raise InterfaceError(f"illegal action {action} for the current mask")

        self._steps += 1
        if not legal:
            # Do not touch the engine: executing an invalid action is unsafe. The
            # state is unchanged, so no shaping delta and no reward accrue, but the
            # step still counts as one interaction and advances the anneal clock.
            self._global_step += 1
            truncated = self._steps >= self._max_episode_steps
            info = self._build_info(invalid_action=True, shaping_terms=zero_shaping_terms())
            if truncated:
                self._finalize_terminal_info(info, won=False)
            return self._observation(), 0.0, False, truncated, info

        assert engine_action is not None  # legal implies a mapped, valid action
        # Route by the mode the action was taken in; the handlers settle to the
        # next decision, which may change self._mode / self._bc.
        if in_combat:
            shaping_terms = self._step_combat(engine_action)
        else:
            shaping_terms = self._step_overworld(engine_action)

        terminated = is_run_over(self._gc)
        won = run_won(self._gc)
        terminal = 0.0
        if terminated:
            terminal = TERMINAL_WIN_REWARD if won else TERMINAL_LOSS_REWARD
        truncated = (not terminated) and self._steps >= self._max_episode_steps

        # beta is indexed by steps already taken, so the first-ever step sees
        # beta(0); the clock advances after the reward is computed.
        reward = terminal + shaping_reward(shaping_terms, self._global_step, self._reward_config)
        self._global_step += 1
        self._ep_return += reward

        info = self._build_info(invalid_action=False, shaping_terms=shaping_terms)
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
        """Set the shaping anneal clock to the true global env-step count.

        Mirrors :meth:`sts_rl.env.adapter.StsEnv.set_global_step`: under parallel
        workers the training loop should broadcast the shared global step so the
        anneal ``beta(t)`` stays on its intended horizon. Subsequent steps advance
        from the value set here.
        """
        if t < 0:
            raise InterfaceError(f"global_step must be >= 0, got {t}")
        self._global_step = int(t)

    def render(self) -> str | None:
        if self.render_mode is None or self._gc is None:
            return None
        text = (
            str(self._bc) if self._mode == _MODE_COMBAT and self._bc is not None else str(self._gc)
        )
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        self._gc = None
        self._bc = None

    # -- Internals ----------------------------------------------------------

    def _step_combat(self, engine_action: Any) -> dict[str, float]:
        """Execute one combat action, resolve, and settle across a combat end.

        Returns the combat shaping terms for this step. On combat end, syncs the
        result back to the overworld and settles to the next decision or terminal;
        otherwise refreshes the combat mask and advances the combat baseline.
        """
        prev = self._prev_combat
        assert prev is not None  # combat mode always has a combat baseline
        engine_action.execute(self._bc)
        # Advance past engine-driven states the agent cannot represent (e.g. a
        # multi-select confirm the played card triggered), then read the outcome.
        auto_resolve(self._bc)
        curr = read_combat(self._bc)
        terms = combat_shaping_terms(prev, curr, self._reward_config)

        if self._bc.outcome != sts.BattleOutcome.UNDECIDED:
            # Combat over: write HP / gold / deck / relics back and hand control to
            # the overworld (regainControl opens rewards, advances the act, or ends
            # the run in victory; a loss sets the run outcome directly).
            self._gc.sync_from_battle_context(self._bc)
            self._bc = None
            self._settle()
        else:
            self._mode = _MODE_COMBAT
            self._mask = build_mask(self._bc)
            assert_valid_mask(self._mask)
            self._prev_combat = curr
        return terms

    def _step_overworld(self, engine_action: Any) -> dict[str, float]:
        """Execute one overworld action, settle, and return run shaping terms.

        Run shaping compares against the persisted run baseline (last overworld
        step), so floor / act progress is credited even when a combat intervened
        between the two overworld decisions.
        """
        prev = self._prev_run
        assert prev is not None  # reset() populates the run baseline before any step
        execute_overworld_action(self._gc, engine_action)
        self._settle()
        curr = read_run(self._gc)
        terms = run_shaping_terms(prev, curr, self._reward_config)
        self._prev_run = curr
        return terms

    def _settle(self) -> None:
        """Advance the engine to the next agent decision, or a terminal outcome.

        Crosses combat / overworld boundaries and auto-advances forced overworld
        continues (screens with exactly one legal engine action) and states with
        no representable move, until it reaches either a terminal run outcome or a
        decision point with a non-empty representable mask. Sets ``self._mode``,
        ``self._bc``, ``self._mask``, and, on combat entry, the combat baseline.
        """
        for _ in range(_SETTLE_CAP):
            if is_run_over(self._gc):
                self._mode = _MODE_OVERWORLD
                self._bc = None
                self._mask = np.zeros(ACTION_DIM, dtype=np.bool_)
                return

            if self._gc.screen_state == sts.ScreenState.BATTLE:
                if self._bc is None:
                    self._bc = self._gc.create_battle_context()
                auto_resolve(self._bc)
                if self._bc.outcome != sts.BattleOutcome.UNDECIDED:
                    # Resolved before the agent acted (e.g. a battle that ends on
                    # entry); sync back and continue from the overworld.
                    self._gc.sync_from_battle_context(self._bc)
                    self._bc = None
                    continue
                self._mode = _MODE_COMBAT
                self._mask = build_mask(self._bc)
                assert_valid_mask(self._mask)
                self._prev_combat = read_combat(self._bc)
                return

            # Overworld: clear any stale combat handle, then advance past screens
            # with no representable move. After this the state is terminal, a
            # battle, or offers at least one representable action.
            self._bc = None
            auto_resolve_overworld(self._gc)
            if is_run_over(self._gc) or self._gc.screen_state == sts.ScreenState.BATTLE:
                continue

            actions = overworld_actions(self._gc)
            if len(actions) == 1:
                # Exactly one legal action: a forced continue with no decision.
                # Advance it ourselves rather than emitting an agent step. (It is
                # representable: an unrepresentable lone action leaves an all-False
                # mask, which auto_resolve_overworld above already advanced past.)
                execute_overworld_action(self._gc, actions[0])
                continue

            self._mode = _MODE_OVERWORLD
            self._mask = build_overworld_mask(self._gc)
            assert_valid_mask(self._mask)
            return

        raise InterfaceError(f"_settle exceeded {_SETTLE_CAP} transitions; possible state loop")

    def _observation(self) -> Obs:
        """Encode the live engine state: combat view when a battle is live, else run view."""
        return encode_observation(self._gc, self._bc)

    def _build_info(self, *, invalid_action: bool, shaping_terms: dict[str, float]) -> Info:
        """Assemble the always-present info handshake for the current decision point."""
        run_snapshot = read_run(self._gc)
        combat_snapshot: CombatSnapshot | None
        if self._mode == _MODE_COMBAT and self._bc is not None:
            combat_snapshot = read_combat(self._bc)
            screen = _MODE_COMBAT
            turn = combat_snapshot.turn
            hp = combat_snapshot.player_hp
        else:
            combat_snapshot = None
            screen = self._gc.screen_state.name.lower()
            turn = 0
            hp = int(self._gc.cur_hp)

        info: Info = {
            "action_mask": self._mask.copy(),
            "screen": screen,
            "turn": turn,
            "floor": int(self._gc.floor_num),
            "act": int(self._gc.act),
            "hp": hp,
            "ascension": self._ascension,
            "shaping_terms": shaping_terms,
            "seed": self._episode_seed,
            "interface_version": self.interface_version,
            "rng_state": None,
            "engine_commit": self._engine_commit,
            "invalid_action": invalid_action,
            "run": run_snapshot,
        }
        if combat_snapshot is not None:
            info["combat"] = combat_snapshot
        return info

    def _finalize_terminal_info(self, info: Info, *, won: bool) -> None:
        info["won"] = bool(won)
        info["episode"] = {"r": float(self._ep_return), "l": int(self._steps)}
