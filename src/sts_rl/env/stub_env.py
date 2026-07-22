"""A fake environment that satisfies the interface contract without the engine.

``StubEnv`` is a drop-in ``gymnasium.Env`` that emits contract-conformant,
random-but-legal data instead of reading a real ``sts_lightspeed``
``BattleContext``. Its purpose is to let the agent side build and smoke-test the
full learning pipeline (encoder, network, rollout buffer, PPO update) before the
C++ engine exists, then swap in the real ``StsEnv`` at integration by changing a
single import.

Nothing here understands Slay the Spire. The training machinery only consumes
tensors of a fixed shape, a legal-action mask, a reward scalar, and the
``terminated``/``truncated`` flags, all of which are defined by
:mod:`sts_rl.interface`. That is exactly what this env produces.

Two reward modes are offered:

``"learnable"`` (default)
    A contextual-bandit signal with a known optimal policy. Each step exposes,
    inside ``player_scalars``, the index of the currently active action block
    and a scalar cue; the reward is ``+1`` when the chosen action equals the
    target index derived from that cue and ``0`` otherwise. Because the mapping
    from observation to the correct action is deterministic, a correct PPO
    implementation will drive the toy return upward, so the loop's ability to
    learn is observable. :meth:`StubEnv.optimal_action` returns the ground-truth
    action for the current state, which tests can follow to confirm the signal.

``"random"``
    Per-step reward is ``0`` and a random ``+1``/``-1`` terminal reward fires on
    a ``terminated`` step. This exercises the sparse-terminal path and GAE value
    bootstrapping without providing anything to learn.

Both modes emit ``terminated`` episodes (via ``terminate_prob``) and
``truncated`` episodes (via the step cap), so the two distinct GAE tail cases are
exercised. The env only depends on ``numpy`` and ``gymnasium``; it pulls in no
engine and no learning framework.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding

from sts_rl.env.spaces import build_spaces
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_BLOCKS,
    ACTION_DIM,
    INTERFACE_VERSION,
    OBS_FIELDS,
    SHAPING_TERMS,
    TERMINAL_LOSS_REWARD,
    TERMINAL_WIN_REWARD,
    ActionBlock,
    InterfaceError,
    Info,
    Obs,
    assert_valid_mask,
)

# Episode-shape defaults chosen so a normal rollout contains a mix of truncated
# and terminated episodes (both GAE tail cases).
DEFAULT_MAX_EPISODE_STEPS = 32
DEFAULT_TERMINATE_PROB = 0.05

REWARD_MODES: tuple[str, ...] = ("learnable", "random")

# Toy per-step reward for a correct action in "learnable" mode.
CORRECT_ACTION_REWARD = 1.0
WRONG_ACTION_REWARD = 0.0

# Slots inside player_scalars repurposed to carry the learnable cues. The active
# block index tells the agent which block is legal (hence its start offset); the
# cue selects the target offset within that block. Remaining scalars stay random
# so the agent must ignore noise. These indices are < PLAYER_SCALAR_DIM (8).
BLOCK_INDEX_SCALAR = 0
CUE_SCALAR = 1

# Opaque replay/debug markers (the contract only requires these keys to exist).
STUB_ENGINE_COMMIT = "stub-no-engine"

# Representative fixed values for scalar info fields that carry no meaning in a
# stub but must be present for the agent's logging to read.
_FAKE_FLOOR = 0
_FAKE_ACT = 1
_FAKE_HP = 80
_FAKE_ASCENSION = 0

# Scale of the random shaping-term values reported in info (audit plumbing only;
# the stub reward does not use them).
_SHAPING_NOISE_SCALE = 0.01


class StubEnv(gym.Env):
    """Contract-conformant fake env emitting random-but-legal observations.

    The observation and action spaces, the mask shape and rules, and the info
    keys all come from :mod:`sts_rl.interface`, so an agent written against the
    real env runs unchanged here.

    Args:
        max_episode_steps: step cap; hitting it ends the episode with
            ``truncated=True``.
        terminate_prob: per-step probability of ending with ``terminated=True``
            (a win/loss), evaluated only when the step cap is not reached.
        reward_mode: ``"learnable"`` or ``"random"`` (see the module docstring).
        active_blocks: optional names of the action blocks that may be the legal
            block on a step. Defaults to all blocks, exercising every screen's
            masking. Restrict to a small block (for example
            ``("CARD_REWARD_SELECT",)``) for a faster-converging toy task.
        strict: if ``True``, an illegal action passed to :meth:`step` raises
            :class:`~sts_rl.interface.InterfaceError` instead of only setting the
            ``invalid_action`` info flag.
        render_mode: one of ``None``, ``"ansi"``, ``"human"``.
    """

    # Gymnasium convention lists only real render modes here; render_mode=None
    # ("no rendering") is always accepted but omitted from the list.
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        *,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        terminate_prob: float = DEFAULT_TERMINATE_PROB,
        reward_mode: str = "learnable",
        active_blocks: Iterable[str] | None = None,
        strict: bool = False,
        render_mode: str | None = None,
    ) -> None:
        if reward_mode not in REWARD_MODES:
            raise InterfaceError(
                f"reward_mode {reward_mode!r} invalid; expected one of {REWARD_MODES}"
            )
        if max_episode_steps < 1:
            raise InterfaceError(f"max_episode_steps must be >= 1, got {max_episode_steps}")
        if not 0.0 <= terminate_prob <= 1.0:
            raise InterfaceError(f"terminate_prob must be in [0, 1], got {terminate_prob}")
        valid_render_modes = (None, *self.metadata["render_modes"])
        if render_mode not in valid_render_modes:
            raise InterfaceError(
                f"render_mode {render_mode!r} invalid; expected one of " f"{valid_render_modes}"
            )

        self.observation_space, self.action_space = build_spaces()
        self.interface_version = INTERFACE_VERSION
        self.render_mode = render_mode

        self._max_episode_steps = max_episode_steps
        self._terminate_prob = terminate_prob
        self._reward_mode = reward_mode
        self._strict = strict
        self._active_blocks: tuple[ActionBlock, ...] = self._resolve_blocks(active_blocks)

        # A learnable task built only from count-1 blocks is degenerate: the sole
        # legal action is always optimal, so a random policy already scores the
        # maximum and there is nothing to learn. Warn rather than fail, since this
        # is still valid for plumbing-only smoke tests.
        if reward_mode == "learnable" and all(block.count == 1 for block in self._active_blocks):
            warnings.warn(
                "StubEnv learnable mode with only count-1 active blocks yields a "
                "degenerate task (nothing to learn); use a multi-action block such "
                "as 'CARD_REWARD_SELECT' to exercise learning.",
                stacklevel=2,
            )

        # Per-episode / per-step state, populated by reset() before use.
        self._steps = 0
        self._ep_return = 0.0
        self._episode_seed: int | None = None
        self._reset_rng_state: object = None
        # State describing the observation the agent currently holds, used to
        # score its next action.
        self._obs: Obs = {}
        self._active_block: ActionBlock = self._active_blocks[0]
        self._cue: float = 0.0
        self._target_index: int = self._active_block.start
        self._mask: np.ndarray = np.zeros(ACTION_DIM, dtype=np.bool_)

    @staticmethod
    def _resolve_blocks(names: Iterable[str] | None) -> tuple[ActionBlock, ...]:
        if names is None:
            return ACTION_BLOCKS
        resolved: list[ActionBlock] = []
        for name in names:
            block = ACTION_BLOCK_BY_NAME.get(name)
            if block is None:
                raise InterfaceError(
                    f"active_blocks contains unknown block {name!r}; valid names: "
                    f"{tuple(ACTION_BLOCK_BY_NAME)}"
                )
            resolved.append(block)
        if not resolved:
            raise InterfaceError("active_blocks is empty; at least one block is required")
        return tuple(resolved)

    # -- Gymnasium API ------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Obs, Info]:
        super().reset(seed=seed)
        self._episode_seed = seed
        # Snapshot the generator state right after seeding so info can advertise a
        # reproducible replay handle for the episode.
        self._reset_rng_state = self.np_random.bit_generator.state
        self._steps = 0
        self._ep_return = 0.0
        self._advance()
        return self._obs, self._build_info(invalid_action=False)

    def step(self, action: int) -> tuple[Obs, float, bool, bool, Info]:
        action = int(action)
        invalid_action = not self._is_legal(action)
        if invalid_action and self._strict:
            raise InterfaceError(
                f"illegal action {action} for the current mask (legal block "
                f"{self._active_block.name} = [{self._active_block.start}, "
                f"{self._active_block.stop}))"
            )

        # Score the action against the state the agent actually observed.
        correct = (not invalid_action) and action == self._target_index
        if self._reward_mode == "learnable":
            reward = CORRECT_ACTION_REWARD if correct else WRONG_ACTION_REWARD
        else:
            reward = 0.0

        self._steps += 1
        truncated = self._steps >= self._max_episode_steps
        terminated = (not truncated) and (self.np_random.random() < self._terminate_prob)

        won: bool | None = None
        if terminated:
            won = bool(self.np_random.random() < 0.5)
            if self._reward_mode == "random":
                reward += TERMINAL_WIN_REWARD if won else TERMINAL_LOSS_REWARD
        elif truncated:
            # Truncation ends the episode without a win/loss verdict.
            won = False

        self._ep_return += reward

        # Advance to the next presented state so legal_actions()/info describe the
        # observation now being returned. On a terminal step the returned obs is
        # unused by the agent (the vector env auto-resets), but must still be a
        # valid, in-space observation. Note: on truncation this "next" obs is a
        # fresh iid draw, not a real successor of the truncated state, so a GAE
        # tail that bootstraps V(next_obs) is only unbiased because the toy task
        # is stationary.
        self._advance()
        info = self._build_info(invalid_action=invalid_action)
        if terminated or truncated:
            info["won"] = bool(won)
            info["episode"] = {"r": float(self._ep_return), "l": int(self._steps)}

        return self._obs, float(reward), terminated, truncated, info

    def seed(self, seed: int | None = None) -> list[int]:
        """Reseed the RNG and return the seed used.

        ``reset(seed=...)`` is the canonical seeding path; this is retained for
        explicit reseeding per the interface contract Env API, since
        ``gymnasium.Env`` no longer provides a ``seed`` method.
        """
        self.np_random, actual_seed = seeding.np_random(seed)
        return [actual_seed]

    def legal_actions(self) -> np.ndarray:
        """Return the current ``(ACTION_DIM,)`` bool mask (a defensive copy)."""
        return self._mask.copy()

    def action_masks(self) -> np.ndarray:
        """Alias of :meth:`legal_actions` for MaskablePPO-style callers."""
        return self.legal_actions()

    def optimal_action(self) -> int:
        """Return the reward-maximizing action for the current state.

        In ``"learnable"`` mode this is the target index; following it every step
        yields the maximum toy return. In ``"random"`` mode the reward does not
        depend on the action, so this returns an arbitrary legal action.
        """
        return self._target_index

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        text = (
            f"StubEnv[{self._reward_mode}] step={self._steps} "
            f"screen={self._active_block.name} "
            f"legal={self._active_block.start}..{self._active_block.stop - 1} "
            f"target={self._target_index}"
        )
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        return None

    # -- Internals ----------------------------------------------------------

    def _is_legal(self, action: int) -> bool:
        return 0 <= action < ACTION_DIM and bool(self._mask[action])

    def _advance(self) -> None:
        """Sample the next active block, cue, target, mask, and observation."""
        block = self._active_blocks[int(self.np_random.integers(len(self._active_blocks)))]
        cue = float(self.np_random.random())
        # Map the cue to a target offset within the block; the last bucket
        # absorbs the cue==~1.0 edge so the index stays in range.
        target_offset = min(int(cue * block.count), block.count - 1)

        self._active_block = block
        self._cue = cue
        self._target_index = block.start + target_offset
        self._mask = self._build_mask(block)
        self._obs = self._random_obs(block, cue)

    @staticmethod
    def _build_mask(block: ActionBlock) -> np.ndarray:
        """Unmask exactly the given block, matching the contract's screen rule."""
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        mask[block.start : block.stop] = True
        assert_valid_mask(mask)
        return mask

    def _random_obs(self, block: ActionBlock, cue: float) -> Obs:
        """Build one random-but-legal observation dict.

        Every field matches its :class:`~sts_rl.interface.ObsField` dtype, shape,
        and bounds: unit fields are random {0, 1} indicators, real fields are
        standard-normal noise, and id fields are integers in ``[0, id_high]`` (so
        embedding lookups never go out of range; ``PAD`` id 0 arises naturally).
        The learnable cues are then written into ``player_scalars``.
        """
        rng = self.np_random
        obs: Obs = {}
        for field in OBS_FIELDS:
            if field.bounds == "id":
                assert field.id_high is not None  # guaranteed by ObsField
                obs[field.name] = rng.integers(
                    0, field.id_high + 1, size=field.shape, dtype=np.int32
                )
            elif field.bounds == "unit":
                obs[field.name] = rng.integers(0, 2, size=field.shape).astype(np.float32)
            else:  # "real"
                obs[field.name] = rng.standard_normal(field.shape).astype(np.float32)

        # A valid single-hot screen indicator. The chosen index is an arbitrary
        # function of the block (blocks outnumber screens, so it is not the real
        # block-to-screen mapping); the authoritative block signal the agent
        # learns from is player_scalars[BLOCK_INDEX_SCALAR] below.
        screen_onehot = obs["screen_onehot"]
        screen_onehot[:] = 0.0
        screen_onehot[block.start % screen_onehot.shape[0]] = 1.0

        # Expose the learnable cues. The agent must read these (and ignore the
        # rest of player_scalars) to recover the optimal action.
        player_scalars = obs["player_scalars"]
        player_scalars[BLOCK_INDEX_SCALAR] = float(self._active_blocks.index(block))
        player_scalars[CUE_SCALAR] = cue

        return obs

    def _build_info(self, *, invalid_action: bool) -> Info:
        shaping_terms = {
            term: float(self.np_random.standard_normal() * _SHAPING_NOISE_SCALE)
            for term in SHAPING_TERMS
        }
        return {
            "action_mask": self._mask.copy(),
            "screen": self._active_block.name,
            "turn": int(self._steps),
            "floor": _FAKE_FLOOR,
            "act": _FAKE_ACT,
            "hp": _FAKE_HP,
            "ascension": _FAKE_ASCENSION,
            "shaping_terms": shaping_terms,
            "seed": self._episode_seed,
            "interface_version": self.interface_version,
            "rng_state": self._reset_rng_state,
            "engine_commit": STUB_ENGINE_COMMIT,
            "invalid_action": invalid_action,
        }
