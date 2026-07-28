"""Potential-based deck/economy reward shaping for full-run training (agent lane).

An opt-in ``gymnasium.Wrapper`` over :class:`~sts_rl.env.run_adapter.StsRunEnv` that
augments the env reward with a potential-based shaping term
``F(s, s') = gamma * Phi(s') - Phi(s)`` (Ng, Harada and Russell 1999). The potential
``Phi`` scores the agent's economy (owned relics, gold, held potions), so the policy is
nudged toward acquiring lasting resources without changing what it ultimately learns: a
potential-based term telescopes over any trajectory, so it adds no bias to the optimal
policy, only to how fast the agent reaches it.

This is a training-time experiment lever, wired behind ``--deck-economy-shaping`` in
``scripts/train_run.py`` and applied to the TRAINING env only; the eval env is left
unwrapped so the reported metric stays the clean terminal-win signal.

Potential-based-shaping contract (each point is load bearing):

- ``Phi`` is a PURE function of the current observation (no running sums, no path
  dependence), so the shaping is a genuine potential and stays policy-invariant.
- ``gamma`` is REQUIRED at construction and must be the SAME discount the trainer uses
  for GAE; a term shaped with a different discount would not telescope against the
  return and would bias the policy, so there is deliberately no internal default.
- ``Phi(terminal) := 0`` by the shaping convention, applied on BOTH ``terminated`` (a
  true episode end) and ``truncated`` (a time-limit cutoff). This repo's single-env
  collector bootstraps ``V`` of the POST-RESET observation on a truncation (not the
  truncated state's true successor; see the ``rollout_collector`` module docstring) and
  does not sever GAE at that boundary, so emitting the real ``gamma * Phi(s')`` on
  truncation would inject an economy-correlated residual that does not telescope. Zeroing
  ``Phi(s')`` there instead yields ``-gamma * Phi(reset)``, a policy-invariant CONSTANT
  because the Ironclad run reset is deterministic (fixed starting relic, ~99 gold, 0
  potions). Truncation is a rare backstop at the default step cap.
- The term is UN-annealed (constant ``scale``): unlike the env's heuristic shaping, a
  potential-based term needs no anneal schedule to stay unbiased.
- Deck COMPOSITION is deliberately excluded from ``Phi``: the ``deck_ids`` observation is
  populated only in the overworld (PAD in combat), so scoring it would make ``Phi`` jump
  at every combat/overworld boundary, a spurious path-dependent discontinuity. The
  economy quantities used (relics, gold, potions) are present in BOTH views.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from sts_rl.interface import PAD_ID, PLAYER_SCALAR_DIM, Info, InterfaceError, Obs

# --- Potential weights ------------------------------------------------------
# Phi = scale * (C_RELIC*num_relics + C_GOLD*gold/GOLD_NORM + C_POTION*num_potions).
# Each weight is sized so one economy gain moves the per-step shaping
# gamma*Phi(s') - Phi(s) into the same band as the env's own shaping terms
# (floor_progress 0.02, boss_kill 0.20; see sts_rl.env.reward), nudging acquisition
# without swamping the +/-1 terminal signal. gold is stored RAW in player_scalars
# (O(100) over a run), so GOLD_NORM scales it to O(1) first; relics and potions are
# already small integer counts.
C_RELIC: float = 0.05  # one relic ~ one enemy fully removed; ~10-15 relics/run keeps it < 1
GOLD_NORM: float = 100.0  # Ironclad starts near 99 gold, so one starting purse ~ one unit
C_GOLD: float = 0.1  # gold term 0.1*gold/100; a ~25-gold combat reward -> +0.025 per gain
C_POTION: float = 0.02  # one potion ~ floor_progress; the 5-slot belt caps its total at 0.1

# Gold occupies this index of player_scalars per the interface's documented layout
# (hp_cur, hp_max, block, energy, gold, floor, ascension, turn). The interface exposes the
# block width but not per-field indices, and this agent-lane module must not edit the shared
# interface, so the index is mirrored here and guarded against a too-short layout.
GOLD_SCALAR_INDEX: int = 4
if GOLD_SCALAR_INDEX >= PLAYER_SCALAR_DIM:
    raise InterfaceError(
        f"GOLD_SCALAR_INDEX {GOLD_SCALAR_INDEX} does not fit PLAYER_SCALAR_DIM "
        f"{PLAYER_SCALAR_DIM}; the interface player_scalars layout changed"
    )


class DeckEconomyShapingWrapper(gym.Wrapper):
    """Augment ``StsRunEnv`` reward with potential-based deck/economy shaping.

    Args:
        env: the run-mode env to wrap (a :class:`~sts_rl.env.run_adapter.StsRunEnv`).
        gamma: the discount the trainer uses for GAE. REQUIRED and single-sourced (pass
            ``args.gamma``); a potential term must use the same discount to stay unbiased,
            so there is no internal default.
        scale: constant multiplier on the whole potential (the experiment lever's strength
            knob). Never annealed.
    """

    def __init__(self, env: gym.Env, *, gamma: float, scale: float = 1.0) -> None:
        super().__init__(env)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if scale < 0.0:
            raise ValueError(f"scale must be >= 0, got {scale}")
        self.gamma = float(gamma)
        self.scale = float(scale)
        # Cached potential of the last observation (a pure Phi value, NOT a running reward
        # sum): the only state kept, needed to form the telescoping gamma*Phi' - Phi.
        self._prev_phi = 0.0

    def _potential(self, obs: Obs) -> float:
        """Economy potential Phi(obs): a pure function of the observation only.

        ``Phi = scale * (C_RELIC*num_relics + C_GOLD*gold/GOLD_NORM + C_POTION*num_potions)``.
        num_relics is the owned-relic multihot sum; gold is the raw ``player_scalars`` gold;
        num_potions counts non-PAD potion slots (potion id 0 == INVALID == empty slot). Deck
        composition is excluded on purpose (``deck_ids`` is overworld-only; see the module
        docstring), so Phi is continuous across the combat/overworld boundary.
        """
        num_relics = float(obs["relics_multihot"].sum())
        gold = float(obs["player_scalars"][GOLD_SCALAR_INDEX])
        num_potions = float(np.count_nonzero(obs["potion_ids"] != PAD_ID))
        return self.scale * (
            C_RELIC * num_relics + C_GOLD * (gold / GOLD_NORM) + C_POTION * num_potions
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, Info]:
        obs, info = self.env.reset(seed=seed, options=options)
        # Baseline the potential at the first state so the first step's difference is well
        # defined; reset itself carries no shaping reward.
        self._prev_phi = self._potential(obs)
        return obs, info

    def step(self, action: Any) -> tuple[Obs, float, bool, bool, Info]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_phi = self._potential(obs)
        # Potential-based shaping F = gamma*Phi(s') - Phi(s), with Phi(terminal) := 0 on BOTH
        # terminated and truncated. On truncation the collector bootstraps V of the post-reset
        # observation (not the truncated state's true successor; see rollout_collector) and
        # does not sever GAE, so the real gamma*Phi(s') would leave an economy-correlated
        # residual that does not telescope; zeroing yields -gamma*Phi(reset), a policy-invariant
        # constant (the run reset is deterministic). Truncation is a rare backstop at the cap.
        shaped = (0.0 if terminated or truncated else self.gamma * new_phi) - self._prev_phi
        self._prev_phi = new_phi
        # Additive: a NEW reward dimension (the base reward has no relic/gold/potion term), so
        # no double count. Expose the term for diagnostics without clobbering existing keys.
        info["deck_economy_shaping"] = shaped
        return obs, float(reward) + shaped, terminated, truncated, info

    # Gymnasium 1.x Wrapper no longer auto-forwards non-gym-API attributes, so re-expose
    # StsRunEnv's masking / anneal-clock methods explicitly, keeping the wrapper a drop-in for
    # the env (the collector reads the mask from info["action_mask"], but a MaskablePPO-style
    # or vectorized caller invokes these directly).
    def legal_actions(self) -> np.ndarray:
        return self.env.legal_actions()  # type: ignore[attr-defined]

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()  # type: ignore[attr-defined]

    def set_global_step(self, t: int) -> None:
        self.env.set_global_step(t)  # type: ignore[attr-defined]
