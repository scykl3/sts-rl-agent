"""Auxiliary perception targets for the shared-encoder actor-critic.

Dense, per-step, observation-derivable combat quantities that the optional aux
head (:class:`~sts_rl.agent.actor_critic.ActorCritic`'s ``aux_head``) regresses,
to shape the shared encoder toward survival-relevant features. The aux loss that
consumes these is OFF by default (``PPOConfig.aux_coef == 0``); when enabled it is
a per-column masked MSE (see :func:`~sts_rl.agent.ppo_update.ppo_update`).

The OBS-DERIVABLE targets are pure functions of the stored observation dict - no
engine call, no collector change - so they add no rollout state. Two are defined:

- ``incoming_damage``: the total base damage the enemies' committed intents would
  deal next, ``sum_e intent_val_e * intent_hits_e`` over live enemies.
- ``survival_margin``: ``cur_hp + block - incoming_damage`` - how much headroom the
  player has against the telegraphed hit, the single most survival-relevant scalar.

Both are HP-unit quantities, normalized by :data:`HP_SCALE` so the aux MSE is O(1)
and comparable in scale to the value loss.

A third target, ``end_combat_hp``, is NOT obs-derivable: it is the player's HP at
the END of the combat each step belongs to - a future outcome the rollout collector
backfills onto every step of a completed combat and the buffer carries to loss time.
It is listed last in :data:`AUX_TARGETS` (so the aux head predicts it as an extra
column) but is NOT produced by :func:`compute_aux_targets`;
:func:`~sts_rl.agent.ppo_update.ppo_update` assembles it into the target and masks it
with its own per-step validity flag. Being a future outcome routed through the shared
encoder, it carries more regression risk than the obs-derived columns, which is why
the aux loss stays OFF by default. The backfill treats each combat as a maximal run of
consecutive in-combat steps and assumes no mid-combat all-dead observation; see
:mod:`~sts_rl.agent.rollout_collector` for that load-bearing engine invariant.

Enable gate: turning the aux loss on (``aux_coef > 0``) backprops through the shared
encoder that the policy and value heads read, so it can move the policy - this is the
mechanism by which the earlier outcome-regression pretrain regressed the policy. Any
``aux_coef > 0`` run MUST therefore be gated on a matched-seed paired eval against the
aux-off baseline: do not trust or save a checkpoint whose greedy Act-1 clear rate
regresses relative to that baseline.

Approximation (v1): ``intent_val`` is the engine's BASE move damage; per-target
Strength and Vulnerable modifiers are NOT folded in, so predicted incoming damage
is a lower bound when those powers are in play. A power-aware target is a
deliberate follow-on.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sts_rl.interface import ENEMY_SCALAR_DIM, PLAYER_SCALAR_DIM, InterfaceError

# Obs-derivable aux columns compute_aux_targets produces, in fixed column order.
# The order is load-bearing: column k of this block maps to OBS_AUX_TARGETS[k].
OBS_AUX_TARGETS: tuple[str, ...] = ("incoming_damage", "survival_margin")

# Collector-backfilled column name: the player's HP at the END of the combat each
# step belongs to. NOT obs-derivable (a future outcome), so the rollout buffer
# supplies it at loss time rather than compute_aux_targets. See ppo_update.
END_COMBAT_HP_TARGET: str = "end_combat_hp"

# Full target set the aux head predicts, in the fixed column order the head's output
# and the assembled target tensor share: the obs-derivable columns first, then the
# collector-provided end_combat_hp column last. Column k of the aux head maps to
# AUX_TARGETS[k]; ppo_update assembles the target in this exact order.
AUX_TARGETS: tuple[str, ...] = OBS_AUX_TARGETS + (END_COMBAT_HP_TARGET,)

# Column index of the collector-provided end_combat_hp target within AUX_TARGETS (it
# is appended last, after the obs-derivable columns). ppo_update slots
# mb.end_combat_hp into this column and masks it with mb.end_combat_hp_valid rather
# than the obs combat_mask.
END_COMBAT_HP_COLUMN: int = len(OBS_AUX_TARGETS)

# Scalar-column indices into the stored obs tensors. The interface exposes the
# block WIDTHS (ENEMY_SCALAR_DIM, PLAYER_SCALAR_DIM) but not per-field indices, and
# this module must not edit the shared interface, so the indices are mirrored here
# and guarded below. Mirrored from the interface scalar layout: enemy_scalars =
# (hp_cur, hp_max, block, intent_val, intent_hits); player_scalars =
# (hp_cur, hp_max, block, energy, gold, floor, ascension, turn).
ENEMY_INTENT_VAL_INDEX: int = 3  # per-hit BASE move damage
ENEMY_INTENT_HITS_INDEX: int = 4  # number of hits the committed move deals
PLAYER_CUR_HP_INDEX: int = 0
PLAYER_BLOCK_INDEX: int = 2

# Fail at import if a mirrored index no longer fits its interface block width, so a
# shrink of the scalar layout surfaces here rather than as a silently wrong target.
# This width guard does NOT catch a same-width column REORDER; the indices are
# verified against the env observation builder, so a same-width reorder there would
# need re-mirroring here (a producer-locked test is a follow-on).
for _idx, _dim, _const_name in (
    (ENEMY_INTENT_VAL_INDEX, ENEMY_SCALAR_DIM, "ENEMY_INTENT_VAL_INDEX"),
    (ENEMY_INTENT_HITS_INDEX, ENEMY_SCALAR_DIM, "ENEMY_INTENT_HITS_INDEX"),
    (PLAYER_CUR_HP_INDEX, PLAYER_SCALAR_DIM, "PLAYER_CUR_HP_INDEX"),
    (PLAYER_BLOCK_INDEX, PLAYER_SCALAR_DIM, "PLAYER_BLOCK_INDEX"),
):
    if _idx >= _dim:
        raise InterfaceError(
            f"{_const_name} ({_idx}) does not fit its interface block width ({_dim}); "
            "the interface scalar layout changed"
        )

# HP-unit normalization scale ~ Ironclad base max HP. Divides the raw incoming-
# damage / survival-margin targets so the aux MSE is O(1) and on the same scale as
# the value loss (an ~80-damage swing maps to ~1.0). Not the exact max HP (it rises
# with rest-site / relic gains); a fixed reference is all normalization needs.
HP_SCALE: float = 80.0


def compute_aux_targets(obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Normalized aux targets and the combat-step mask from a batched obs dict.

    ``obs`` is a batched interface observation: ``enemy_scalars`` ``(B, MAX_ENEMIES,
    ENEMY_SCALAR_DIM)``, ``enemy_alive`` ``(B, MAX_ENEMIES)``, and ``player_scalars``
    ``(B, PLAYER_SCALAR_DIM)``. Returns ``(targets, combat_mask)``:

    - ``targets`` ``(B, len(OBS_AUX_TARGETS))``: column 0 normalized incoming damage,
      column 1 normalized survival margin, both divided by :data:`HP_SCALE`. These
      are the OBS-DERIVABLE columns only; the ``end_combat_hp`` column of the full
      :data:`AUX_TARGETS` is a future outcome the rollout buffer supplies at loss
      time (see :func:`~sts_rl.agent.ppo_update.ppo_update`), not computed here.
    - ``combat_mask`` ``(B,)`` bool: True where at least one enemy is alive - the
      steps the obs-derived aux columns apply over (overworld steps have no live
      enemy).

    ``incoming_damage`` gates ``intent_val * intent_hits`` by ``enemy_alive`` so a
    dead or PAD enemy contributes exactly zero regardless of any stale intent the
    engine leaves in a downed monster's ``enemy_scalars`` row. Float fields are
    coerced to float32 (a numpy-sampled obs is commonly float64).
    """
    enemy_scalars = obs["enemy_scalars"].float()  # (B, MAX_ENEMIES, ENEMY_SCALAR_DIM)
    enemy_alive = obs["enemy_alive"].float()  # (B, MAX_ENEMIES)
    player_scalars = obs["player_scalars"].float()  # (B, PLAYER_SCALAR_DIM)

    intent_val = enemy_scalars[..., ENEMY_INTENT_VAL_INDEX]  # (B, MAX_ENEMIES)
    intent_hits = enemy_scalars[..., ENEMY_INTENT_HITS_INDEX]  # (B, MAX_ENEMIES)
    # Gate by alive so dead/PAD enemies contribute zero by construction.
    incoming_damage = (intent_val * intent_hits * enemy_alive).sum(dim=-1)  # (B,)

    cur_hp = player_scalars[..., PLAYER_CUR_HP_INDEX]  # (B,)
    block = player_scalars[..., PLAYER_BLOCK_INDEX]  # (B,)
    survival_margin = cur_hp + block - incoming_damage  # (B,)

    # Key by name so the column order provably matches OBS_AUX_TARGETS (a new obs
    # column cannot silently land in the wrong slot). end_combat_hp is NOT here: it
    # is a future-outcome target the rollout buffer supplies at loss time.
    by_name = {"incoming_damage": incoming_damage, "survival_margin": survival_margin}
    targets = torch.stack([by_name[name] for name in OBS_AUX_TARGETS], dim=-1) / HP_SCALE
    combat_mask = enemy_alive.gt(0.0).any(dim=-1)  # (B,) bool
    return targets, combat_mask
