"""Tests for the optional auxiliary-perception head and its masked PPO loss.

Engine-free: observations come from the interface's own space via the shared
``sample_observation_batch`` helper, and the buffer harness mirrors
``test_ppo_update``'s (real ``act`` log-probs/values squeezed to scalars, then
GAE). The suite covers the target math (on a hand-built obs), the interface-index
lock, the aux head's shape and presence, gradient flow, the empty-combat guard,
and - most importantly - that with ``aux_coef == 0`` the aux plumbing is a true
no-op: the default net has no aux params and a PPO step matches the aux-free
baseline exactly.
"""

from __future__ import annotations

import math

import pytest
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.aux_heads import (
    AUX_TARGETS,
    END_COMBAT_HP_COLUMN,
    ENEMY_INTENT_HITS_INDEX,
    ENEMY_INTENT_VAL_INDEX,
    HP_SCALE,
    OBS_AUX_TARGETS,
    PLAYER_BLOCK_INDEX,
    PLAYER_CUR_HP_INDEX,
    compute_aux_targets,
)
from sts_rl.agent.ppo_update import PPOConfig, ppo_update
from sts_rl.agent.rollout_buffer import RolloutBuffer
from sts_rl.interface import ACTION_DIM, ENEMY_SCALAR_DIM, PLAYER_SCALAR_DIM
from conftest import sample_observation_batch

BATCH = 4
T = 16
MINIBATCH = 8
N_EPOCHS = 2
# Legal prefix width: mask[:, :K] True guarantees >=1 legal action per row.
K = 5


def _legal_prefix_mask(batch: int = BATCH, k: int = K) -> torch.Tensor:
    """Bool mask (B, ACTION_DIM) with the first ``k`` actions legal in every row."""
    mask = torch.zeros(batch, ACTION_DIM, dtype=torch.bool)
    mask[:, :k] = True
    return mask


def _fill_buffer(
    actor_critic: ActorCritic,
    length: int = T,
    *,
    all_overworld: bool = False,
    force_combat: bool = False,
) -> RolloutBuffer:
    """Fill a buffer with real act() log-probs/values (as scalars); mirrors test_ppo_update.

    ``all_overworld`` zeroes ``enemy_alive`` in every stored obs (no live enemy, so
    the aux loss hits its empty-combat guard); ``force_combat`` marks slot 0 alive
    in every step (so the masked aux loss is always non-empty). The two are
    mutually exclusive.
    """
    buffer = RolloutBuffer()
    full = sample_observation_batch(length)
    full["enemy_alive"] = full["enemy_alive"].float()
    if all_overworld:
        full["enemy_alive"] = torch.zeros_like(full["enemy_alive"])
    if force_combat:
        full["enemy_alive"][:, 0] = 1.0
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    mask[:K] = True
    for t in range(length):
        obs = {key: value[t] for key, value in full.items()}
        batched_obs = {key: value.unsqueeze(0) for key, value in obs.items()}
        action, log_prob, _, value = actor_critic.act(batched_obs, mask.unsqueeze(0))
        buffer.add(
            obs=obs,
            action=action.squeeze(0),
            log_prob=log_prob.squeeze(0),
            value=value.squeeze(0),
            reward=float(torch.randn(())),
            done=1.0 if t == length - 1 else 0.0,
            mask=mask,
        )
    buffer.compute_advantages(torch.zeros(()))
    return buffer


# --- Target math + interface lock ------------------------------------------


def test_aux_target_columns():
    """AUX_TARGETS pins the column set and order: obs-derivable pair, then end_combat_hp.

    compute_aux_targets produces only the OBS_AUX_TARGETS columns; end_combat_hp is
    the collector-backfilled column appended last, at END_COMBAT_HP_COLUMN.
    """
    assert OBS_AUX_TARGETS == ("incoming_damage", "survival_margin")
    assert AUX_TARGETS == ("incoming_damage", "survival_margin", "end_combat_hp")
    assert AUX_TARGETS[END_COMBAT_HP_COLUMN] == "end_combat_hp"
    assert END_COMBAT_HP_COLUMN == len(OBS_AUX_TARGETS)


def test_scalar_column_indices_match_interface_layout():
    """Pin the mirrored scalar-column indices to the documented interface layout.

    A reorder/shrink of enemy_scalars / player_scalars in interface.py must be a
    deliberate, reviewed edit here (the width guard in aux_heads fires on a
    shrink; this pins the exact columns). The columns' MEANING is additionally
    locked functionally by test_compute_aux_targets_on_hand_built_obs.
    """
    assert (ENEMY_INTENT_VAL_INDEX, ENEMY_INTENT_HITS_INDEX) == (3, 4)
    assert (PLAYER_CUR_HP_INDEX, PLAYER_BLOCK_INDEX) == (0, 2)
    assert ENEMY_INTENT_VAL_INDEX < ENEMY_SCALAR_DIM
    assert ENEMY_INTENT_HITS_INDEX < ENEMY_SCALAR_DIM
    assert PLAYER_CUR_HP_INDEX < PLAYER_SCALAR_DIM
    assert PLAYER_BLOCK_INDEX < PLAYER_SCALAR_DIM


def test_compute_aux_targets_on_hand_built_obs():
    """Hand-built obs -> incoming/margin/mask computed by hand, incl. the dead-enemy gate.

    Row 0 (combat): two live enemies, intents 6x2 and 5x1 -> incoming 17; cur_hp
    50 + block 8 -> margin 41; combat_mask True. Row 1 (no live enemy): a DEAD
    enemy carries a large stale intent (99x9) that must be gated OUT by
    enemy_alive -> incoming 0; cur_hp 30 -> margin 30; combat_mask False. Both
    targets are divided by HP_SCALE.
    """
    obs = sample_observation_batch(2)
    obs["enemy_scalars"] = torch.zeros_like(obs["enemy_scalars"].float())
    obs["enemy_alive"] = torch.zeros_like(obs["enemy_alive"].float())
    obs["player_scalars"] = torch.zeros_like(obs["player_scalars"].float())

    # Row 0: two live enemies with committed intents.
    obs["enemy_alive"][0, 0] = 1.0
    obs["enemy_alive"][0, 1] = 1.0
    obs["enemy_scalars"][0, 0, ENEMY_INTENT_VAL_INDEX] = 6.0
    obs["enemy_scalars"][0, 0, ENEMY_INTENT_HITS_INDEX] = 2.0
    obs["enemy_scalars"][0, 1, ENEMY_INTENT_VAL_INDEX] = 5.0
    obs["enemy_scalars"][0, 1, ENEMY_INTENT_HITS_INDEX] = 1.0
    obs["player_scalars"][0, PLAYER_CUR_HP_INDEX] = 50.0
    obs["player_scalars"][0, PLAYER_BLOCK_INDEX] = 8.0

    # Row 1: a DEAD enemy (alive 0) with a large stale intent that must NOT count.
    obs["enemy_scalars"][1, 0, ENEMY_INTENT_VAL_INDEX] = 99.0
    obs["enemy_scalars"][1, 0, ENEMY_INTENT_HITS_INDEX] = 9.0
    obs["player_scalars"][1, PLAYER_CUR_HP_INDEX] = 30.0

    targets, combat_mask = compute_aux_targets(obs)

    assert combat_mask.dtype == torch.bool
    assert combat_mask.tolist() == [True, False]
    # Row 0: incoming 17, margin 41 (normalized).
    assert targets[0, 0].item() == pytest.approx(17.0 / HP_SCALE)
    assert targets[0, 1].item() == pytest.approx(41.0 / HP_SCALE)
    # Row 1: dead enemy gated out -> incoming 0, margin == cur_hp.
    assert targets[1, 0].item() == pytest.approx(0.0)
    assert targets[1, 1].item() == pytest.approx(30.0 / HP_SCALE)


def test_compute_aux_targets_shapes_from_sampled_obs():
    """Driven off the sampled space: (B, len(OBS_AUX_TARGETS)) targets, (B,) bool mask.

    compute_aux_targets emits ONLY the obs-derivable columns; the full AUX_TARGETS is
    wider by the collector-provided end_combat_hp column (assembled in ppo_update).
    """
    obs = sample_observation_batch(BATCH)
    targets, combat_mask = compute_aux_targets(obs)
    assert targets.shape == (BATCH, len(OBS_AUX_TARGETS))
    assert combat_mask.shape == (BATCH,)
    assert combat_mask.dtype == torch.bool
    assert torch.isfinite(targets).all()


# --- Aux head shape / presence ---------------------------------------------


def test_aux_head_output_shape_when_enabled():
    """evaluate_actions returns aux_pred of shape (B, len(AUX_TARGETS)) when aux is on."""
    ac = ActorCritic(aux_targets=AUX_TARGETS)
    obs = sample_observation_batch(BATCH)
    actions = torch.zeros(BATCH, dtype=torch.long)
    _lp, _e, _v, aux_pred = ac.evaluate_actions(obs, _legal_prefix_mask(), actions)
    assert aux_pred is not None
    assert aux_pred.shape == (BATCH, len(AUX_TARGETS))
    assert torch.isfinite(aux_pred).all()


def test_evaluate_actions_aux_pred_presence_tracks_aux_targets():
    """4-tuple always; aux_pred is None without an aux head, a tensor with one."""
    obs = sample_observation_batch(BATCH)
    mask = _legal_prefix_mask()
    actions = torch.zeros(BATCH, dtype=torch.long)

    disabled = ActorCritic().evaluate_actions(obs, mask, actions)
    assert len(disabled) == 4
    assert disabled[3] is None

    enabled = ActorCritic(aux_targets=AUX_TARGETS).evaluate_actions(obs, mask, actions)
    assert len(enabled) == 4
    assert enabled[3] is not None
    assert enabled[3].shape == (BATCH, len(AUX_TARGETS))


def test_aux_loss_gradient_reaches_head_and_encoder():
    """A masked aux MSE backprops into the aux head AND the shared encoder trunk.

    Forces a live enemy so the masked loss is non-empty, then checks grads reach
    both the aux head and the CLS input MLP (directly upstream of the pooled
    context the aux head reads).
    """
    ac = ActorCritic(aux_targets=AUX_TARGETS)
    obs = sample_observation_batch(BATCH)
    obs["enemy_alive"] = obs["enemy_alive"].float()
    obs["enemy_alive"][0, 0] = 1.0  # guarantee at least one combat step
    actions = torch.zeros(BATCH, dtype=torch.long)
    _lp, _e, _v, aux_pred = ac.evaluate_actions(obs, _legal_prefix_mask(), actions)
    assert aux_pred is not None

    # Assemble the full-width target ppo_update builds: obs columns + the (fabricated
    # here) end_combat_hp column, both HP_SCALE-normalized, so aux_pred (all columns)
    # matches the target width.
    obs_targets, combat_mask = compute_aux_targets(obs)
    end_hp_col = torch.full((BATCH, 1), 30.0) / HP_SCALE
    targets = torch.cat([obs_targets, end_hp_col], dim=-1)
    aux_loss = ((aux_pred[combat_mask] - targets[combat_mask]) ** 2).mean()
    aux_loss.backward()

    assert ac.aux_head is not None
    assert ac.aux_head.weight.grad is not None
    assert torch.isfinite(ac.aux_head.weight.grad).all()
    # Grad flows into the shared trunk (the CLS input MLP feeds the pooled context).
    cls_grad = ac.encoder.cls_mlp[0].weight.grad
    assert cls_grad is not None
    assert torch.isfinite(cls_grad).all()


# --- PPO update integration ------------------------------------------------


def test_aux_coef_positive_trains_aux_head_and_reports_loss():
    """aux_coef > 0 over combat steps: aux_loss is finite/positive and the head moves."""
    torch.manual_seed(0)
    ac = ActorCritic(aux_targets=AUX_TARGETS)
    buffer = _fill_buffer(ac, force_combat=True)
    assert ac.aux_head is not None
    aux_before = ac.aux_head.weight.detach().clone()

    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, aux_coef=0.5)
    stats = ppo_update(ac, buffer, optimizer, config)

    assert math.isfinite(stats.aux_loss)
    assert stats.aux_loss > 0.0
    assert not torch.equal(aux_before, ac.aux_head.weight)


def test_masked_aux_loss_empty_combat_guard_is_zero_and_finite():
    """An all-overworld minibatch (no live enemy) yields aux_loss 0 with no NaN.

    combat_mask.sum() == 0 takes the guard (a masked mean over zero rows would be
    NaN), so aux_loss is exactly 0, total_loss stays finite, and no parameter is
    NaN-poisoned.
    """
    torch.manual_seed(0)
    ac = ActorCritic(aux_targets=AUX_TARGETS)
    buffer = _fill_buffer(ac, all_overworld=True)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, aux_coef=0.5)

    stats = ppo_update(ac, buffer, optimizer, config)

    assert stats.aux_loss == 0.0
    assert math.isfinite(stats.total_loss)
    assert all(torch.isfinite(p).all() for p in ac.parameters())


def test_ppo_update_trains_end_combat_hp_column_when_valid():
    """aux_coef>0 with valid end_combat_hp: the end_combat_hp head column gets a grad.

    force_combat makes the obs columns non-empty; backfilling the whole rollout makes
    the end_combat_hp column valid on every step. The head's end_combat_hp output row
    (END_COMBAT_HP_COLUMN) must then move, proving the collector-provided column is
    assembled into the target (normalized by HP_SCALE) and reaches the head + encoder.
    """
    torch.manual_seed(0)
    ac = ActorCritic(aux_targets=AUX_TARGETS)
    buffer = _fill_buffer(ac, force_combat=True)
    buffer.backfill_end_combat_hp(0, len(buffer), 40.0)  # every step: valid end HP
    assert ac.aux_head is not None
    ech_before = ac.aux_head.weight[END_COMBAT_HP_COLUMN].detach().clone()

    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, aux_coef=0.5)
    stats = ppo_update(ac, buffer, optimizer, config)

    assert math.isfinite(stats.aux_loss)
    assert stats.aux_loss > 0.0
    assert not torch.equal(ech_before, ac.aux_head.weight[END_COMBAT_HP_COLUMN])


def test_end_combat_hp_column_frozen_when_all_invalid():
    """Per-column mask: the end_combat_hp column contributes nothing when no step is valid.

    force_combat keeps the obs columns non-empty (they train), but with end_combat_hp
    never backfilled every step is invalid, so its per-column validity mask
    (end_combat_hp_valid, NOT combat_mask) takes the all-False guard and zeroes that
    column: its head row must NOT move while an obs-column row does. This is the guard
    that also keeps the NaN placeholder out of the loss (no NaN parameter poisoning).
    """
    torch.manual_seed(0)
    ac = ActorCritic(aux_targets=AUX_TARGETS)
    buffer = _fill_buffer(ac, force_combat=True)  # end_combat_hp left invalid
    assert ac.aux_head is not None
    ech_before = ac.aux_head.weight[END_COMBAT_HP_COLUMN].detach().clone()
    obs_col_before = ac.aux_head.weight[0].detach().clone()

    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, aux_coef=0.5)
    stats = ppo_update(ac, buffer, optimizer, config)

    # end_combat_hp column frozen (all steps invalid -> all-False guard, zero grad)...
    assert torch.equal(ech_before, ac.aux_head.weight[END_COMBAT_HP_COLUMN])
    # ...while an obs-derived column moved (combat_mask active via force_combat).
    assert not torch.equal(obs_col_before, ac.aux_head.weight[0])
    assert math.isfinite(stats.total_loss)
    assert all(torch.isfinite(p).all() for p in ac.parameters())


# --- No-op when disabled (the critical guarantee) --------------------------


def test_default_actor_critic_has_no_aux_params():
    """Empty aux_targets is a true no-op: no aux head, unchanged state_dict key set."""
    default = ActorCritic()
    empty = ActorCritic(aux_targets=())
    assert default.aux_head is None
    assert empty.aux_head is None
    assert set(default.state_dict()) == set(empty.state_dict())
    assert not any("aux" in key for key in default.state_dict())

    # Enabling aux ADDS exactly the aux head's params and nothing else.
    enabled = ActorCritic(aux_targets=AUX_TARGETS)
    added = set(enabled.state_dict()) - set(default.state_dict())
    assert added == {"aux_head.weight", "aux_head.bias"}


def test_aux_coef_zero_update_matches_no_aux_baseline():
    """aux_coef == 0 makes an aux-headed net's PPO step byte-identical to a no-aux net.

    Same shared weights (copied in) + same buffer + same shuffle seed: with the aux
    loss off, the aux head is never in the objective, so total_loss and every
    shared-param post-update value match the aux-free baseline exactly. The aux head
    itself never gets a grad (Adam and grad-clip skip a None grad), so it stays at
    its init. This is the core no-op guarantee: adding the aux plumbing does not
    perturb the default training path.
    """
    torch.manual_seed(0)
    baseline = ActorCritic()
    buffer = _fill_buffer(baseline, force_combat=True)

    candidate = ActorCritic(aux_targets=AUX_TARGETS)
    # Copy shared trunk/head weights; aux_head keeps its own init (unused at coef 0).
    # strict=False because baseline carries no aux keys.
    result = candidate.load_state_dict(baseline.state_dict(), strict=False)
    assert set(result.missing_keys) == {"aux_head.weight", "aux_head.bias"}
    assert result.unexpected_keys == []
    assert candidate.aux_head is not None
    aux_before = candidate.aux_head.weight.detach().clone()

    # Default PPOConfig has aux_coef 0.0; make the candidate's explicit for clarity.
    torch.manual_seed(123)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    stats_b = ppo_update(
        baseline, buffer, opt_b, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)
    )

    torch.manual_seed(123)
    opt_c = torch.optim.Adam(candidate.parameters(), lr=1e-3)
    stats_c = ppo_update(
        candidate,
        buffer,
        opt_c,
        PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, aux_coef=0.0),
    )

    assert stats_c.aux_loss == 0.0
    assert stats_c.total_loss == stats_b.total_loss  # exact: identical float ops
    # Every SHARED parameter updated identically; the aux head never moved.
    for (name_b, param_b), (name_c, param_c) in zip(
        baseline.named_parameters(), candidate.named_parameters()
    ):
        if name_c.startswith("aux_head"):
            continue
        assert name_b == name_c
        assert torch.equal(param_b, param_c)
    assert torch.equal(aux_before, candidate.aux_head.weight)
