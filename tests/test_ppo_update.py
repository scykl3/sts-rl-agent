"""Tests for the PPO update pass.

Engine-free: a helper fills a :class:`RolloutBuffer` from one batched
``sample_observation_batch(T)`` call (per-step obs are the batch slices), a legal
prefix mask, actions sampled by a real :meth:`ActorCritic.act`, and its log-prob
/value squeezed to scalars, then runs GAE. The suite pins the minibatch-step
count, proves parameters move, checks stat sanity, both normalization paths, the
empty-buffer zero-division guard, and run-to-run determinism.
"""

from __future__ import annotations

import math

import pytest
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.ppo_update import PPOConfig, PPOStats, ppo_update
from sts_rl.agent.rollout_buffer import RolloutBuffer
from sts_rl.agent.running_moments import RunningMoments
from sts_rl.interface import ACTION_DIM
from conftest import sample_observation_batch

T = 16
MINIBATCH = 8
N_EPOCHS = 2
# Legal prefix width: mask[:K] True guarantees the sampled action lands in [0, K).
K = 5
EXPECTED_UPDATES = N_EPOCHS * math.ceil(T / MINIBATCH)  # 2 * 2 = 4


def _fill_buffer(actor_critic: ActorCritic, length: int = T) -> RolloutBuffer:
    """Add ``length`` transitions with real act() log-probs/values (as scalars)."""
    buffer = RolloutBuffer()
    full = sample_observation_batch(length)
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    mask[:K] = True
    for t in range(length):
        obs = {key: value[t] for key, value in full.items()}
        # act expects a batch dim; squeeze its (1,) outputs back to scalars.
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


def test_n_updates_matches_epochs_times_minibatches():
    """n_updates == n_epochs * ceil(T / minibatch_size); all stats finite."""
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)

    stats = ppo_update(ac, buffer, optimizer, config)
    assert isinstance(stats, PPOStats)
    assert stats.n_updates == EXPECTED_UPDATES
    for field in (
        stats.policy_loss,
        stats.value_loss,
        stats.entropy,
        stats.total_loss,
        stats.approx_kl,
        stats.clip_fraction,
        stats.grad_norm,
        stats.explained_variance,
    ):
        assert math.isfinite(field)


def test_parameters_change_after_update():
    """A real optimizer step must move at least one encoder and/or head weight."""
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    encoder_before = ac.encoder.card_embed.weight.detach().clone()
    head_before = ac.value.value.weight.detach().clone()

    ppo_update(ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH))

    encoder_changed = not torch.equal(encoder_before, ac.encoder.card_embed.weight)
    head_changed = not torch.equal(head_before, ac.value.value.weight)
    assert encoder_changed or head_changed


def test_stats_are_sane():
    """entropy >= 0 and clip_fraction in [0, 1]; losses finite."""
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    stats = ppo_update(
        ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)
    )
    assert stats.entropy >= 0.0
    assert 0.0 <= stats.clip_fraction <= 1.0
    assert stats.approx_kl >= 0.0  # Schulman's k3 estimator is non-negative
    assert stats.grad_norm >= 0.0
    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.value_loss)


def test_both_normalization_paths_run():
    """normalize_advantages True and False both return finite stats."""
    for normalize in (True, False):
        ac = ActorCritic()
        buffer = _fill_buffer(ac)
        optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
        config = PPOConfig(
            n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, normalize_advantages=normalize
        )
        stats = ppo_update(ac, buffer, optimizer, config)
        assert stats.n_updates == EXPECTED_UPDATES
        assert math.isfinite(stats.total_loss)


def test_normalize_advantages_standardizes_against_full_batch():
    """decay=1.0 folds the whole rollout, so the tracker standardizes that batch.

    With adv_norm_decay=1.0 the running moments equal the full rollout's
    population mean/std, so normalizing that same rollout yields ~0 mean / ~1
    std - the observable that the running-stats path replaced the old
    per-minibatch mean/std.
    """
    torch.manual_seed(0)
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    adv = buffer.advantages.detach().clone()
    moments = RunningMoments(decay=1.0)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(
        n_epochs=1, minibatch_size=MINIBATCH, normalize_advantages=True, adv_norm_decay=1.0
    )

    ppo_update(ac, buffer, optimizer, config, adv_moments=moments)

    normalized = moments.normalize(adv)
    assert normalized.mean().item() == pytest.approx(0.0, abs=1e-4)
    assert normalized.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-3)


def test_running_moments_persist_across_calls():
    """A shared RunningMoments carries the EWMA forward across two update calls.

    The tracker is folded once per call with that call's full rollout, so after
    two calls its running mean equals the hand-computed two-batch EWMA (first
    call seeds, second blends with per-item retain) and differs from a fresh
    tracker that saw only the second rollout - proving persistence, not a reset.
    """
    torch.manual_seed(0)
    ac = ActorCritic()
    buffer1 = _fill_buffer(ac)
    buffer2 = _fill_buffer(ac)
    adv1 = buffer1.advantages.detach().clone()
    adv2 = buffer2.advantages.detach().clone()
    decay = 0.1
    shared = RunningMoments(decay=decay)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(
        n_epochs=1, minibatch_size=MINIBATCH, normalize_advantages=True, adv_norm_decay=decay
    )

    ppo_update(ac, buffer1, optimizer, config, adv_moments=shared)
    ppo_update(ac, buffer2, optimizer, config, adv_moments=shared)

    # Reference: call 1 seeds from buffer1, call 2 blends buffer2 with retain.
    expected = float(adv1.mean())
    retain = (1.0 - decay) ** adv2.numel()
    expected = retain * expected + (1.0 - retain) * float(adv2.mean())
    assert shared.mean == pytest.approx(expected)

    # A fresh tracker over only the second rollout differs -> the shared one
    # genuinely carried buffer1's contribution forward.
    fresh = RunningMoments(decay=decay)
    fresh.update(adv2)
    assert shared.mean != pytest.approx(fresh.mean)


def test_advantage_fold_is_once_per_call_not_per_epoch():
    """The rollout's advantages are folded into the tracker ONCE per call, not per epoch.

    Pre-seed a shared tracker with a distinct batch (a known prior mean), then run
    a single ppo_update with n_epochs=4. The rollout is folded once before the
    epochs, so the tracker's mean is a SINGLE EWMA blend of the prior and this
    rollout's mean with retain = (1 - decay) ** len(buffer). Folding once per
    epoch would compound retain n_epochs times (geometric), a materially different
    mean - so this pins the fold-once placement.

    Revert-verify: move ``moments.update(adv_batch.advantages)`` inside the epoch
    loop and the mean picks up retain ** n_epochs, failing the equality below.
    """
    torch.manual_seed(0)
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    decay = 0.1
    shared = RunningMoments(decay=decay)
    # Seed with a distinct batch so the prior mean is known and non-degenerate;
    # only that the tracker is initialized matters, not the exact values.
    shared.update(torch.tensor([2.0, -1.0, 4.0, 0.5]))
    prior_mean = shared.mean

    n_epochs = 4
    config = PPOConfig(
        n_epochs=n_epochs,
        minibatch_size=MINIBATCH,
        normalize_advantages=True,
        adv_norm_decay=decay,
    )
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    ppo_update(ac, buffer, optimizer, config, adv_moments=shared)

    # adv_mean over the SAME full-width unshuffled pass ppo_update folds; the
    # update only reads advantages, so recomputing here reproduces its fold input.
    (adv_batch,) = buffer.iter_minibatches(len(buffer), shuffle=False)
    adv_mean = float(adv_batch.advantages.mean())
    retain = (1.0 - decay) ** len(buffer)
    expected_once = retain * prior_mean + (1.0 - retain) * adv_mean

    # What a per-epoch fold would produce: the same rollout folded n_epochs times
    # compounds the retain geometrically.
    retain_per_epoch = retain**n_epochs
    expected_per_epoch = retain_per_epoch * prior_mean + (1.0 - retain_per_epoch) * adv_mean

    # The two hypotheses must be distinguishable or the test has no teeth.
    assert expected_once != pytest.approx(expected_per_epoch)
    # Folded once per call: the mean matches the single-blend reference, and NOT
    # the per-epoch reference (which is what the fold-in-loop bug would yield).
    assert shared.mean == pytest.approx(expected_once)
    assert shared.mean != pytest.approx(expected_per_epoch)


def test_empty_buffer_returns_zeroed_stats():
    """A buffer yielding zero minibatches returns zeroed stats, not a ZeroDivisionError.

    compute_advantages rejects an empty buffer, so we reproduce the smallest
    zero-minibatch case directly: an empty buffer whose advantages/returns are
    set (mirroring iter_minibatches' own length==0 early return, which yields
    nothing). ppo_update must then take the n_updates == 0 guard.
    """
    ac = ActorCritic()
    buffer = RolloutBuffer()
    buffer.advantages = torch.zeros(0)
    buffer.returns = torch.zeros(0)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    stats = ppo_update(
        ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)
    )
    assert stats.n_updates == 0
    assert stats.policy_loss == 0.0
    assert stats.value_loss == 0.0
    assert stats.total_loss == 0.0


def test_update_is_deterministic_under_fixed_seed():
    """Two identically seeded runs produce equal stats (model, buffer, and shuffle)."""

    def run() -> PPOStats:
        torch.manual_seed(1234)
        ac = ActorCritic()
        buffer = _fill_buffer(ac)
        optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
        return ppo_update(
            ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)
        )

    first = run()
    second = run()
    assert first == second


def test_singleton_tail_minibatch_does_not_nan():
    """A size-1 remainder minibatch must not NaN advantage normalization.

    With T % minibatch_size == 1 the final minibatch holds one transition. The
    numel() > 0 guard normalizes that singleton against the persistent running
    stats (population moments, N=1-safe: no Bessel term to divide by N-1), so it
    is standardized like its peers and the update stays finite. length=9, mb=4 ->
    sizes [4, 4, 1].
    """
    ac = ActorCritic()
    buffer = _fill_buffer(ac, length=9)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=1, minibatch_size=4, normalize_advantages=True)
    stats = ppo_update(ac, buffer, optimizer, config)
    assert stats.n_updates == 3  # [4, 4, 1]
    assert math.isfinite(stats.total_loss)
    assert math.isfinite(stats.policy_loss)
    # The update must leave the WHOLE network finite, not NaN-poisoned.
    assert all(torch.isfinite(p).all() for p in ac.parameters())


def test_target_kl_early_stops_after_first_epoch():
    """A near-zero target_kl trips the end-of-epoch guard, halting after epoch 1.

    The k3 approx_kl is >= 0 and strictly positive once an optimizer step has
    moved the policy (the second minibatch onward), so epoch 1's mean exceeds
    target_kl=0.0 and the epoch loop breaks before epoch 2. Only one epoch's
    minibatches run: n_updates == ceil(T / MINIBATCH) < EXPECTED_UPDATES.

    Revert-verify: delete the early-stop break in ppo_update and every epoch
    runs, so n_updates == EXPECTED_UPDATES and this assertion fails.
    """
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, target_kl=0.0)

    stats = ppo_update(ac, buffer, optimizer, config)

    one_epoch_updates = math.ceil(T / MINIBATCH)
    assert stats.n_updates == one_epoch_updates
    assert stats.n_updates < EXPECTED_UPDATES


def test_target_kl_none_runs_all_epochs():
    """target_kl=None (the default) disables the guard: every epoch runs."""
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, target_kl=None)

    stats = ppo_update(ac, buffer, optimizer, config)
    assert stats.n_updates == EXPECTED_UPDATES


def test_high_target_kl_never_triggers():
    """A target_kl far above any realistic approx_kl never trips: all epochs run."""
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)
    config = PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH, target_kl=1e9)

    stats = ppo_update(ac, buffer, optimizer, config)
    assert stats.n_updates == EXPECTED_UPDATES


def test_explained_variance_is_one_when_values_match_returns():
    """old_values == returns over the full rollout -> explained_variance == 1.0.

    EV reads the buffer's stored collection-time values (``_values``, no public
    setter) and GAE returns, so overwrite both to identical real-variance arrays:
    the residual variance is 0 and EV is exactly 1.0 regardless of the update.
    """
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    target = torch.arange(T, dtype=torch.float32)  # genuine variance in y_true
    buffer.returns = target.clone()
    buffer._values = [target[t].clone() for t in range(T)]
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    stats = ppo_update(
        ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)
    )
    assert math.isfinite(stats.explained_variance)
    assert stats.explained_variance == 1.0


def test_explained_variance_is_finite_zero_on_constant_returns():
    """All-equal returns (Var == 0) -> explained_variance is a finite 0.0, not nan.

    On a constant-return target the EV ratio divides by ~0; SB3/CleanRL yield nan
    there. The EXPLAINED_VAR_EPS guard clamps it to 0.0 so a near-constant-return
    stub does not break training-loop finite-value assertions.

    Revert-verify: remove the ``var_y < EXPLAINED_VAR_EPS`` guard in
    ``_explained_variance`` and EV becomes 0/0 -> nan, failing isfinite below.
    """
    ac = ActorCritic()
    buffer = _fill_buffer(ac)
    buffer.returns = torch.full((T,), 3.0)  # zero-variance y_true
    buffer._values = [torch.tensor(1.0) for _ in range(T)]
    optimizer = torch.optim.Adam(ac.parameters(), lr=1e-3)

    stats = ppo_update(
        ac, buffer, optimizer, PPOConfig(n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)
    )
    assert math.isfinite(stats.explained_variance)
    assert stats.explained_variance == 0.0
