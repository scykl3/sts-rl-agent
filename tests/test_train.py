"""Tests for the PPO outer training loop.

Engine-free: every test drives :func:`~sts_rl.agent.train.train` on
:class:`~sts_rl.env.stub_env.StubEnv` (no native engine build required). The
suite locks the loop's composition and behaviour: history/step accounting, the
config guards, run-to-run determinism (trained-param checksum and first-iteration
return), diagnostic finiteness/sanity, that the optimizer actually moves weights,
one structured log line per iteration, and that the loop never references a
concrete env. Observation/action widths are pulled from the interface, never
hardcoded.

Learning signal. :func:`test_critic_value_loss_decreases` exercises the
value-learning path end to end - rollout returns -> value head -> value loss ->
backprop -> optimizer step - and asserts the optimizer reduces that real
regression objective while the loss stays finite. It proves the machinery is
wired and optimizes, NOT that the encoder learns useful features: the single-step
bandit return is a near-constant (the terminal win/loss bonus is added only in the
random reward mode, and a sub-optimal policy's P(correct) is cue-independent), so
the value head's bias alone can fit it and the loss drop implies nothing about the
trunk's representation.
The policy's episode return is deliberately NOT asserted to climb on this toy:
the reward-optimal action is a function of one low-variance cue
(``player_scalars[1]``) buried in a mostly-noise observation, and a shared-trunk
policy-gradient learner cannot extract that cue within a unit-test budget
(direct supervised extraction of the same cue needs ~100k samples; the RL return
sits at the ~1/5 random baseline even given 100k env steps). Genuine evidence
that the encoder learns useful features is deferred to the engine-env phase.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import math

import pytest
import torch

import sts_rl.agent.train as train_module
from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.encoder import HIDDEN_DIM
from sts_rl.agent.ppo import DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA
from sts_rl.agent.ppo_update import PPOConfig
from sts_rl.agent.train import (
    DEFAULT_LEARNING_RATE,
    IterationRecord,
    TrainConfig,
    TrainHistory,
    train,
)
from sts_rl.env.stub_env import StubEnv

# Small, fast, deterministic settings for the loop under test. HIDDEN is well
# below the interface default so a run is cheap; it is a training knob, not an
# interface width, so a literal is fine here.
SEED = 0
HIDDEN = 32
NUM_ITERATIONS = 5
N_STEPS = 128
LEARNING_RATE = 1e-2
CARD_REWARD_BLOCK = "CARD_REWARD_SELECT"

# Minibatch-step count per iteration is a function of the rollout length and the
# PPO schedule, recomputed rather than hardcoded so a default change propagates.
_PPO_DEFAULTS = PPOConfig()
EXPECTED_UPDATES = _PPO_DEFAULTS.n_epochs * math.ceil(N_STEPS / _PPO_DEFAULTS.minibatch_size)

# A valid baseline the helpers mutate via dataclasses.replace. ent_coef=0 keeps
# the toy policy from being pushed toward uniform, so runs stay deterministic and
# quick.
_BASE_CONFIG = TrainConfig(
    num_iterations=NUM_ITERATIONS,
    n_steps=N_STEPS,
    learning_rate=LEARNING_RATE,
    seed=SEED,
    hidden_dim=HIDDEN,
    ppo=PPOConfig(ent_coef=0.0),
)


def _task_env() -> StubEnv:
    """The exact task env: learnable reward gated on one CARD_REWARD_SELECT block."""
    return StubEnv(reward_mode="learnable", active_blocks=(CARD_REWARD_BLOCK,))


def _bandit_env() -> StubEnv:
    """Single-step variant (``terminate_prob=1.0``) of the task env.

    Every episode is one step, so ``mean_episode_return`` is the per-iteration
    fraction of correct actions and the critic target is the near-constant
    bandit value - the clean signal :func:`test_critic_value_loss_decreases`
    reads. ``max_episode_steps`` is set high so the step cap never truncates
    before the per-step terminate fires.
    """
    return StubEnv(
        reward_mode="learnable",
        active_blocks=(CARD_REWARD_BLOCK,),
        terminate_prob=1.0,
        max_episode_steps=10_000,
    )


def _config(**overrides: object) -> TrainConfig:
    """Build a fast :class:`TrainConfig`, overriding fields per test.

    Uses :func:`dataclasses.replace` on a valid base so overrides stay typed
    (``replace`` accepts arbitrary field changes) without restating every field.
    """
    return dataclasses.replace(_BASE_CONFIG, **overrides)


def _param_checksum(actor_critic: ActorCritic) -> float:
    """Order-stable scalar fingerprint of every parameter, for determinism checks."""
    return float(sum(p.detach().double().sum().item() for p in actor_critic.parameters()))


@pytest.fixture(scope="module")
def standard_history() -> TrainHistory:
    """One short shared run on the task env, reused by the read-only structural tests."""
    return train(_task_env(), _config())


def test_history_length_and_step_accounting(standard_history: TrainHistory) -> None:
    """One record per iteration; global_step is the running single-env step count."""
    records = standard_history.records
    assert len(records) == NUM_ITERATIONS
    assert len(standard_history.mean_episode_returns()) == NUM_ITERATIONS
    for i, record in enumerate(records):
        assert isinstance(record, IterationRecord)
        assert record.iteration == i
        # Single env: exactly n_steps environment steps per iteration.
        assert record.global_step == (i + 1) * N_STEPS
    assert records[-1].global_step == NUM_ITERATIONS * N_STEPS


def test_records_carry_collect_and_ppo_stats(standard_history: TrainHistory) -> None:
    """Each record preserves the whole CollectStats/PPOStats blocks, unflattened."""
    for record in standard_history.records:
        assert record.collect.n_steps == N_STEPS
        # n_updates == n_epochs * ceil(n_steps / minibatch_size); recomputed above.
        assert record.ppo.n_updates == EXPECTED_UPDATES


def test_diagnostics_finite_and_sane(standard_history: TrainHistory) -> None:
    """Every per-iteration diagnostic is finite and within its valid range."""
    saw_episode = False
    for record in standard_history.records:
        ppo = record.ppo
        for value in (
            ppo.policy_loss,
            ppo.value_loss,
            ppo.entropy,
            ppo.total_loss,
            ppo.approx_kl,
            ppo.clip_fraction,
            ppo.grad_norm,
        ):
            assert math.isfinite(value)
        assert ppo.entropy >= 0.0
        assert ppo.approx_kl >= 0.0  # Schulman's k3 estimator is non-negative
        assert ppo.grad_norm >= 0.0
        assert 0.0 <= ppo.clip_fraction <= 1.0
        # Finite (not just positive): the collector floors a zero-elapsed collect
        # to inf, which would slip past a bare > 0 check.
        sps = record.collect.steps_per_second
        assert math.isfinite(sps) and sps > 0.0
        if record.collect.n_episodes > 0:
            saw_episode = True
            assert record.collect.mean_episode_return is not None
            assert record.collect.mean_episode_length is not None
    # The short run must complete at least one episode, or the return signal is vacuous.
    assert saw_episode


def test_parameters_change_after_training(standard_history: TrainHistory) -> None:
    """Training moves weights: the trained net differs from its seeded init.

    ``train`` seeds the global RNG then constructs the network, so reseeding with
    the same seed and rebuilding reproduces the exact initial weights; any
    difference afterward is the optimizer's doing.
    """
    torch.manual_seed(SEED)
    initial = ActorCritic(hidden_dim=HIDDEN)
    assert _param_checksum(initial) != _param_checksum(standard_history.actor_critic)


def test_determinism_same_seed_reproduces_run() -> None:
    """Two identically configured runs match on trained params and first-iter return."""
    config = _config(num_iterations=3, n_steps=N_STEPS)
    first = train(_bandit_env(), config)
    second = train(_bandit_env(), config)
    assert _param_checksum(first.actor_critic) == _param_checksum(second.actor_critic)
    assert first.mean_episode_returns()[0] == second.mean_episode_returns()[0]


def test_different_seed_changes_run() -> None:
    """A different seed yields a different trained network (seeding actually plumbs through)."""
    base = train(_bandit_env(), _config(num_iterations=3, seed=SEED))
    other = train(_bandit_env(), _config(num_iterations=3, seed=SEED + 1))
    assert _param_checksum(base.actor_critic) != _param_checksum(other.actor_critic)


def test_different_gamma_changes_run() -> None:
    """TrainConfig.gamma reaches GAE end to end: a different gamma retrains differently.

    Same seed/env/config but a different ``gamma`` changes the discounted GAE
    returns -> different value targets and advantages -> different updates ->
    different trained params. Guards the train->collector gamma wire end to end:
    dropping ``gamma=config.gamma`` from the ``RolloutCollector`` call silently
    falls back to ``DEFAULT_GAMMA`` for both runs, collapsing the checksums to
    equal and failing this assertion (revert-verified). Uses the multi-step task
    env, not the single-step bandit where every transition is terminal and gamma
    cancels out of the returns.
    """
    # 0.5 is far from the 0.99 default, so the discounted returns differ enough
    # to move the trained params observably.
    base = train(_task_env(), _config(num_iterations=3, gamma=DEFAULT_GAMMA))
    other = train(_task_env(), _config(num_iterations=3, gamma=0.5))
    assert _param_checksum(base.actor_critic) != _param_checksum(other.actor_critic)


@pytest.mark.parametrize(
    "field, value",
    [
        ("num_iterations", 0),
        ("num_iterations", -1),
        ("n_steps", 0),
        ("n_steps", -4),
        ("learning_rate", 0.0),
        ("learning_rate", -1e-3),
        ("hidden_dim", 0),
        ("hidden_dim", -8),
    ],
)
def test_config_rejects_non_positive(field: str, value: object) -> None:
    """TrainConfig fails fast on a degenerate budget rather than running an empty loop."""
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(_BASE_CONFIG, **{field: value})


def test_config_defaults_are_symbolic() -> None:
    """Defaults come from the shared constants, not re-declared literals."""
    config = TrainConfig(num_iterations=1, n_steps=1)
    assert config.hidden_dim == HIDDEN_DIM
    assert config.learning_rate == DEFAULT_LEARNING_RATE
    assert config.gamma == DEFAULT_GAMMA
    assert config.gae_lambda == DEFAULT_GAE_LAMBDA
    assert config.anneal_lr is False  # constant LR is the default first cut
    assert config.ppo == PPOConfig()


def test_anneal_lr_decays_learning_rate() -> None:
    """anneal_lr=True keeps iteration 0 at the full LR and decays it monotonically.

    CleanRL linear schedule: frac = 1 - iteration/num_iterations, so iteration 0
    runs at the full configured LR and every later iteration is strictly lower.
    ``IterationRecord.learning_rate`` is the rate actually applied that iteration,
    which is what makes the schedule observable/testable.

    Revert-verify: without the anneal branch every iteration keeps the constant
    LR, so records[-1] == records[0] and the strict-decay assertion fails.
    """
    config = _config(anneal_lr=True)
    history = train(_task_env(), config)
    lrs = [record.learning_rate for record in history.records]
    # Iteration 0 -> full LR (frac = 1 - 0/N = 1).
    assert lrs[0] == config.learning_rate
    # The last iteration is strictly decayed below the first...
    assert lrs[-1] < lrs[0]
    # ...and the schedule is monotonically non-increasing across the run.
    assert all(later <= earlier for earlier, later in zip(lrs, lrs[1:]))


def test_constant_lr_when_annealing_disabled() -> None:
    """anneal_lr defaults False: the recorded LR is constant across every iteration."""
    config = _config()  # anneal_lr defaults False
    history = train(_task_env(), config)
    lrs = [record.learning_rate for record in history.records]
    assert all(lr == config.learning_rate for lr in lrs)


def test_logs_one_line_per_iteration(caplog: pytest.LogCaptureFixture) -> None:
    """Exactly one INFO diagnostic line is emitted per iteration (logging, not print)."""
    iterations = 3
    with caplog.at_level(logging.INFO, logger=train_module.__name__):
        train(_task_env(), _config(num_iterations=iterations, n_steps=N_STEPS))
    iteration_lines = [r for r in caplog.records if r.name == train_module.__name__]
    assert len(iteration_lines) == iterations
    assert all("global_step=" in r.getMessage() for r in iteration_lines)
    # The LR used and the explained variance are surfaced on every diagnostic line.
    assert all("lr=" in r.getMessage() for r in iteration_lines)
    assert all("explained_var=" in r.getMessage() for r in iteration_lines)


def test_no_episode_completes_reports_none_and_logs_na(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No episode ending in a collect keeps the return means None and logs 'n/a'.

    ``terminate_prob=0.0`` with a step cap far above ``n_steps`` means the single
    episode never ends inside an iteration, so every ``CollectStats`` reports
    ``n_episodes == 0``. This exercises the ``mean_episode_return is None`` branch
    of ``_log_iteration`` (rendered ``ep_return=n/a``) that the other tests, each
    completing an episode per iteration, never reach.
    """
    # terminate_prob=0.0 disables per-step termination and the high step cap keeps
    # a 4-step collect from truncating, so no episode boundary occurs.
    non_terminating_env = StubEnv(
        reward_mode="learnable",
        active_blocks=(CARD_REWARD_BLOCK,),
        terminate_prob=0.0,
        max_episode_steps=10_000,
    )
    with caplog.at_level(logging.INFO, logger=train_module.__name__):
        history = train(non_terminating_env, _config(num_iterations=3, n_steps=4))

    # No episode ended, so the per-iteration means stay None (guarded, not a fake 0).
    assert None in history.mean_episode_returns()
    assert all(record.collect.n_episodes == 0 for record in history.records)
    assert all(record.collect.mean_episode_length is None for record in history.records)

    # The None branch renders 'n/a' rather than crashing the %f-style log format.
    iteration_lines = [r for r in caplog.records if r.name == train_module.__name__]
    assert iteration_lines
    assert all("ep_return=n/a" in r.getMessage() for r in iteration_lines)
    assert all("ep_len=n/a" in r.getMessage() for r in iteration_lines)


def test_train_is_env_agnostic() -> None:
    """The loop never names a concrete env, so it drives stub and engine unchanged."""
    source = inspect.getsource(train_module)
    assert "StubEnv" not in source
    assert "stub_env" not in source


def test_critic_value_loss_decreases() -> None:
    """The critic learns the near-constant bandit value: value loss falls from init.

    In-budget proof that the value-learning path executes end to end and the
    optimizer reduces a real regression objective - rollout returns -> value head
    -> value loss -> backprop -> optimizer step, staying finite. It does NOT prove
    the encoder learns useful features: the bandit return is near-constant, so the
    value head's bias alone can fit it. Iteration 0's mean value loss (starting
    from random init) is the baseline; after training on the single-step bandit
    the tail drops well below it. Policy-return convergence is not
    asserted here - see the module docstring for why the buried-cue toy is not
    RL-learnable within a unit-test budget.
    """
    config = _config(num_iterations=16, n_steps=256, learning_rate=1e-2, seed=SEED)
    history = train(_bandit_env(), config)
    value_losses = [record.ppo.value_loss for record in history.records]
    assert all(math.isfinite(v) for v in value_losses)
    initial = value_losses[0]
    trained_tail = sum(value_losses[-5:]) / 5
    # Locked under this seed at ratio ~0.57; 0.8 leaves clear headroom so a minor
    # numeric perturbation (torch build) does not flip it.
    assert trained_tail < 0.8 * initial
