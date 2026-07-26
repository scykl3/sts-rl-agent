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
from pathlib import Path

import pytest
import torch
from conftest import make_stub_vec_env

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
from sts_rl.eval import EvalReport
from sts_rl.interface import INTERFACE_VERSION

# Small, fast, deterministic settings for the loop under test. HIDDEN is well
# below the interface default so a run is cheap; it is a training knob, not an
# interface width, so a literal is fine here.
SEED = 0
HIDDEN = 32
NUM_ITERATIONS = 5
N_STEPS = 128
LEARNING_RATE = 1e-2
CARD_REWARD_BLOCK = "REWARD_SELECT"
# Tiny holdout for the eval/checkpoint tests: the bandit eval env terminates every
# episode in one step, so a handful of episodes keeps these tests fast.
EVAL_EPISODES = 4

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
    """The exact task env: learnable reward gated on one REWARD_SELECT block."""
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


@pytest.mark.parametrize(
    "field, value",
    [
        ("gamma", 1.5),
        ("gamma", -0.1),
        ("gae_lambda", 1.5),
        ("gae_lambda", -0.1),
    ],
)
def test_config_rejects_out_of_range_discount(field: str, value: float) -> None:
    """gamma/gae_lambda are range-bound to [0, 1]; outside it silently corrupts GAE.

    Revert-verify: drop the two __post_init__ range guards and this fails -
    gamma=1.5 constructs without raising.
    """
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(_BASE_CONFIG, **{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("gamma", 0.0),
        ("gamma", 1.0),
        ("gae_lambda", 0.0),
        ("gae_lambda", 1.0),
    ],
)
def test_config_accepts_inclusive_discount_endpoints(field: str, value: float) -> None:
    """The [0, 1] endpoints are valid: gamma=1.0 (undiscounted) and gae_lambda in
    {0, 1} (TD(0) / Monte Carlo) are legitimate, so construction must not reject them."""
    config = TrainConfig(num_iterations=1, n_steps=1, **{field: value})
    assert getattr(config, field) == value


@pytest.mark.parametrize("value", [0, -1, -8])
def test_config_rejects_non_positive_eval_every(value: int) -> None:
    """eval_every, when set, must be a positive cadence; None (disabled) is fine.

    Revert-verify: drop the __post_init__ eval_every guard and this fails -
    eval_every=0 constructs without raising (and would make the loop's modulo
    cadence a divide-by-zero).
    """
    with pytest.raises(ValueError, match="eval_every"):
        dataclasses.replace(_BASE_CONFIG, eval_every=value)


def test_config_defaults_are_symbolic() -> None:
    """Defaults come from the shared constants, not re-declared literals."""
    config = TrainConfig(num_iterations=1, n_steps=1)
    assert config.hidden_dim == HIDDEN_DIM
    assert config.learning_rate == DEFAULT_LEARNING_RATE
    assert config.gamma == DEFAULT_GAMMA
    assert config.gae_lambda == DEFAULT_GAE_LAMBDA
    assert config.anneal_lr is False  # constant LR is the default first cut
    assert config.num_envs == 1  # single-env path by default (opt-in vectorization)
    assert config.ppo == PPOConfig()
    # Eval + checkpointing default OFF, so an unset config is behaviour-identical
    # to the pre-feature loop; the band defaults come from the module constants.
    assert config.eval_every is None
    assert config.checkpoint_dir is None
    assert config.eval_episodes == train_module.DEFAULT_EVAL_EPISODES
    assert config.eval_seed_base == train_module.DEFAULT_EVAL_SEED_BASE


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


def _expected_eval_iterations(num_iterations: int, eval_every: int) -> list[int]:
    """Iterations that trigger an eval: the cadence hits plus the final iteration.

    Mirrors train()'s predicate symbolically so the cadence assertions never
    hardcode a count.
    """
    return [
        i for i in range(num_iterations) if (i + 1) % eval_every == 0 or (i + 1) == num_iterations
    ]


@pytest.mark.parametrize("eval_every", [1, 2, 3, 5])
def test_periodic_eval_populates_eval_reports(eval_every: int) -> None:
    """eval_every + a separate eval_env records one EvalReport per eval iteration.

    Locks the cadence (every eval_every iterations plus always the final one, so
    eval_every > num_iterations still evaluates at least once) and that each
    report is a well-formed win rate over the configured band. The eval env is a
    SEPARATE instance from the training env, per train()'s contract.
    """
    num_iterations = 3
    config = _config(
        num_iterations=num_iterations,
        eval_every=eval_every,
        eval_episodes=EVAL_EPISODES,
    )
    history = train(_bandit_env(), config, eval_env=_bandit_env())

    expected_iters = _expected_eval_iterations(num_iterations, eval_every)
    assert expected_iters  # every parametrization evaluates at least once
    assert [rec.iteration for rec in history.eval_reports] == expected_iters
    for rec in history.eval_reports:
        assert 0.0 <= rec.report.win_rate <= 1.0
        assert rec.report.n_episodes == EVAL_EPISODES
        # global_step is the single-env running step count at the eval iteration.
        assert rec.global_step == (rec.iteration + 1) * config.n_steps


def test_periodic_eval_requires_separate_eval_env() -> None:
    """eval_every set without an eval_env raises: eval must not share the train env.

    Revert-verify: drop the eval_env guard in train() and this no longer raises
    (evaluate() would then reset the training env mid-run and corrupt the
    collector's rollout stream).
    """
    config = _config(num_iterations=2, eval_every=1, eval_episodes=EVAL_EPISODES)
    with pytest.raises(ValueError, match="eval_env"):
        train(_bandit_env(), config)  # no eval_env passed


def test_periodic_eval_does_not_perturb_training() -> None:
    """Periodic eval is side-effect-free on training: eval-on and eval-off match.

    Greedy (deterministic) eval takes the masked argmax and draws no samples, so
    running periodic eval must not consume the training torch RNG - the trained
    params are bit-identical with eval ON and OFF given the same seed/config. Locks
    the RNG-safety property that makes eval a pure readout. eval ON uses a SEPARATE
    eval_env (per train()'s contract); checkpoint_dir is omitted so nothing is
    written.

    Revert-verify: switch the periodic evaluate() call to deterministic=False and
    the sampling consumes the shared torch RNG, diverging the two checksums and
    failing this.
    """
    eval_on = train(
        _bandit_env(),
        _config(num_iterations=3, eval_every=1, eval_episodes=EVAL_EPISODES),
        eval_env=_bandit_env(),
    )
    eval_off = train(_bandit_env(), _config(num_iterations=3))
    # Equal => periodic eval did not perturb the training RNG stream.
    assert _param_checksum(eval_on.actor_critic) == _param_checksum(eval_off.actor_critic)


def test_checkpointing_writes_best_and_last(tmp_path: Path) -> None:
    """checkpoint_dir + eval writes reloadable best.pt/last.pt, atomically.

    Asserts both files exist, no *.tmp is left behind (atomic temp->replace), the
    payload carries the state_dict + hidden_dim + interface_version + iteration +
    win_rate schema, a fresh ActorCritic rebuilt from the SAVED hidden_dim can load
    the saved state_dict, best.pt records the max eval win rate, and last.pt records
    the final iteration's eval win rate. Loading with weights_only=True also proves
    we saved a state_dict, not a pickled module.
    """
    num_iterations = 3
    config = _config(
        num_iterations=num_iterations,
        eval_every=1,
        eval_episodes=EVAL_EPISODES,
        checkpoint_dir=str(tmp_path),
    )
    history = train(_bandit_env(), config, eval_env=_bandit_env())

    best_path = tmp_path / train_module.BEST_CHECKPOINT_NAME
    last_path = tmp_path / train_module.LAST_CHECKPOINT_NAME
    assert best_path.exists()
    assert last_path.exists()
    # Atomic write leaves no temp behind on the success path.
    assert list(tmp_path.glob("*" + train_module.CHECKPOINT_TMP_SUFFIX)) == []

    best_ckpt = torch.load(best_path, weights_only=True)
    assert train_module.CHECKPOINT_MODEL_KEY in best_ckpt
    assert train_module.CHECKPOINT_WIN_RATE_KEY in best_ckpt
    assert train_module.CHECKPOINT_ITERATION_KEY in best_ckpt
    # Self-describing: the checkpoint records the trunk width and interface
    # provenance, so a reload needs no external --hidden-dim.
    assert best_ckpt[train_module.CHECKPOINT_HIDDEN_DIM_KEY] == HIDDEN
    assert best_ckpt[train_module.CHECKPOINT_INTERFACE_VERSION_KEY] == INTERFACE_VERSION

    # A fresh network rebuilt from the SAVED hidden_dim (not a hardcoded literal)
    # loads the saved state_dict without error - the checkpoint reloads itself
    # (state_dict, not a pickled module).
    fresh = ActorCritic(hidden_dim=best_ckpt[train_module.CHECKPOINT_HIDDEN_DIM_KEY])
    fresh.load_state_dict(best_ckpt[train_module.CHECKPOINT_MODEL_KEY])

    # best.pt records the max win rate across the eval curve; TrainHistory.best_eval
    # agrees with the saved best.
    max_win = max(rec.report.win_rate for rec in history.eval_reports)
    best_eval = history.best_eval()
    assert best_eval is not None
    assert best_ckpt[train_module.CHECKPOINT_WIN_RATE_KEY] == max_win
    assert best_ckpt[train_module.CHECKPOINT_WIN_RATE_KEY] == best_eval.report.win_rate

    # last.pt captures the final iteration's weights and its eval win rate.
    last_ckpt = torch.load(last_path, weights_only=True)
    assert last_ckpt[train_module.CHECKPOINT_ITERATION_KEY] == num_iterations - 1
    assert (
        last_ckpt[train_module.CHECKPOINT_WIN_RATE_KEY] == history.eval_reports[-1].report.win_rate
    )


@pytest.mark.parametrize(
    "win_seq",
    [
        [0.1, 0.5, 0.3],  # unique peak mid-run: best is the peak iteration, not the last
        [0.5, 0.5],  # tie at the max: first-maximal (earliest) iteration keeps best.pt
    ],
)
def test_best_checkpoint_tracks_increasing_win_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, win_seq: list[float]
) -> None:
    """best.pt holds the max eval win rate (strictly-greater overwrite); last.pt the final.

    StubEnv's win outcome is drawn from env RNG and is policy-independent, so its
    eval win rate is constant across iterations and cannot exercise the "new best
    overwrites" path. Stub evaluate() with a scripted win-rate sequence to lock it:
    best.pt must settle on the max captured at its FIRST-maximal iteration and NOT
    be overwritten by a later equal-or-lower value, while last.pt tracks the final
    eval. The tie sequence ([0.5, 0.5]) pins the strictly-greater rule specifically:
    under a >= overwrite best.pt would move to the later iteration, so this fails
    if ties overwrite.
    """
    reports = iter(win_seq)

    def _fake_evaluate(
        _policy: object, _env: object, seeds: object, **_kwargs: object
    ) -> EvalReport:
        win_rate = next(reports)
        return EvalReport(
            n_episodes=EVAL_EPISODES,
            win_rate=win_rate,
            avg_floor=0.0,
            avg_hp=0.0,
            avg_ep_len=1.0,
            avg_return=win_rate,
        )

    monkeypatch.setattr(train_module, "evaluate", _fake_evaluate)

    config = _config(
        num_iterations=len(win_seq),
        eval_every=1,
        eval_episodes=EVAL_EPISODES,
        checkpoint_dir=str(tmp_path),
    )
    history = train(_bandit_env(), config, eval_env=_bandit_env())

    # Expectations derived from the sequence, not hardcoded.
    expected_best = max(win_seq)
    expected_best_iter = win_seq.index(expected_best)
    expected_last = win_seq[-1]

    best_ckpt = torch.load(tmp_path / train_module.BEST_CHECKPOINT_NAME, weights_only=True)
    last_ckpt = torch.load(tmp_path / train_module.LAST_CHECKPOINT_NAME, weights_only=True)
    assert best_ckpt[train_module.CHECKPOINT_WIN_RATE_KEY] == expected_best
    assert best_ckpt[train_module.CHECKPOINT_ITERATION_KEY] == expected_best_iter
    assert last_ckpt[train_module.CHECKPOINT_WIN_RATE_KEY] == expected_last

    best_eval = history.best_eval()
    assert best_eval is not None
    assert best_eval.report.win_rate == expected_best
    assert best_eval.iteration == expected_best_iter


def test_checkpoint_dir_without_eval_writes_only_last(tmp_path: Path) -> None:
    """checkpoint_dir set but eval_every None writes only last.pt (no eval win rates).

    Documents the checkpoint-without-eval case: no eval runs (no eval_env needed),
    so there is no win rate to rank a best - best.pt is absent and last.pt records
    win_rate None.
    """
    config = _config(num_iterations=2, checkpoint_dir=str(tmp_path))  # eval_every None
    history = train(_bandit_env(), config)  # no eval_env required when eval is off

    assert history.eval_reports == []
    last_path = tmp_path / train_module.LAST_CHECKPOINT_NAME
    best_path = tmp_path / train_module.BEST_CHECKPOINT_NAME
    assert last_path.exists()
    assert not best_path.exists()  # best requires eval win rates
    assert list(tmp_path.glob("*" + train_module.CHECKPOINT_TMP_SUFFIX)) == []

    last_ckpt = torch.load(last_path, weights_only=True)
    assert last_ckpt[train_module.CHECKPOINT_WIN_RATE_KEY] is None
    assert last_ckpt[train_module.CHECKPOINT_ITERATION_KEY] == config.num_iterations - 1


def test_eval_and_checkpoint_default_off(tmp_path: Path) -> None:
    """Unset eval_every/checkpoint_dir: no eval records and no files written.

    The default-OFF contract - an unset config runs the prior loop unchanged (the
    existing loop tests cover behaviour equivalence). No eval_env is needed.
    """
    config = _config(num_iterations=3, checkpoint_dir=None)  # eval_every None by default
    history = train(_bandit_env(), config)

    assert history.eval_reports == []
    assert history.best_eval() is None
    # checkpoint_dir was None, so nothing is written anywhere (tmp_path stays empty).
    assert list(tmp_path.iterdir()) == []


# -- Vectorized (num_envs > 1) path ------------------------------------------

# terminate_prob=1.0 with a high step cap makes every StubEnv episode a single
# terminal step, keeping the vectorized runs fast and every iteration full of
# completed episodes.
VEC_NUM_ENVS = 3


def _vec_task_env(num_envs: int = VEC_NUM_ENVS):
    """A StubVecEnv of single-step bandit envs for the vectorized-path tests."""
    return make_stub_vec_env(num_envs, terminate_prob=1.0, max_episode_steps=10_000)


def test_num_envs_defaults_to_one() -> None:
    """num_envs defaults to 1 (the single-env path)."""
    assert TrainConfig(num_iterations=1, n_steps=1).num_envs == 1


@pytest.mark.parametrize("value", [0, -1, -4])
def test_config_rejects_non_positive_num_envs(value: int) -> None:
    """num_envs must be positive; a degenerate count fails at construction."""
    with pytest.raises(ValueError, match="num_envs"):
        dataclasses.replace(_BASE_CONFIG, num_envs=value)


def test_vectorized_global_step_accounting() -> None:
    """num_envs>1 advances global_step by num_envs*n_steps per iteration.

    Locks the vectorized step accounting end to end: each iteration collects
    n_steps per env across num_envs envs, so global_step is the running
    num_envs*n_steps total and CollectStats.n_steps reports that per-iteration
    total (== len of the flattened buffer).
    """
    n_steps = 16
    num_iterations = 3
    config = _config(num_iterations=num_iterations, n_steps=n_steps, num_envs=VEC_NUM_ENVS)

    history = train(_vec_task_env(), config)

    assert len(history.records) == num_iterations
    for i, record in enumerate(history.records):
        assert record.global_step == (i + 1) * VEC_NUM_ENVS * n_steps
        assert record.collect.n_steps == VEC_NUM_ENVS * n_steps  # total transitions collected
    assert history.records[-1].global_step == num_iterations * VEC_NUM_ENVS * n_steps


def test_vectorized_num_envs_mismatch_raises() -> None:
    """config.num_envs must match the passed vec env's num_envs, else a ValueError.

    Revert-verify: drop the num_envs equality guard in train() and this no longer
    raises (the collector would then batch the wrong number of envs).
    """
    vec_env = make_stub_vec_env(2, terminate_prob=1.0, max_episode_steps=10_000)
    config = _config(num_iterations=1, n_steps=4, num_envs=3)  # 3 != 2
    with pytest.raises(ValueError, match="num_envs"):
        train(vec_env, config)


def test_num_envs_one_rejects_vectorized_env() -> None:
    """num_envs==1 with a vectorized env passed raises a clear ValueError.

    The single-env branch requires a plain gym.Env; a VecEnvProtocol (StubVecEnv
    here) is rejected with a ValueError - matching the adjacent num_envs-mismatch
    check - not a bare AssertionError that python -O would strip. Revert-verify:
    restore the bare assert isinstance(env, gym.Env) and this raises AssertionError,
    not ValueError, so the ValueError match fails.
    """
    vec_env = make_stub_vec_env(1, terminate_prob=1.0, max_episode_steps=10_000)
    config = _config(num_iterations=1, n_steps=4, num_envs=1)
    with pytest.raises(ValueError, match="single-env path"):
        train(vec_env, config)


def test_num_envs_gt_one_rejects_plain_env() -> None:
    """num_envs>1 with a plain gym.Env passed raises a clear ValueError.

    The vectorized branch requires a VecEnvProtocol; a plain gym.Env (StubEnv here)
    is rejected with a ValueError - mirroring the single-env branch and the
    num_envs-mismatch check - not a bare AssertionError that python -O would strip.
    Revert-verify: restore the bare assert not isinstance(env, gym.Env) and this
    raises AssertionError, not ValueError, so the ValueError match fails.
    """
    config = _config(num_iterations=1, n_steps=4, num_envs=VEC_NUM_ENVS)
    with pytest.raises(ValueError, match="vectorized path"):
        train(_bandit_env(), config)


def test_vectorized_training_moves_weights() -> None:
    """The vectorized path trains end to end: weights differ from the seeded init.

    Exercises collect -> per-env GAE -> flattened minibatches -> ppo_update over the
    vectorized buffer, and confirms the optimizer actually steps. ``train`` seeds the
    global RNG then builds the network, so reseeding with the same seed reproduces
    the exact init; any difference afterward is the update's doing.
    """
    config = _config(num_iterations=3, n_steps=32, num_envs=VEC_NUM_ENVS)

    history = train(_vec_task_env(), config)

    torch.manual_seed(SEED)
    initial = ActorCritic(hidden_dim=HIDDEN)
    assert _param_checksum(initial) != _param_checksum(history.actor_critic)


def test_vectorized_diagnostics_finite_and_sane() -> None:
    """Every per-iteration diagnostic on the vectorized path is finite and in range."""
    history = train(_vec_task_env(), _config(num_iterations=3, n_steps=32, num_envs=VEC_NUM_ENVS))
    for record in history.records:
        ppo = record.ppo
        for value in (ppo.policy_loss, ppo.value_loss, ppo.entropy, ppo.approx_kl, ppo.grad_norm):
            assert math.isfinite(value)
        assert 0.0 <= ppo.clip_fraction <= 1.0
        sps = record.collect.steps_per_second
        assert math.isfinite(sps) and sps > 0.0
        # Single-step bandit envs terminate on every transition, so the pooled
        # episode count equals the total transitions collected this iteration.
        assert record.collect.n_episodes == record.collect.n_steps
