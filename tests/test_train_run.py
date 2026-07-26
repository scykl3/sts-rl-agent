"""Tests for the ``train_run`` full-run training entry point.

Two layers, mirroring ``test_train_combat``:

* Engine-free: the pure helpers (``_final_eval_report`` reuse/recompute,
  ``_validate_num_envs``, ``build_arg_parser`` defaults) are locked without a
  native build. The module imports engine-free (its ``StsRunEnv`` import is
  deferred into ``main``), and ``load_checkpoint``/``evaluate`` are engine-free,
  so importing the helpers needs no engine. ``evaluate`` is monkeypatched to a
  counting stub so the reuse path is shown to make NO eval call.
* Engine-gated: a tiny end-to-end smoke drives the real ``StsRunEnv`` through
  ``train`` for a couple of iterations and asserts the run's eval report carries
  ``act1_clear_rate``. Skipped cleanly when the engine is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import gymnasium as gym
import pytest

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.train import EvalRecord, TrainConfig, TrainHistory, train
from sts_rl.eval import EvalReport, make_holdout_seeds

# scripts/ is not an importable package, so put it on sys.path to import the
# train_run entry-point module directly (matching test_train_combat).
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train_run  # noqa: E402  (imported after the sys.path insert above)

# The smoke test drives the native engine; the rest of this module is engine-free
# and must still run without a build. Probe once and skip just that test when the
# engine is absent (matching the suite's ImportError -> skip idiom).
try:
    import sts_rl.env._engine  # noqa: F401,E402

    _ENGINE_BUILT = True
except ImportError:  # pragma: no cover - exercised only without a build
    _ENGINE_BUILT = False

# Arbitrary holdout band the recompute path would request. Small; the stubbed
# evaluate never runs it, so the values only feed the seed-band assertion.
_EVAL_SEED_BASE = 1_000_000
_EVAL_EPISODES = 4


def _make_report(win_rate: float, act1_clear_rate: float = 0.0) -> EvalReport:
    """A fully-populated EvalReport with distinguishing win / clear rates."""
    return EvalReport(
        n_episodes=_EVAL_EPISODES,
        win_rate=win_rate,
        avg_floor=1.0,
        avg_hp=10.0,
        avg_ep_len=5.0,
        avg_return=win_rate,
        act1_clear_rate=act1_clear_rate,
    )


def _dummy_eval_env() -> gym.Env:
    """Sentinel eval env: the helper never steps it (reuse path) or only hands it
    to the stubbed ``evaluate`` (recompute path), so a bare object stands in."""
    return cast(gym.Env, object())


def test_final_eval_report_reuses_periodic_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """With periodic eval present, the helper returns the LAST snapshot's report
    and does NOT recompute: the monkeypatched evaluate must be called 0 times."""
    calls = 0

    def _counting_evaluate(*_args: object, **_kwargs: object) -> EvalReport:
        nonlocal calls
        calls += 1
        return _make_report(0.99)  # distinct from any reused report

    monkeypatch.setattr(train_run, "evaluate", _counting_evaluate)

    reused = _make_report(0.5, act1_clear_rate=0.8)
    history = TrainHistory(
        records=[],
        actor_critic=ActorCritic(),
        eval_reports=[
            EvalRecord(iteration=0, global_step=128, report=_make_report(0.25)),
            EvalRecord(iteration=1, global_step=256, report=reused),
        ],
    )

    result = train_run._final_eval_report(
        history, _dummy_eval_env(), _EVAL_SEED_BASE, _EVAL_EPISODES
    )

    assert result is reused
    assert calls == 0


def test_final_eval_report_evaluates_when_no_periodic_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no periodic eval, the helper recomputes exactly once and returns that
    result, requesting the ``make_holdout_seeds`` band with ``deterministic=True``."""
    recomputed = _make_report(0.75, act1_clear_rate=0.9)
    calls = 0
    seen_seeds: object = None
    seen_kwargs: dict[str, object] = {}

    def _counting_evaluate(
        _policy: object, _env: object, seeds: object, **kwargs: object
    ) -> EvalReport:
        nonlocal calls, seen_seeds
        calls += 1
        seen_seeds = seeds
        seen_kwargs.update(kwargs)
        return recomputed

    monkeypatch.setattr(train_run, "evaluate", _counting_evaluate)

    history = TrainHistory(records=[], actor_critic=ActorCritic(), eval_reports=[])

    result = train_run._final_eval_report(
        history, _dummy_eval_env(), _EVAL_SEED_BASE, _EVAL_EPISODES
    )

    assert result is recomputed
    assert calls == 1
    assert seen_seeds == make_holdout_seeds(_EVAL_SEED_BASE, _EVAL_EPISODES)
    assert seen_kwargs.get("deterministic") is True


def test_validate_num_envs_accepts_single_env() -> None:
    """Single-env is the supported path, so num_envs == 1 does not raise."""
    train_run._validate_num_envs(1)  # no raise


@pytest.mark.parametrize("num_envs", [2, 8, 16])
def test_validate_num_envs_rejects_vectorized(num_envs: int) -> None:
    """num_envs > 1 is not supported yet and fails fast with a clear pointer.

    The vectorized StsRunEnv path is blocked on an env-owner widening of
    SubprocVecEnv.make_env's type hint; the driver rejects it rather than reaching
    for a cast.
    """
    with pytest.raises(NotImplementedError, match="single-env"):
        train_run._validate_num_envs(num_envs)


def test_arg_parser_run_scale_defaults() -> None:
    """The parser exposes run-scale defaults and the warm-start / num-envs knobs.

    Locks the run-mode contract: max_episode_steps defaults to the run-length cap
    (3000, not the combat 500), num_envs defaults to the single-env 1, and
    --warm-start defaults to None (train from scratch). Built engine-free.
    """
    args = train_run.build_arg_parser().parse_args([])
    assert args.max_episode_steps == train_run.DEFAULT_MAX_EPISODE_STEPS == 3000
    assert args.num_envs == 1
    assert args.warm_start is None
    assert args.eval_every is None
    assert args.checkpoint_dir is None


def test_arg_parser_accepts_warm_start_path() -> None:
    """--warm-start captures the checkpoint path for load_checkpoint."""
    args = train_run.build_arg_parser().parse_args(["--warm-start", "/tmp/best.pt"])
    assert args.warm_start == "/tmp/best.pt"


def test_load_warm_start_returns_net_and_trunk_width(tmp_path: Path) -> None:
    """_load_warm_start loads a checkpoint and derives its encoder trunk width.

    Builds a small ActorCritic, saves it via ``train._save_checkpoint`` (the same
    payload the driver writes), loads it back through the driver helper, and
    asserts the returned ``hidden_dim`` equals the loaded net's
    ``encoder.output_dim`` and that the net is a real ``ActorCritic`` at that
    width. Closes the warm-start glue gap without the engine (both
    ``load_checkpoint`` and ``_save_checkpoint`` are engine-free).
    """
    from sts_rl.agent.train import _save_checkpoint

    hidden = 32
    net = ActorCritic(hidden_dim=hidden)
    _save_checkpoint(
        tmp_path,
        "warm.pt",
        net,
        hidden_dim=hidden,
        iteration=0,
        global_step=0,
        win_rate=None,
    )

    loaded, hidden_dim = train_run._load_warm_start(str(tmp_path / "warm.pt"))
    assert isinstance(loaded, ActorCritic)
    assert hidden_dim == loaded.encoder.output_dim == hidden


def test_arg_parser_has_no_encounters_knob() -> None:
    """Run mode has no encounter pool, so --encounters (combat-only) is absent."""
    with pytest.raises(SystemExit):
        train_run.build_arg_parser().parse_args(["--encounters", "GREMLIN_NOB"])


@pytest.mark.skipif(not _ENGINE_BUILT, reason="engine not built")
def test_default_max_episode_steps_matches_run_adapter() -> None:
    """The driver's default step cap mirrors the adapter's own run-length cap.

    ``train_run`` defines ``DEFAULT_MAX_EPISODE_STEPS`` as a module constant (so
    the module imports engine-free) rather than importing it from the
    engine-coupled ``run_adapter``. This engine-gated check guards that the two
    never silently desync if the env owner changes the adapter cap. ``run_adapter``
    is imported inside the test (not at module scope) so the driver's own import
    stays engine-free.
    """
    from sts_rl.env.run_adapter import DEFAULT_MAX_EPISODE_STEPS

    assert train_run.DEFAULT_MAX_EPISODE_STEPS == DEFAULT_MAX_EPISODE_STEPS


@pytest.mark.skipif(not _ENGINE_BUILT, reason="engine not built")
def test_run_training_smoke_reports_act1_clear_rate() -> None:
    """A tiny real full-run training loop returns a TrainHistory whose eval report
    carries act1_clear_rate.

    Plumbing only: 2 iterations, a 64-step rollout, a tiny trunk, and short
    episodes (max_episode_steps=64) keep it fast. A separate eval_env holds the
    greedy holdout. The clear rate is a valid fraction (near 0 for an untrained
    net, which dies or truncates inside Act 1); the assertion is that the metric
    exists and is well-formed, not that the net clears anything.
    """
    from sts_rl.env.run_adapter import StsRunEnv

    env = StsRunEnv(ascension=0, max_episode_steps=64)
    eval_env = StsRunEnv(ascension=0, max_episode_steps=64)
    config = TrainConfig(
        num_iterations=2,
        n_steps=64,
        hidden_dim=32,
        eval_every=2,
        eval_episodes=2,
        eval_seed_base=_EVAL_SEED_BASE,
    )
    try:
        history = train(env, config, eval_env=eval_env)
    finally:
        env.close()
        eval_env.close()

    assert isinstance(history, TrainHistory)
    assert history.eval_reports  # eval ran on the final iteration
    report = history.eval_reports[-1].report
    assert hasattr(report, "act1_clear_rate")
    assert 0.0 <= report.act1_clear_rate <= 1.0
