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
import types
from pathlib import Path
from typing import cast

import gymnasium as gym
import pytest

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.train import EvalRecord, TrainConfig, TrainHistory, train
from sts_rl.env.reward import RewardConfig
from sts_rl.env.reward_cli import reward_config_from_args
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


def _make_report(
    win_rate: float, act1_clear_rate: float = 0.0, n_episodes: int = _EVAL_EPISODES
) -> EvalReport:
    """A fully-populated EvalReport with distinguishing win / clear rates.

    ``n_episodes`` defaults to the small ``_EVAL_EPISODES`` band; a caller that must
    match a driver's default eval band (so the reuse guard short-circuits) passes
    the driver constant explicitly.
    """
    return EvalReport(
        n_episodes=n_episodes,
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


class _StubRunEnv:
    """Engine-free stand-in for ``StsRunEnv``: records its construction kwargs and
    serves the provenance ``info`` keys ``main()`` reads off the initial reset, so
    ``main()`` runs the whole warm-start wiring without a native engine build.

    ``reward_config`` mirrors the real ``StsRunEnv`` signature (``main()`` now passes
    it); it is stored but otherwise unused by the stub.
    """

    def __init__(
        self,
        *,
        ascension: int,
        max_episode_steps: int,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.ascension = ascension
        self.max_episode_steps = max_episode_steps
        self.reward_config = reward_config

    def reset(
        self, *, seed: int | None = None, options: object = None
    ) -> tuple[object, dict[str, object]]:
        # main() reads engine_commit/interface_version off this initial reset.
        return None, {"engine_commit": "stub-commit", "interface_version": "stub-iface"}

    def close(self) -> None:
        pass


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
    # Both records carry eval_seed_base=_EVAL_SEED_BASE and (via _make_report)
    # report.n_episodes=_EVAL_EPISODES, matching the band the call below requests,
    # so the guarded reuse short-circuit fires and no recompute happens.
    history = TrainHistory(
        records=[],
        actor_critic=ActorCritic(),
        eval_reports=[
            EvalRecord(
                iteration=0,
                global_step=128,
                eval_seed_base=_EVAL_SEED_BASE,
                report=_make_report(0.25),
            ),
            EvalRecord(
                iteration=1,
                global_step=256,
                eval_seed_base=_EVAL_SEED_BASE,
                report=reused,
            ),
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


@pytest.mark.parametrize(
    "call_seed_base, call_episodes",
    [
        (_EVAL_SEED_BASE + 1, _EVAL_EPISODES),  # seed base differs -> recompute
        (_EVAL_SEED_BASE, _EVAL_EPISODES + 1),  # episode count differs -> recompute
    ],
)
def test_final_eval_report_recomputes_on_band_mismatch(
    monkeypatch: pytest.MonkeyPatch, call_seed_base: int, call_episodes: int
) -> None:
    """A cached snapshot from a DIFFERENT band is not reused: the helper recomputes.

    The reuse short-circuit fires only when the recorded ``eval_seed_base`` AND the
    report's ``n_episodes`` both match the requested band; a mismatch on either
    falls through to a fresh ``evaluate`` over the requested band rather than
    returning the stale cached report (the contract this guard enforces: a future
    caller passing a different band gets a fresh recompute rather than the stale
    holdout number).

    Revert-verify: replace the guarded reuse with an unconditional
    ``return history.eval_reports[-1].report`` and this fails - the stale cached
    report is returned and the sentinel ``evaluate`` is called 0 != 1 times.
    """
    sentinel = _make_report(0.99, act1_clear_rate=0.99)
    calls = 0

    def _counting_evaluate(*_args: object, **_kwargs: object) -> EvalReport:
        nonlocal calls
        calls += 1
        return sentinel

    monkeypatch.setattr(train_run, "evaluate", _counting_evaluate)

    # Cached over the KNOWN band (_EVAL_SEED_BASE, and _EVAL_EPISODES via
    # _make_report); the call below requests a DIFFERENT band, so the guard must
    # not short-circuit to the cached report.
    cached = _make_report(0.25, act1_clear_rate=0.1)
    history = TrainHistory(
        records=[],
        actor_critic=ActorCritic(),
        eval_reports=[
            EvalRecord(
                iteration=0,
                global_step=128,
                eval_seed_base=_EVAL_SEED_BASE,
                report=cached,
            )
        ],
    )

    result = train_run._final_eval_report(history, _dummy_eval_env(), call_seed_base, call_episodes)

    assert result is sentinel
    assert result is not cached
    assert calls == 1


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
    width. Covers the checkpoint-to-network load path (load and migrate, then
    return the net with its derived trunk width) without the engine (both
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


def test_main_wires_warm_start_net_and_derived_width_into_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() hands the loaded warm-start net AND its derived width to train().

    Engine-free integration test of the wiring the helper-level tests do not
    cover: main() must pass the SAME ActorCritic that _load_warm_start produced to
    train() as init_actor_critic, and the trunk width it derives from that
    checkpoint (net.encoder.output_dim, which overrides --hidden-dim) must reach
    train() as config.hidden_dim. StsRunEnv (the engine env), load_checkpoint, and
    train are all stubbed, so it runs without a native build.
    """
    # Warm-start net at a NON-DEFAULT trunk width, so asserting the DERIVED width
    # (not the --hidden-dim default of HIDDEN_DIM) reaches train() is sharp.
    warm_width = 32
    assert warm_width != train_run.HIDDEN_DIM  # else the width-override path is untested
    warm_net = ActorCritic(hidden_dim=warm_width)
    # Grab the trunk width now: the identity assert below (`is warm_net`) narrows
    # warm_net's static type, so read encoder.output_dim before it.
    warm_trunk_width = warm_net.encoder.output_dim

    # Stub load_checkpoint to hand back our net; the REAL _load_warm_start then
    # derives the width from net.encoder.output_dim, so main()'s actual
    # width-derivation path is exercised, not a stubbed shortcut.
    monkeypatch.setattr(train_run, "load_checkpoint", lambda _path: warm_net)

    captured: dict[str, object] = {}

    def _fake_train(
        _env: object,
        config: TrainConfig,
        *,
        eval_env: object = None,
        init_actor_critic: ActorCritic | None = None,
    ) -> TrainHistory:
        captured["config"] = config
        captured["init_actor_critic"] = init_actor_critic
        # A last eval snapshot recorded over the driver's DEFAULT band (base seed +
        # episode count, which main() passes _final_eval_report when only
        # --warm-start is given) keeps it on the reuse path, so the final eval needs
        # neither the engine nor a stubbed evaluate.
        return TrainHistory(
            records=[],
            actor_critic=init_actor_critic if init_actor_critic is not None else ActorCritic(),
            eval_reports=[
                EvalRecord(
                    iteration=0,
                    global_step=1,
                    eval_seed_base=train_run.DEFAULT_EVAL_BASE_SEED,
                    report=_make_report(0.0, n_episodes=train_run.DEFAULT_EVAL_EPISODES),
                )
            ],
        )

    monkeypatch.setattr(train_run, "train", _fake_train)

    # Bind main()'s deferred `from sts_rl.env.run_adapter import StsRunEnv` to the
    # engine-free stub by injecting a fake module, so no native engine is imported.
    fake_run_adapter = types.ModuleType("sts_rl.env.run_adapter")
    setattr(fake_run_adapter, "StsRunEnv", _StubRunEnv)
    monkeypatch.setitem(sys.modules, "sts_rl.env.run_adapter", fake_run_adapter)

    monkeypatch.setattr(sys, "argv", ["train_run", "--warm-start", "/tmp/warm.pt"])

    train_run.main()

    # (a) the loaded net is handed to train() unchanged (SAME object), and (b) the
    # width derived from that checkpoint reaches train() as config.hidden_dim.
    assert captured["init_actor_critic"] is warm_net
    config = cast(TrainConfig, captured["config"])
    assert config.hidden_dim == warm_trunk_width == warm_width


def test_main_builds_config_with_run_best_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() builds its TrainConfig with best_metric=RUN_BEST_METRIC (act1_clear_rate).

    Run mode ranks best.pt / TrainHistory.best_eval on the Act 1 clear rate, not the
    default win_rate: full-run win_rate is ~0 for a long time, so every early eval
    ties at 0 and best.pt would be degenerate. Engine-free: train and StsRunEnv are
    stubbed. Revert-verify: drop best_metric=RUN_BEST_METRIC from the TrainConfig
    build and config.best_metric falls back to the win_rate default, failing this.
    """
    captured: dict[str, object] = {}

    def _fake_train(
        _env: object,
        config: TrainConfig,
        *,
        eval_env: object = None,
        init_actor_critic: ActorCritic | None = None,
    ) -> TrainHistory:
        captured["config"] = config
        # A last eval snapshot over the driver's DEFAULT band keeps
        # _final_eval_report on the reuse path, so the final eval needs neither the
        # engine nor a stubbed evaluate.
        return TrainHistory(
            records=[],
            actor_critic=ActorCritic(),
            eval_reports=[
                EvalRecord(
                    iteration=0,
                    global_step=1,
                    eval_seed_base=train_run.DEFAULT_EVAL_BASE_SEED,
                    report=_make_report(0.0, n_episodes=train_run.DEFAULT_EVAL_EPISODES),
                )
            ],
        )

    monkeypatch.setattr(train_run, "train", _fake_train)

    # Bind main()'s deferred StsRunEnv import to the engine-free stub.
    fake_run_adapter = types.ModuleType("sts_rl.env.run_adapter")
    setattr(fake_run_adapter, "StsRunEnv", _StubRunEnv)
    monkeypatch.setitem(sys.modules, "sts_rl.env.run_adapter", fake_run_adapter)

    monkeypatch.setattr(sys, "argv", ["train_run"])

    train_run.main()

    config = cast(TrainConfig, captured["config"])
    assert config.best_metric == train_run.RUN_BEST_METRIC == "act1_clear_rate"


def test_arg_parser_warmup_and_early_stop_defaults() -> None:
    """The parser exposes the warmup + early-stop knobs, all defaulting OFF.

    Locks the opt-in contract: --value-warmup-iters defaults 0, --early-stop-patience
    defaults None, --early-stop-min-delta defaults 0.0, so the driver's default behavior
    is unchanged. Built engine-free.
    """
    args = train_run.build_arg_parser().parse_args([])
    assert args.value_warmup_iters == 0
    assert args.early_stop_patience is None
    assert args.early_stop_min_delta == 0.0


def test_main_wires_warmup_and_early_stop_into_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() flows the warmup + early-stop CLI flags into the TrainConfig.

    Engine-free: train and StsRunEnv are stubbed. Passing non-default values on argv and
    asserting they reach the config guards the CLI->TrainConfig wire.

    Revert-verify: drop any of the three TrainConfig kwargs in main() and that field
    falls back to its OFF default, failing the matching assertion.
    """
    captured: dict[str, object] = {}

    def _fake_train(
        _env: object,
        config: TrainConfig,
        *,
        eval_env: object = None,
        init_actor_critic: ActorCritic | None = None,
    ) -> TrainHistory:
        captured["config"] = config
        # A last eval snapshot over the driver's DEFAULT band keeps _final_eval_report on
        # the reuse path, so the final eval needs neither the engine nor a stubbed evaluate.
        return TrainHistory(
            records=[],
            actor_critic=ActorCritic(),
            eval_reports=[
                EvalRecord(
                    iteration=0,
                    global_step=1,
                    eval_seed_base=train_run.DEFAULT_EVAL_BASE_SEED,
                    report=_make_report(0.0, n_episodes=train_run.DEFAULT_EVAL_EPISODES),
                )
            ],
        )

    monkeypatch.setattr(train_run, "train", _fake_train)

    # Bind main()'s deferred StsRunEnv import to the engine-free stub.
    fake_run_adapter = types.ModuleType("sts_rl.env.run_adapter")
    setattr(fake_run_adapter, "StsRunEnv", _StubRunEnv)
    monkeypatch.setitem(sys.modules, "sts_rl.env.run_adapter", fake_run_adapter)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_run",
            "--value-warmup-iters",
            "3",
            # early_stop_patience requires eval_every (TrainConfig rejects the no-op
            # combination), so pass --eval-every alongside the early-stop flags.
            "--eval-every",
            "2",
            "--early-stop-patience",
            "5",
            "--early-stop-min-delta",
            "0.02",
        ],
    )

    train_run.main()

    config = cast(TrainConfig, captured["config"])
    assert config.value_warmup_iters == 3
    assert config.eval_every == 2
    assert config.early_stop_patience == 5
    assert config.early_stop_min_delta == 0.02


def test_arg_parser_has_no_encounters_knob() -> None:
    """Run mode has no encounter pool, so --encounters (combat-only) is absent."""
    with pytest.raises(SystemExit):
        train_run.build_arg_parser().parse_args(["--encounters", "GREMLIN_NOB"])


def test_arg_parser_wires_reward_shaping_flags() -> None:
    """build_arg_parser wires the shared reward-shaping flags: an empty argv maps to
    RewardConfig() and --boss-kill-coef overrides only boss_kill.

    Guards that build_arg_parser calls add_reward_shaping_args (drop the call and
    --boss-kill-coef becomes unrecognized). Engine-free: train_run.build_arg_parser
    uses a module constant for the step cap, so it needs no engine (unlike
    train_combat's). The mapping itself is locked in test_reward_cli.
    """
    parser = train_run.build_arg_parser()
    assert reward_config_from_args(parser.parse_args([])) == RewardConfig()
    overridden = reward_config_from_args(parser.parse_args(["--boss-kill-coef", "1.0"]))
    assert overridden.boss_kill == 1.0
    # Overriding one coefficient leaves the others at their RewardConfig default.
    assert overridden.enemy_hp_removed == RewardConfig().enemy_hp_removed


def test_main_wires_reward_config_into_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() builds a RewardConfig from the shaping flags and passes it to BOTH the
    training and eval envs, so --boss-kill-coef reaches the live reward.

    Engine-free: train and StsRunEnv are stubbed. The stub records its
    reward_config kwarg, so the CLI -> env wire is checked without a native build.

    Revert-verify: drop reward_config=reward_config from either StsRunEnv build in
    main() and that env's recorded coefficient falls back to the default, failing
    the matching assertion.
    """
    captured_reward_configs: list[RewardConfig | None] = []

    class _RewardStubRunEnv(_StubRunEnv):
        def __init__(
            self,
            *,
            ascension: int,
            max_episode_steps: int,
            reward_config: RewardConfig | None = None,
        ) -> None:
            super().__init__(
                ascension=ascension,
                max_episode_steps=max_episode_steps,
                reward_config=reward_config,
            )
            captured_reward_configs.append(reward_config)

    def _fake_train(
        _env: object,
        config: TrainConfig,
        *,
        eval_env: object = None,
        init_actor_critic: ActorCritic | None = None,
    ) -> TrainHistory:
        return TrainHistory(
            records=[],
            actor_critic=ActorCritic(),
            eval_reports=[
                EvalRecord(
                    iteration=0,
                    global_step=1,
                    eval_seed_base=train_run.DEFAULT_EVAL_BASE_SEED,
                    report=_make_report(0.0, n_episodes=train_run.DEFAULT_EVAL_EPISODES),
                )
            ],
        )

    monkeypatch.setattr(train_run, "train", _fake_train)
    fake_run_adapter = types.ModuleType("sts_rl.env.run_adapter")
    setattr(fake_run_adapter, "StsRunEnv", _RewardStubRunEnv)
    monkeypatch.setitem(sys.modules, "sts_rl.env.run_adapter", fake_run_adapter)
    monkeypatch.setattr(sys, "argv", ["train_run", "--boss-kill-coef", "1.0"])

    train_run.main()

    # Both the training env and the eval env received the overridden shaping config.
    assert len(captured_reward_configs) == 2
    assert all(rc is not None and rc.boss_kill == 1.0 for rc in captured_reward_configs)


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
