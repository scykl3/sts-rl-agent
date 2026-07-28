"""Tests for the ``train_combat`` entry point's final-eval helper.

Engine-free: these lock :func:`train_combat._final_eval_report`'s
reuse-or-recompute logic without driving the native engine. The module imports
engine-free (its ``StsEnv``/adapter import is deferred into the CLI functions),
so importing the helper needs no build. ``evaluate`` is monkeypatched to a
counting stub so the recompute path is observable and, crucially, so the reuse
path can be shown to make NO eval call at all - the property that makes periodic
eval a genuine dedup rather than a second identical holdout pass.

TrainHistory/EvalRecord/EvalReport are built with their real constructors; the
eval env is a sentinel because the helper never touches it (reuse path) or only
forwards it to the stubbed ``evaluate`` (recompute path).
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import cast

import gymnasium as gym
import pytest

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.train import EvalRecord, TrainHistory
from sts_rl.env.reward import RewardConfig
from sts_rl.eval import EvalReport, make_holdout_seeds

# scripts/ is not an importable package, so put it on sys.path to import the
# train_combat entry-point module directly (the repo has no other convention for
# importing script modules from tests).
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train_combat  # noqa: E402  (imported after the sys.path insert above)

# _parse_encounters imports the native engine lazily, so its test needs a build;
# the rest of this module is engine-free and must still run without one. Probe
# once and skip just that test when the engine is absent (matching the suite's
# ImportError -> skip idiom; importorskip would miss a broken-build ImportError).
try:
    import sts_rl.env._engine  # noqa: F401,E402

    _ENGINE_BUILT = True
except ImportError:  # pragma: no cover - exercised only without a build
    _ENGINE_BUILT = False

# Arbitrary holdout band the recompute path would request. Small; the stubbed
# evaluate never runs it, so the values only feed the seed-band assertion.
_EVAL_SEED_BASE = 1_000_000
_EVAL_EPISODES = 4


def _make_report(win_rate: float) -> EvalReport:
    """A fully-populated EvalReport with a distinguishing ``win_rate``."""
    return EvalReport(
        n_episodes=_EVAL_EPISODES,
        win_rate=win_rate,
        avg_floor=1.0,
        avg_hp=10.0,
        avg_ep_len=5.0,
        avg_return=win_rate,
    )


def _dummy_eval_env() -> gym.Env:
    """Sentinel eval env: the helper never steps it (reuse path) or only hands it
    to the stubbed ``evaluate`` (recompute path), so a bare object stands in."""
    return cast(gym.Env, object())


def test_final_eval_report_reuses_periodic_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """With periodic eval present, the helper returns the LAST snapshot's report
    and does NOT recompute: the monkeypatched evaluate must be called 0 times.

    Revert-verify: make the reuse branch fall through to recompute (always
    evaluate) and this fails (evaluate called 1 != 0), proving the test locks the
    no-recompute dedup.
    """
    calls = 0

    def _counting_evaluate(*_args: object, **_kwargs: object) -> EvalReport:
        nonlocal calls
        calls += 1
        return _make_report(0.99)  # distinct from any reused report

    monkeypatch.setattr(train_combat, "evaluate", _counting_evaluate)

    reused = _make_report(0.5)
    # Two snapshots so the assertion also pins that the LAST one is taken. Both
    # record eval_seed_base=_EVAL_SEED_BASE and (via _make_report) report.n_episodes
    # =_EVAL_EPISODES, matching the band the call below requests, so the guarded
    # reuse short-circuit fires.
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

    result = train_combat._final_eval_report(
        history, _dummy_eval_env(), _EVAL_SEED_BASE, _EVAL_EPISODES
    )

    assert result is reused
    assert calls == 0


def test_final_eval_report_evaluates_when_no_periodic_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no periodic eval, the helper recomputes exactly once and returns that
    result, requesting the ``make_holdout_seeds`` band with ``deterministic=True``
    (so a recompute matches train()'s greedy eval regardless of evaluate()'s
    default).
    """
    recomputed = _make_report(0.75)
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

    monkeypatch.setattr(train_combat, "evaluate", _counting_evaluate)

    history = TrainHistory(records=[], actor_critic=ActorCritic(), eval_reports=[])

    result = train_combat._final_eval_report(
        history, _dummy_eval_env(), _EVAL_SEED_BASE, _EVAL_EPISODES
    )

    assert result is recomputed
    assert calls == 1
    # Recompute uses the same contiguous holdout band the periodic eval would,
    # and greedily (deterministic=True is passed explicitly, not defaulted).
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
    sentinel = _make_report(0.99)
    calls = 0

    def _counting_evaluate(*_args: object, **_kwargs: object) -> EvalReport:
        nonlocal calls
        calls += 1
        return sentinel

    monkeypatch.setattr(train_combat, "evaluate", _counting_evaluate)

    # Cached over the KNOWN band (_EVAL_SEED_BASE, and _EVAL_EPISODES via
    # _make_report); the call below requests a DIFFERENT band, so the guard must
    # not short-circuit to the cached report.
    cached = _make_report(0.25)
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

    result = train_combat._final_eval_report(
        history, _dummy_eval_env(), call_seed_base, call_episodes
    )

    assert result is sentinel
    assert result is not cached
    assert calls == 1


def _reward_ns(**overrides: float) -> argparse.Namespace:
    """A Namespace carrying the four reward-shaping flag dests, each defaulting to
    the matching ``RewardConfig()`` value; overrides replace individual coefficients.

    Lets ``_reward_config_from_args`` be exercised without the parser (and so
    without the engine), since the helper only reads these four attributes.
    """
    base = RewardConfig()
    values: dict[str, float] = {
        "enemy_hp_removed_coef": base.enemy_hp_removed,
        "damage_taken_coef": base.damage_taken,
        "floor_progress_coef": base.floor_progress,
        "boss_kill_coef": base.boss_kill,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_reward_config_from_args_maps_all_four_coefficients() -> None:
    """Each shaping flag maps to its RewardConfig field, and the un-exposed anneal
    schedule (beta_min / t_anneal) keeps the RewardConfig default. Engine-free: reads
    a plain Namespace, no parser or engine.
    """
    cfg = train_combat._reward_config_from_args(
        _reward_ns(
            enemy_hp_removed_coef=0.1,
            damage_taken_coef=-0.3,
            floor_progress_coef=0.4,
            boss_kill_coef=1.0,
        )
    )
    assert (cfg.enemy_hp_removed, cfg.damage_taken, cfg.floor_progress, cfg.boss_kill) == (
        0.1,
        -0.3,
        0.4,
        1.0,
    )
    assert cfg.beta_min == RewardConfig().beta_min
    assert cfg.t_anneal == RewardConfig().t_anneal


def test_reward_config_from_args_defaults_reproduce_reward_config() -> None:
    """With every flag at its default, the helper reproduces RewardConfig() exactly,
    so an unspecified run keeps the default shaping. Engine-free (plain Namespace)."""
    assert train_combat._reward_config_from_args(_reward_ns()) == RewardConfig()


def test_main_wires_reward_config_into_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() builds a RewardConfig from the shaping flags and passes it to BOTH the
    training and eval envs, so a shaping flag reaches the live reward on the combat
    side too (parity with test_train_run's same-named test).

    Engine-free: the adapter module (StsEnv + DEFAULT_MAX_EPISODE_STEPS, both pulled
    via deferred imports), the encounter pool, and train are all stubbed, so main()
    runs its whole env-construction path without a native build. The stub StsEnv
    records its reward_config kwarg.

    Revert-verify: drop reward_config=reward_config from either StsEnv build in
    main() and that env captures the RewardConfig() default (boss_kill 0.20 != 1.0),
    failing the assertion.
    """
    captured_reward_configs: list[RewardConfig | None] = []

    class _StubEnv:
        def __init__(
            self,
            *,
            ascension: int,
            max_episode_steps: int,
            encounters: object,
            reward_config: RewardConfig | None = None,
        ) -> None:
            captured_reward_configs.append(reward_config)

        def reset(
            self, *, seed: int | None = None, options: object = None
        ) -> tuple[object, dict[str, object]]:
            return None, {"engine_commit": "stub-commit", "interface_version": "stub-iface"}

        def close(self) -> None:
            pass

    # Bind both deferred `from sts_rl.env.adapter import ...` sites (build_arg_parser's
    # DEFAULT_MAX_EPISODE_STEPS and main's StsEnv) to an engine-free fake module.
    fake_adapter = types.ModuleType("sts_rl.env.adapter")
    fake_adapter.DEFAULT_MAX_EPISODE_STEPS = 500  # type: ignore[attr-defined]
    fake_adapter.StsEnv = _StubEnv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sts_rl.env.adapter", fake_adapter)

    # act1_encounter_pool() builds from engine enums; stub it so the default (unset
    # --encounters) path needs no engine.
    monkeypatch.setattr(train_combat, "act1_encounter_pool", lambda: ())

    def _fake_train(_env: object, config: object, *, eval_env: object = None) -> TrainHistory:
        # A last eval snapshot over the driver's DEFAULT band (base seed + the
        # --eval-episodes passed on argv below) keeps _final_eval_report on the reuse
        # path, so the final eval needs neither the engine nor a stubbed evaluate.
        return TrainHistory(
            records=[],
            actor_critic=ActorCritic(),
            eval_reports=[
                EvalRecord(
                    iteration=0,
                    global_step=1,
                    eval_seed_base=train_combat.DEFAULT_EVAL_BASE_SEED,
                    report=_make_report(0.0),
                )
            ],
        )

    monkeypatch.setattr(train_combat, "train", _fake_train)
    # --eval-episodes 4 matches _make_report's n_episodes so the reuse guard fires.
    monkeypatch.setattr(
        sys, "argv", ["train_combat", "--boss-kill-coef", "1.0", "--eval-episodes", "4"]
    )

    train_combat.main()

    # Both the training env and the eval env received the overridden shaping config.
    assert len(captured_reward_configs) == 2
    assert all(rc is not None and rc.boss_kill == 1.0 for rc in captured_reward_configs)


@pytest.mark.skipif(not _ENGINE_BUILT, reason="engine not built")
def test_arg_parser_reward_shaping_flags_wire_to_config() -> None:
    """The parser's reward-shaping flags default to RewardConfig() and --boss-kill-coef
    overrides only boss_kill.

    Engine-gated: train_combat.build_arg_parser imports the adapter (for
    DEFAULT_MAX_EPISODE_STEPS), which pulls the native engine. The pure-Namespace
    tests above already lock the mapping without a build.
    """
    parser = train_combat.build_arg_parser()
    assert train_combat._reward_config_from_args(parser.parse_args([])) == RewardConfig()
    overridden = train_combat._reward_config_from_args(
        parser.parse_args(["--boss-kill-coef", "1.0"])
    )
    assert overridden.boss_kill == 1.0
    # Overriding one coefficient leaves the others at their RewardConfig default.
    assert overridden.enemy_hp_removed == RewardConfig().enemy_hp_removed


@pytest.mark.skipif(not _ENGINE_BUILT, reason="engine not built")
def test_parse_encounters_validates_against_enum_members() -> None:
    """The --encounters parser accepts a real member, rejects a bogus name, and
    rejects the engine's INVALID sentinel (which dir()-based validation, admitting
    every attribute name, wrongly accepted).
    """
    from sts_rl.env._engine import slaythespire as sts

    # A real member parses to exactly that enum value.
    assert train_combat._parse_encounters("GREMLIN_NOB") == (sts.MonsterEncounter.GREMLIN_NOB,)

    # A genuinely unknown name is still rejected (unchanged behavior).
    with pytest.raises(argparse.ArgumentTypeError):
        train_combat._parse_encounters("NOT_A_REAL_ENCOUNTER")

    # INVALID is a real __members__ entry but the engine's sentinel, so the parser
    # must reject it explicitly rather than build a battle from it.
    with pytest.raises(argparse.ArgumentTypeError):
        train_combat._parse_encounters("INVALID")
