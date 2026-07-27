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
from pathlib import Path
from typing import cast

import gymnasium as gym
import pytest

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.train import EvalRecord, TrainHistory
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
