"""Error-path tests for the engine wrapper and the module loader.

These do not need the native engine build. ``start_combat``'s three failure
branches (run ended, no legal actions, navigation budget exhausted) are driven
with a small fake engine injected in place of the real bindings, so they run on
any machine. The loader's not-built path is exercised by loading a fresh copy of
``_engine.py`` with the underlying import forced to fail, so it runs whether or
not the engine is present.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_ENGINE_SOURCE = Path(__file__).resolve().parents[1] / "src" / "sts_rl" / "env" / "_engine.py"

# Sentinel screen/outcome values for the fake engine. Their identity is all that
# matters: ``start_combat`` only compares them against the fake's own enums.
_BATTLE = "BATTLE"
_NON_BATTLE = "MAP"
_UNDECIDED = "UNDECIDED"
_RUN_ENDED = "PLAYER_LOSS"


class _FakeAction:
    """A navigation action whose ``execute`` never advances toward battle."""

    def execute(self, gc: object) -> None:  # noqa: D102 - no-op by design
        return None


def _make_fake_sts(gc: SimpleNamespace) -> SimpleNamespace:
    """Build a stand-in ``slaythespire`` module that hands out ``gc``."""
    return SimpleNamespace(
        CharacterClass=SimpleNamespace(IRONCLAD="IRONCLAD"),
        ScreenState=SimpleNamespace(BATTLE=_BATTLE),
        GameOutcome=SimpleNamespace(UNDECIDED=_UNDECIDED),
        GameContext=lambda *args, **kwargs: gc,
        GameAction=SimpleNamespace(getAllActionsInState=lambda g: g.actions),
    )


def _import_engine_with_fake(monkeypatch: pytest.MonkeyPatch, gc: SimpleNamespace) -> ModuleType:
    """Import ``sts_rl.env.engine`` bound to a fake engine built around ``gc``.

    The fake is installed as ``sts_rl.env._engine`` before importing so the
    ``from sts_rl.env._engine import slaythespire`` line in ``engine`` picks it
    up; both module entries are restored on teardown by ``monkeypatch``.
    """
    fake_engine = ModuleType("sts_rl.env._engine")
    fake_engine.slaythespire = _make_fake_sts(gc)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sts_rl.env._engine", fake_engine)
    monkeypatch.delitem(sys.modules, "sts_rl.env.engine", raising=False)
    return importlib.import_module("sts_rl.env.engine")


def test_start_combat_raises_when_run_ends_before_combat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that reaches a terminal outcome before battle raises EngineError."""
    gc = SimpleNamespace(screen_state=_NON_BATTLE, outcome=_RUN_ENDED, actions=[_FakeAction()])
    engine = _import_engine_with_fake(monkeypatch, gc)

    with pytest.raises(engine.EngineError, match="run ended"):
        engine.start_combat(seed=1)


def test_start_combat_raises_when_no_actions_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-combat screen with no legal actions raises EngineError."""
    gc = SimpleNamespace(screen_state=_NON_BATTLE, outcome=_UNDECIDED, actions=[])
    engine = _import_engine_with_fake(monkeypatch, gc)

    with pytest.raises(engine.EngineError, match="no legal actions"):
        engine.start_combat(seed=1)


def test_start_combat_raises_when_nav_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never reaching battle within max_nav_actions raises EngineError."""
    # Actions are always available but never advance to battle, so navigation
    # runs until the budget is spent.
    gc = SimpleNamespace(screen_state=_NON_BATTLE, outcome=_UNDECIDED, actions=[_FakeAction()])
    engine = _import_engine_with_fake(monkeypatch, gc)

    max_nav = 3
    with pytest.raises(engine.EngineError, match=f"within {max_nav} navigation actions"):
        engine.start_combat(seed=1, max_nav_actions=max_nav)


def test_load_wraps_import_error_as_engine_not_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader wraps a failed engine import as EngineNotBuiltError.

    Build-independent: rather than importing the installed module (which only
    fails when the engine is genuinely absent, and whose import error this
    pytest surfaces as an error rather than a skip), load a fresh copy of
    ``_engine.py`` from source with ``importlib.import_module`` patched to fail.
    The module's top-level ``_load()`` then takes the error-wrapping path whether
    or not the native engine is present.
    """

    def _raise(_name: str, _package: str | None = None) -> ModuleType:
        raise ImportError("simulated missing engine module")

    monkeypatch.setattr(importlib, "import_module", _raise)

    spec = importlib.util.spec_from_file_location("sts_rl_engine_under_test", _ENGINE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    # EngineNotBuiltError subclasses ImportError; assert the wrapped message names
    # the build command. _BUILD_COMMAND is assigned before _load() runs, so it is
    # populated on the partially-initialized module even though exec_module raised.
    with pytest.raises(ImportError) as excinfo:
        spec.loader.exec_module(module)
    assert module._BUILD_COMMAND in str(excinfo.value)
