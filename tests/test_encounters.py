"""Tests for the canonical Act 1 encounter pool constant and its resolver.

The pool is a pure-Python tuple of enum-member names, so its structural
invariants - the exact name set, its length, no duplicates, and the three elites
- are asserted without the compiled engine. Only mapping those names to live
``MonsterEncounter`` values (member validity and the resolver's reject rule)
needs the native binding (``scripts/build_engine.sh``); those tests skip cleanly
without it, matching the suite's ImportError -> skip idiom.
"""

from __future__ import annotations

import pytest

from sts_rl.env.encounters import (
    ACT1_ENCOUNTER_NAMES,
    act1_encounter_pool,
    resolve_encounter_names,
)
from sts_rl.interface import InterfaceError

# Probe the compiled engine once. Importing _engine raises EngineNotBuiltError
# (an ImportError) when the native module is absent; only the name-validity and
# resolver tests below actually need it, so gate just those with skipif and let
# the pure-Python invariants run engine-free. (pytest.importorskip only skips on
# a missing module, not on an ImportError raised during import; a module-level
# skip would wrongly take the constant-only invariants down with it.)
try:
    import sts_rl.env._engine  # noqa: F401

    _ENGINE_AVAILABLE = True
    _ENGINE_SKIP_REASON = ""
except ImportError as exc:  # pragma: no cover - exercised only without a build
    _ENGINE_AVAILABLE = False
    _ENGINE_SKIP_REASON = f"engine not built ({exc})"

requires_engine = pytest.mark.skipif(not _ENGINE_AVAILABLE, reason=_ENGINE_SKIP_REASON)

# The three Act 1 elites the pool must carry alongside the hallway fights,
# asserted explicitly so the "hallway fights + the three elites" claim is locked.
ACT1_ELITE_NAMES = ("GREMLIN_NOB", "LAGAVULIN", "THREE_SENTRIES")

# The full Act 1 pool pinned literally, independent of the production constant, so
# a wrong, dropped, reordered, or added name is caught. Comparing the constant to
# itself (even mapped through the engine) is tautological and cannot catch that.
EXPECTED_ACT1_NAMES = (
    "CULTIST",
    "JAW_WORM",
    "TWO_LOUSE",
    "SMALL_SLIMES",
    "GREMLIN_GANG",
    "LOTS_OF_SLIMES",
    "RED_SLAVER",
    "EXORDIUM_THUGS",
    "EXORDIUM_WILDLIFE",
    "BLUE_SLAVER",
    "LOOTER",
    "LARGE_SLIME",
    "THREE_LOUSE",
    "TWO_FUNGI_BEASTS",
    "GREMLIN_NOB",
    "LAGAVULIN",
    "THREE_SENTRIES",
)
EXPECTED_ACT1_POOL_SIZE = 17


# Pure-Python invariants: inspect only the constant, so they run without the engine.


def test_act1_pool_names_are_the_expected_seventeen() -> None:
    """The pool is exactly the 17 pinned names, in order, so a wrong, dropped, or
    reordered hallway name is caught (the old constant-vs-itself check could not).
    The size pin is a second, independent tripwire against a coordinated edit."""
    assert ACT1_ENCOUNTER_NAMES == EXPECTED_ACT1_NAMES
    assert len(ACT1_ENCOUNTER_NAMES) == EXPECTED_ACT1_POOL_SIZE


def test_act1_pool_has_no_duplicate_names() -> None:
    """No duplicate names: a copy-paste dupe would silently over-weight a fight."""
    assert len(ACT1_ENCOUNTER_NAMES) == len(set(ACT1_ENCOUNTER_NAMES))


def test_act1_pool_includes_the_three_elites() -> None:
    """The pool carries the three Act 1 elites (not only hallway fights)."""
    for elite in ACT1_ELITE_NAMES:
        assert elite in ACT1_ENCOUNTER_NAMES


def test_resolve_rejects_empty_names() -> None:
    """resolve_encounter_names([]) raises rather than returning (): the empty-pool
    rejection is the resolver's own rule, matching --encounters and StsEnv, so its
    docstring's "same validity rule" claim holds. Unmarked (not engine-gated)
    because the empty guard runs before the lazy engine import, so it is pure-Python."""
    with pytest.raises(InterfaceError):
        resolve_encounter_names([])


# Engine-dependent checks: mapping names to live enum values needs the native binding.


@requires_engine
def test_act1_pool_is_non_empty_and_all_valid_members() -> None:
    """The pool is non-empty and every entry is a real ``MonsterEncounter`` value."""
    from sts_rl.env._engine import slaythespire as sts

    pool = act1_encounter_pool()
    assert len(pool) > 0
    valid_members = set(sts.MonsterEncounter.__members__.values())
    for enc in pool:
        assert isinstance(enc, sts.MonsterEncounter)
        assert enc in valid_members
        assert enc != sts.MonsterEncounter.INVALID


@requires_engine
def test_resolve_rejects_invalid_and_unknown() -> None:
    """resolve_encounter_names enforces the same rule as the --encounters parser:
    reject the engine's INVALID sentinel and any unknown name. The "INVALID"
    literal here pins the user-facing rejected string, deliberately independent of
    the production _INVALID_SENTINEL constant."""
    with pytest.raises(InterfaceError):
        resolve_encounter_names(["INVALID"])
    with pytest.raises(InterfaceError):
        resolve_encounter_names(["NOT_A_REAL_ENCOUNTER"])
