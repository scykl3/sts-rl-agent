"""Tests for the canonical Act 1 encounter pool constant and its resolver.

Every entry is validated against the live engine's ``MonsterEncounter`` enum, so
this module needs the built engine (``scripts/build_engine.sh``) and skips
cleanly without it, matching the suite's ImportError -> skip idiom.
"""

from __future__ import annotations

import pytest

# Skip the whole module unless the compiled engine is importable: resolving the
# pool to enum values and checking membership both need the native binding.
# Importing _engine raises EngineNotBuiltError (an ImportError) when the native
# module is absent; catch it and skip at module load. (pytest.importorskip only
# skips on a missing module, not on an ImportError raised during import.)
try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.encounters import (
    ACT1_ENCOUNTER_NAMES,
    act1_encounter_pool,
    resolve_encounter_names,
)
from sts_rl.interface import InterfaceError

# The three Act 1 elites the pool must carry alongside the hallway fights,
# asserted explicitly so the "hallway fights + the three elites" claim is locked.
ACT1_ELITE_NAMES = ("GREMLIN_NOB", "LAGAVULIN", "THREE_SENTRIES")


def test_act1_pool_is_non_empty_and_all_valid_members() -> None:
    """The pool is non-empty and every entry is a real ``MonsterEncounter`` value."""
    pool = act1_encounter_pool()
    assert len(pool) > 0
    valid_members = set(sts.MonsterEncounter.__members__.values())
    for enc in pool:
        assert isinstance(enc, sts.MonsterEncounter)
        assert enc in valid_members
        assert enc != sts.MonsterEncounter.INVALID


def test_act1_pool_matches_names_one_to_one() -> None:
    """The resolved pool is exactly ``ACT1_ENCOUNTER_NAMES`` mapped to enum values."""
    members = sts.MonsterEncounter.__members__
    expected = tuple(members[name] for name in ACT1_ENCOUNTER_NAMES)
    assert act1_encounter_pool() == expected


def test_act1_pool_has_no_duplicate_names() -> None:
    """No duplicate names: a copy-paste dupe would silently over-weight a fight."""
    assert len(ACT1_ENCOUNTER_NAMES) == len(set(ACT1_ENCOUNTER_NAMES))


def test_act1_pool_includes_the_three_elites() -> None:
    """The pool carries the three Act 1 elites (not only hallway fights)."""
    for elite in ACT1_ELITE_NAMES:
        assert elite in ACT1_ENCOUNTER_NAMES


def test_resolve_rejects_invalid_and_unknown() -> None:
    """resolve_encounter_names enforces the same rule as the --encounters parser:
    reject the engine's INVALID sentinel and any unknown name."""
    with pytest.raises(InterfaceError):
        resolve_encounter_names(["INVALID"])
    with pytest.raises(InterfaceError):
        resolve_encounter_names(["NOT_A_REAL_ENCOUNTER"])
