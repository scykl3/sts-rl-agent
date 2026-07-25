"""Canonical Slay the Spire encounter pools, named as MonsterEncounter members.

The pools are defined as engine enum-member *names* (SCREAMING_SNAKE_CASE, the
same strings as ``sts.MonsterEncounter.__members__``) so this module imports
without a native engine build. :func:`resolve_encounter_names` maps those names
to live enum values lazily, and :func:`act1_encounter_pool` returns the Act 1
pool ready to hand to :class:`~sts_rl.env.adapter.StsEnv`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sts_rl.interface import InterfaceError

# The engine's sentinel: a real ``__members__`` entry but not a spawnable
# encounter, rejected exactly as the --encounters CLI parser rejects it.
_INVALID_SENTINEL = "INVALID"

# The "full sampled Act 1" training pool: every Ironclad Act 1 (Exordium) hallway
# monster room plus the three Act 1 elites. These are exactly the engine's Act 1
# pools in constants/MonsterEncounters.h (MonsterEncounterPool: weakEnemies[0],
# strongEnemies[0], elites[0]). Act 1 bosses (SLIME_BOSS, THE_GUARDIAN,
# HEXAGHOST) are intentionally excluded: they are boss rooms, not hallway fights.
ACT1_ENCOUNTER_NAMES: tuple[str, ...] = (
    # Weak pool: the act's first monster room.
    "CULTIST",
    "JAW_WORM",
    "TWO_LOUSE",
    "SMALL_SLIMES",
    # Strong pool: the act's later monster rooms.
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
    # Elites.
    "GREMLIN_NOB",
    "LAGAVULIN",
    "THREE_SENTRIES",
)


def resolve_encounter_names(names: Sequence[str]) -> tuple[Any, ...]:
    """Resolve ``MonsterEncounter`` member names to live engine enum values.

    Validates each name against ``sts.MonsterEncounter.__members__`` (the real
    enum members: unlike ``dir()`` this excludes Python attributes such as
    ``name``/``value``) and rejects the engine's ``INVALID`` sentinel. This is
    the same validity rule the ``--encounters`` CLI parser applies, so a
    hardcoded pool and a user-supplied list are accepted or rejected the same
    way. The engine is imported lazily so this module stays importable without a
    native build.

    Raises :class:`~sts_rl.interface.InterfaceError` naming any unknown or
    sentinel entry.
    """
    from sts_rl.env._engine import slaythespire as sts

    members = sts.MonsterEncounter.__members__
    resolved: list[Any] = []
    for name in names:
        if name == _INVALID_SENTINEL:
            raise InterfaceError(
                f"{_INVALID_SENTINEL} is the engine's sentinel, not a real MonsterEncounter"
            )
        if name not in members:
            raise InterfaceError(f"unknown MonsterEncounter {name!r}")
        resolved.append(members[name])
    return tuple(resolved)


def act1_encounter_pool() -> tuple[Any, ...]:
    """Return the canonical full-sampled Act 1 pool as live ``MonsterEncounter`` values.

    This is the default combat pool for the single-combat training entry point
    (``scripts/train_combat.py``); each :class:`~sts_rl.env.adapter.StsEnv` reset
    samples one member deterministically from the reset seed.
    """
    return resolve_encounter_names(ACT1_ENCOUNTER_NAMES)
