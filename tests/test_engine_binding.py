"""Smoke tests for the raw engine binding wrapper.

These require the native engine module to be built (scripts/build_engine.sh).
The ``importorskip`` below skips the whole module cleanly when it is absent, so
the suite stays green on machines without a native build (for example CI).
"""

from __future__ import annotations

import pytest

# Skip the entire module unless the compiled engine is importable. Importing
# _engine raises EngineNotBuiltError (an ImportError) when the native module is
# absent; catch it and skip at module load so the suite stays green without a
# build. (pytest.importorskip only skips on a missing module, not on an
# ImportError raised during import, so guard explicitly.)
try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.env.engine import CombatSnapshot, engine_commit, read_combat, start_combat

# Ironclad Ascension 0 starting values, verified against the pinned engine build.
# The exact per-seed values below are regression locks against the pinned commit;
# a change here means the engine or the navigation path moved.
REGRESSION_SEED = 42
IRONCLAD_MAX_HP = 80
IRONCLAD_START_ENERGY = 3
OPENING_HAND_SIZE = 5

# A spread of seeds that should each reach a first combat without error.
REACHABLE_SEEDS = (1, 7, 42, 123, 2024)

# The first combat for the regression seed is a single Cultist at full HP.
REGRESSION_FIRST_ENEMY = "Cultist"
CULTIST_MAX_HP = 52

# A seed whose first combat has more than one enemy, to exercise indexing.
MULTI_ENEMY_SEED = 3

COMMIT_SHA_LEN = 40


def test_start_combat_reads_raw_state() -> None:
    """A seeded run reaches combat and exposes readable BattleContext state."""
    _, bc = start_combat(seed=REGRESSION_SEED)
    snap = read_combat(bc)

    assert snap.turn == 0
    assert snap.player_hp == snap.player_max_hp == IRONCLAD_MAX_HP
    assert snap.player_energy == IRONCLAD_START_ENERGY
    assert snap.player_block == 0
    assert snap.monster_count >= 1
    assert snap.monsters_alive >= 1
    assert snap.hand_size == OPENING_HAND_SIZE == len(snap.hand_card_ids)
    # Card ids read as engine enum names (e.g. "STRIKE_RED"), never empty.
    assert all(isinstance(name, str) and name for name in snap.hand_card_ids)

    # Per-enemy state is readable: this seed's first combat is one full-HP Cultist.
    assert len(snap.monsters) == snap.monster_count == 1
    enemy = snap.monsters[0]
    assert enemy.idx == 0
    assert enemy.name == REGRESSION_FIRST_ENEMY
    assert enemy.hp == enemy.max_hp == CULTIST_MAX_HP
    assert enemy.alive is True


def test_start_combat_is_deterministic() -> None:
    """The same seed produces an identical initial combat readout."""
    _, bc_a = start_combat(seed=REGRESSION_SEED)
    _, bc_b = start_combat(seed=REGRESSION_SEED)
    assert read_combat(bc_a) == read_combat(bc_b)


@pytest.mark.parametrize("seed", REACHABLE_SEEDS)
def test_various_seeds_reach_combat(seed: int) -> None:
    """Several seeds each reach a valid first combat (structural invariants only)."""
    _, bc = start_combat(seed=seed)
    snap = read_combat(bc)
    assert isinstance(snap, CombatSnapshot)
    assert 0 < snap.player_hp <= snap.player_max_hp
    assert snap.monster_count >= 1
    assert snap.hand_size >= 1
    # Per-enemy readout is present and slot-aligned for every enemy.
    assert len(snap.monsters) == snap.monster_count
    for slot, enemy in enumerate(snap.monsters):
        assert enemy.idx == slot
        assert enemy.name and enemy.monster_id
        assert 0 <= enemy.hp <= enemy.max_hp


def test_multi_enemy_combat_indexes_each_monster() -> None:
    """A multi-enemy combat exposes an indexed, distinct readout per enemy."""
    _, bc = start_combat(seed=MULTI_ENEMY_SEED)
    snap = read_combat(bc)
    assert snap.monster_count >= 2
    assert len(snap.monsters) == snap.monster_count
    assert [m.idx for m in snap.monsters] == list(range(snap.monster_count))


def test_engine_commit_is_pinned() -> None:
    """The pinned engine commit is a full-length SHA."""
    assert len(engine_commit()) == COMMIT_SHA_LEN
