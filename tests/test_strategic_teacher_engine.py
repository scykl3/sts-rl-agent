"""Engine-present cross-checks for StrategicTeacher.

The agent-side teacher duplicates the env encoder's ``map_context`` /
``map_lookahead`` layout with named constants (it cannot import
``sts_rl.env.observation``, which loads the native engine). These tests import
BOTH modules and assert the teacher's per-field offsets land on the encoder's
ACTUAL fill positions, and that the potion tables use real ``Potion`` enum names -
so a future encoder reorder or a mis-typed potion name fails loudly instead of
silently degrading the teacher's choice. Skips cleanly when the engine is absent.
"""

from __future__ import annotations

import pytest

try:
    import sts_rl.env._engine  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without a build
    pytest.skip(f"engine not built ({exc})", allow_module_level=True)

from sts_rl.agent import strategic_teacher as st
from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.observation import (
    MAP_LOOKAHEAD_ELITE_CAP,
    _MAP_AGG_GLOBAL,
    _MAP_AGG_PER_COL,
    _MAP_CUR_BLOCK,
    _MAP_NO_REST_DIST,
    _MAP_PER_COL_FEATS,
    _map_dp,
    encode_observation,
)
from sts_rl.env.run import execute_overworld_action, overworld_actions, start_run

_SEED = 42
_MAX_DRIVE = 50


def _drive_to_first_map_screen(seed: int = _SEED):
    """Return a run positioned at its first map screen (still before the first row)."""
    gc = start_run(seed=seed)
    for _ in range(_MAX_DRIVE):
        if gc.screen_state == sts.ScreenState.MAP_SCREEN:
            return gc
        execute_overworld_action(gc, overworld_actions(gc)[0])
    raise AssertionError("run never reached a map screen")


def test_map_block_layout_matches_encoder() -> None:
    """The agent-side block widths equal the encoder's (structural cross-check)."""
    assert st._MAP_CTX_COL_BASE == _MAP_CUR_BLOCK
    assert st._MAP_CTX_PER_COL == _MAP_PER_COL_FEATS
    assert st._MAP_LA_GLOBAL == _MAP_AGG_GLOBAL
    assert st._MAP_LA_PER_COL == _MAP_AGG_PER_COL


def test_map_per_field_offsets_match_encoder_fill_positions() -> None:
    """Teacher is_combat/is_elite and elites/rest_dist offsets hit the real fills.

    The sum assertions in strategic_teacher only guard total width; this pins the
    per-field ordering against the encoder's actual output and the engine's map, so
    a future encoder reorder (e.g. swapping is_combat and is_elite) fails here.
    """
    gc = _drive_to_first_map_screen()
    assert gc.cur_map_node_y < 0  # pre-first-row: reachable == row-0 rooms
    obs = encode_observation(gc, None)
    mc = obs["map_context"]
    look = obs["map_lookahead"]

    legal_cols = {action.idx1 for action in overworld_actions(gc)}
    assert legal_cols, "expected at least one legal map column"

    elite = int(sts.Room.ELITE)
    combat_rooms = (int(sts.Room.MONSTER), elite, int(sts.Room.BOSS))
    emin, _emax, drest = _map_dp(gc.map)

    for col in legal_cols:
        room_id = int(gc.map.get_room_type(col, 0))

        # map_context per-column: the teacher's is_combat / is_elite reads must equal
        # the encoder's fill for this column's real room type.
        base = st._MAP_CTX_COL_BASE + col * st._MAP_CTX_PER_COL
        assert mc[base + st._MAP_CTX_IS_COMBAT] == (1.0 if room_id in combat_rooms else 0.0)
        assert mc[base + st._MAP_CTX_IS_ELITE] == (1.0 if room_id == elite else 0.0)

        # map_lookahead per-column: the teacher's elites / rest_dist reads must equal
        # the encoder's DP fill for this column (pre-map -> row 0).
        la = st._MAP_LA_GLOBAL + col * st._MAP_LA_PER_COL
        expected_elites = min(emin[0][col], MAP_LOOKAHEAD_ELITE_CAP) / MAP_LOOKAHEAD_ELITE_CAP
        expected_rest = drest[0][col] / _MAP_NO_REST_DIST
        assert look[la + st._MAP_LA_ELITES] == pytest.approx(expected_elites)
        assert look[la + st._MAP_LA_REST_DIST] == pytest.approx(expected_rest)


def test_potion_tables_use_real_engine_enum_names() -> None:
    """Every potion-table name is a real Potion enum member (catches a typo)."""
    members = set(sts.Potion.__members__)
    missing_defensive = st.DEFENSIVE_UNTARGETED_POTIONS - members
    assert not missing_defensive, f"unknown defensive potion names: {sorted(missing_defensive)}"
    missing_valued = set(st.POTION_VALUES) - members
    assert not missing_valued, f"unknown potion-value names: {sorted(missing_valued)}"
