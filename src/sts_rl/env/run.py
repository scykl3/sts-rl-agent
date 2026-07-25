"""Thin, raw wrapper over the engine's overworld: start a run, read run state,
enumerate and execute the non-combat moves that connect one battle to the next.

This is the run-mode sibling of :mod:`sts_rl.env.engine`, which covers combat.
Where that module drives a ``BattleContext``, this one drives the ``GameContext``
overworld state machine: the map, events, shops, campfires, and reward screens.
It deliberately does *not* encode observations, apply the interface action
layout, or implement the Gymnasium API; those live in higher modules built on
top of this one. It only:

- starts a seeded Ironclad run at its first overworld decision, and
- reads raw ``GameContext`` values and enumerates / executes the legal overworld
  moves the engine reports for the current screen.

Safety: an overworld move is executed only after the engine confirms it legal via
``GameAction.isValidAction``. ``GameAction.execute`` on an invalid action is
undefined under the engine's asserts (it can abort the process), so this module
never hands the engine an unvalidated action. This mirrors the combat action
legality contract enforced in :mod:`sts_rl.env.actions`.

All engine access goes through :mod:`sts_rl.env._engine`, so importing this
module raises if the native engine has not been built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.engine import EngineError

# The engine module ships no type stubs, so its classes (GameContext, GameAction,
# ...) are untyped at the boundary and annotated as Any here.


def start_run(
    seed: int,
    *,
    ascension: int = 0,
    neow_mini_blessing: bool = False,
) -> Any:
    """Start a seeded Ironclad run and return its ``GameContext``.

    The run is returned at its first overworld decision (the engine opens a new
    run on the Neow reward screen); no navigation is performed, so the caller
    drives every subsequent choice. Unlike :func:`~sts_rl.env.engine.start_combat`
    this does not walk into a battle: run mode makes the map / event / shop / rest
    choices that combat mode skips over.

    Raises :class:`~sts_rl.env.engine.EngineError` if the freshly created run is
    already terminal or offers no legal overworld action (either would mean a
    broken engine build).
    """
    gc = sts.GameContext(
        sts.CharacterClass.IRONCLAD,
        int(seed),
        int(ascension),
        neow_mini_blessing=neow_mini_blessing,
    )
    if gc.outcome != sts.GameOutcome.UNDECIDED:
        raise EngineError(f"run started already terminal (outcome={gc.outcome})")
    if not overworld_actions(gc):
        raise EngineError(f"run started with no legal action at screen {gc.screen_state}")
    return gc


def overworld_actions(gc: Any) -> tuple[Any, ...]:
    """Return the engine's legal ``GameAction`` list for the current overworld screen.

    Wraps ``GameAction.getAllActionsInState``, which enumerates only actions that
    pass ``isValidAction`` for the current screen (and returns an empty list once
    the run is terminal). In the engine's ``BATTLE`` screen this is empty: combat
    moves are ``search::Action`` objects handled by :mod:`sts_rl.env.actions`, not
    overworld ``GameAction`` objects.
    """
    return tuple(sts.GameAction.getAllActionsInState(gc))


def execute_overworld_action(gc: Any, action: Any) -> None:
    """Execute one overworld ``GameAction`` after confirming it legal.

    Gates on ``action.isValidAction(gc)`` before ``execute`` so an invalid move
    is never handed to the engine. Raises :class:`~sts_rl.env.engine.EngineError`
    (rather than tripping an uncatchable engine assert) if ``action`` is not legal
    in the current state. Mutates ``gc``.
    """
    if not action.isValidAction(gc):
        raise EngineError(
            f"illegal overworld action {describe_action(gc, action)!r} at screen {gc.screen_state}"
        )
    action.execute(gc)


def describe_action(gc: Any, action: Any) -> str:
    """Return the engine's human-readable description of ``action`` in ``gc``."""
    return str(action.getDesc(gc))


def is_run_over(gc: Any) -> bool:
    """Whether the run has reached a terminal outcome (win or loss)."""
    return gc.outcome != sts.GameOutcome.UNDECIDED


def run_won(gc: Any) -> bool:
    """Whether the run ended in victory. Meaningful only when :func:`is_run_over`."""
    return gc.outcome == sts.GameOutcome.PLAYER_VICTORY


@dataclass(frozen=True)
class RunSnapshot:
    """Raw, unnormalized readout of overworld ``GameContext`` state.

    Values are exactly as the engine reports them (no interface encoding). This
    is a debugging / verification view of run state, not the observation the
    agent consumes. Enum-valued fields (``screen``, ``outcome``, ``cur_room``,
    ``cur_event``) are stored as their engine enum names.

    ``map_node_x`` / ``map_node_y`` are the current map coordinates; the engine
    uses ``map_node_y == -1`` for a run positioned before the first map row.
    ``action_count`` is the number of legal overworld moves in this state.
    """

    screen: str
    outcome: str
    floor: int
    act: int
    ascension: int
    player_hp: int
    player_max_hp: int
    gold: int
    map_node_x: int
    map_node_y: int
    cur_room: str
    cur_event: str
    deck_size: int
    relic_count: int
    action_count: int


def read_run(gc: Any) -> RunSnapshot:
    """Read a :class:`RunSnapshot` of raw values from a ``GameContext``. Reads only."""
    return RunSnapshot(
        screen=gc.screen_state.name,
        outcome=gc.outcome.name,
        floor=int(gc.floor_num),
        act=int(gc.act),
        ascension=int(gc.ascension),
        player_hp=int(gc.cur_hp),
        player_max_hp=int(gc.max_hp),
        gold=int(gc.gold),
        map_node_x=int(gc.cur_map_node_x),
        map_node_y=int(gc.cur_map_node_y),
        cur_room=gc.cur_room.name,
        cur_event=gc.cur_event.name,
        deck_size=len(gc.deck),
        relic_count=len(gc.relics),
        action_count=len(overworld_actions(gc)),
    )
