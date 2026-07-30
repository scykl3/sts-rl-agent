"""Thin, raw wrapper over the engine bindings: start a seeded combat, read state.

This is the lowest layer of the environment adapter. It deliberately does *not*
encode observations, apply the action contract, or implement the Gymnasium API;
those live in higher modules built on top of this one. It only:

- starts a seeded Ironclad run and advances it into the run's first combat, and
- reads raw :class:`BattleContext` values (no normalization, no tensors).

All engine access goes through :mod:`sts_rl.env._engine`, so importing this
module raises if the native engine has not been built.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts_rl.env._engine import slaythespire as sts
from sts_rl.env.combat_state import StateSpec, resolve_deck, resolve_relics
from sts_rl.env.encounters import resolve_encounter_names

# The engine module ships no type stubs, so its classes (GameContext,
# BattleContext, ...) are untyped at the boundary and annotated as Any here.

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_CONFIG = _REPO_ROOT / "configs" / "engine.toml"

# Safety cap on actions taken to walk from the run start to the first combat.
# A normal path to the first monster room is a handful of actions; this only
# guards against an unexpected state machine that never enters battle.
DEFAULT_MAX_NAV_ACTIONS = 500


class EngineError(RuntimeError):
    """Raised when the engine cannot be driven into the expected state."""


def start_combat(
    seed: int,
    *,
    ascension: int = 0,
    max_nav_actions: int = DEFAULT_MAX_NAV_ACTIONS,
    encounter: Any = None,  # sts.MonsterEncounter | None
) -> tuple[Any, Any]:
    """Start a seeded Ironclad run and advance it to the first combat.

    Returns ``(game_context, battle_context)``. Navigation takes the first legal
    action at each pre-combat screen; this is a fixed function of engine state,
    so for a given ``seed`` it reproducibly lands in the run's first battle.

    When ``encounter`` (a :class:`MonsterEncounter`) is given, that battle is
    built directly on the fresh run state (floor 0) and navigation is skipped, so
    the returned combat is the chosen encounter rather than the run's first
    monster room. That fresh state carries the starting deck, full HP, and no
    act-earned relics, so the chosen encounter is a single-combat training setup,
    not a mid-act state.

    Raises :class:`EngineError` if the run ends, offers no actions, or fails to
    reach a battle within ``max_nav_actions`` steps (default, navigated path only).
    """
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, int(seed), int(ascension))
    if encounter is not None:
        # Chosen-encounter path: build the battle directly on the fresh run state
        # (floor 0); no pre-combat navigation is needed, so the nav guards below
        # do not apply.
        return gc, gc.create_battle_context(encounter)
    steps = 0
    while gc.screen_state != sts.ScreenState.BATTLE:
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            raise EngineError(f"run ended (outcome={gc.outcome}) before reaching combat")
        if steps >= max_nav_actions:
            raise EngineError(f"did not reach combat within {max_nav_actions} navigation actions")
        actions = sts.GameAction.getAllActionsInState(gc)
        if not actions:
            raise EngineError(f"no legal actions at screen {gc.screen_state}")
        actions[0].execute(gc)
        steps += 1
    return gc, gc.create_battle_context()


def start_combat_from_state(spec: StateSpec, *, seed: int = 0) -> tuple[Any, Any]:
    """Build a combat from a mid-run :class:`~sts_rl.env.combat_state.StateSpec`.

    Mirrors :func:`start_combat`'s contract - returns ``(game_context,
    battle_context)`` - but instead of navigating a fresh run to its first fight it
    constructs the run state the spec describes and builds the spec's chosen
    encounter directly (no navigation). The steps, in order:

    1. Create a fresh Ironclad ``GameContext`` at ``spec.ascension`` seeded by
       ``seed`` (the seed drives the battle RNG: enemy HP rolls, move sequence).
    2. Clear the starter deck and obtain the spec's cards with their upgrades.
    3. Obtain the spec's relics (in addition to the always-present starter relic;
       see :mod:`sts_rl.env.combat_state`).
    4. Set ``max_hp`` then ``cur_hp`` (max first so lowering it never clamps the
       current value below the intended one).
    5. Build the spec's ``MonsterEncounter`` on that state.

    Raises :class:`~sts_rl.interface.InterfaceError` (via the resolvers) if any
    card, relic, or encounter name is unknown or the engine's ``INVALID`` sentinel.
    """
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, int(seed), int(spec.ascension))
    gc.clear_deck()
    for card in resolve_deck(spec.deck):
        gc.obtain_card(card)
    for relic in resolve_relics(spec.relics):
        gc.obtain_relic(relic)
    # max_hp before cur_hp: setting max below the current HP would otherwise clamp it.
    gc.max_hp = int(spec.max_hp)
    gc.cur_hp = int(spec.effective_cur_hp)
    encounter = resolve_encounter_names([spec.encounter])[0]
    return gc, gc.create_battle_context(encounter)


@dataclass(frozen=True)
class MonsterSnapshot:
    """Raw, unnormalized readout of one enemy in a :class:`BattleContext`.

    ``intent_damage`` and ``intent_hits`` are the engine's predicted per-hit base
    damage and hit count for the monster's current move, before strength and
    vulnerable are applied. ``intent_hits == 0`` means the current move is not an
    attack.
    """

    idx: int
    name: str
    monster_id: str
    hp: int
    max_hp: int
    block: int
    alive: bool
    intent_damage: int
    intent_hits: int


@dataclass(frozen=True)
class CombatSnapshot:
    """Raw, unnormalized readout of a :class:`BattleContext`.

    Values are exactly as the engine reports them (no contract encoding). This
    is a debugging/verification view of combat state, not the observation the
    agent consumes.
    """

    turn: int
    player_hp: int
    player_max_hp: int
    player_block: int
    player_energy: int
    monster_count: int
    monsters_alive: int
    monsters: tuple[MonsterSnapshot, ...]
    hand_size: int
    hand_card_ids: tuple[str, ...]


def read_combat(bc: Any) -> CombatSnapshot:
    """Read a :class:`CombatSnapshot` of raw values from a BattleContext."""
    player = bc.player
    group = bc.monsters
    monsters = tuple(_read_monster(group[i], bc) for i in range(len(group)))
    return CombatSnapshot(
        turn=int(bc.turn),
        player_hp=int(player.curHp),
        player_max_hp=int(player.maxHp),
        player_block=int(player.block),
        player_energy=int(player.energy),
        monster_count=int(group.monsterCount),
        monsters_alive=int(group.getAliveCount()),
        monsters=monsters,
        hand_size=int(bc.cards.cardsInHand),
        hand_card_ids=tuple(card.id.name for card in bc.cards.hand),
    )


def _read_monster(monster: Any, bc: Any) -> MonsterSnapshot:
    intent_damage, intent_hits = monster.get_move_base_damage(bc)
    return MonsterSnapshot(
        idx=int(monster.idx),
        name=str(monster.getName()),
        monster_id=monster.id.name,
        hp=int(monster.curHp),
        max_hp=int(monster.maxHp),
        block=int(monster.block),
        alive=bool(monster.isAlive()),
        intent_damage=int(intent_damage),
        intent_hits=int(intent_hits),
    )


def engine_commit() -> str:
    """Return the pinned engine commit SHA recorded in ``configs/engine.toml``."""
    with _ENGINE_CONFIG.open("rb") as handle:
        return str(tomllib.load(handle)["engine"]["commit"])
