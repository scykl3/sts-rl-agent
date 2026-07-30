"""Specify a mid-run combat start: an evolved deck, relics, HP, and ascension.

Every combat the adapter starts today begins from a fresh floor-0 run: the
10-card starter deck, full HP, and only the starting relic. That makes the hard
fights impossible to drill honestly - an Act-1 boss cannot be won with the
starter deck - so this module lets a caller describe a realistic mid-run state
and build the chosen encounter from it.

A :class:`StateSpec` names its cards, relics, and encounter as engine enum-member
*names* (SCREAMING_SNAKE_CASE strings, the same convention as
:mod:`sts_rl.env.encounters`), so this module imports and its specs validate
without a native engine build. The name-to-enum resolvers (:func:`resolve_deck`,
:func:`resolve_relics`) import the engine lazily, and
:func:`sts_rl.env.engine.start_combat_from_state` drives a ``GameContext`` from a
resolved spec.

The starting relic is not removed: an Ironclad ``GameContext`` always carries its
starter relic (``BURNING_BLOOD``), and the engine exposes no way to clear relics,
so a spec's relics are obtained *in addition* to it. This matches every real
Ironclad run, which always holds ``BURNING_BLOOD``.

Fidelity caveat - relics that alter the deck on obtain: a few relics add or change
cards the moment they are obtained, so a spec that lists one will NOT build the
deck it names. ``WAR_PAINT`` and ``WHETSTONE`` upgrade random cards, ``TINY_HOUSE``
upgrades a random card, ``PANDORAS_BOX`` transforms every starter Strike/Defend,
and ``CALLING_BELL`` adds a curse. These are unsupported for exact-deck
reproduction: cards are obtained before relics (so the relic sees the full deck),
and the random-upgrade effects need the cards already present, so no reordering
fixes it. Leave them out of a spec whose deck must match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sts_rl.interface import InterfaceError

# The engine's sentinel enum member: a real ``__members__`` entry but not a real
# card / relic, rejected the same way the encounter resolver rejects it.
_INVALID_SENTINEL = "INVALID"

# Ironclad base maximum HP at Ascension 0, the engine's starting value. Used as
# the default ``max_hp`` so a minimal spec (deck + encounter) starts at full base
# HP; a mid-run spec overrides it with the run's own max HP.
IRONCLAD_BASE_MAX_HP = 80

# Highest ascension level the game supports; specs are bounded to [0, MAX_ASCENSION].
MAX_ASCENSION = 20

# The engine deck is a fixed-capacity buffer (fixed_list<Card, 96>) whose push_back
# does not bounds-check, so obtaining more than this many cards writes out of bounds
# (undefined behavior). A spec's deck length is capped here to keep the build safe.
MAX_DECK_SIZE = 96


@dataclass(frozen=True)
class CardSpec:
    """One deck card: an engine ``CardId`` member name plus an upgrade count.

    ``card_id`` is a ``CardId`` member name (e.g. ``"STRIKE_RED"``, ``"INFLAME"``);
    ``upgrades`` is the number of times the card is upgraded (0 = unupgraded, the
    common case; 1 = the single upgrade most Ironclad cards have; higher values
    are meaningful only for the few cards that upgrade repeatedly, e.g. Searing
    Blow). Resolution to a live ``sts.Card`` happens in :func:`resolve_deck`.
    """

    card_id: str
    upgrades: int = 0

    def __post_init__(self) -> None:
        if not self.card_id:
            raise InterfaceError("CardSpec.card_id is empty; name a CardId member")
        if self.upgrades < 0:
            raise InterfaceError(
                f"CardSpec.upgrades must be >= 0, got {self.upgrades} for {self.card_id!r}"
            )


@dataclass(frozen=True)
class StateSpec:
    """A realistic mid-run state to build a single combat from.

    Fields:
        deck: the run deck as :class:`CardSpec` cards (ids + upgrades). Must be
            non-empty (a combat needs a deck to draw from) and at most
            :data:`MAX_DECK_SIZE` cards (the engine's fixed deck capacity).
        encounter: the ``MonsterEncounter`` member name to fight (e.g.
            ``"GREMLIN_NOB"``, ``"HEXAGHOST"``), resolved via
            :func:`sts_rl.env.encounters.resolve_encounter_names`.
        relics: ``RelicId`` member names to obtain, in addition to the always-present
            Ironclad starter relic (see the module docstring).
        max_hp: maximum HP; defaults to :data:`IRONCLAD_BASE_MAX_HP`.
        cur_hp: current HP; ``None`` (the default) means full HP (``max_hp``).
            Otherwise must satisfy ``1 <= cur_hp <= max_hp``.
        ascension: ascension level in ``[0, MAX_ASCENSION]``. Scales enemy HP and
            behavior, so it is part of the state, not a separate env knob.

    Names are validated for existence when resolved against the live engine (see
    :func:`resolve_deck` / :func:`resolve_relics`); the numeric and structural
    invariants here are checked eagerly and need no engine.
    """

    deck: tuple[CardSpec, ...]
    encounter: str
    relics: tuple[str, ...] = ()
    max_hp: int = IRONCLAD_BASE_MAX_HP
    cur_hp: int | None = None
    ascension: int = 0

    def __post_init__(self) -> None:
        if not self.deck:
            raise InterfaceError("StateSpec.deck is empty; a combat needs at least one card")
        if len(self.deck) > MAX_DECK_SIZE:
            raise InterfaceError(
                f"StateSpec.deck has {len(self.deck)} cards; the engine deck holds at most "
                f"{MAX_DECK_SIZE}"
            )
        if not self.encounter:
            raise InterfaceError("StateSpec.encounter is empty; name a MonsterEncounter member")
        if self.max_hp < 1:
            raise InterfaceError(f"StateSpec.max_hp must be >= 1, got {self.max_hp}")
        if self.cur_hp is not None and not (1 <= self.cur_hp <= self.max_hp):
            raise InterfaceError(
                f"StateSpec.cur_hp must be in [1, max_hp={self.max_hp}], got {self.cur_hp}"
            )
        if not (0 <= self.ascension <= MAX_ASCENSION):
            raise InterfaceError(
                f"StateSpec.ascension must be in [0, {MAX_ASCENSION}], got {self.ascension}"
            )

    @property
    def effective_cur_hp(self) -> int:
        """Current HP to apply: ``cur_hp`` if set, else full HP (``max_hp``)."""
        return self.max_hp if self.cur_hp is None else self.cur_hp


def resolve_deck(deck: tuple[CardSpec, ...]) -> list[Any]:  # -> list[sts.Card]
    """Resolve deck ``CardSpec`` names to live ``sts.Card`` objects (with upgrades).

    Validates each ``card_id`` against ``sts.CardId.__members__`` and rejects the
    engine's ``INVALID`` sentinel, the same validity rule
    :func:`sts_rl.env.encounters.resolve_encounter_names` applies to encounter
    names. The engine is imported lazily so this module stays importable without a
    native build.

    Raises :class:`~sts_rl.interface.InterfaceError` on an unknown or sentinel
    card name.
    """
    from sts_rl.env._engine import slaythespire as sts

    members = sts.CardId.__members__
    cards: list[Any] = []
    for spec in deck:
        if spec.card_id == _INVALID_SENTINEL:
            raise InterfaceError(f"{_INVALID_SENTINEL} is the engine's sentinel, not a real CardId")
        if spec.card_id not in members:
            raise InterfaceError(f"unknown CardId {spec.card_id!r}")
        cards.append(sts.Card(members[spec.card_id], int(spec.upgrades)))
    return cards


def resolve_relics(relics: tuple[str, ...]) -> list[Any]:  # -> list[sts.RelicId]
    """Resolve relic member names to live ``sts.RelicId`` values.

    Validates each name against ``sts.RelicId.__members__`` and rejects the
    ``INVALID`` sentinel, mirroring :func:`resolve_deck`. Duplicates are allowed
    through (the engine no-ops obtaining a relic already held); the engine is
    imported lazily.

    Raises :class:`~sts_rl.interface.InterfaceError` on an unknown or sentinel
    relic name.
    """
    from sts_rl.env._engine import slaythespire as sts

    members = sts.RelicId.__members__
    resolved: list[Any] = []
    for name in relics:
        if name == _INVALID_SENTINEL:
            raise InterfaceError(
                f"{_INVALID_SENTINEL} is the engine's sentinel, not a real RelicId"
            )
        if name not in members:
            raise InterfaceError(f"unknown RelicId {name!r}")
        resolved.append(members[name])
    return resolved


def state_spec_from_game_context(gc: Any, *, encounter: str) -> StateSpec:
    """Snapshot a live ``GameContext``'s deck, relics, and HP into a :class:`StateSpec`.

    Reads the run's current deck (each card's id name and upgrade count), owned
    relics, current/max HP, and ascension, pairing them with the ``encounter`` to
    fight. Duck-typed (reads ``.id.name`` / ``.upgrade_count`` / int fields), so it
    needs no direct engine import; the caller supplies the live ``gc``.

    Round-trips through :func:`sts_rl.env.engine.start_combat_from_state`: the
    starter relic the snapshot records is re-obtained (a no-op, since it is always
    present), so a rebuilt state carries the same deck, relics, and HP.

    Round-trip limitation - relic counters are not captured: this records relic id
    names only, not their stored per-relic values (Neow's Lament charges, Pen Nib's
    attack count, Nunchaku's counter, Girya's lift count, and similar). On rebuild
    each relic resets to its fresh-obtain default, so a stateful relic loses its
    accumulated charge. Restore those explicitly with the engine's
    ``set_relic_value`` if exact relic state matters.
    """
    deck = tuple(CardSpec(card.id.name, int(card.upgrade_count)) for card in gc.deck)
    relics = tuple(relic.id.name for relic in gc.relics)
    return StateSpec(
        deck=deck,
        encounter=encounter,
        relics=relics,
        max_hp=int(gc.max_hp),
        cur_hp=int(gc.cur_hp),
        ascension=int(gc.ascension),
    )
