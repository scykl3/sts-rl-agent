"""Tests for mid-run combat-state specs (StateSpec) and building a battle from one.

Structural / validation invariants of ``StateSpec`` and ``CardSpec`` are
pure-Python and run without the engine. Building a battle from a spec
(``start_combat_from_state``, ``StsEnv.from_state``) needs the native binding
(``scripts/build_engine.sh``); those tests skip cleanly without it, matching the
suite's ImportError -> skip idiom.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from sts_rl.env.combat_state import (
    IRONCLAD_BASE_MAX_HP,
    MAX_ASCENSION,
    MAX_DECK_SIZE,
    CardSpec,
    StateSpec,
)
from sts_rl.interface import ACTION_BLOCK_BY_NAME, InterfaceError

# Probe the compiled engine once; only the build/rollout tests below need it, so
# gate just those and let the pure-Python validation invariants run engine-free.
try:
    import sts_rl.env._engine  # noqa: F401

    _ENGINE_AVAILABLE = True
    _ENGINE_SKIP_REASON = ""
except ImportError as exc:  # pragma: no cover - exercised only without a build
    _ENGINE_AVAILABLE = False
    _ENGINE_SKIP_REASON = f"engine not built ({exc})"

requires_engine = pytest.mark.skipif(not _ENGINE_AVAILABLE, reason=_ENGINE_SKIP_REASON)

REGRESSION_SEED = 42
MAX_SCRIPTED_STEPS = 400
_END_TURN = ACTION_BLOCK_BY_NAME["END_TURN"].start

# The Ironclad starter relic, always present on a fresh GameContext and not
# removable, so a spec's relics are obtained in addition to it.
STARTER_RELIC = "BURNING_BLOOD"

# A small evolved deck with upgrade variety, including the same card both upgraded
# and not and a duplicate, so the assertions pin the full multiset with upgrades.
_SAMPLE_DECK = (
    CardSpec("STRIKE_RED", 1),
    CardSpec("STRIKE_RED", 0),
    CardSpec("DEFEND_RED", 0),
    CardSpec("DEFEND_RED", 0),
    CardSpec("BASH", 1),
    CardSpec("INFLAME", 1),
    CardSpec("DEMON_FORM", 0),
)
_SAMPLE_RELICS = ("AKABEKO", "ANCHOR")

# A hand-crafted strong deck for the "plays to a terminal outcome" boss test.
_STRONG_DECK = (
    tuple(CardSpec("STRIKE_RED", 1) for _ in range(4))
    + tuple(CardSpec("DEFEND_RED", 1) for _ in range(4))
    + (
        CardSpec("BASH", 1),
        CardSpec("INFLAME", 1),
        CardSpec("DEMON_FORM", 0),
        CardSpec("POMMEL_STRIKE", 1),
        CardSpec("IRON_WAVE", 1),
        CardSpec("SHRUG_IT_OFF", 1),
    )
)
ACT1_BOSSES = ("SLIME_BOSS", "THE_GUARDIAN", "HEXAGHOST")

# Ironclad Ascension-0 starter combat: 5 Strikes + 4 Defends + 1 Bash.
STARTER_DECK_SIZE = 10


# -- Pure-Python validation (no engine) -------------------------------------


def test_card_spec_rejects_negative_upgrades() -> None:
    with pytest.raises(InterfaceError):
        CardSpec("STRIKE_RED", -1)


def test_card_spec_rejects_empty_id() -> None:
    with pytest.raises(InterfaceError):
        CardSpec("")


def test_state_spec_rejects_empty_deck() -> None:
    with pytest.raises(InterfaceError):
        StateSpec(deck=(), encounter="GREMLIN_NOB")


def test_state_spec_rejects_oversized_deck() -> None:
    # The engine deck is a fixed-capacity buffer with no push_back bounds check, so
    # a deck past MAX_DECK_SIZE would write out of bounds; the guard must reject it.
    ok = tuple(CardSpec("STRIKE_RED") for _ in range(MAX_DECK_SIZE))
    StateSpec(deck=ok, encounter="GREMLIN_NOB")  # exactly at the cap is allowed
    with pytest.raises(InterfaceError):
        StateSpec(deck=ok + (CardSpec("STRIKE_RED"),), encounter="GREMLIN_NOB")


def test_state_spec_rejects_empty_encounter() -> None:
    with pytest.raises(InterfaceError):
        StateSpec(deck=_SAMPLE_DECK, encounter="")


def test_state_spec_rejects_bad_hp() -> None:
    with pytest.raises(InterfaceError):
        StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", max_hp=0)
    with pytest.raises(InterfaceError):  # cur_hp above max
        StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", max_hp=50, cur_hp=51)
    with pytest.raises(InterfaceError):  # cur_hp below 1 (0 HP is dead)
        StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", max_hp=50, cur_hp=0)


def test_state_spec_rejects_out_of_range_ascension() -> None:
    with pytest.raises(InterfaceError):
        StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", ascension=-1)
    with pytest.raises(InterfaceError):
        StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", ascension=MAX_ASCENSION + 1)


def test_effective_cur_hp_defaults_to_max() -> None:
    spec = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", max_hp=63)
    assert spec.cur_hp is None
    assert spec.effective_cur_hp == 63
    assert spec.max_hp == 63

    partial = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", max_hp=63, cur_hp=40)
    assert partial.effective_cur_hp == 40


def test_state_spec_defaults_to_ironclad_base_hp() -> None:
    spec = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB")
    assert spec.max_hp == IRONCLAD_BASE_MAX_HP
    assert spec.effective_cur_hp == IRONCLAD_BASE_MAX_HP
    assert spec.ascension == 0
    assert spec.relics == ()


# -- Engine-dependent: building a battle from a spec ------------------------


def _combat_card_multiset(bc) -> Counter:
    """Multiset of ``(card_id_name, upgrade_count)`` over every pile in the battle.

    At battle start a card is in the hand, draw, discard, or exhaust pile, so
    their union is the full combat deck the spec produced.
    """
    cards: list[tuple[str, int]] = []
    for pile in (bc.cards.hand, bc.cards.drawPile, bc.cards.discardPile, bc.cards.exhaustPile):
        cards += [(c.id.name, int(c.upgrade_count)) for c in pile]
    return Counter(cards)


def _spec_multiset(deck: tuple[CardSpec, ...]) -> Counter:
    return Counter((c.card_id, c.upgrades) for c in deck)


@requires_engine
def test_build_from_state_has_exactly_the_specified_deck() -> None:
    from sts_rl.env.engine import start_combat_from_state

    spec = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB")
    _, bc = start_combat_from_state(spec, seed=REGRESSION_SEED)
    # Exact multiset, upgrades included: (STRIKE_RED,1) and (STRIKE_RED,0) are
    # distinct entries, so this fails if upgrades are dropped or a card is missing.
    assert _combat_card_multiset(bc) == _spec_multiset(_SAMPLE_DECK)


@requires_engine
def test_build_from_state_has_specified_relics_plus_starter() -> None:
    from sts_rl.env.engine import start_combat_from_state

    spec = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", relics=_SAMPLE_RELICS)
    gc, _ = start_combat_from_state(spec, seed=REGRESSION_SEED)
    owned = {relic.id.name for relic in gc.relics}
    assert owned == {STARTER_RELIC, *_SAMPLE_RELICS}


@requires_engine
def test_build_from_state_sets_hp() -> None:
    from sts_rl.env.engine import read_combat, start_combat_from_state

    spec = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", max_hp=71, cur_hp=44)
    _, bc = start_combat_from_state(spec, seed=REGRESSION_SEED)
    snap = read_combat(bc)
    assert snap.player_hp == 44
    assert snap.player_max_hp == 71


@requires_engine
def test_build_from_state_builds_requested_encounter() -> None:
    from sts_rl.env.engine import read_combat, start_combat_from_state

    # Single-monster elite: exactly one enemy, the Gremlin Nob.
    nob = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB")
    _, bc = start_combat_from_state(nob, seed=REGRESSION_SEED)
    snap = read_combat(bc)
    assert snap.monster_count == 1
    assert snap.monsters[0].monster_id == "GREMLIN_NOB"

    # An Act-1 boss builds too (unwinnable with the starter deck, which is why
    # building from an evolved-deck spec exists).
    boss = StateSpec(deck=_SAMPLE_DECK, encounter="HEXAGHOST")
    _, bc_boss = start_combat_from_state(boss, seed=REGRESSION_SEED)
    boss_snap = read_combat(bc_boss)
    assert "HEXAGHOST" in {m.monster_id for m in boss_snap.monsters}


@requires_engine
def test_ascension_scales_the_built_encounter() -> None:
    # Ascension is part of the state: a higher ascension raises enemy max HP, so
    # the same encounter built at A0 vs A(max) is not identical. Guards that the
    # spec's ascension actually reaches the GameContext.
    from sts_rl.env.engine import read_combat, start_combat_from_state

    def _enemy_max_hp(ascension: int) -> int:
        spec = StateSpec(deck=_SAMPLE_DECK, encounter="THE_GUARDIAN", ascension=ascension)
        _, bc = start_combat_from_state(spec, seed=REGRESSION_SEED)
        return read_combat(bc).monsters[0].max_hp

    assert _enemy_max_hp(MAX_ASCENSION) > _enemy_max_hp(0)


@requires_engine
def test_resolve_rejects_unknown_and_invalid_names() -> None:
    from sts_rl.env.engine import start_combat_from_state

    with pytest.raises(InterfaceError):  # unknown card
        start_combat_from_state(StateSpec(deck=(CardSpec("NOT_A_CARD"),), encounter="GREMLIN_NOB"))
    with pytest.raises(InterfaceError):  # INVALID card sentinel
        start_combat_from_state(StateSpec(deck=(CardSpec("INVALID"),), encounter="GREMLIN_NOB"))
    with pytest.raises(InterfaceError):  # unknown relic
        start_combat_from_state(
            StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", relics=("NOT_A_RELIC",))
        )
    with pytest.raises(InterfaceError):  # unknown encounter
        start_combat_from_state(StateSpec(deck=_SAMPLE_DECK, encounter="NOT_AN_ENCOUNTER"))


@requires_engine
def test_snapshot_from_game_context_round_trips() -> None:
    # state_spec_from_game_context(build(spec)) rebuilds the same deck/relics/HP.
    from sts_rl.env.combat_state import state_spec_from_game_context
    from sts_rl.env.engine import read_combat, start_combat_from_state

    spec = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", relics=_SAMPLE_RELICS, cur_hp=55)
    gc, bc = start_combat_from_state(spec, seed=REGRESSION_SEED)

    recovered = state_spec_from_game_context(gc, encounter="GREMLIN_NOB")
    gc2, bc2 = start_combat_from_state(recovered, seed=REGRESSION_SEED)
    assert _combat_card_multiset(bc2) == _combat_card_multiset(bc)
    assert {r.id.name for r in gc2.relics} == {r.id.name for r in gc.relics}
    snap2 = read_combat(bc2)
    assert (snap2.player_hp, snap2.player_max_hp) == (55, IRONCLAD_BASE_MAX_HP)


# -- Engine-dependent: adapter integration ----------------------------------


@requires_engine
def test_from_state_env_reset_builds_the_spec() -> None:
    # Regression: reset() must build the spec's encounter AND deck. If state_spec
    # were ignored (fell through to the fresh navigated start), the fight would be
    # the seed's first hallway room with the 10-card starter deck, not this.
    from sts_rl.env.adapter import StsEnv

    spec = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", relics=_SAMPLE_RELICS)
    env = StsEnv.from_state(spec)
    _, info = env.reset(seed=REGRESSION_SEED)
    assert info["combat"].monster_count == 1
    assert info["combat"].monsters[0].monster_id == "GREMLIN_NOB"
    assert _combat_card_multiset(env._bc) == _spec_multiset(_SAMPLE_DECK)
    env.close()


@requires_engine
def test_default_path_is_unchanged() -> None:
    # Backward-compat regression: a no-spec env's first combat is byte-for-byte the
    # navigated fresh start (start_combat), i.e. the 10-card starter deck at full HP.
    from sts_rl.env.adapter import StsEnv
    from sts_rl.env.engine import read_combat, start_combat

    env = StsEnv()
    _, info = env.reset(seed=REGRESSION_SEED)
    _, bc_ref = start_combat(REGRESSION_SEED)
    assert info["combat"] == read_combat(bc_ref)
    assert _combat_card_multiset(env._bc).total() == STARTER_DECK_SIZE
    assert info["combat"].player_hp == info["combat"].player_max_hp == IRONCLAD_BASE_MAX_HP
    env.close()


@requires_engine
def test_state_spec_ascension_overrides_arg_in_info() -> None:
    from sts_rl.env.adapter import StsEnv

    spec = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB", ascension=7)
    # The ascension argument is redundant with the spec; the spec wins.
    env = StsEnv(state_spec=spec, ascension=0)
    _, info = env.reset(seed=REGRESSION_SEED)
    assert info["ascension"] == 7
    env.close()


@requires_engine
def test_encounters_and_state_spec_are_mutually_exclusive() -> None:
    from sts_rl.env._engine import slaythespire as sts
    from sts_rl.env.adapter import StsEnv

    spec = StateSpec(deck=_SAMPLE_DECK, encounter="GREMLIN_NOB")
    with pytest.raises(InterfaceError):
        StsEnv(state_spec=spec, encounters=[sts.MonsterEncounter.GREMLIN_NOB])


@requires_engine
@pytest.mark.parametrize("boss", ACT1_BOSSES)
def test_strong_deck_vs_act1_boss_plays_to_terminal(boss: str) -> None:
    # A hand-crafted strong deck vs each Act-1 boss plays to a terminal outcome
    # (win or loss) with no error - impossible with the 10-card starter deck.
    from sts_rl.env.adapter import StsEnv

    spec = StateSpec(
        deck=_STRONG_DECK, encounter=boss, max_hp=IRONCLAD_BASE_MAX_HP, cur_hp=IRONCLAD_BASE_MAX_HP
    )
    env = StsEnv.from_state(spec, max_episode_steps=MAX_SCRIPTED_STEPS + 100)
    _, info = env.reset(seed=REGRESSION_SEED)
    terminated = truncated = False
    for _ in range(MAX_SCRIPTED_STEPS + 100):
        legal = np.flatnonzero(info["action_mask"])
        play = [i for i in legal if i != _END_TURN]
        action = int(play[0]) if play else _END_TURN
        _, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["won"] == (reward > 0)
    env.close()
