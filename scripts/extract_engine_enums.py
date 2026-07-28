"""Measure the engine's real enum sizes and check them against the contract.

The observation contract in :mod:`sts_rl.interface` sizes its embedding tables
from enum cardinalities that were estimated before the engine was built (every
one tagged ``confirm against engine enums``). This script measures the real
values from the compiled engine and reports, for each table, whether the
contract size still fits the engine's highest id.

It prefers the compiled module's bound enums (``EnumType.__members__``) as the
source of truth and falls back to parsing the C++ headers under
``engine/sts_lightspeed/include/constants`` for enums the bindings do not
expose. It then calls the contract's own :func:`validate_engine_enums` guard so
the report shows exactly what that startup check will do, and writes a Markdown
findings report for the (jointly owned) contract change.

This script does not edit the contract. Run it after building the engine:

    scripts/build_engine.sh
    python scripts/extract_engine_enums.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sts_rl import interface
from sts_rl.env._engine import slaythespire as sts

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_DIR = REPO_ROOT / "engine" / "sts_lightspeed" / "include" / "constants"
REPORT_PATH = REPO_ROOT / "configs" / "engine_enum_report.md"


# Members that are placeholders, not ids the engine emits into an observation
# (a leading INVALID=0 doubles as PAD; a trailing INVALID/NONE just terminates
# the enum). They must be excluded from the max-embeddable-id measurement, or a
# trailing sentinel inflates the id space: RelicId's real ids end at 179 but
# RelicId.INVALID=180 would falsely demand a 181-row table.
SENTINEL_NAMES = frozenset({"INVALID", "NONE"})


def _max_embeddable(members: dict[str, int]) -> tuple[int | None, list[str]]:
    """Return (highest non-sentinel id, sentinels sitting above that id).

    The second element lists ``NAME=value`` for sentinel members whose value
    exceeds the real max, so the report can show exactly what was excluded.
    """
    real = {name: v for name, v in members.items() if name.upper() not in SENTINEL_NAMES}
    if not real:
        return None, []
    max_id = max(real.values())
    excluded = [
        f"{name}={v}"
        for name, v in members.items()
        if name.upper() in SENTINEL_NAMES and v > max_id
    ]
    return max_id, excluded


def bound_enum_members(enum_name: str) -> dict[str, int] | None:
    """Return a compiled-module enum's ``{name: value}`` map, or None if unbound."""
    obj = getattr(sts, enum_name, None)
    members = getattr(obj, "__members__", None)
    if not members:
        return None
    return {name: int(value) for name, value in members.items()}


def header_enum_members(header: str, enum_name: str) -> dict[str, int] | None:
    """Parse ``enum class <enum_name>`` from a constants header into {name: value}.

    Tracks a running counter so both explicit ``= N`` assignments and implicit
    auto-increment members are handled; comments are stripped first. Raises
    :class:`ValueError` on a member whose initializer is not a plain integer
    literal (e.g. an expression or a reference to another member), rather than
    silently desyncing the counter.
    """
    path = CONSTANTS_DIR / header
    if not path.is_file():
        return None
    text = path.read_text()
    # Strip comments from the whole header before locating the enum body. The
    # body match below is non-greedy up to the first ``}``, so a ``}`` inside a
    # comment in the enum body would truncate the match and under-report the max
    # id. Removing comments first makes the body match see only real code.
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    match = re.search(
        r"enum\s+class\s+" + re.escape(enum_name) + r"\s*(?::[^{]+)?\{(.*?)\}",
        text,
        re.S,
    )
    if not match:
        return None
    body = match.group(1)
    members: dict[str, int] = {}
    value = -1
    for token in body.split(","):
        token = token.strip()
        if not token:
            continue
        member = re.match(r"([A-Za-z_]\w*)\s*(?:=\s*(.+))?$", token, re.S)
        if not member:
            raise ValueError(f"{header}:{enum_name}: cannot parse enumerator {token!r}")
        name, initializer = member.group(1), member.group(2)
        if initializer is not None:
            try:
                value = int(initializer.strip(), 0)
            except ValueError as exc:
                raise ValueError(
                    f"{header}:{enum_name}: enumerator {name!r} has a non-literal "
                    f"initializer {initializer.strip()!r}; measure it from the "
                    f"compiled module instead of the header"
                ) from exc
        else:
            value += 1
        members[name] = value
    return members or None


@dataclass
class EnumProbe:
    """One contract table and how to measure the engine enum(s) behind it."""

    contract_key: str
    # (module enum name, header file, header enum name) candidates, tried in order.
    sources: tuple[tuple[str, str, str], ...]
    note: str = ""

    def measure(self) -> tuple[int | None, str]:
        """Return (max embeddable id, source description). None if unmeasurable.

        The id excludes trailing INVALID/NONE sentinels; the description records
        both the enum used and any sentinel that was excluded.
        """
        best: int | None = None
        used: list[str] = []
        for module_name, header, header_enum in self.sources:
            members = bound_enum_members(module_name)
            where = f"{module_name} (module)"
            if members is None and header:
                members = header_enum_members(header, header_enum)
                where = f"{header_enum} ({header})"
            if members is None:
                continue
            max_id, excluded = _max_embeddable(members)
            if max_id is None:
                continue
            label = f"{where}={max_id}"
            if excluded:
                label += f" [excl. sentinel {', '.join(excluded)}]"
            used.append(label)
            best = max_id if best is None else max(best, max_id)
        return best, ", ".join(used) if used else "no engine source"


# One probe per contract table. Player and monster statuses are separate enums
# with their own tables; enemy intent is the raw MonsterMoveId (no Intent enum).
PROBES: tuple[EnumProbe, ...] = (
    EnumProbe("N_CARD_IDS", (("CardId", "Cards.h", "CardId"),)),
    EnumProbe("N_RELIC_IDS", (("RelicId", "Relics.h", "RelicId"),)),
    EnumProbe("N_POTION_IDS", (("PotionId", "Potions.h", "Potion"),)),
    EnumProbe("N_PLAYER_POWER_IDS", (("PlayerStatus", "PlayerStatusEffects.h", "PlayerStatus"),)),
    EnumProbe(
        "N_MONSTER_POWER_IDS", (("MonsterStatus", "MonsterStatusEffects.h", "MonsterStatus"),)
    ),
    EnumProbe("N_MONSTER_IDS", (("MonsterId", "MonsterIds.h", "MonsterId"),)),
    EnumProbe("N_MONSTER_MOVE_IDS", (("MonsterMoveId", "MonsterMoves.h", "MonsterMoveId"),)),
    EnumProbe("N_NODE_TYPES", (("Room", "Rooms.h", "Room"),)),
    EnumProbe("N_SCREENS", (("ScreenState", "", ""),)),
    EnumProbe("N_EVENT_IDS", (("Event", "", ""),)),
    EnumProbe("N_NEOW_BONUS", (("NeowBonus", "", ""),)),
    EnumProbe("N_NEOW_DRAWBACK", (("NeowDrawback", "", ""),)),
)


def main() -> None:
    rows: list[dict[str, object]] = []
    engine_max_ids: dict[str, int] = {}

    for probe in PROBES:
        table_size = interface.EXPECTED_TABLE_SIZES[probe.contract_key]
        max_id, source = probe.measure()
        if max_id is not None:
            engine_max_ids[probe.contract_key] = max_id
            fits = max_id < table_size
            verdict = "OK" if fits else f"OVERFLOW (need N >= {max_id + 1})"
        else:
            verdict = "NO ENGINE SOURCE"
        rows.append(
            {
                "key": probe.contract_key,
                "contract_n": table_size,
                "engine_max_id": max_id,
                "verdict": verdict,
                "source": source,
                "note": probe.note,
            }
        )

    # Console table.
    print(f"{'table':16s} {'contract N':>10s} {'engine max_id':>13s}  verdict")
    print("-" * 72)
    for r in rows:
        mid = "-" if r["engine_max_id"] is None else str(r["engine_max_id"])
        print(f"{r['key']:16s} {r['contract_n']:>10} {mid:>13}  {r['verdict']}")

    # Run the contract's own guard so the report shows what startup validation does.
    guard_output = "validate_engine_enums() passed: every measured id fits."
    try:
        interface.validate_engine_enums(engine_max_ids)
    except interface.InterfaceError as exc:
        guard_output = str(exc)

    _write_report(rows, guard_output)
    print(f"\nwrote {REPORT_PATH.relative_to(REPO_ROOT)}")


def _write_report(rows: list[dict[str, object]], guard_output: str) -> None:
    lines: list[str] = []
    lines.append("# Engine enum reconciliation")
    lines.append("")
    lines.append(
        "Measured from the compiled engine module (bound enum `__members__`) and, "
        "where an enum is not exposed to Python, from the C++ headers under "
        "`engine/sts_lightspeed/include/constants`. Generated by "
        "`scripts/extract_engine_enums.py`."
    )
    lines.append("")
    lines.append(
        "Each embedding table holds ids `0..N-1`, so the invariant is "
        "`engine max_id < N`. Rows marked OVERFLOW or NO ENGINE SOURCE require a "
        "contract change, which is jointly owned and needs both reviewers."
    )
    lines.append("")
    lines.append("| Table | Contract N | Engine max_id | Verdict | Source |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        mid = "-" if r["engine_max_id"] is None else r["engine_max_id"]
        lines.append(
            f"| `{r['key']}` | {r['contract_n']} | {mid} | {r['verdict']} | {r['source']} |"
        )
    notes = [f"- `{r['key']}`: {r['note']}" for r in rows if r["note"]]
    if notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.extend(notes)
    lines.append("")
    lines.append("## What the contract's startup guard reports")
    lines.append("")
    lines.append("```")
    lines.append(guard_output)
    lines.append("```")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
