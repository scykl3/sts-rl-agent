"""Render logged run metrics as a self-contained HTML dashboard.

Training and evaluation runs write a ``manifest.json`` plus a ``metrics.jsonl``
stream via :class:`sts_rl.utils.logging.RunLogger`. This module reads one or more
of those run directories and renders every numeric metric it finds as a time
series (x = ``step``), overlaying the runs on a shared chart so they can be
compared across a training campaign.

The output is a single static HTML file with inline SVG charts: no JavaScript, no
external assets, and no plotting dependency. It is a pure function of the input
records, so the same run directories always produce byte-identical HTML - which
also makes it testable without a browser.

Usage::

    python -m sts_rl.eval.dashboard --out dashboard.html runs/exp-a runs/exp-b

or programmatically via :func:`write_dashboard` / :func:`render_dashboard`.
"""

from __future__ import annotations

import argparse
import html
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from sts_rl.utils.logging import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    STEP_KEY,
    read_manifest,
    read_metrics,
)

DASHBOARD_TITLE_DEFAULT = "STS RL - Evaluation Dashboard"

# Headline metrics shown first (in this order) when present; every other numeric
# metric follows, sorted alphabetically. Mirrors the fields of EvalReport so the
# most-watched signals lead the page.
_PRIORITY_METRICS: tuple[str, ...] = (
    "win_rate",
    "avg_floor",
    "avg_hp",
    "avg_return",
    "avg_ep_len",
    "n_episodes",
)

# Per-run line colors, cycled by run index so a run keeps one color across every
# chart and the shared legend. Chosen to stay distinguishable on white.
_PALETTE: tuple[str, ...] = (
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#9c755f",
)

# Chart geometry (SVG user units). The plot area is the chart box minus padding
# that leaves room for axis tick labels.
_CHART_WIDTH = 640
_CHART_HEIGHT = 300
_PAD_LEFT = 60
_PAD_RIGHT = 18
_PAD_TOP = 14
_PAD_BOTTOM = 34
_PLOT_WIDTH = _CHART_WIDTH - _PAD_LEFT - _PAD_RIGHT
_PLOT_HEIGHT = _CHART_HEIGHT - _PAD_TOP - _PAD_BOTTOM

# A single point can't draw a polyline, so mark it with a dot of this radius.
_MARKER_RADIUS = 3.0
_LINE_STROKE_WIDTH = 2  # width of a run's series line
_YTICK_LABEL_GAP = 6  # px left of the axis for right-aligned y tick labels
_TICK_LABEL_BASELINE_DY = 3  # nudge y tick text down so it centers on its line
_XTICK_LABEL_DY = 18  # px below the plot for x tick labels
_FLAT_RANGE_PAD_FRACTION = 0.05  # pad a constant series by +/- this * |value|
# Characters of a commit SHA shown in the provenance table.
_SHA_DISPLAY_LEN = 10


@dataclass(frozen=True)
class RunSeries:
    """One run's metrics, ready to plot.

    ``series`` maps each numeric metric name to its ``(step, value)`` points,
    sorted by step, with non-finite values dropped. Metrics are stored per-metric
    rather than aligned to a single step axis so a metric logged on only some
    records (a ragged stream) still plots correctly.
    """

    label: str
    run_dir: str
    n_records: int
    series: dict[str, list[tuple[int, float]]]
    manifest: dict[str, Any]

    @property
    def metric_names(self) -> list[str]:
        return sorted(self.series)


def _load_manifest_or_empty(run_path: Path) -> dict[str, Any]:
    """Read ``manifest.json``, or return ``{}`` if it is unusable.

    Provenance is a header detail; a missing, half-written, or otherwise unusable
    manifest should degrade to a blank one, not sink the whole dashboard. That
    includes valid JSON that is not an object (e.g. a list), which has no ``.get``
    for the render path. (``JSONDecodeError`` is a ``ValueError``.)
    """
    if not (run_path / MANIFEST_FILENAME).exists():
        return {}
    try:
        manifest = read_manifest(run_path)
    except (ValueError, OSError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def load_run_series(run_dir: str | Path, *, label: str | None = None) -> RunSeries:
    """Load one run directory into a :class:`RunSeries`.

    Loading is best-effort: a missing file, an unparseable manifest, or a
    malformed metrics line (what a crashed run can leave behind) is skipped
    rather than raised, so a partial run still renders. ``label`` defaults to the
    directory's base name.
    """
    run_path = Path(run_dir)
    manifest = _load_manifest_or_empty(run_path)
    records = (
        read_metrics(run_path, skip_malformed=True)
        if (run_path / METRICS_FILENAME).exists()
        else []
    )

    series: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        step = record.get(STEP_KEY)
        # A record with no usable (numeric, finite) step can't be placed on the
        # x-axis. Finiteness matters: a NaN/inf step is valid JSON that parses
        # back as a float and would crash int() below (mirrors the value guard).
        if not _is_number(step) or not math.isfinite(step):
            continue
        step_int = int(step)
        for key, value in record.items():
            if key == STEP_KEY or not _is_number(value) or not math.isfinite(value):
                continue
            series.setdefault(key, []).append((step_int, float(value)))
    for points in series.values():
        points.sort(key=lambda point: point[0])

    return RunSeries(
        label=label if label is not None else run_path.name,
        run_dir=str(run_path),
        n_records=len(records),
        series=series,
        manifest=manifest,
    )


def load_runs(run_dirs: Sequence[str | Path]) -> list[RunSeries]:
    """Load several run directories, labelled by base name, preserving order."""
    return [load_run_series(run_dir) for run_dir in run_dirs]


def render_dashboard(runs: Sequence[RunSeries], *, title: str = DASHBOARD_TITLE_DEFAULT) -> str:
    """Render the runs into a complete, self-contained HTML document string."""
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        _STYLE,
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
    ]

    if not runs:
        parts.append('<p class="empty">No runs to display.</p>')
        parts.extend(["</body>", "</html>", ""])
        return "\n".join(parts)

    colors = {index: _PALETTE[index % len(_PALETTE)] for index in range(len(runs))}
    parts.append(_render_legend(runs, colors))
    parts.append(_render_provenance(runs, colors))

    metric_names = _ordered_metrics(runs)
    if not metric_names:
        parts.append('<p class="empty">No numeric metrics were logged.</p>')
    else:
        parts.append('<div class="charts">')
        for metric in metric_names:
            parts.append('<section class="chart">')
            parts.append(f"<h2>{html.escape(metric)}</h2>")
            parts.append(_render_chart(metric, runs, colors))
            parts.append("</section>")
        parts.append("</div>")

    parts.extend(["</body>", "</html>", ""])
    return "\n".join(parts)


def write_dashboard(
    run_dirs: Sequence[str | Path],
    out_path: str | Path,
    *,
    title: str = DASHBOARD_TITLE_DEFAULT,
) -> Path:
    """Load ``run_dirs``, render the dashboard, and write it to ``out_path``.

    Returns the written path. The parent directory is created if missing.
    """
    runs = load_runs(run_dirs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_dashboard(runs, title=title), encoding="utf-8")
    return out


# -- internals --------------------------------------------------------------


def _is_number(value: Any) -> TypeGuard[float]:
    """True for real int/float values, excluding bool (a metric, not a flag)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ordered_metrics(runs: Iterable[RunSeries]) -> list[str]:
    """Union of metric names across runs: priority metrics first, then sorted."""
    names: set[str] = set()
    for run in runs:
        names.update(run.series)
    priority = [name for name in _PRIORITY_METRICS if name in names]
    rest = sorted(names.difference(priority))
    return priority + rest


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    """Linear map ``value`` from ``[lo, hi]`` onto ``[out_lo, out_hi]``.

    A degenerate input range (``hi == lo``) maps to the output midpoint so a
    single distinct x or a constant series lands centered instead of dividing by
    zero.
    """
    if hi == lo:
        return (out_lo + out_hi) / 2.0
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def _fmt(value: float) -> str:
    """Compact numeric label: integers without a decimal, else ~4 sig figs."""
    if value == int(value):
        return str(int(value))
    return f"{value:.4g}"


def _short_sha(value: Any) -> str:
    text = str(value)
    return text[:_SHA_DISPLAY_LEN] if len(text) > _SHA_DISPLAY_LEN else text


def _render_legend(runs: Sequence[RunSeries], colors: dict[int, str]) -> str:
    items = [
        f'<span class="legend-item"><span class="swatch" style="background:{colors[index]}">'
        f"</span>{html.escape(run.label)}</span>"
        for index, run in enumerate(runs)
    ]
    return '<div class="legend">' + "".join(items) + "</div>"


def _render_provenance(runs: Sequence[RunSeries], colors: dict[int, str]) -> str:
    header = (
        "<tr><th>Run</th><th>Seed</th><th>Engine</th><th>Commit</th>"
        "<th>Created</th><th>Points</th></tr>"
    )
    rows = [header]
    for index, run in enumerate(runs):
        manifest = run.manifest
        # A hand-edited manifest could set "git" to a non-dict; fall back rather
        # than raise on the .get below.
        git = manifest.get("git")
        if not isinstance(git, dict):
            git = {}
        cells = [
            f'<span class="swatch" style="background:{colors[index]}"></span>'
            + html.escape(run.label),
            html.escape(str(manifest.get("seed", "-"))),
            html.escape(_short_sha(manifest.get("engine_commit", "-"))),
            html.escape(_short_sha(git.get("commit", "-"))),
            html.escape(str(manifest.get("created_at", "-"))),
            str(run.n_records),
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return '<table class="provenance">' + "".join(rows) + "</table>"


def _render_chart(metric: str, runs: Sequence[RunSeries], colors: dict[int, str]) -> str:
    """Render one metric's overlaid time-series chart as an inline ``<svg>``."""
    plotted = [
        (index, run.series[metric]) for index, run in enumerate(runs) if metric in run.series
    ]
    # _ordered_metrics only lists a metric when some run carries finite points,
    # so `plotted` is non-empty here.
    xs = [step for _, points in plotted for step, _ in points]
    ys = [value for _, points in plotted for _, value in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    # Pad a flat range so a constant series draws a centered horizontal line with
    # honest, distinct tick labels rather than three identical ones.
    if y_hi == y_lo:
        pad = abs(y_hi) * _FLAT_RANGE_PAD_FRACTION if y_hi != 0 else 1.0
        y_lo, y_hi = y_lo - pad, y_hi + pad

    body: list[str] = []

    # Horizontal gridlines + y tick labels at low / mid / high.
    for tick in (y_lo, (y_lo + y_hi) / 2.0, y_hi):
        py = _scale(tick, y_lo, y_hi, _PAD_TOP + _PLOT_HEIGHT, _PAD_TOP)
        body.append(
            f'<line class="grid" x1="{_PAD_LEFT}" y1="{py:.2f}" '
            f'x2="{_PAD_LEFT + _PLOT_WIDTH}" y2="{py:.2f}"></line>'
        )
        label = html.escape(_fmt(tick))
        body.append(
            f'<text class="ytick" x="{_PAD_LEFT - _YTICK_LABEL_GAP}" '
            f'y="{py + _TICK_LABEL_BASELINE_DY:.2f}">{label}</text>'
        )

    # x tick labels at first / last step.
    for tick in dict.fromkeys((x_lo, x_hi)):
        px = _scale(tick, x_lo, x_hi, _PAD_LEFT, _PAD_LEFT + _PLOT_WIDTH)
        body.append(
            f'<text class="xtick" x="{px:.2f}" '
            f'y="{_PAD_TOP + _PLOT_HEIGHT + _XTICK_LABEL_DY:.2f}">{html.escape(_fmt(tick))}</text>'
        )

    # One polyline per run (plus a dot when a run has a single point).
    for index, points in plotted:
        color = colors[index]
        coords = " ".join(
            f"{_scale(step, x_lo, x_hi, _PAD_LEFT, _PAD_LEFT + _PLOT_WIDTH):.2f},"
            f"{_scale(value, y_lo, y_hi, _PAD_TOP + _PLOT_HEIGHT, _PAD_TOP):.2f}"
            for step, value in points
        )
        body.append(
            f'<polyline fill="none" stroke="{color}" '
            f'stroke-width="{_LINE_STROKE_WIDTH}" points="{coords}"></polyline>'
        )
        if len(points) == 1:
            cx, cy = coords.split(",")
            body.append(
                f'<circle cx="{cx}" cy="{cy}" r="{_MARKER_RADIUS}" fill="{color}"></circle>'
            )

    plot_frame = (
        f'<rect class="plot" x="{_PAD_LEFT}" y="{_PAD_TOP}" '
        f'width="{_PLOT_WIDTH}" height="{_PLOT_HEIGHT}"></rect>'
    )
    return (
        f'<svg viewBox="0 0 {_CHART_WIDTH} {_CHART_HEIGHT}" '
        f'role="img" aria-label="{html.escape(metric)} by step">'
        + plot_frame
        + "".join(body)
        + "</svg>"
    )


_STYLE = """<style>
  :root { font-family: -apple-system, Segoe UI, Roboto, sans-serif; }
  body { margin: 24px; color: #1d1d1f; }
  h1 { font-size: 20px; }
  h2 { font-size: 14px; margin: 0 0 4px; font-weight: 600; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0 18px; font-size: 13px; }
  .legend-item { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 2px; vertical-align: middle; margin-right: 4px; }
  table.provenance { border-collapse: collapse; font-size: 12px; margin-bottom: 22px; }
  table.provenance th, table.provenance td { border: 1px solid #d2d2d7; padding: 4px 8px; text-align: left; }
  table.provenance th { background: #f5f5f7; }
  .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; }
  .chart svg { width: 100%; height: auto; border: 1px solid #e5e5ea; border-radius: 6px; background: #fff; }
  .chart .plot { fill: #fbfbfd; stroke: #d2d2d7; }
  .chart .grid { stroke: #ececf0; stroke-width: 1; }
  .chart text { font-size: 10px; fill: #6e6e73; }
  .chart .ytick { text-anchor: end; }
  .chart .xtick { text-anchor: middle; }
  .empty { color: #6e6e73; font-style: italic; }
</style>"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sts_rl.eval.dashboard",
        description="Render logged run metrics as a self-contained HTML dashboard.",
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Run directories, each holding manifest.json and metrics.jsonl.",
    )
    parser.add_argument("-o", "--out", required=True, help="Path to write the HTML dashboard to.")
    parser.add_argument("--title", default=DASHBOARD_TITLE_DEFAULT, help="Dashboard page title.")
    args = parser.parse_args(argv)
    out = write_dashboard(args.run_dirs, args.out, title=args.title)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI entrypoint
    raise SystemExit(main())
