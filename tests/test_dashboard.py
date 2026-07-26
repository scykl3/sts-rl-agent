"""Loading and HTML/SVG rendering for the eval metrics dashboard."""

from __future__ import annotations

import json
import math
from pathlib import Path

from sts_rl.eval import dashboard as dash
from sts_rl.utils.logging import METRICS_FILENAME, RunLogger

_ENGINE_SHA = "engsha1234567"  # 13 chars: longer than _SHA_DISPLAY_LEN, so it truncates


def _write_run(
    path: Path,
    steps_metrics: list[tuple[int, dict[str, float]]],
    *,
    seed: int = 0,
) -> Path:
    """Create a run directory (manifest + metrics stream) via the real logger."""
    with RunLogger(path, {"lr": 1e-3}, seed=seed, engine_commit=_ENGINE_SHA) as logger:
        for step, metrics in steps_metrics:
            logger.log_metrics(step, metrics)
    return path


# -- loading ----------------------------------------------------------------


def test_load_reads_points_and_manifest(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "r",
        [(10, {"win_rate": 0.4}), (20, {"win_rate": 0.6})],
        seed=7,
    )
    series = dash.load_run_series(run)
    assert series.label == "r"
    assert series.n_records == 2
    assert series.series["win_rate"] == [(10, 0.4), (20, 0.6)]
    assert series.manifest["seed"] == 7
    assert series.manifest["engine_commit"] == _ENGINE_SHA


def test_load_sorts_points_by_step(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(20, {"m": 2.0}), (10, {"m": 1.0})])
    assert dash.load_run_series(run).series["m"] == [(10, 1.0), (20, 2.0)]


def test_load_drops_non_finite_values(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "r",
        [
            (0, {"win_rate": math.nan, "bad": math.inf}),
            (1, {"win_rate": 0.5, "bad": math.nan}),
        ],
    )
    series = dash.load_run_series(run)
    # Only the finite win_rate point survives; an all-non-finite metric is absent.
    assert series.series["win_rate"] == [(1, 0.5)]
    assert "bad" not in series.series


def test_load_excludes_bool_metrics(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"flag": True, "win_rate": 0.5})])
    series = dash.load_run_series(run)
    assert "flag" not in series.series
    assert "win_rate" in series.series


def test_load_skips_records_without_usable_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "raw"
    run_dir.mkdir()
    lines = [
        {"win_rate": 0.1},  # no step key
        {"win_rate": 0.2, "step": "oops"},  # non-numeric step
        {"win_rate": 0.3, "step": 5},  # the only usable record
    ]
    (run_dir / METRICS_FILENAME).write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    series = dash.load_run_series(run_dir)
    assert series.series["win_rate"] == [(5, 0.3)]


def test_load_drops_non_finite_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "raw"
    run_dir.mkdir()
    lines = [
        {"win_rate": 0.1, "step": float("nan")},  # NaN step: parses, would crash int()
        {"win_rate": 0.2, "step": float("inf")},  # inf step: same
        {"win_rate": 0.3, "step": 5},  # the only usable record
    ]
    (run_dir / METRICS_FILENAME).write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    series = dash.load_run_series(run_dir)
    assert series.series["win_rate"] == [(5, 0.3)]


def test_load_tolerates_missing_files(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    series = dash.load_run_series(empty)
    assert series.n_records == 0
    assert series.series == {}
    assert series.manifest == {}


def test_load_default_label_is_dir_name(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "exp-42", [(0, {"m": 1.0})])
    assert dash.load_run_series(run).label == "exp-42"


# -- metric ordering --------------------------------------------------------


def test_ordered_metrics_priority_first_then_sorted(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "r",
        [(0, {"zeta": 1.0, "alpha": 1.0, "win_rate": 0.5, "avg_hp": 30.0})],
    )
    series = dash.load_run_series(run)
    ordered = dash._ordered_metrics([series])
    # Priority metrics keep their declared order; the rest follow alphabetically.
    assert ordered == ["win_rate", "avg_hp", "alpha", "zeta"]


# -- rendering --------------------------------------------------------------


def test_render_has_document_and_chart_structure(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"win_rate": 0.4}), (1, {"win_rate": 0.7})])
    html_out = dash.render_dashboard([dash.load_run_series(run)])
    assert html_out.startswith("<!DOCTYPE html>")
    assert "<svg" in html_out and "<polyline" in html_out
    assert "win_rate" in html_out
    assert "</html>" in html_out


def test_render_overlays_multiple_runs(tmp_path: Path) -> None:
    a = _write_run(tmp_path / "a", [(0, {"win_rate": 0.4}), (1, {"win_rate": 0.5})])
    b = _write_run(tmp_path / "b", [(0, {"win_rate": 0.6}), (1, {"win_rate": 0.7})])
    runs = dash.load_runs([a, b])
    html_out = dash.render_dashboard(runs)
    assert "a" in html_out and "b" in html_out
    # Distinct palette colors for the two runs both appear.
    assert dash._PALETTE[0] in html_out
    assert dash._PALETTE[1] in html_out
    assert html_out.count("<polyline") == 2


def test_render_escapes_labels(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"win_rate": 0.5})])
    series = dash.load_run_series(run, label="<script>x&y")
    html_out = dash.render_dashboard([series])
    assert "<script>x&y" not in html_out
    assert "&lt;script&gt;x&amp;y" in html_out


def test_render_escapes_metric_names(tmp_path: Path) -> None:
    # Metric names are external (dict keys from the log) and reach both the <h2>
    # and the SVG aria-label, so both must be escaped.
    run = _write_run(tmp_path / "r", [(0, {"<b>&x": 0.5})])
    html_out = dash.render_dashboard([dash.load_run_series(run)])
    assert "<b>&x" not in html_out
    assert "&lt;b&gt;&amp;x" in html_out


def test_render_escapes_provenance_fields(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"win_rate": 0.5})])
    series = dash.load_run_series(run)
    # Simulate a manifest carrying markup-bearing provenance.
    poisoned = dash.RunSeries(
        label=series.label,
        run_dir=series.run_dir,
        n_records=series.n_records,
        series=series.series,
        manifest={"seed": "<x>", "engine_commit": "a&b", "git": "not-a-dict"},
    )
    html_out = dash.render_dashboard([poisoned])
    assert "<x>" not in html_out and "&lt;x&gt;" in html_out
    assert "a&b" not in html_out and "a&amp;b" in html_out  # engine_commit escaped
    # A non-dict "git" must not raise; it falls back to a blank commit cell.


def test_render_empty_runs_is_valid_and_noted() -> None:
    html_out = dash.render_dashboard([])
    assert html_out.startswith("<!DOCTYPE html>")
    assert "</html>" in html_out
    assert "No runs to display." in html_out
    assert "<svg" not in html_out


def test_render_no_numeric_metrics_noted(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {}), (1, {})])  # steps only, no metrics
    html_out = dash.render_dashboard([dash.load_run_series(run)])
    assert "No numeric metrics were logged." in html_out
    assert "<svg" not in html_out


def test_render_single_point_draws_marker(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"win_rate": 0.5})])
    html_out = dash.render_dashboard([dash.load_run_series(run)])
    assert "<circle" in html_out


def test_render_constant_metric_does_not_crash(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"win_rate": 0.5}), (1, {"win_rate": 0.5})])
    html_out = dash.render_dashboard([dash.load_run_series(run)])
    # No division by zero on a flat series; the line still renders.
    assert "<polyline" in html_out


def test_render_ragged_metric_across_runs(tmp_path: Path) -> None:
    a = _write_run(tmp_path / "a", [(0, {"win_rate": 0.4, "loss": 2.0})])
    b = _write_run(tmp_path / "b", [(0, {"win_rate": 0.6})])  # no "loss"
    html_out = dash.render_dashboard(dash.load_runs([a, b]))
    assert ">win_rate<" in html_out
    assert ">loss<" in html_out  # panel exists even though only run a has it


def test_render_is_deterministic(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"win_rate": 0.4}), (1, {"win_rate": 0.6})])
    runs = dash.load_runs([run])
    assert dash.render_dashboard(runs) == dash.render_dashboard(runs)


def test_render_shows_provenance(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"win_rate": 0.5})], seed=99)
    html_out = dash.render_dashboard([dash.load_run_series(run)])
    assert "99" in html_out  # seed
    assert _ENGINE_SHA[: dash._SHA_DISPLAY_LEN] in html_out  # truncated engine sha


# -- write + CLI ------------------------------------------------------------


def test_write_dashboard_creates_file(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"win_rate": 0.5}), (1, {"win_rate": 0.7})])
    out = dash.write_dashboard([run], tmp_path / "out" / "dash.html")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_cli_main_writes_and_returns_zero(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "r", [(0, {"win_rate": 0.5})])
    out = tmp_path / "dash.html"
    code = dash.main(["--out", str(out), str(run)])
    assert code == 0
    assert out.exists()


def test_load_runs_preserves_order(tmp_path: Path) -> None:
    a = _write_run(tmp_path / "a", [(0, {"m": 1.0})])
    b = _write_run(tmp_path / "b", [(0, {"m": 2.0})])
    labels = [series.label for series in dash.load_runs([b, a])]
    assert labels == ["b", "a"]
