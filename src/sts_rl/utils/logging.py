"""Experiment logging: a per-run manifest plus a metrics stream, for replay.

Every training or evaluation run gets its own directory holding two files:

- ``manifest.json`` - the fixed provenance of the run: the exact config, the
  base seed, the pinned engine commit, this repo's git revision, the interface
  version, and the host/library environment. This is the record you need to
  replay a run: re-seed from the manifest's ``seed`` against the same
  ``engine_commit`` and config and the run reproduces.
- ``metrics.jsonl`` - one JSON object per line, ``{"step": ..., <metrics>}``,
  appended as training progresses. Line-delimited so a crashed run still leaves
  a valid partial log and downstream tooling can stream it.

The module reads the pinned engine commit straight from ``configs/engine.toml``
rather than importing the env layer, so it carries no native-engine dependency
and stays importable (and testable) without a built engine.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sts_rl.interface import INTERFACE_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_CONFIG = _REPO_ROOT / "configs" / "engine.toml"

MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.jsonl"

# The per-record x-axis key in metrics.jsonl. Written by log_metrics and read
# back by downstream tooling (e.g. the dashboard), so it lives as one constant
# rather than a literal repeated on both sides.
STEP_KEY = "step"

# Sentinel recorded when a provenance field cannot be resolved (e.g. git is
# absent, or the run is not inside a checkout). Kept explicit so a replay reader
# can tell "unknown" apart from a real value.
UNKNOWN = "unknown"


def read_engine_commit() -> str:
    """Return the pinned engine commit SHA from ``configs/engine.toml``.

    Mirrors ``sts_rl.env.engine.engine_commit`` but without importing the native
    binding, so provenance capture works on a machine that has not built the
    engine.
    """
    with _ENGINE_CONFIG.open("rb") as handle:
        return str(tomllib.load(handle)["engine"]["commit"])


def repo_git_revision() -> dict[str, Any]:
    """Return this repo's HEAD SHA and dirty flag, or sentinels if unavailable.

    ``dirty`` is ``True`` when the working tree has uncommitted changes, so a
    manifest records not just which commit but whether it was modified - the
    difference between a reproducible run and a "close enough" one.
    """
    return {
        "commit": _git_output(["rev-parse", "HEAD"]),
        "dirty": _git_is_dirty(),
    }


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN
    return result.stdout.strip() or UNKNOWN


def _git_is_dirty() -> bool | None:
    status = _git_output(["status", "--porcelain"])
    if status == UNKNOWN:
        return None
    return bool(status)


def _config_to_dict(config: Any) -> Any:
    """Coerce a config (dataclass, mapping, or plain value) to a JSON-ready form.

    Dataclasses are unrolled recursively; mappings are copied. Anything else is
    returned as-is and left to the JSON encoder's string fallback, so a caller
    is never blocked from logging by an exotic config type.
    """
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    return config


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


class RunLogger:
    """Write a run's manifest and stream its metrics to a run directory.

    Construct once at the start of a run; the manifest is written immediately so
    provenance survives an early crash. Call :meth:`log_metrics` each time there
    are metrics to record, and :meth:`close` (or use the logger as a context
    manager) to release the metrics file.
    """

    def __init__(
        self,
        run_dir: str | Path,
        config: Any,
        seed: int,
        *,
        engine_commit: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Create ``run_dir`` and write ``manifest.json``.

        ``config`` may be a dataclass or a mapping; it is serialized verbatim.
        ``engine_commit`` defaults to the pinned commit in ``configs/engine.toml``.
        ``extra`` is merged into the manifest under ``"extra"`` for run-specific
        provenance (e.g. a phase label or a checkpoint path).
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Bound before the open below so close() is safe even if the open raises
        # (e.g. the run dir becomes unwritable).
        self._metrics_file: Any = None

        if (self.run_dir / MANIFEST_FILENAME).exists():
            # Resuming an existing run: preserve the original provenance rather
            # than recapturing git/engine/timestamp, which may have drifted since
            # the run first started and would misrepresent what it ran on.
            self.manifest: dict[str, Any] = read_manifest(self.run_dir)
        else:
            self.manifest = {
                "interface_version": INTERFACE_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "engine_commit": (
                    engine_commit if engine_commit is not None else read_engine_commit()
                ),
                "git": repo_git_revision(),
                "environment": _environment(),
                "config": _config_to_dict(config),
            }
            if extra is not None:
                self.manifest["extra"] = dict(extra)
            self._write_manifest()

        # Manifest is the single source of truth; on resume this is the original
        # seed, not the (possibly mismatched) argument passed to this reopen.
        self.seed = self.manifest["seed"]

        # Append so re-opening an existing run dir extends its metrics rather
        # than silently truncating a prior run's log.
        self._metrics_file = (self.run_dir / METRICS_FILENAME).open("a", encoding="utf-8")

    def _write_manifest(self) -> None:
        path = self.run_dir / MANIFEST_FILENAME
        # default=str so an unexpected non-serializable config value degrades to
        # its string form instead of aborting the whole run.
        path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def log_metrics(self, step: int, metrics: Mapping[str, float]) -> None:
        """Append one ``{**metrics, "step": step}`` record to ``metrics.jsonl``.

        Flushed per call so a run interrupted mid-training still leaves every
        already-logged step on disk. The positional ``step`` is authoritative:
        it is applied last so a stray ``"step"`` key in ``metrics`` cannot
        override the record's index.
        """
        if self._metrics_file is None or self._metrics_file.closed:
            raise ValueError("cannot log metrics after the logger is closed")
        record = {**dict(metrics), STEP_KEY: step}
        self._metrics_file.write(json.dumps(record, default=str) + "\n")
        self._metrics_file.flush()

    def close(self) -> None:
        """Close the metrics file. Idempotent."""
        if self._metrics_file is not None and not self._metrics_file.closed:
            self._metrics_file.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Load a run's ``manifest.json`` (the provenance needed to replay it)."""
    path = Path(run_dir) / MANIFEST_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(run_dir: str | Path, *, skip_malformed: bool = False) -> list[dict[str, Any]]:
    """Load ``metrics.jsonl`` as a list of records, skipping blank lines.

    With ``skip_malformed=True`` a line that is not valid JSON is skipped rather
    than raised. The jsonl format is meant to survive a crash mid-write, which
    usually leaves a truncated final line; a reader that only wants to plot what
    was logged should pass this. Off by default so a caller relying on strict
    provenance still sees the error.
    """
    path = Path(run_dir) / METRICS_FILENAME
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if not skip_malformed:
                raise
    return records
