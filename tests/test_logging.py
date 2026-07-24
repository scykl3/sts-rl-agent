"""Run manifest and metrics-stream behavior for the experiment logger."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from sts_rl.interface import INTERFACE_VERSION
from sts_rl.utils import logging as run_logging


@dataclass
class _FakeConfig:
    lr: float = 3e-4
    num_envs: int = 64


def test_read_engine_commit_matches_pinned_toml() -> None:
    # The logger must record the same pinned commit the env layer reports.
    commit = run_logging.read_engine_commit()
    assert isinstance(commit, str) and commit


def test_manifest_written_on_construction(tmp_path: Path) -> None:
    with run_logging.RunLogger(tmp_path, _FakeConfig(), seed=7):
        pass
    manifest = run_logging.read_manifest(tmp_path)
    assert manifest["seed"] == 7
    assert manifest["interface_version"] == INTERFACE_VERSION
    assert manifest["engine_commit"] == run_logging.read_engine_commit()
    assert "git" in manifest and "commit" in manifest["git"]
    assert "created_at" in manifest


def test_manifest_serializes_dataclass_config(tmp_path: Path) -> None:
    with run_logging.RunLogger(tmp_path, _FakeConfig(lr=1e-3), seed=0):
        pass
    manifest = run_logging.read_manifest(tmp_path)
    assert manifest["config"] == {"lr": 1e-3, "num_envs": 64}


def test_manifest_serializes_mapping_config(tmp_path: Path) -> None:
    with run_logging.RunLogger(tmp_path, {"a": 1, "b": "two"}, seed=0):
        pass
    manifest = run_logging.read_manifest(tmp_path)
    assert manifest["config"] == {"a": 1, "b": "two"}


def test_extra_provenance_merged(tmp_path: Path) -> None:
    with run_logging.RunLogger(tmp_path, {}, seed=0, extra={"phase": "combat"}):
        pass
    assert run_logging.read_manifest(tmp_path)["extra"] == {"phase": "combat"}


def test_explicit_engine_commit_overrides_pin(tmp_path: Path) -> None:
    with run_logging.RunLogger(tmp_path, {}, seed=0, engine_commit="deadbeef"):
        pass
    assert run_logging.read_manifest(tmp_path)["engine_commit"] == "deadbeef"


def test_metrics_stream_round_trip(tmp_path: Path) -> None:
    with run_logging.RunLogger(tmp_path, {}, seed=0) as logger:
        logger.log_metrics(100, {"loss": 1.5, "kl": 0.01})
        logger.log_metrics(200, {"loss": 1.0, "kl": 0.02})
    records = run_logging.read_metrics(tmp_path)
    assert records == [
        {"step": 100, "loss": 1.5, "kl": 0.01},
        {"step": 200, "loss": 1.0, "kl": 0.02},
    ]


def test_positional_step_overrides_metrics_step(tmp_path: Path) -> None:
    # The positional step is the authoritative record index; a stray "step" key
    # in the metrics payload must not override it.
    with run_logging.RunLogger(tmp_path, {}, seed=0) as logger:
        logger.log_metrics(42, {"step": -1, "loss": 0.5})
    (record,) = run_logging.read_metrics(tmp_path)
    assert record["step"] == 42
    assert record["loss"] == 0.5


def test_log_after_close_raises(tmp_path: Path) -> None:
    logger = run_logging.RunLogger(tmp_path, {}, seed=0)
    logger.close()
    with pytest.raises(ValueError):
        logger.log_metrics(0, {"loss": 1.0})


def test_close_is_idempotent(tmp_path: Path) -> None:
    logger = run_logging.RunLogger(tmp_path, {}, seed=0)
    logger.close()
    logger.close()


def test_reopen_appends_metrics(tmp_path: Path) -> None:
    # A resumed run must extend the metrics log, not truncate the prior one.
    with run_logging.RunLogger(tmp_path, {}, seed=0) as logger:
        logger.log_metrics(1, {"loss": 3.0})
    with run_logging.RunLogger(tmp_path, {}, seed=0) as logger:
        logger.log_metrics(2, {"loss": 2.0})
    steps = [r["step"] for r in run_logging.read_metrics(tmp_path)]
    assert steps == [1, 2]


def test_reopen_preserves_original_provenance(tmp_path: Path) -> None:
    # Resuming a run must keep the manifest the run first started under, not
    # recapture config/seed/timestamp that may have drifted since.
    with run_logging.RunLogger(tmp_path, {"a": 1}, seed=1):
        pass
    original = run_logging.read_manifest(tmp_path)
    with run_logging.RunLogger(tmp_path, {"a": 999}, seed=2):
        pass
    assert run_logging.read_manifest(tmp_path) == original


def test_run_dir_created_when_absent(tmp_path: Path) -> None:
    nested = tmp_path / "runs" / "exp-1"
    with run_logging.RunLogger(nested, {}, seed=0):
        pass
    assert (nested / run_logging.MANIFEST_FILENAME).is_file()
