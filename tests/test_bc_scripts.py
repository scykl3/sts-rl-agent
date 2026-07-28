"""Smoke tests for the BC CLI entry points (collect_bc, train_bc).

Engine-free: both scripts defer their engine-dependent imports into ``main``, so
their argument parsers can be imported and exercised without a native build.
These tests only parse args (they never call ``main``), locking the CLI surface
(required flags, defaults) without running collection or training.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ is not an importable package, so put it on sys.path to import the BC
# entry-point modules directly (matching test_train_run / test_train_combat).
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import collect_bc  # noqa: E402  (imported after the sys.path insert above)
import train_bc  # noqa: E402  (imported after the sys.path insert above)


class TestCollectBcArgs:
    """collect_bc.build_arg_parser exposes the documented flags and defaults."""

    def test_minimal_args_use_defaults(self) -> None:
        args = collect_bc.build_arg_parser().parse_args(["--warm-start", "ckpt/best.pt"])
        assert args.warm_start == "ckpt/best.pt"
        assert args.output == collect_bc.DEFAULT_OUTPUT
        assert args.n_episodes == collect_bc.DEFAULT_N_EPISODES
        assert args.seed == collect_bc.DEFAULT_SEED
        assert args.ascension == 0
        assert args.deterministic is True

    def test_overrides_are_parsed(self) -> None:
        args = collect_bc.build_arg_parser().parse_args(
            [
                "--warm-start",
                "ckpt/best.pt",
                "--output",
                "out.npz",
                "--n-episodes",
                "5",
                "--seed",
                "7",
                "--no-deterministic",
            ]
        )
        assert args.output == "out.npz"
        assert args.n_episodes == 5
        assert args.seed == 7
        assert args.deterministic is False


class TestTrainBcArgs:
    """train_bc.build_arg_parser exposes the documented flags and defaults."""

    def test_minimal_args_use_defaults(self) -> None:
        args = train_bc.build_arg_parser().parse_args(
            ["--dataset", "bc.npz", "--warm-start", "ckpt/best.pt"]
        )
        assert args.dataset == "bc.npz"
        assert args.warm_start == "ckpt/best.pt"
        assert args.output == train_bc.DEFAULT_OUTPUT
        assert args.epochs == train_bc.DEFAULT_EPOCHS
        assert args.lr == train_bc.DEFAULT_LR
        assert args.batch_size == train_bc.DEFAULT_BATCH_SIZE
        assert args.val_frac == train_bc.DEFAULT_VAL_FRAC
        assert args.seed == train_bc.DEFAULT_SEED
        assert args.patience == train_bc.DEFAULT_PATIENCE
        assert args.freeze_encoder is False
        assert args.device == "cpu"

    def test_overrides_are_parsed(self) -> None:
        args = train_bc.build_arg_parser().parse_args(
            [
                "--dataset",
                "bc.npz",
                "--warm-start",
                "ckpt/best.pt",
                "--epochs",
                "3",
                "--lr",
                "0.01",
                "--freeze-encoder",
                "--device",
                "mps",
            ]
        )
        assert args.epochs == 3
        assert args.lr == 0.01
        assert args.freeze_encoder is True
        assert args.device == "mps"
