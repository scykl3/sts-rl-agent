"""Evaluation harness: greedy holdout rollouts and aggregate metrics."""

from __future__ import annotations

from sts_rl.eval.evaluate import (
    EpisodeResult,
    EvalReport,
    PairedEvalReport,
    evaluate,
    make_holdout_seeds,
    paired_evaluate,
    run_episode,
)

__all__ = [
    "EpisodeResult",
    "EvalReport",
    "PairedEvalReport",
    "evaluate",
    "make_holdout_seeds",
    "paired_evaluate",
    "run_episode",
]
