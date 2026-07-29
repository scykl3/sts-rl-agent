"""CLI: Outcome-regression supervised pretraining over self-play run games.

Loads a warm-start (behavior-cloned) policy, generates self-play full-run
episodes with it, labels each decision step by its episode's Act-1-clear outcome,
fine-tunes the SAME net with a BCEWithLogits loss on the chosen-action logit, and
saves a checkpoint compatible with ``train_run.py --warm-start`` (no train_run
change: its ``--warm-start`` already loads this 3-key format via load_checkpoint).

Usage:
    python scripts/train_sl.py --warm-start <bc_checkpoint> --out sl_best.pt \
        --num-episodes 500 --epochs 30 --seed 42 --ascension 0

    # Reuse a previously collected dataset instead of self-playing again:
    python scripts/train_sl.py --dataset sp_data.npz --out sl_best.pt --epochs 30
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import torch

# The ~0.40 flat-MLP behavior-cloned checkpoint used to generate self-play, so its
# outcomes carry real ~40/60 Act-1-clear variance.
DEFAULT_WARM_START = "runs/bc-headtohead-20260727-213829/bc_best.pt"
DEFAULT_NUM_EPISODES = 500
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_VAL_FRAC = 0.15
DEFAULT_PATIENCE = 8
DEFAULT_ASCENSION = 0
DEFAULT_OUTPUT = "sl_best.pt"
# Save gate: after fine-tuning, a matched-seed paired eval compares the SL net to
# the input warm-start and the checkpoint is saved only on non-regression (see
# main() and the self_play_sl module docstring). The eval band is a distinct,
# reproducible run of seeds from --gate-eval-base-seed, chosen large so it does not
# overlap the collection seeds [--seed, --seed + --num-episodes).
DEFAULT_GATE_EVAL_EPISODES = 100
DEFAULT_GATE_EVAL_BASE_SEED = 1_000_000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outcome-regression SL pretraining over self-play run games."
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=DEFAULT_WARM_START,
        help=(
            "Path to the behavior-cloned checkpoint that generates self-play AND is "
            f"fine-tuned. Default: {DEFAULT_WARM_START}"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path for the SL checkpoint. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Optional .npz path. If it exists, load it and skip self-play; otherwise "
            "collect self-play and save it here (when given)."
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=f"Number of self-play episodes to collect. Default: {DEFAULT_NUM_EPISODES}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base seed for collection and the train/val split. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Maximum training epochs. Default: {DEFAULT_EPOCHS}",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate. Default: {DEFAULT_LR}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Mini-batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help=f"Validation fraction. Default: {DEFAULT_VAL_FRAC}",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early-stop patience (epochs). Default: {DEFAULT_PATIENCE}",
    )
    parser.add_argument(
        "--ascension",
        type=int,
        default=DEFAULT_ASCENSION,
        help=f"Ascension level for the run env. Default: {DEFAULT_ASCENSION}",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional per-episode step cap for the run env (defaults to the env's own).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (cpu, cuda, mps). Default: cpu",
    )
    parser.add_argument(
        "--gate-eval-episodes",
        type=int,
        default=DEFAULT_GATE_EVAL_EPISODES,
        help=(
            "Greedy holdout episodes for the save gate's paired eval (SL vs the input "
            f"warm-start). Default: {DEFAULT_GATE_EVAL_EPISODES}"
        ),
    )
    parser.add_argument(
        "--gate-eval-base-seed",
        type=int,
        default=DEFAULT_GATE_EVAL_BASE_SEED,
        help=(
            "First seed of the reproducible gate-eval band, distinct from --seed so it "
            f"does not overlap collection seeds. Default: {DEFAULT_GATE_EVAL_BASE_SEED}"
        ),
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help=(
            "Skip the paired-eval save gate and save the fine-tuned checkpoint "
            "unconditionally (debugging only; the fine-tune can regress the policy)."
        ),
    )
    return parser


def _seed_global_rngs(seed: int) -> None:
    """Seed the process-global RNGs so self-play collection is reproducible.

    ``collect_self_play_dataset`` drives ``policy.act`` with ``deterministic=False``;
    the sampling draws from the GLOBAL torch generator (the collector seeds only the
    env reset stream, by the repo contract that seeding the global generators is the
    training entry point's job). Seeding here makes a run's collected dataset
    reproducible from ``--seed``. Mirrors the one-time global seeding in
    :func:`sts_rl.agent.train.train`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()
    log = logging.getLogger(__name__)

    # Seed the process-global RNGs up front so self-play collection - which samples
    # from the global torch generator via policy.act(deterministic=False) - is
    # reproducible from --seed. The collector itself seeds only env resets.
    _seed_global_rngs(args.seed)

    # Engine-dependent imports deferred to here so the CLI is parseable without the
    # engine (tests import and inspect the arg parser without running main).
    from sts_rl.agent.checkpoint_migration import load_checkpoint
    from sts_rl.agent.self_play_sl import (
        SelfPlaySLDataset,
        collect_self_play_dataset,
        outcome_regression_pretrain,
        save_bc_checkpoint,
    )

    # Paired-eval save gate helpers (engine-free: pure eval harness over any env).
    from sts_rl.eval import make_holdout_seeds, paired_evaluate

    # Load (and migrate) the warm-start policy. This SAME net both generates the
    # self-play data and is fine-tuned by the outcome regression. Move it to the
    # requested device up front so collection runs there too (collection reads the
    # device off the policy; only fine-tuning moved it before).
    log.info("Loading warm-start checkpoint from %s", args.warm_start)
    device = torch.device(args.device)
    model = load_checkpoint(args.warm_start).to(device)

    # Collect self-play, or reuse a previously saved dataset.
    if args.dataset is not None and Path(args.dataset).exists():
        log.info("Loading self-play dataset from %s", args.dataset)
        dataset = SelfPlaySLDataset.load(args.dataset)
    else:
        from sts_rl.env.run_adapter import StsRunEnv

        # max_episode_steps is optional: pass it only when set, else use the env's
        # own default (keeps the kwarg types statically checkable).
        if args.max_episode_steps is not None:
            env = StsRunEnv(ascension=args.ascension, max_episode_steps=args.max_episode_steps)
        else:
            env = StsRunEnv(ascension=args.ascension)
        # Greedy driving would collapse to one trajectory; sample (the default) for
        # action variety. eval() only affects modules with train/eval-dependent
        # behavior; the net has none, but it mirrors the collector's eval posture.
        model.eval()
        log.info("Collecting %d self-play episodes (seed=%d)", args.num_episodes, args.seed)
        dataset = collect_self_play_dataset(
            model, env, n_episodes=args.num_episodes, seed=args.seed
        )
        if args.dataset is not None:
            dataset.save(args.dataset)

    log.info(
        "Dataset: %d samples over %d episodes, act1_clear_rate=%.3f",
        len(dataset),
        dataset.manifest.n_episodes,
        dataset.manifest.act1_clear_rate,
    )

    # Snapshot the input warm-start policy BEFORE fine-tuning, as the save gate's
    # non-regression baseline (outcome_regression_pretrain mutates `model` in
    # place). Skipped under --no-gate, which saves unconditionally. Collection above
    # runs under no_grad and does not change the weights, so this still captures the
    # pristine BC net.
    bc_baseline = None if args.no_gate else copy.deepcopy(model)

    # Fine-tune the SAME net with the outcome-regression objective.
    model, stats = outcome_regression_pretrain(
        model,
        dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        patience=args.patience,
    )
    log.info(
        "SL training complete: %d epochs, best_val_loss=%.4f at epoch %d",
        stats.total_epochs,
        stats.best_val_loss,
        stats.best_epoch + 1,
    )

    # hidden_dim is recorded in the 3-key checkpoint format load_checkpoint (and
    # train_run.py --warm-start) reads.
    hidden_dim = model.encoder.output_dim

    # --no-gate: skip the paired eval and save unconditionally (debugging only).
    if args.no_gate:
        log.warning(
            "--no-gate: skipping the paired-eval save gate and saving "
            "unconditionally; the outcome-regression fine-tune CAN regress the "
            "warm-started policy (see the self_play_sl module docstring)."
        )
        save_bc_checkpoint(model, args.out, hidden_dim)
        log.info("SL checkpoint saved to %s (hidden_dim=%d)", args.out, hidden_dim)
        return

    # Save gate. Run a matched-seed paired eval of the fine-tuned net (policy_a) vs
    # the input warm-start (policy_b) on a SEPARATE eval env over a reproducible
    # holdout band, and save only when the SL Act-1 clear rate did not regress
    # (sl >= bc). This is the guard that a degraded fine-tune (the BCE-on-chosen-
    # logit objective can flatten the BC policy) never ships a worse-than-input
    # checkpoint to PPO. Greedy (deterministic=True) so the comparison is
    # reproducible and does not consume the global RNG.
    assert bc_baseline is not None  # set whenever not --no-gate
    # Imported here (not with the other engine-free imports) so a --dataset reuse
    # with --no-gate needs no engine build: the eval env is the only engine
    # dependency on that path.
    from sts_rl.env.run_adapter import StsRunEnv

    if args.max_episode_steps is not None:
        eval_env = StsRunEnv(ascension=args.ascension, max_episode_steps=args.max_episode_steps)
    else:
        eval_env = StsRunEnv(ascension=args.ascension)
    gate_seeds = make_holdout_seeds(args.gate_eval_base_seed, args.gate_eval_episodes)
    log.info(
        "Save gate: paired eval (SL vs BC warm-start) over %d holdout episodes (base_seed=%d)",
        args.gate_eval_episodes,
        args.gate_eval_base_seed,
    )
    report = paired_evaluate(
        model, bc_baseline, eval_env, gate_seeds, device=device, deterministic=True
    )
    eval_env.close()

    sl_clear = report.act1_clear_rate_a
    bc_clear = report.act1_clear_rate_b
    log.info(
        "GATE RESULT: sl_act1_clear_rate=%.3f bc_act1_clear_rate=%.3f delta=%+.3f "
        "(se=%.3f) over %d episodes",
        sl_clear,
        bc_clear,
        report.act1_clear_rate_delta,
        report.act1_clear_rate_delta_se,
        report.n_episodes,
    )

    if sl_clear < bc_clear:
        log.warning(
            "GATE FAIL: the outcome-regression SL pretrain REGRESSED the policy "
            "(sl_act1_clear_rate=%.3f < bc_act1_clear_rate=%.3f); NOT saving %s. "
            "Re-run with --no-gate to save anyway for debugging.",
            sl_clear,
            bc_clear,
            args.out,
        )
        raise SystemExit(1)

    save_bc_checkpoint(model, args.out, hidden_dim)
    log.info(
        "GATE PASS: sl_act1_clear_rate=%.3f >= bc_act1_clear_rate=%.3f; SL "
        "checkpoint saved to %s (hidden_dim=%d)",
        sl_clear,
        bc_clear,
        args.out,
        hidden_dim,
    )


if __name__ == "__main__":
    main()
