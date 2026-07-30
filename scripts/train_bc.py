"""CLI: Train the strategic decision heads via behavior cloning.

Loads a pre-collected BC dataset and a warm-start checkpoint, runs supervised
pretraining over the full masked action space (each sample teaches only its
screen's legal actions), and saves a checkpoint compatible with
``train_run.py --warm-start``.

Usage:
    python scripts/train_bc.py --dataset bc_dataset.npz \
        --warm-start <combat_checkpoint> --output bc_best.pt \
        --epochs 30 --lr 1e-4 --seed 42
"""

from __future__ import annotations

import argparse
import logging

DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_VAL_FRAC = 0.15
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 8
DEFAULT_OUTPUT = "bc_best.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the strategic decision heads via behavior cloning."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a BC dataset (.npz) from collect_bc.py.",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        required=True,
        help="Path to the combat/card-vision checkpoint to warm-start from.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path for the BC checkpoint. Default: {DEFAULT_OUTPUT}",
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
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early-stop patience (epochs). Default: {DEFAULT_PATIENCE}",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=False,
        help="Freeze the encoder (NOT recommended for strategic BC).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (cpu, cuda, mps). Default: cpu",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()
    log = logging.getLogger(__name__)

    from sts_rl.agent.bc import BCDataset, bc_pretrain, save_bc_checkpoint
    from sts_rl.agent.checkpoint_migration import load_checkpoint

    # Load dataset
    log.info("Loading BC dataset from %s", args.dataset)
    dataset = BCDataset.load(args.dataset)
    log.info("Dataset: %d samples, skip_rate=%.3f", len(dataset), dataset.manifest.skip_rate)

    # Load warm-start model
    log.info("Loading warm-start checkpoint from %s", args.warm_start)
    model = load_checkpoint(args.warm_start)

    # Train
    stats = bc_pretrain(
        model,
        dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
        freeze_encoder=args.freeze_encoder,
        patience=args.patience,
    )

    log.info(
        "BC training complete: %d epochs, best_val_acc=%.4f at epoch %d",
        stats.total_epochs,
        stats.best_val_accuracy,
        stats.best_epoch + 1,
    )

    # Save BC checkpoint
    hidden_dim = model.encoder.output_dim
    save_bc_checkpoint(model, args.output, hidden_dim)
    log.info("BC checkpoint saved to %s (hidden_dim=%d)", args.output, hidden_dim)


if __name__ == "__main__":
    main()
