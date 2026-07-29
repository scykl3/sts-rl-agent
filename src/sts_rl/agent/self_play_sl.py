"""Outcome-regression supervised pretraining over self-play run-mode games.

A pre-PPO bootstrap for the policy. It generates self-play full-run episodes with
an existing (behavior-cloned) policy, labels each decision step by its episode's
Act-1-clear outcome (reached Act 2 = 1, else 0), and fine-tunes the SAME net so
each step's CHOSEN action logit regresses toward that 0/1 outcome under a
``binary_cross_entropy_with_logits`` loss. The resulting checkpoint warm-starts
PPO (it is saved in the same 3-key format the warm-start path reads).

Provides:
- ``SelfPlaySLDataset``: container for (obs, chosen_action, full ACTION_DIM mask,
  outcome) tuples with npz persistence and a provenance manifest.
- ``collect_self_play_dataset``: run-mode episode loop that drives the policy
  STOCHASTICALLY (for action variety), logs every decision step, and backfills
  the per-episode Act-1-clear label onto all of that episode's steps.
- ``outcome_regression_pretrain``: BCEWithLogits on the chosen-action logit, with
  a train/val split and early stopping on validation loss.

Known approximation and degradation risk: BCE on individual chosen logits is a
crude bootstrap, not a calibrated policy objective, and it can make the policy
WORSE than the warm start rather than better. The label is the whole run's 0/1
outcome shared across every decision, so per-decision signal-to-noise is low and
the dominant gradient pulls chosen logits toward the dataset base rate (flattening
the BC preferences); there is no baseline or credit assignment (a strong move in a
lost run is pushed down and a blunder in a won run is pushed up); and only the
chosen action's logit gets gradient, so un-chosen actions are never calibrated.
Because a degraded checkpoint would hand PPO a worse start than the input, the
``scripts/train_sl.py`` entry point does NOT save unconditionally: it runs a
matched-seed paired eval of the fine-tuned net against the input warm-start and
saves only when the Act-1 clear rate did not regress. PPO refines the surviving
checkpoint afterward from these warm-started weights.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse save_bc_checkpoint's 3-key payload so the SL output warm-starts PPO with
# no train_run change, exactly like a BC checkpoint.
from sts_rl.agent.bc import save_bc_checkpoint
from sts_rl.agent.rollout_collector import observation_to_batched_tensors

# Reuse the Act-1-clear labelling mechanism (do not hardcode the index or key):
# the terminal act is read from the final info and compared to ACT2_INDEX, the
# same way sts_rl.eval.evaluate.run_episode derives its act1_clear_rate.
from sts_rl.eval.evaluate import ACT2_INDEX, ACT_INFO_KEY, MASK_INFO_KEY
from sts_rl.interface import ACTION_DIM, INTERFACE_VERSION, OBS_FIELDS, assert_valid_mask

logger = logging.getLogger(__name__)

__all__ = [
    "SelfPlaySLManifest",
    "SelfPlaySLDataset",
    "SelfPlaySLStats",
    "collect_self_play_dataset",
    "outcome_regression_pretrain",
    "save_bc_checkpoint",
]


# --- Dataset ----------------------------------------------------------------


@dataclass
class SelfPlaySLManifest:
    """Provenance metadata for a self-play SL dataset."""

    n_episodes: int
    n_samples: int
    act1_clear_rate: float  # fraction of collected EPISODES that cleared Act 1
    interface_version: str
    seed: int | None


class SelfPlaySLDataset:
    """Container for outcome-regression samples.

    Each sample is (obs_dict, chosen_action, full ACTION_DIM action mask,
    outcome). Stored as stacked numpy arrays for persistence and yielded as torch
    tensors for training. The mask is the full-space legality at that step; it is
    kept for provenance and is NOT part of the loss (the outcome-regression loss
    reads only the chosen action's logit).
    """

    def __init__(
        self,
        obs_arrays: dict[str, np.ndarray],
        actions: np.ndarray,
        masks: np.ndarray,
        outcomes: np.ndarray,
        manifest: SelfPlaySLManifest,
    ) -> None:
        n = int(actions.shape[0])
        # Boundary shape guard: a malformed dataset should fail here, not with a
        # cryptic mismatch deep in the training loop.
        if masks.shape != (n, ACTION_DIM):
            raise ValueError(
                f"masks must be (N, ACTION_DIM)=({n}, {ACTION_DIM}), got {tuple(masks.shape)}"
            )
        if outcomes.shape != (n,):
            raise ValueError(f"outcomes must be (N,)=({n},), got {tuple(outcomes.shape)}")
        for name, arr in obs_arrays.items():
            if arr.shape[0] != n:
                raise ValueError(
                    f"obs field {name!r} has {arr.shape[0]} rows, expected {n} to match actions"
                )
        self.obs_arrays = obs_arrays  # {field_name: (N, *field_shape)}
        self.actions = actions  # (N,) int64, full-space action indices
        self.masks = masks  # (N, ACTION_DIM) bool
        self.outcomes = outcomes  # (N,) float32, 0.0/1.0 Act-1-clear label
        self.manifest = manifest

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def save(self, path: str | Path) -> None:
        """Persist to a .npz file with obs arrays, actions, masks, outcomes, manifest."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {}
        for name, arr in self.obs_arrays.items():
            arrays[f"obs_{name}"] = arr
        arrays["actions"] = self.actions
        arrays["masks"] = self.masks
        arrays["outcomes"] = self.outcomes
        # Scalar manifest fields as small typed arrays so the whole file round-trips
        # under allow_pickle=False (seed -1 encodes None).
        arrays["manifest_n_episodes"] = np.array([self.manifest.n_episodes], dtype=np.int64)
        arrays["manifest_n_samples"] = np.array([self.manifest.n_samples], dtype=np.int64)
        arrays["manifest_act1_clear_rate"] = np.array(
            [self.manifest.act1_clear_rate], dtype=np.float64
        )
        arrays["manifest_seed"] = np.array(
            [self.manifest.seed if self.manifest.seed is not None else -1], dtype=np.int64
        )
        arrays["manifest_interface_version"] = np.array(
            [self.manifest.interface_version], dtype="U32"
        )
        np.savez(str(path), **arrays)
        logger.info("self-play SL dataset saved to %s (%d samples)", path, len(self))

    @classmethod
    def load(cls, path: str | Path) -> "SelfPlaySLDataset":
        """Load a self-play SL dataset from a .npz file."""
        data = np.load(path, allow_pickle=False)
        obs_arrays: dict[str, np.ndarray] = {}
        for key in data.files:
            if key.startswith("obs_"):
                obs_arrays[key[4:]] = data[key]
        actions = data["actions"]
        masks = data["masks"]
        outcomes = data["outcomes"]
        seed_val = int(data["manifest_seed"][0])
        manifest = SelfPlaySLManifest(
            n_episodes=int(data["manifest_n_episodes"][0]),
            n_samples=int(data["manifest_n_samples"][0]),
            act1_clear_rate=float(data["manifest_act1_clear_rate"][0]),
            interface_version=str(data["manifest_interface_version"][0]),
            seed=seed_val if seed_val >= 0 else None,
        )
        return cls(obs_arrays, actions, masks, outcomes, manifest)


# --- Collection -------------------------------------------------------------


def collect_self_play_dataset(
    policy: Any,
    env: Any,
    *,
    n_episodes: int,
    seed: int | None = None,
    deterministic: bool = False,
) -> SelfPlaySLDataset:
    """Run self-play run episodes, logging every step and labelling by outcome.

    The ``policy`` drives every decision. Sampling is STOCHASTIC by default
    (``deterministic=False``) so the collected actions have variety rather than
    collapsing onto the single greedy trajectory. Every decision step's (obs,
    chosen action, full ACTION_DIM mask) is recorded. At episode end the terminal
    ``act`` is read from the final ``info`` exactly as
    :func:`sts_rl.eval.evaluate.run_episode` does, and the Act-1-clear label
    (terminal ``act >= ACT2_INDEX``) is BACKFILLED onto every decision step of
    that episode - the run outcome is only known at termination, so all steps of
    one episode share one 0/1 label.

    Args:
        policy: An ActorCritic-like module with ``.act(obs_batched, mask_batched,
            deterministic)`` returning ``(action, ...)`` and ``.parameters()`` (for
            device inference).
        env: A run-mode Gymnasium env honouring the interface (``reset``/``step``,
            the ``action_mask`` and terminal ``act`` info keys).
        n_episodes: Number of full run episodes to collect (>= 1).
        seed: Optional base seed for env resets (incremented per episode).
        deterministic: If True the policy acts greedily; default False (sampled)
            for action variety across the self-play data.

    Returns:
        A SelfPlaySLDataset with one sample per decision step.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    obs_buffers: dict[str, list[np.ndarray]] = {f.name: [] for f in OBS_FIELDS}
    action_buffer: list[int] = []
    mask_buffer: list[np.ndarray] = []
    outcome_buffer: list[float] = []
    n_clear = 0

    device = next(policy.parameters()).device

    t0 = time.time()
    for ep_idx in range(n_episodes):
        ep_seed = (seed + ep_idx) if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        mask = info[MASK_INFO_KEY]
        terminated = truncated = False
        ep_start = len(action_buffer)  # first buffer index of this episode's steps

        while not (terminated or truncated):
            # Validate the decision mask at the env boundary before acting on it,
            # matching evaluate.run_episode / RolloutCollector: a malformed or
            # all-illegal mask fails here with a clear message rather than as a
            # cryptic error deeper in policy.act.
            assert_valid_mask(mask)
            obs_batched = observation_to_batched_tensors(obs, device)
            mask_batched = torch.as_tensor(mask, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy.act(obs_batched, mask_batched, deterministic=deterministic)[0]
            chosen = int(action.item())
            # Record the decision BEFORE stepping: obs/mask are the state the
            # policy actually acted on. copy() so a later env mutation of its own
            # buffers cannot alias into the dataset. The outcome is a placeholder
            # here; it is backfilled once the episode terminates.
            for f in OBS_FIELDS:
                obs_buffers[f.name].append(obs[f.name].copy())
            action_buffer.append(chosen)
            mask_buffer.append(np.asarray(mask, dtype=np.bool_).copy())
            outcome_buffer.append(0.0)
            obs, _reward, terminated, truncated, info = env.step(chosen)
            mask = info[MASK_INFO_KEY]

        # Act-1-clear label via the evaluate.run_episode mechanism: read the
        # terminal act from the final info and compare to ACT2_INDEX. Backfill it
        # onto EVERY step of this episode - all steps of one episode share its
        # single 0/1 outcome, since a run's result is only known at termination.
        terminal_act = int(info[ACT_INFO_KEY])
        outcome = 1.0 if terminal_act >= ACT2_INDEX else 0.0
        if outcome:
            n_clear += 1
        for i in range(ep_start, len(action_buffer)):
            outcome_buffer[i] = outcome

        if (ep_idx + 1) % max(1, n_episodes // 10) == 0:
            logger.info(
                "collect_self_play: %d/%d episodes, %d steps, %.1fs",
                ep_idx + 1,
                n_episodes,
                len(action_buffer),
                time.time() - t0,
            )

    n_samples = len(action_buffer)
    # Defensive: every episode with >= 1 step logs a sample, so this only trips on
    # a pathological env that terminates before any decision step is taken.
    if n_samples == 0:
        raise ValueError(
            "collect_self_play_dataset collected no decision steps (n_samples == 0); "
            "the env terminated before any action was taken"
        )

    obs_arrays = {name: np.stack(arrs, axis=0) for name, arrs in obs_buffers.items()}
    actions = np.array(action_buffer, dtype=np.int64)
    masks = np.stack(mask_buffer, axis=0)
    outcomes = np.array(outcome_buffer, dtype=np.float32)
    # Episode-level clear rate (fraction of runs that cleared Act 1), matching
    # sts_rl.eval.evaluate's per-episode metric - not the step-weighted mean of
    # `outcomes`, which would over-count longer episodes.
    act1_clear_rate = n_clear / n_episodes

    logger.info(
        "self-play collection done: %d episodes, %d steps, act1_clear_rate=%.3f",
        n_episodes,
        n_samples,
        act1_clear_rate,
    )

    manifest = SelfPlaySLManifest(
        n_episodes=n_episodes,
        n_samples=n_samples,
        act1_clear_rate=act1_clear_rate,
        interface_version=INTERFACE_VERSION,
        seed=seed,
    )
    return SelfPlaySLDataset(obs_arrays, actions, masks, outcomes, manifest)


# --- Training dataset (PyTorch) ---------------------------------------------


class _SelfPlaySLTorchDataset(Dataset):
    """Wraps a SelfPlaySLDataset for a DataLoader, yielding (obs, action, outcome).

    The mask is deliberately NOT yielded: the outcome-regression loss reads only
    the chosen action's logit, so the stored mask (provenance) plays no part in
    training.
    """

    def __init__(self, ds: SelfPlaySLDataset, indices: list[int] | np.ndarray) -> None:
        self._ds = ds
        self._indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        i = int(self._indices[idx])
        obs: dict[str, torch.Tensor] = {}
        id_fields = {f.name for f in OBS_FIELDS if f.bounds == "id"}
        for f in OBS_FIELDS:
            arr = self._ds.obs_arrays[f.name][i]
            dtype = torch.long if f.name in id_fields else torch.float32
            obs[f.name] = torch.as_tensor(arr, dtype=dtype)
        action = torch.tensor(self._ds.actions[i], dtype=torch.long)
        outcome = torch.tensor(self._ds.outcomes[i], dtype=torch.float32)
        return obs, action, outcome


def _collate_self_play(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Collate a list of (obs_dict, action, outcome) into batched tensors."""
    obs_batch: dict[str, list[torch.Tensor]] = {}
    actions = []
    outcomes = []
    for obs, action, outcome in batch:
        for key, val in obs.items():
            obs_batch.setdefault(key, []).append(val)
        actions.append(action)
        outcomes.append(outcome)
    obs_stacked = {key: torch.stack(vals) for key, vals in obs_batch.items()}
    return obs_stacked, torch.stack(actions), torch.stack(outcomes)


# --- Training loop ----------------------------------------------------------


@dataclass
class SelfPlaySLStats:
    """Training statistics from outcome_regression_pretrain."""

    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_epochs: int = 0


def outcome_regression_pretrain(
    model: Any,
    dataset: SelfPlaySLDataset,
    *,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 64,
    val_frac: float = 0.15,
    seed: int = 42,
    device: torch.device | str = "cpu",
    patience: int = 8,
) -> tuple[Any, SelfPlaySLStats]:
    """Fine-tune ``model`` so each chosen action's logit regresses toward its outcome.

    Loss: ``binary_cross_entropy_with_logits`` between the CHOSEN action's raw
    (pre-mask) logit and that step's 0/1 Act-1-clear outcome. Only the chosen
    logit participates - the full-space mask is not applied here, since the target
    is a single logit, not a distribution.

    Known approximation and degradation risk (see the module docstring): this
    per-logit BCE is a crude bootstrap, not a calibrated policy objective, and it
    CAN degrade the warm-started policy rather than improve it - the shared
    whole-run label gives a low per-decision signal-to-noise ratio that pulls
    chosen logits toward the base rate, there is no credit assignment, and only
    the chosen action's logit receives gradient (un-chosen actions are never
    calibrated). This function only runs the fine-tune; guarding against a
    regression is the caller's job. ``scripts/train_sl.py`` gates the saved
    checkpoint on a matched-seed paired eval showing no Act-1-clear regression vs
    the input warm-start. PPO refines the surviving checkpoint afterward.

    Trains end-to-end (encoder + policy head) from the warm-started weights with a
    modest LR, mirroring ``bc_pretrain``'s recipe. Early stops on validation loss
    and restores the best-val-loss weights before returning.

    Args:
        model: ActorCritic with ``encoder`` and ``policy.logits``.
        dataset: SelfPlaySLDataset from collect_self_play_dataset.
        epochs: Maximum training epochs.
        lr: Learning rate (modest default to preserve the warm-started features).
        batch_size: Mini-batch size.
        val_frac: Fraction held for validation. At least one sample is always held
            out (n_val = max(1, int(n * val_frac))), so val_frac=0.0 still reserves
            a single validation sample rather than training on all data.
        seed: Random seed for the split and DataLoader shuffle (reproducible).
        device: Torch device.
        patience: Early-stop on validation-loss plateau.

    Returns:
        ``(model, stats)``: ``model`` is the same net, fine-tuned in place and
        restored to its best-val-loss weights (returned for chaining); ``stats``
        carries the per-epoch loss history (mirrors ``bc_pretrain``'s BCStats).

    Raises:
        ValueError: if the outcomes have no variance (all cleared or all not
            cleared), so BCE has no discriminative signal; or if the dataset is
            too small to form a train/val split.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.train()

    # Train/val split sizing. Always hold out at least one validation sample, so
    # val_frac=0.0 reserves one rather than training on all data (this guard
    # requires n >= 2). Checked before the outcome-variance guard so a single-row
    # dataset fails on size first.
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    if n < 2 or n_train < 1 or n_val < 1:
        raise ValueError(
            f"self-play SL dataset too small for a train/val split: n={n}, "
            f"n_train={n_train}, n_val={n_val} (val_frac={val_frac}); need n >= 2 "
            f"with at least one train and one val sample"
        )

    # Degenerate-outcome guard: BCE toward a constant target has no discriminative
    # signal - with every outcome equal, all chosen logits are pushed the same way
    # and the "bias toward clear-correlated actions" bootstrap is meaningless.
    # Require both a cleared (1.0) and a non-cleared (0.0) episode in the data.
    unique_outcomes = np.unique(dataset.outcomes)
    if unique_outcomes.size < 2:
        raise ValueError(
            f"outcome_regression_pretrain needs both cleared (1.0) and non-cleared "
            f"(0.0) outcomes to learn a discriminative signal; all {len(dataset)} "
            f"samples share outcome={unique_outcomes.tolist()} (zero variance)"
        )
    logger.info("self-play SL train/val split: n_train=%d, n_val=%d", n_train, n_val)

    gen = torch.Generator().manual_seed(seed)
    all_indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)
    train_idx = all_indices[:n_train].tolist()
    val_idx = all_indices[n_train:].tolist()

    train_loader = DataLoader(
        _SelfPlaySLTorchDataset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_self_play,
        generator=gen,
        drop_last=False,
    )
    val_loader = DataLoader(
        _SelfPlaySLTorchDataset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_self_play,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats = SelfPlaySLStats()
    # inf so epoch 0's val loss always improves on the initial best and captures
    # best_state (lower is better); otherwise best_state could stay None.
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state: dict[str, torch.Tensor] | None = None

    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1

        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for obs_batch, action_batch, outcome_batch in train_loader:
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            action_batch = action_batch.to(device)
            outcome_batch = outcome_batch.to(device)

            features = model.encoder(obs_batch)
            raw_logits = model.policy.logits(features)  # (B, ACTION_DIM), pre-mask
            # Regress ONLY the chosen action's logit toward the episode outcome.
            chosen_logits = raw_logits.gather(1, action_batch[:, None]).squeeze(1)  # (B,)
            loss = F.binary_cross_entropy_with_logits(chosen_logits, outcome_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = action_batch.shape[0]
            epoch_loss += loss.item() * bs
            epoch_samples += bs

        avg_train_loss = epoch_loss / max(1, epoch_samples)
        stats.train_loss_history.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for obs_batch, action_batch, outcome_batch in val_loader:
                obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
                action_batch = action_batch.to(device)
                outcome_batch = outcome_batch.to(device)

                features = model.encoder(obs_batch)
                raw_logits = model.policy.logits(features)
                chosen_logits = raw_logits.gather(1, action_batch[:, None]).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(chosen_logits, outcome_batch)

                bs = action_batch.shape[0]
                val_loss += loss.item() * bs
                val_total += bs

        avg_val_loss = val_loss / max(1, val_total)
        stats.val_loss_history.append(avg_val_loss)

        logger.info(
            "self-play SL epoch %d/%d: train_loss=%.4f, val_loss=%.4f",
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_val_loss,
        )

        # Early stopping on validation loss (lower is better).
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stats.best_val_loss = avg_val_loss
            stats.best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "self-play SL early stopping at epoch %d (patience=%d, "
                    "best_val_loss=%.4f at epoch %d)",
                    epoch + 1,
                    patience,
                    best_val_loss,
                    stats.best_epoch + 1,
                )
                break

    stats.total_epochs = epochs_run

    # Restore best weights.
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, stats
