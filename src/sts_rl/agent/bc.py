"""Behavior-cloning pretraining for the reward-card-pick policy head.

Provides:
- ``BCDataset``: container for (obs, teacher_action, mask) tuples with
  persistence (npz) and a provenance manifest.
- ``collect_bc_dataset``: episode loop that drives non-card decisions with a
  base policy and records teacher actions at card-reward steps.
- ``bc_pretrain``: surgical cross-entropy loss over the card-pick + skip
  sub-slice, with train/val split and early stopping.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sts_rl.agent.card_teacher import (
    CARD_PICK_END,
    CARD_PICK_START,
    CARD_SKIP_IDX,
    CardRewardTeacher,
)
from sts_rl.agent.policy_head import MASKED_LOGIT
from sts_rl.agent.train import (
    CHECKPOINT_HIDDEN_DIM_KEY,
    CHECKPOINT_INTERFACE_VERSION_KEY,
    CHECKPOINT_MODEL_KEY,
)
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    INTERFACE_VERSION,
    MAX_REWARD_CARD_SLOTS,
    OBS_FIELDS,
)

logger = logging.getLogger(__name__)

# The surgical sub-slice: card pick slots + skip.
_RS = ACTION_BLOCK_BY_NAME["REWARD_SELECT"]
# Indices into the full ACTION_DIM mask for the card+skip sub-slice.
_CARD_SUBSLICE_INDICES: list[int] = list(range(CARD_PICK_START, CARD_PICK_END)) + [CARD_SKIP_IDX]
_SUBSLICE_SIZE: int = len(_CARD_SUBSLICE_INDICES)  # MAX_REWARD_CARD_SLOTS + 1


def _action_to_subslice_idx(action: int) -> int:
    """Map a full-space action index to the local sub-slice index (0..8).

    Card slot k -> index k; skip -> index MAX_REWARD_CARD_SLOTS.
    """
    if CARD_PICK_START <= action < CARD_PICK_END:
        return action - CARD_PICK_START
    if action == CARD_SKIP_IDX:
        return MAX_REWARD_CARD_SLOTS
    raise ValueError(
        f"action {action} is outside the card-pick+skip sub-slice "
        f"[{CARD_PICK_START}..{CARD_PICK_END}) ∪ {{{CARD_SKIP_IDX}}}"
    )


def _has_legal_card_slot(mask: np.ndarray) -> bool:
    """True if at least one card-pick slot is legal in the full-space mask."""
    return bool(mask[CARD_PICK_START:CARD_PICK_END].any())


# --- Dataset ----------------------------------------------------------------


@dataclass
class BCManifest:
    """Provenance metadata for a BC dataset."""

    n_samples: int
    interface_version: str
    seed: int | None
    teacher_config: dict[str, Any]
    tier_histogram: dict[str, int]  # tier_name -> count of teacher picks at that tier
    skip_rate: float  # fraction of samples where teacher chose skip


class BCDataset:
    """Container for behavior-cloning samples.

    Each sample is (obs_dict, teacher_action_subslice_idx, card_subslice_mask).
    Stored as stacked numpy arrays for persistence, and yields torch tensors for
    training.
    """

    def __init__(
        self,
        obs_arrays: dict[str, np.ndarray],
        actions: np.ndarray,
        masks: np.ndarray,
        manifest: BCManifest,
    ) -> None:
        self.obs_arrays = obs_arrays  # {field_name: (N, *field_shape)}
        self.actions = actions  # (N,) int64, sub-slice indices [0..8]
        self.masks = masks  # (N, _SUBSLICE_SIZE) bool
        self.manifest = manifest

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def save(self, path: str | Path) -> None:
        """Persist to a .npz file with obs arrays, actions, masks, and manifest."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {}
        for name, arr in self.obs_arrays.items():
            arrays[f"obs_{name}"] = arr
        arrays["actions"] = self.actions
        arrays["masks"] = self.masks
        # Manifest as structured metadata
        arrays["manifest_n_samples"] = np.array([self.manifest.n_samples], dtype=np.int64)
        arrays["manifest_seed"] = np.array(
            [self.manifest.seed if self.manifest.seed is not None else -1], dtype=np.int64
        )
        arrays["manifest_skip_rate"] = np.array([self.manifest.skip_rate], dtype=np.float64)
        arrays["manifest_interface_version"] = np.array(
            [self.manifest.interface_version], dtype="U32"
        )
        # teacher_config / tier_histogram are variable-size dicts; serialize to
        # JSON and store as auto-sized unicode scalar arrays so they round-trip
        # under allow_pickle=False (a fixed "U32" would truncate).
        arrays["manifest_teacher_config"] = np.array(json.dumps(self.manifest.teacher_config))
        arrays["manifest_tier_histogram"] = np.array(json.dumps(self.manifest.tier_histogram))
        np.savez(str(path), **arrays)
        logger.info("BC dataset saved to %s (%d samples)", path, len(self))

    @classmethod
    def load(cls, path: str | Path) -> "BCDataset":
        """Load a BC dataset from a .npz file."""
        data = np.load(path, allow_pickle=False)
        obs_arrays: dict[str, np.ndarray] = {}
        for key in data.files:
            if key.startswith("obs_"):
                obs_arrays[key[4:]] = data[key]
        actions = data["actions"]
        masks = data["masks"]
        n_samples = int(data["manifest_n_samples"][0])
        seed_val = int(data["manifest_seed"][0])
        seed = seed_val if seed_val >= 0 else None
        skip_rate = float(data["manifest_skip_rate"][0])
        iface_version = str(data["manifest_interface_version"][0])
        teacher_config = json.loads(str(data["manifest_teacher_config"]))
        tier_histogram = json.loads(str(data["manifest_tier_histogram"]))
        manifest = BCManifest(
            n_samples=n_samples,
            interface_version=iface_version,
            seed=seed,
            teacher_config=teacher_config,
            tier_histogram=tier_histogram,
            skip_rate=skip_rate,
        )
        return cls(obs_arrays, actions, masks, manifest)


# --- Collection -------------------------------------------------------------


def collect_bc_dataset(
    env: Any,
    policy: Any,
    teacher: CardRewardTeacher,
    *,
    n_episodes: int,
    seed: int | None = None,
    deterministic: bool = True,
) -> BCDataset:
    """Run episodes collecting teacher actions at card-reward decision steps.

    The base ``policy`` drives all non-card decisions. At steps where at least one
    card-pick slot is legal, the teacher's action is executed AND logged.

    Args:
        env: A Gymnasium env implementing the STS RL interface (reset/step).
        policy: An ActorCritic with .act(obs_batched, mask_batched, deterministic).
        teacher: CardRewardTeacher that provides select_action.
        n_episodes: Number of full episodes to collect.
        seed: Optional base seed for env resets (incremented per episode).
        deterministic: Whether the base policy acts greedily.

    Returns:
        A BCDataset with all collected card-reward samples.
    """
    obs_buffers: dict[str, list[np.ndarray]] = {f.name: [] for f in OBS_FIELDS}
    action_buffer: list[int] = []
    mask_buffer: list[np.ndarray] = []
    tier_counts: Counter[int] = Counter()
    skip_count = 0
    total_steps = 0

    device = next(policy.parameters()).device

    t0 = time.time()
    for ep_idx in range(n_episodes):
        ep_seed = (seed + ep_idx) if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        mask = info["action_mask"]
        terminated = truncated = False

        while not (terminated or truncated):
            total_steps += 1
            is_card_step = _has_legal_card_slot(mask)

            if is_card_step:
                # Teacher decides; log the sample
                teacher_action = teacher.select_action(obs, mask)
                # Record sub-slice mask and sub-slice action
                subslice_mask = np.array(
                    [bool(mask[i]) for i in _CARD_SUBSLICE_INDICES], dtype=np.bool_
                )
                subslice_action = _action_to_subslice_idx(teacher_action)
                for f in OBS_FIELDS:
                    obs_buffers[f.name].append(obs[f.name].copy())
                action_buffer.append(subslice_action)
                mask_buffer.append(subslice_mask)
                # Track tier stats
                if teacher_action == CARD_SKIP_IDX:
                    skip_count += 1
                else:
                    slot = teacher_action - CARD_PICK_START
                    card_id = int(obs["reward_card_ids"][slot])
                    card_name = teacher._card_id_to_name(card_id)
                    tier = teacher._resolve_tier(card_name)
                    if tier is not None:
                        tier_counts[tier] += 1
                # Execute teacher action
                obs, _reward, terminated, truncated, info = env.step(teacher_action)
            else:
                # Base policy decides
                obs_batched = _obs_to_batched(obs, device)
                mask_batched = torch.as_tensor(mask, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = policy.act(
                        obs_batched, mask_batched, deterministic=deterministic
                    )
                obs, _reward, terminated, truncated, info = env.step(int(action.item()))

            mask = info["action_mask"]

        if (ep_idx + 1) % max(1, n_episodes // 10) == 0:
            elapsed = time.time() - t0
            logger.info(
                "collect_bc: %d/%d episodes, %d samples, %.1fs",
                ep_idx + 1,
                n_episodes,
                len(action_buffer),
                elapsed,
            )

    n_samples = len(action_buffer)
    total_card_decisions = n_samples
    skip_rate = skip_count / max(1, total_card_decisions)
    tier_hist = {str(k): v for k, v in sorted(tier_counts.items())}

    logger.info(
        "BC collection done: %d episodes, %d steps, %d card-decision samples, "
        "skip_rate=%.3f, tier_hist=%s",
        n_episodes,
        total_steps,
        n_samples,
        skip_rate,
        tier_hist,
    )

    # Stack arrays
    if n_samples == 0:
        raise ValueError(
            "collect_bc_dataset collected no card-reward decisions "
            "(n_samples == 0); increase n_episodes"
        )
    obs_arrays = {name: np.stack(arrs, axis=0) for name, arrs in obs_buffers.items()}
    actions = np.array(action_buffer, dtype=np.int64)
    masks = np.stack(mask_buffer, axis=0)

    manifest = BCManifest(
        n_samples=n_samples,
        interface_version=INTERFACE_VERSION,
        seed=seed,
        teacher_config={
            "min_keep_tier": teacher.min_keep_tier,
            "default_tier": teacher.default_tier,
        },
        tier_histogram=tier_hist,
        skip_rate=skip_rate,
    )
    return BCDataset(obs_arrays, actions, masks, manifest)


def _obs_to_batched(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Single obs -> batched tensors (B=1), matching the network's expectations."""
    id_fields = {f.name for f in OBS_FIELDS if f.bounds == "id"}
    batched: dict[str, torch.Tensor] = {}
    for f in OBS_FIELDS:
        dtype = torch.long if f.name in id_fields else torch.float32
        tensor = torch.as_tensor(obs[f.name], dtype=dtype, device=device)
        batched[f.name] = tensor.unsqueeze(0)
    return batched


# --- BC Training Dataset (PyTorch) ------------------------------------------


class _BCTorchDataset(Dataset):
    """Wraps a BCDataset for PyTorch DataLoader."""

    def __init__(self, bc_dataset: BCDataset, indices: list[int] | np.ndarray) -> None:
        self._ds = bc_dataset
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
        mask = torch.as_tensor(self._ds.masks[i], dtype=torch.bool)
        return obs, action, mask


def _collate_bc(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Collate a list of (obs_dict, action, mask) into batched tensors."""
    obs_batch: dict[str, list[torch.Tensor]] = {}
    actions = []
    masks = []
    for obs, action, mask in batch:
        for key, val in obs.items():
            obs_batch.setdefault(key, []).append(val)
        actions.append(action)
        masks.append(mask)
    obs_stacked = {key: torch.stack(vals) for key, vals in obs_batch.items()}
    return obs_stacked, torch.stack(actions), torch.stack(masks)


# --- BC Training Loop -------------------------------------------------------


@dataclass
class BCStats:
    """Training statistics from bc_pretrain."""

    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    val_accuracy_history: list[float] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_epoch: int = 0
    total_epochs: int = 0


def bc_pretrain(
    model: Any,
    dataset: BCDataset,
    *,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 64,
    val_frac: float = 0.15,
    seed: int = 42,
    device: torch.device | str = "cpu",
    freeze_encoder: bool = False,
    patience: int = 8,
) -> BCStats:
    """Run BC pretraining on the card-pick + skip sub-slice.

    Loss: negative log-prob of the teacher action under the masked policy
    distribution restricted to the card sub-slice. Masking is applied within the
    sub-slice (not the full ACTION_DIM), so only card + skip logits participate.

    Default trains end-to-end from a combat warm-start. The freeze_encoder
    option is exposed but NOT recommended: the card-vision encoder's suffix
    columns in trunk.0 are zero-initialized, so a frozen trunk cannot propagate
    card identity to the head. End-to-end training with a modest LR is the
    intended recipe.

    Args:
        model: ActorCritic with encoder, policy head.
        dataset: BCDataset from collect_bc_dataset.
        epochs: Maximum training epochs.
        lr: Learning rate (modest default to preserve combat features).
        batch_size: Mini-batch size.
        val_frac: Fraction of data held for validation.
        seed: Random seed for reproducibility.
        device: Torch device.
        freeze_encoder: If True, freeze encoder parameters (not recommended).
        patience: Early-stop on val accuracy plateau.

    Returns:
        BCStats with loss/accuracy histories.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.train()

    if freeze_encoder:
        logger.warning(
            "freeze_encoder=True: card-vision suffix columns in trunk.0 are "
            "zero-initialized, so a frozen trunk cannot propagate card identity. "
            "This is NOT recommended for BC pretraining."
        )
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Train/val split
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    if n < 2 or n_train < 1 or n_val < 1:
        raise ValueError(
            f"BC dataset too small for a train/val split: n={n}, n_train={n_train}, "
            f"n_val={n_val} (val_frac={val_frac}); need n >= 2 with at least one "
            f"train and one val sample"
        )
    logger.info("BC train/val split: n_train=%d, n_val=%d", n_train, n_val)
    gen = torch.Generator().manual_seed(seed)
    # Shuffle indices deterministically, then split
    all_indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)
    train_idx = all_indices[:n_train].tolist()
    val_idx = all_indices[n_train:].tolist()

    train_ds = _BCTorchDataset(dataset, train_idx)
    val_ds = _BCTorchDataset(dataset, val_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_bc,
        generator=gen,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_bc,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    stats = BCStats()
    # -inf (not 0.0) so epoch 0 always captures best_state, even if val accuracy
    # is 0.0 throughout; otherwise best_state stays None and the last-epoch
    # weights are returned instead of the tracked best.
    best_val_acc = float("-inf")
    epochs_no_improve = 0
    best_state: dict[str, torch.Tensor] | None = None

    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for obs_batch, action_batch, mask_batch in train_loader:
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            action_batch = action_batch.to(device)
            mask_batch = mask_batch.to(device)

            # Forward: get raw logits from the policy head's linear layer
            features = model.encoder(obs_batch)
            raw_logits = model.policy.logits(features)  # (B, ACTION_DIM)

            # Extract the card sub-slice logits
            subslice_logits = raw_logits[:, _CARD_SUBSLICE_INDICES]  # (B, 9)
            # Apply sub-slice mask
            subslice_logits = subslice_logits.masked_fill(~mask_batch, MASKED_LOGIT)

            # Cross-entropy over the sub-slice
            loss = F.cross_entropy(subslice_logits, action_batch)

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
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for obs_batch, action_batch, mask_batch in val_loader:
                obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
                action_batch = action_batch.to(device)
                mask_batch = mask_batch.to(device)

                features = model.encoder(obs_batch)
                raw_logits = model.policy.logits(features)
                subslice_logits = raw_logits[:, _CARD_SUBSLICE_INDICES]
                subslice_logits = subslice_logits.masked_fill(~mask_batch, MASKED_LOGIT)

                loss = F.cross_entropy(subslice_logits, action_batch)
                preds = subslice_logits.argmax(dim=-1)

                bs = action_batch.shape[0]
                val_loss += loss.item() * bs
                val_correct += (preds == action_batch).sum().item()
                val_total += bs

        avg_val_loss = val_loss / max(1, val_total)
        val_accuracy = val_correct / max(1, val_total)
        stats.val_loss_history.append(avg_val_loss)
        stats.val_accuracy_history.append(val_accuracy)

        logger.info(
            "BC epoch %d/%d: train_loss=%.4f, val_loss=%.4f, val_acc=%.4f",
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_val_loss,
            val_accuracy,
        )

        # Early stopping on val accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            stats.best_val_accuracy = val_accuracy
            stats.best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "BC early stopping at epoch %d (patience=%d, best_acc=%.4f at epoch %d)",
                    epoch + 1,
                    patience,
                    best_val_acc,
                    stats.best_epoch + 1,
                )
                break

    stats.total_epochs = epochs_run

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Unfreeze if we froze
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = True

    return stats


def save_bc_checkpoint(model: Any, path: str | Path, hidden_dim: int) -> None:
    """Save a BC-trained model as a checkpoint loadable by load_checkpoint.

    The checkpoint contains exactly the keys expected by the warm-start path:
    model_state_dict, hidden_dim, interface_version.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        CHECKPOINT_MODEL_KEY: model.state_dict(),
        CHECKPOINT_HIDDEN_DIM_KEY: hidden_dim,
        CHECKPOINT_INTERFACE_VERSION_KEY: INTERFACE_VERSION,
    }
    torch.save(payload, path)
    logger.info("BC checkpoint saved to %s", path)
