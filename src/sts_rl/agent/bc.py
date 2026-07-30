"""Behavior-cloning pretraining for the strategic decision heads.

Provides:
- ``BCDataset``: container for (obs, teacher_action, mask) tuples with
  persistence (npz) and a provenance manifest. The teacher action is a
  full-space action index and the mask is the full ``(ACTION_DIM,)`` legality
  mask, so a sample can teach ANY screen the teacher covers (card pick, campfire,
  map, potion), not only the reward-card sub-slice.
- ``collect_bc_dataset``: episode loop that asks the teacher on every step and
  records a sample whenever the teacher acts (returns an action), driving the
  remaining decisions with a base policy (the teacher defers with ``None``).
- ``bc_pretrain``: cross-entropy of the teacher action under the masked policy
  distribution over the full action space, with train/val split and early stopping.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sts_rl.agent.card_teacher import (
    CARD_PICK_END,
    CARD_PICK_START,
    CARD_SKIP_IDX,
)
from sts_rl.agent.train import (
    CHECKPOINT_HIDDEN_DIM_KEY,
    CHECKPOINT_INTERFACE_VERSION_KEY,
    CHECKPOINT_MODEL_KEY,
)
from sts_rl.interface import (
    ACTION_BLOCK_BY_NAME,
    ACTION_DIM,
    INTERFACE_VERSION,
    MAX_REWARD_POTIONS,
    OBS_FIELDS,
    REWARD_POTION_OFFSET,
)

logger = logging.getLogger(__name__)


class BCTeacher(Protocol):
    """A teacher the collector can consult on every step.

    ``select_action`` returns a full-space action index to teach on a recognized
    screen, or ``None`` to defer to the base policy (see
    :class:`~sts_rl.agent.strategic_teacher.StrategicTeacher`).
    """

    def select_action(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int | None: ...


# --- Decision-type classification (for provenance / coverage stats) ---------
# Every teacher action is labeled by the action block / offset it falls in, so a
# collected dataset carries a histogram of which decision types it spans. Derived
# from the interface layout, never hardcoded indices.
_RS = ACTION_BLOCK_BY_NAME["REWARD_SELECT"]
_REST = ACTION_BLOCK_BY_NAME["REST_SELECT"]
_MAP = ACTION_BLOCK_BY_NAME["MAP_SELECT"]
_USE_POTION_UNTARGETED = ACTION_BLOCK_BY_NAME["USE_POTION_UNTARGETED"]
_REWARD_POTION_START = _RS.start + REWARD_POTION_OFFSET

DT_CARD_PICK = "card_pick"
DT_CARD_SKIP = "card_skip"
DT_CAMPFIRE = "campfire"
DT_MAP = "map"
DT_REWARD_POTION = "reward_potion"
DT_COMBAT_POTION = "combat_potion"
DT_OTHER = "other"


def _decision_type_of(action: int) -> str:
    """Label a full-space teacher action by the decision it represents."""
    if CARD_PICK_START <= action < CARD_PICK_END:
        return DT_CARD_PICK
    if action == CARD_SKIP_IDX:
        return DT_CARD_SKIP
    if _REST.contains(action):
        return DT_CAMPFIRE
    if _MAP.contains(action):
        return DT_MAP
    if _REWARD_POTION_START <= action < _REWARD_POTION_START + MAX_REWARD_POTIONS:
        return DT_REWARD_POTION
    if _USE_POTION_UNTARGETED.contains(action):
        return DT_COMBAT_POTION
    return DT_OTHER


# --- Dataset ----------------------------------------------------------------


@dataclass
class BCManifest:
    """Provenance metadata for a BC dataset."""

    n_samples: int
    interface_version: str
    seed: int | None
    teacher_config: dict[str, Any]
    tier_histogram: dict[str, int]  # tier_name -> count of teacher CARD picks at that tier
    decision_type_histogram: dict[str, int]  # decision_type -> count of teacher decisions
    skip_rate: float  # fraction of CARD-reward decisions where the teacher chose skip


class BCDataset:
    """Container for behavior-cloning samples.

    Each sample is (obs_dict, teacher_action, mask), where ``teacher_action`` is a
    full-space action index and ``mask`` is the full ``(ACTION_DIM,)`` legality
    mask for that step. Stored as stacked numpy arrays for persistence, and yields
    torch tensors for training.
    """

    def __init__(
        self,
        obs_arrays: dict[str, np.ndarray],
        actions: np.ndarray,
        masks: np.ndarray,
        manifest: BCManifest,
    ) -> None:
        self.obs_arrays = obs_arrays  # {field_name: (N, *field_shape)}
        self.actions = actions  # (N,) int64, full-space action indices [0..ACTION_DIM)
        self.masks = masks  # (N, ACTION_DIM) bool
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
        # teacher_config / tier_histogram / decision_type_histogram are variable-size
        # dicts; serialize to JSON and store as auto-sized unicode scalar arrays so
        # they round-trip under allow_pickle=False (a fixed "U32" would truncate).
        arrays["manifest_teacher_config"] = np.array(json.dumps(self.manifest.teacher_config))
        arrays["manifest_tier_histogram"] = np.array(json.dumps(self.manifest.tier_histogram))
        arrays["manifest_decision_type_histogram"] = np.array(
            json.dumps(self.manifest.decision_type_histogram)
        )
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
        # decision_type_histogram was added after the initial (card-only) format;
        # tolerate its absence so older datasets still load.
        if "manifest_decision_type_histogram" in data.files:
            decision_type_histogram = json.loads(str(data["manifest_decision_type_histogram"]))
        else:
            decision_type_histogram = {}
        manifest = BCManifest(
            n_samples=n_samples,
            interface_version=iface_version,
            seed=seed,
            teacher_config=teacher_config,
            tier_histogram=tier_histogram,
            decision_type_histogram=decision_type_histogram,
            skip_rate=skip_rate,
        )
        return cls(obs_arrays, actions, masks, manifest)


# --- Collection -------------------------------------------------------------


def _teacher_config(teacher: Any) -> dict[str, Any]:
    """Best-effort provenance of the teacher's tuning knobs (card + strategic)."""
    card_teacher = getattr(teacher, "card_teacher", teacher)
    config: dict[str, Any] = {}
    for attr in ("min_keep_tier", "default_tier"):
        if hasattr(card_teacher, attr):
            config[attr] = getattr(card_teacher, attr)
    for attr in ("rest_hp_fraction", "map_low_hp_fraction", "potion_emergency_hp_fraction"):
        if hasattr(teacher, attr):
            config[attr] = getattr(teacher, attr)
    return config


def collect_bc_dataset(
    env: Any,
    policy: Any,
    teacher: BCTeacher,
    *,
    n_episodes: int,
    seed: int | None = None,
    deterministic: bool = True,
) -> BCDataset:
    """Run episodes recording teacher actions at every decision the teacher owns.

    On each step the teacher is consulted first. If it returns an action, that
    action is executed AND logged as a sample; if it defers (``None``), the base
    ``policy`` decides and no sample is logged. This generalizes the original
    card-only flow to any screen the teacher covers (card pick, campfire, map,
    potion), classified into the manifest's decision-type histogram.

    Args:
        env: A Gymnasium env implementing the STS RL interface (reset/step).
        policy: An ActorCritic with .act(obs_batched, mask_batched, deterministic).
        teacher: A teacher with ``select_action(obs, mask) -> int | None``.
        n_episodes: Number of full episodes to collect.
        seed: Optional base seed for env resets (incremented per episode).
        deterministic: Whether the base policy acts greedily.

    Returns:
        A BCDataset with all collected teacher-decision samples.
    """
    obs_buffers: dict[str, list[np.ndarray]] = {f.name: [] for f in OBS_FIELDS}
    action_buffer: list[int] = []
    mask_buffer: list[np.ndarray] = []
    decision_counts: Counter[str] = Counter()
    tier_counts: Counter[int] = Counter()
    card_decisions = 0
    skip_count = 0
    total_steps = 0

    # A composite teacher exposes its card sub-teacher; a bare card teacher is its
    # own card teacher. Used only to attribute card-pick tier stats.
    card_teacher: Any = getattr(teacher, "card_teacher", teacher)

    device = next(policy.parameters()).device

    t0 = time.time()
    for ep_idx in range(n_episodes):
        ep_seed = (seed + ep_idx) if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        mask = info["action_mask"]
        terminated = truncated = False

        while not (terminated or truncated):
            total_steps += 1
            teacher_action = teacher.select_action(obs, mask)

            if teacher_action is not None:
                # Teacher owns this decision: log the full-space sample and execute.
                teacher_action = int(teacher_action)
                for f in OBS_FIELDS:
                    obs_buffers[f.name].append(obs[f.name].copy())
                action_buffer.append(teacher_action)
                mask_buffer.append(np.asarray(mask, dtype=np.bool_).copy())

                label = _decision_type_of(teacher_action)
                decision_counts[label] += 1
                if label in (DT_CARD_PICK, DT_CARD_SKIP):
                    card_decisions += 1
                    if label == DT_CARD_SKIP:
                        skip_count += 1
                    elif card_teacher is not None:
                        slot = teacher_action - CARD_PICK_START
                        card_id = int(obs["reward_card_ids"][slot])
                        card_name = card_teacher._card_id_to_name(card_id)
                        tier = card_teacher._resolve_tier(card_name)
                        if tier is not None:
                            tier_counts[tier] += 1

                obs, _reward, terminated, truncated, info = env.step(teacher_action)
            else:
                # Teacher deferred: the base policy decides (no sample logged).
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
    skip_rate = skip_count / max(1, card_decisions)
    tier_hist = {str(k): v for k, v in sorted(tier_counts.items())}
    decision_hist = {k: v for k, v in sorted(decision_counts.items())}

    logger.info(
        "BC collection done: %d episodes, %d steps, %d teacher-decision samples, "
        "skip_rate=%.3f, decisions=%s, tier_hist=%s",
        n_episodes,
        total_steps,
        n_samples,
        skip_rate,
        decision_hist,
        tier_hist,
    )

    # Stack arrays
    if n_samples == 0:
        raise ValueError(
            "collect_bc_dataset collected no teacher decisions (n_samples == 0); "
            "increase n_episodes or broaden the teacher's coverage"
        )
    obs_arrays = {name: np.stack(arrs, axis=0) for name, arrs in obs_buffers.items()}
    actions = np.array(action_buffer, dtype=np.int64)
    masks = np.stack(mask_buffer, axis=0)

    manifest = BCManifest(
        n_samples=n_samples,
        interface_version=INTERFACE_VERSION,
        seed=seed,
        teacher_config=_teacher_config(teacher),
        tier_histogram=tier_hist,
        decision_type_histogram=decision_hist,
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
    """Run BC pretraining over the full masked action space.

    Loss: negative log-prob of the teacher action under the masked policy
    distribution over all ``ACTION_DIM`` actions. Because illegal logits are
    driven to the finite floor, only the legal actions of each sample's screen
    participate in the softmax and receive gradient - so a card-pick sample
    updates only the card / skip logits (and the encoder through them), a campfire
    sample only the rest / smith logits, and so on. Each screen is thus supervised
    surgically without a per-screen sub-slice.

    Default trains end-to-end from a combat warm-start. The freeze_encoder
    option is exposed but NOT recommended: a frozen encoder cannot adapt its
    entity-token representations (the card / relic embeddings and the attention)
    to the strategic objectives, so the heads would learn over fixed features.
    End-to-end training with a modest LR is the intended recipe.

    Args:
        model: ActorCritic with encoder, policy head.
        dataset: BCDataset from collect_bc_dataset.
        epochs: Maximum training epochs.
        lr: Learning rate (modest default to preserve combat features).
        batch_size: Mini-batch size.
        val_frac: Fraction of data held for validation. At least one sample is
            always held out (n_val = max(1, int(n * val_frac))), so val_frac=0.0
            still reserves a single validation sample rather than training on all data.
        seed: Random seed for reproducibility.
        device: Torch device.
        freeze_encoder: If True, freeze encoder parameters (not recommended).
        patience: Early-stop on val accuracy plateau.

    Returns:
        BCStats with loss/accuracy histories.
    """
    # Full-space masks are required: each sample teaches over all ACTION_DIM logits.
    # A card-only (sub-slice-width) dataset from an older format would silently
    # mis-align, so reject it with a clear message rather than deep in the head.
    if dataset.masks.ndim != 2 or dataset.masks.shape[1] != ACTION_DIM:
        raise ValueError(
            f"BC dataset masks must be full-space (N, {ACTION_DIM}); got "
            f"{dataset.masks.shape}. Re-collect with the current collect_bc_dataset."
        )

    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.train()

    if freeze_encoder:
        logger.warning(
            "freeze_encoder=True: a frozen encoder cannot adapt its entity-token "
            "representations to the strategic objectives, so only the policy head "
            "learns. This is NOT recommended for BC pretraining."
        )
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Train/val split. Always hold out at least one validation sample, so
    # val_frac=0.0 reserves one rather than training on all data (the guard below
    # requires n >= 2).
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

            # Forward: the encoder returns (per_token, pooled_cls, mask); the pointer
            # head scores + masks over the full action space (same masking as rollout).
            per_token, _features, key_padding_mask = model.encoder(obs_batch)
            logits = model.policy(per_token, key_padding_mask, mask_batch)  # (B, ACTION_DIM)

            # Cross-entropy over the full masked action space. On a reward screen
            # this spans every simultaneously-legal reward action (gold / potion /
            # relic), not just card + skip - intended and deployment-consistent: the
            # same masked head chooses among all legal reward actions at inference.
            loss = F.cross_entropy(logits, action_batch)

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

                per_token, _features, key_padding_mask = model.encoder(obs_batch)
                logits = model.policy(per_token, key_padding_mask, mask_batch)

                loss = F.cross_entropy(logits, action_batch)
                preds = logits.argmax(dim=-1)

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
