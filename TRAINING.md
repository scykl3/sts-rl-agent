# Training

The agent learns to play Ironclad (Acts 1-3) via a from-scratch PPO implementation running against the live `sts_lightspeed` C++ engine. There are two entry points: `scripts/train_combat.py` trains on single Act 1 combats (fast iteration, useful for bootstrapping a policy), and `scripts/train_run.py` trains on full Acts 1-3 runs (the end-to-end objective). A typical workflow trains on isolated combats first, then warm-starts the full-run script from that checkpoint.

## Prerequisites

Training runs against the live engine, so the `slaythespire` C++ Python binding must be built and importable. Build it with:

```bash
scripts/build_engine.sh
```

Then run training scripts with:

```bash
PYTHONPATH=src:engine/sts_lightspeed/build python scripts/train_combat.py ...
```

This differs from the unit tests, most of which run without the engine.

## `scripts/train_combat.py`

Trains the Ironclad agent on single Act 1 combats. Each episode is one fight sampled from the specified encounter pool (or the full Act 1 pool by default). Useful for rapidly learning combat mechanics before tackling the full run.

### Example

```bash
PYTHONPATH=src:engine/sts_lightspeed/build python scripts/train_combat.py \
    --encounters GREMLIN_NOB,LAGAVULIN,THREE_SENTRIES \
    --num-iterations 200 --eval-every 25 --eval-episodes 256 \
    --seed 0 --checkpoint-dir runs/combat/checkpoints
```

### Flags

#### Rollout / model / environment

| Flag | Default | Description |
|------|---------|-------------|
| `--n-steps` | `2048` | Transitions collected per iteration |
| `--num-iterations` | `200` | Number of collect-then-update iterations |
| `--hidden-dim` | `512` | Encoder trunk width |
| `--ascension` | `0` | Ascension level |
| `--max-episode-steps` | `500` | Per-episode step cap (episode truncates when hit) |
| `--seed` | `0` | Global + env reset seed |
| `--encounters` | None | Comma-separated `MonsterEncounter` names to sample from (e.g. `GREMLIN_NOB,LAGAVULIN,THREE_SENTRIES`); unset samples the canonical full Act 1 pool |

#### PPO / optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--learning-rate` | `0.0003` | Adam step size |
| `--anneal-lr` | `False` | Linearly decay the learning rate across the run |
| `--gamma` | `0.99` | GAE discount factor |
| `--gae-lambda` | `0.95` | GAE trace-decay lambda |
| `--clip-coef` | `0.2` | PPO surrogate/value clip coefficient |
| `--vf-coef` | `0.5` | Value-loss weight in the PPO objective |
| `--ent-coef` | `0.01` | Entropy-bonus weight in the PPO objective |
| `--n-epochs` | `4` | PPO epochs per collected rollout |
| `--minibatch-size` | `64` | PPO minibatch size |
| `--max-grad-norm` | `0.5` | Global grad-norm clip |
| `--target-kl` | None | Approximate-KL early-stop threshold (unset disables the early stop) |
| `--adv-norm-decay` | `0.0005` | Per-item EWMA decay for advantage normalization: a persistent running mean/std across iterations (more stable than per-batch, especially single-env). Set `1.0` to revert to per-batch normalization |

#### Reward shaping

Potential weights for the per-step shaping terms; defaults match the built-in `RewardConfig`. Shaping is potential-based (`gamma * Phi(s') - Phi(s)`), so over an episode it telescopes to `-Phi(s_0)` and is policy-invariant, with no anneal; each coefficient scales its term's potential. `--floor-progress-coef` and `--boss-kill-coef` are run-mode signals and stay `0.0` in single-combat (exposed here only for parity with `train_run.py`).

| Flag | Default | Description |
|------|---------|-------------|
| `--enemy-hp-removed-coef` | `RewardConfig.enemy_hp_removed` | Shaping weight for the drop in enemy HP fraction (combat) |
| `--damage-taken-coef` | `RewardConfig.damage_taken` | Shaping weight for the drop in player HP fraction (negative penalizes damage) |
| `--floor-progress-coef` | `RewardConfig.floor_progress` | Shaping weight per new floor descended (run mode; `0.0` in single-combat) |
| `--boss-kill-coef` | `RewardConfig.boss_kill` | Shaping weight per act boss defeated (run mode; `0.0` in single-combat) |

#### Evaluation + checkpointing

| Flag | Default | Description |
|------|---------|-------------|
| `--eval-every` | None | Run a greedy holdout eval every N iterations (unset disables periodic eval) |
| `--eval-episodes` | `256` | Number of greedy holdout episodes |
| `--eval-base-seed` | `1000000` | First seed of the reproducible eval band (distinct from `--seed`) |
| `--checkpoint-dir` | None | Directory for `best.pt`/`last.pt` checkpoints (unset disables checkpointing) |

## `scripts/train_run.py`

Trains the Ironclad agent on full Acts 1-3 runs. Each episode is a complete game from the Act 1 whale bonus through the Act 3 boss. Supports warm-starting from a combat-trained checkpoint and value-head warmup to stabilize the transition to the harder objective.

### Example

Warm-started from a combat checkpoint, with value-head warmup and plateau early-stop:

```bash
PYTHONPATH=src:engine/sts_lightspeed/build python scripts/train_run.py \
    --warm-start runs/combat/checkpoints/best.pt \
    --num-iterations 1000 --n-steps 2048 \
    --learning-rate 1e-4 --anneal-lr --target-kl 0.02 \
    --value-warmup-iters 50 --early-stop-patience 6 --early-stop-min-delta 0.03 \
    --seed 0 --eval-every 50 --eval-episodes 100 --eval-base-seed 1000000 \
    --checkpoint-dir runs/full-run/checkpoints
```

### Flags

#### Rollout / model / environment

| Flag | Default | Description |
|------|---------|-------------|
| `--n-steps` | `2048` | Transitions collected per iteration |
| `--num-iterations` | `1000` | Number of collect-then-update iterations |
| `--hidden-dim` | `512` | Encoder trunk width |
| `--ascension` | `0` | Ascension level |
| `--max-episode-steps` | `3000` | Per-episode step cap (episode truncates when hit) |
| `--seed` | `0` | Global + env reset seed |
| `--num-envs` | `1` | Number of parallel run environments; `>1` runs them in worker processes and collects vectorized rollouts for higher throughput |
| `--stop-after-act` | `None` | End each episode as soon as this act's boss is cleared (the run advances past the act), making "clear act N" a frequent terminal signal instead of a rare late-run event; unset runs the full three-act episode. Applies to both training and the periodic eval |

#### PPO / optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--learning-rate` | `0.0003` | Adam step size |
| `--anneal-lr` | `False` | Linearly decay the learning rate across the run |
| `--gamma` | `1.0` | GAE discount factor; the full-run default is undiscounted so the terminal win/loss signal reaches early-run decisions (combat training keeps `0.99`) |
| `--gae-lambda` | `0.97` | GAE trace-decay lambda |
| `--clip-coef` | `0.2` | PPO surrogate/value clip coefficient |
| `--vf-coef` | `0.5` | Value-loss weight in the PPO objective |
| `--ent-coef` | `0.01` | Entropy-bonus weight in the PPO objective |
| `--n-epochs` | `4` | PPO epochs per collected rollout |
| `--minibatch-size` | `64` | PPO minibatch size |
| `--max-grad-norm` | `0.5` | Global grad-norm clip |
| `--target-kl` | None | Approximate-KL early-stop threshold (unset disables the early stop) |
| `--adv-norm-decay` | `0.0005` | Per-item EWMA decay for advantage normalization: a persistent running mean/std across iterations (more stable than per-batch, especially single-env). Set `1.0` to revert to per-batch normalization |

#### Reward shaping

Potential weights for the per-step shaping terms; defaults match the built-in `RewardConfig`. Shaping is potential-based (`gamma * Phi(s') - Phi(s)`), so over an episode it telescopes to `-Phi(s_0)` and is policy-invariant, with no anneal; each coefficient scales its term's potential. `--boss-kill-coef` weights the act potential (`w * act`), so raising it emphasizes act progress.

| Flag | Default | Description |
|------|---------|-------------|
| `--enemy-hp-removed-coef` | `RewardConfig.enemy_hp_removed` | Shaping weight for the drop in enemy HP fraction (combat) |
| `--damage-taken-coef` | `RewardConfig.damage_taken` | Shaping weight for the drop in player HP fraction (negative penalizes damage) |
| `--floor-progress-coef` | `RewardConfig.floor_progress` | Shaping weight per new floor descended (overworld) |
| `--boss-kill-coef` | `RewardConfig.boss_kill` | Shaping weight per act boss defeated (overworld) |

#### Evaluation + checkpointing

| Flag | Default | Description |
|------|---------|-------------|
| `--eval-every` | None | Run a greedy holdout eval every N iterations (unset disables periodic eval) |
| `--eval-episodes` | `100` | Number of greedy holdout episodes |
| `--eval-base-seed` | `1000000` | First seed of the reproducible eval band (distinct from `--seed`) |
| `--checkpoint-dir` | None | Directory for `best.pt`/`last.pt` checkpoints (unset disables checkpointing) |

#### Warm-start + stabilization

| Flag | Default | Description |
|------|---------|-------------|
| `--warm-start` | None | Path to a checkpoint (`best.pt`/`last.pt`) to warm-start the network from; loaded and migrated to the current action layout via `load_checkpoint`, and its trunk width overrides `--hidden-dim` (unset trains from scratch) |
| `--value-warmup-iters` | `0` | Freeze the trunk + policy head for the first N iterations so only the value head trains (0 disables; calibrates the critic before it can corrupt the shared trunk) |
| `--early-stop-patience` | None | Stop once this many consecutive post-warmup evals fail to improve the ranked metric (unset disables early stop) |
| `--early-stop-min-delta` | `0.0` | Minimum ranked-metric gain that counts as an improvement for early stop; the eval metric is noisy (a ~100-episode clear rate has std ~0.03-0.04), so set this above roughly one eval-std to avoid noise-driven premature or deferred stops |

### Output

When `--checkpoint-dir` is set, the script writes two files:

- `best.pt` - the checkpoint with the highest ranked eval metric (for `train_run.py`, the Act 1 clear rate). Only written when `--eval-every` enables periodic evaluation.
- `last.pt` - the most recent checkpoint, written every iteration.

Without `--eval-every` there is no periodic eval and no `best.pt` ranking.

## Supervised pretraining (optional warm-start)

Before PPO, an optional supervised stage can bootstrap the network so the full run starts from a sensible prior rather than uniform-random exploration. Its output is a checkpoint that `train_run.py --warm-start` loads directly (same 3-key format). The stage has two parts, both starting from a warm-start policy (typically a combat-trained checkpoint from `train_combat.py`):

1. Behavior cloning of the run's strategic decisions - the card-reward pick together with campfire (rest vs. smith), map-path selection, and potion use - in two steps:

```bash
# Collect a dataset: use a warm-start policy for the remaining decisions and record
# the heuristic teacher's action at every supported strategic decision (card reward,
# campfire, map path, potions).
PYTHONPATH=src:engine/sts_lightspeed/build python scripts/collect_bc.py \
    --warm-start runs/combat/checkpoints/best.pt --output bc_data.npz \
    --n-episodes 200 --seed 42

# Train the cloned decisions on that dataset; writes a warm-start-compatible checkpoint.
PYTHONPATH=src python scripts/train_bc.py \
    --dataset bc_data.npz --warm-start runs/combat/checkpoints/best.pt \
    --output bc_best.pt --epochs 30 --lr 1e-4 --seed 42
```

2. Outcome-regression pretraining: self-play full runs with the behavior-cloned policy, label each decision by its episode's Act 1 clear outcome, and fine-tune the value head with a win-probability `BCEWithLogits` loss (the policy head is frozen). The result feeds `train_run.py --warm-start`.

```bash
PYTHONPATH=src:engine/sts_lightspeed/build python scripts/train_sl.py \
    --warm-start bc_best.pt --out sl_best.pt \
    --num-episodes 500 --epochs 30 --seed 42 --ascension 0
# Reuse a previously collected self-play dataset instead of self-playing again:
#   python scripts/train_sl.py --dataset sp_data.npz --out sl_best.pt --epochs 30
```

Then run `scripts/train_run.py --warm-start sl_best.pt` as usual; the discount, potential-based reward, map-lookahead observation, EWMA advantage normalization, and the transformer encoder with pointer policy head are all on by default, so no extra flags are needed to enable them.

## Paired evaluation (A/B comparison)

To compare two checkpoints (for example a candidate against a baseline), `sts_rl.eval.paired_evaluate(policy_a, policy_b, env, seeds)` runs both policies over one shared holdout seed set and returns a `PairedEvalReport` with per-seed paired win-rate, Act 1 clear-rate, and return deltas, their paired standard errors, and an exact McNemar p-value on the discordant win/loss seeds. Because both policies see the same seeds, the difference has far lower variance than differencing two independent `evaluate()` runs, so a small real gap shows through the ~0.03-0.04 per-run win-rate noise. It is a library call (used, for instance, by `train_sl.py`'s outcome gate), not a training flag.

## Reproducibility

- `--seed` seeds both training (PyTorch, numpy) and environment resets, so the same seed replays the same trajectory of rollouts.
- `--eval-base-seed` fixes the holdout evaluation band independently of the training seed, giving a stable comparison across runs.
- Runs pin the engine commit via the git submodule under `engine/sts_lightspeed`, ensuring identical game logic across machines.
