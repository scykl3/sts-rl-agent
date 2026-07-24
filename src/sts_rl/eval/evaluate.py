"""Seeded holdout evaluation: greedy rollouts and aggregate metrics.

The learning loop needs a fixed, reproducible readout of policy quality that is
independent of the noisy training return. :func:`evaluate` runs the current
policy greedily (argmax over the masked logits, no sampling) over a fixed set of
holdout seeds - one episode per seed - and reports win rate, floor reached, HP
retained, episode length, and return.

Greedy action selection is deterministic given the network weights, so a fixed
(policy, seed set) yields the same metrics on every call without touching the
global torch/numpy RNG - the property that makes eval a stable comparison point
across training checkpoints.

The harness is env-agnostic: it drives any gymnasium-style env that honours the
shared interface (the terminal ``info`` keys ``won``/``floor``/``hp`` and the
``episode`` summary). It is therefore engine-free testable against
:class:`~sts_rl.env.stub_env.StubEnv` and swaps in the real
:class:`~sts_rl.env.adapter.StsEnv` at integration by passing a different env.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import gymnasium as gym
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.rollout_collector import observation_to_batched_tensors
from sts_rl.interface import InterfaceError, assert_valid_mask

# Info-dict keys read at the terminal step. All are documented in
# sts_rl.interface: "action_mask" is INFO_KEYS_ALWAYS; "won" and "episode" are
# INFO_KEYS_TERMINAL (present only when terminated or truncated); "floor"/"hp"
# are INFO_KEYS_ALWAYS and carry the terminal values on the last step.
MASK_INFO_KEY = "action_mask"
WON_INFO_KEY = "won"
FLOOR_INFO_KEY = "floor"
HP_INFO_KEY = "hp"
EPISODE_INFO_KEY = "episode"
EPISODE_RETURN_KEY = "r"
EPISODE_LENGTH_KEY = "l"

# Absolute per-episode step ceiling. A well-formed env truncates itself at its
# own max_episode_steps, so this is only a backstop against a misconfigured env
# that never returns terminated/truncated - it caps a hang instead of looping
# forever. Set far above any real combat length.
_EPISODE_STEP_CEILING = 100_000


@dataclass(frozen=True)
class EpisodeResult:
    """Terminal outcome of a single greedy holdout episode."""

    seed: int
    won: bool
    floor: int
    hp: int
    length: int
    ret: float


@dataclass(frozen=True)
class EvalReport:
    """Aggregate metrics over a holdout set (one episode per seed).

    ``avg_hp`` is the mean terminal player HP across all episodes; it includes
    losses (HP near 0), so it trends with, but is not conditioned on, wins.
    """

    n_episodes: int
    win_rate: float
    avg_floor: float
    avg_hp: float
    avg_ep_len: float
    avg_return: float

    def as_dict(self) -> dict[str, float]:
        """Flat ``{metric: value}`` view for experiment logging."""
        return {key: float(value) for key, value in asdict(self).items()}


def make_holdout_seeds(base_seed: int, count: int) -> list[int]:
    """Return ``count`` distinct, reproducible holdout seeds from ``base_seed``.

    A plain contiguous range ``[base_seed, base_seed + count)`` so the holdout
    set is fully determined by ``(base_seed, count)`` and trivially reproducible
    across runs and machines. ``count`` must be positive.
    """
    if count <= 0:
        raise InterfaceError(f"count must be positive, got {count}")
    return [base_seed + offset for offset in range(count)]


def run_episode(
    policy: ActorCritic,
    env: gym.Env,
    seed: int,
    *,
    device: torch.device,
    deterministic: bool = True,
) -> EpisodeResult:
    """Run one episode to termination under the given policy and return its outcome.

    Resets ``env`` with ``seed`` and steps until ``terminated or truncated``,
    selecting actions with :meth:`ActorCritic.act`. Runs under ``no_grad`` in
    eval mode; the network's prior training/eval mode is saved and restored so a
    caller mid-training is unaffected (mirrors the rollout collector).

    ``deterministic=True`` (the eval default) takes the masked argmax, so the
    episode is reproducible for a fixed ``(policy, seed)``. ``deterministic=False``
    samples, which is useful for stochastic-eval diagnostics but then depends on
    the global torch RNG.
    """
    was_training = policy.training
    policy.eval()
    try:
        obs, info = env.reset(seed=seed)
        mask = info[MASK_INFO_KEY]
        terminated = truncated = False
        steps = 0
        with torch.no_grad():
            while not (terminated or truncated):
                if steps >= _EPISODE_STEP_CEILING:
                    raise InterfaceError(
                        f"episode exceeded {_EPISODE_STEP_CEILING} steps without "
                        f"terminating (env not truncating?); seed={seed}"
                    )
                assert_valid_mask(mask)
                obs_batched = observation_to_batched_tensors(obs, device)
                mask_batched = torch.as_tensor(mask, device=device).unsqueeze(0)
                action, _log_prob, _entropy, _value = policy.act(
                    obs_batched, mask_batched, deterministic=deterministic
                )
                # Single env: act returns (B=1,); a genuine scalar is required
                # before .item() steps the env with a python int.
                assert action.shape == (1,), f"expected (1,) action, got {tuple(action.shape)}"
                obs, _reward, terminated, truncated, info = env.step(int(action.item()))
                # Refresh the mask for the state just returned, so the next
                # greedy pick is over the CURRENT legal set, not the reset one.
                # On a terminal step the loop exits before this mask is read;
                # StsEnv's terminal all-False mask is never validated.
                mask = info[MASK_INFO_KEY]
                steps += 1
    finally:
        policy.train(was_training)

    episode = info[EPISODE_INFO_KEY]
    return EpisodeResult(
        seed=seed,
        won=bool(info[WON_INFO_KEY]),
        floor=int(info[FLOOR_INFO_KEY]),
        hp=int(info[HP_INFO_KEY]),
        length=int(episode[EPISODE_LENGTH_KEY]),
        ret=float(episode[EPISODE_RETURN_KEY]),
    )


def evaluate(
    policy: ActorCritic,
    env: gym.Env,
    seeds: Sequence[int],
    *,
    device: torch.device | None = None,
    deterministic: bool = True,
) -> EvalReport:
    """Evaluate ``policy`` over a holdout set: one episode per seed, then aggregate.

    Runs :func:`run_episode` for each seed in ``seeds`` (the authoritative
    holdout set) and averages the outcomes into an :class:`EvalReport`. ``env``
    is reset per seed, so a single env instance suffices; it is never
    auto-reset here. ``device`` is inferred from the network's parameters when
    omitted. ``seeds`` must be non-empty (an empty set has no metric to report).
    """
    if len(seeds) == 0:
        raise InterfaceError("seeds is empty; a holdout set needs at least one seed")
    if device is None:
        try:
            device = next(policy.parameters()).device
        except StopIteration as exc:  # a param-less module gives no device to infer
            raise ValueError(
                "cannot infer device from a policy with no parameters; pass device explicitly"
            ) from exc

    results = [
        run_episode(policy, env, seed, device=device, deterministic=deterministic) for seed in seeds
    ]

    n = len(results)
    return EvalReport(
        n_episodes=n,
        win_rate=sum(r.won for r in results) / n,
        avg_floor=sum(r.floor for r in results) / n,
        avg_hp=sum(r.hp for r in results) / n,
        avg_ep_len=sum(r.length for r in results) / n,
        avg_return=sum(r.ret for r in results) / n,
    )
