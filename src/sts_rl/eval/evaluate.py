"""Seeded holdout evaluation: greedy rollouts and aggregate metrics.

The learning loop needs a fixed, reproducible readout of policy quality that is
independent of the noisy training return. :func:`evaluate` runs the current
policy greedily (argmax over the masked logits, no sampling) over a fixed set of
holdout seeds - one episode per seed - and reports win rate, floor reached, HP
retained, episode length, return, and the Act-1 clear rate (the fraction of
episodes whose terminal ``act`` reached Act 2, i.e. the Act 1 boss was beaten).

Greedy action selection is deterministic given the network weights, so a fixed
(policy, seed set) yields the same metrics on every call without touching the
global torch/numpy RNG - the property that makes eval a stable comparison point
across training checkpoints.

The harness is env-agnostic: it drives any gymnasium-style env that honours the
shared interface (the terminal ``info`` keys ``won``/``floor``/``act``/``hp`` and
the ``episode`` summary). It is therefore engine-free testable against
:class:`~sts_rl.env.stub_env.StubEnv` and swaps in the real
:class:`~sts_rl.env.adapter.StsEnv` at integration by passing a different env.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import gymnasium as gym
import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.rollout_collector import observation_to_batched_tensors
from sts_rl.interface import InterfaceError, assert_valid_mask

# Info-dict keys read at the terminal step. All are documented in
# sts_rl.interface: "action_mask" is INFO_KEYS_ALWAYS; "won" and "episode" are
# INFO_KEYS_TERMINAL (present only when terminated or truncated); "floor"/"act"/"hp"
# are INFO_KEYS_ALWAYS and carry the terminal values on the last step.
MASK_INFO_KEY = "action_mask"
WON_INFO_KEY = "won"
FLOOR_INFO_KEY = "floor"
ACT_INFO_KEY = "act"
HP_INFO_KEY = "hp"
EPISODE_INFO_KEY = "episode"
EPISODE_RETURN_KEY = "r"
EPISODE_LENGTH_KEY = "l"

# gc.act is 1-based (Act 1 == 1) and a full run starts in Act 1. Beating the Act 1
# boss advances the run to Act 2 (floor 17), so a terminal act >= ACT2_INDEX means
# Act 1 was cleared, whether the run then continued, died later, or won the whole
# run outright. "won" (full-run victory) is a strict subset, so a run-mode clear
# rate must key on act, not won.
ACT2_INDEX = 2

# Absolute per-episode step ceiling. A well-formed env truncates itself at its
# own max_episode_steps, so this is only a backstop against a misconfigured env
# that never returns terminated/truncated - it caps a hang instead of looping
# forever. Set far above any real combat length.
_EPISODE_STEP_CEILING = 100_000

# Null-hypothesis probability for McNemar's exact test: under "the two policies are
# equally likely to win a seed the other loses", each discordant pair is a fair coin.
_MCNEMAR_NULL_PROB = 0.5


@dataclass(frozen=True)
class EpisodeResult:
    """Terminal outcome of a single greedy holdout episode."""

    seed: int
    won: bool
    floor: int
    act: int
    hp: int
    length: int
    ret: float


@dataclass(frozen=True)
class EvalReport:
    """Aggregate metrics over a holdout set (one episode per seed).

    ``avg_hp`` is the mean terminal player HP across all episodes; it includes
    losses (HP near 0), so it trends with, but is not conditioned on, wins.

    ``win_rate`` is the fraction of full-run victories (terminal ``won``); on a
    combat env it is the combat win rate, on a full-run env the whole-run win
    rate. ``act1_clear_rate`` is the fraction of episodes that cleared Act 1
    (terminal ``act >= ACT2_INDEX``), a run-mode progress metric: it is a
    superset of ``win_rate`` on a full run and defaults to ``0.0`` on combat/stub
    eval (those episodes never leave Act 1), so it is harmless there.
    """

    n_episodes: int
    win_rate: float
    avg_floor: float
    avg_hp: float
    avg_ep_len: float
    avg_return: float
    # Trailing, defaulted so existing combat callers/tests that build an
    # EvalReport without it are unchanged; evaluate() always fills it.
    act1_clear_rate: float = 0.0

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
        # act is INFO_KEYS_ALWAYS: every conformant env emits it, so read it
        # strictly (like won/floor/hp) - a missing key is a regression to catch,
        # not a default-to-Act-1 case.
        act=int(info[ACT_INFO_KEY]),
        hp=int(info[HP_INFO_KEY]),
        length=int(episode[EPISODE_LENGTH_KEY]),
        ret=float(episode[EPISODE_RETURN_KEY]),
    )


def _infer_device(policy: ActorCritic, device: torch.device | None) -> torch.device:
    """Return ``device`` if given, else the device of ``policy``'s first parameter.

    A param-less module has no device to infer, so this raises rather than
    guessing - the caller must then pass ``device`` explicitly.
    """
    if device is not None:
        return device
    try:
        return next(policy.parameters()).device
    except StopIteration as exc:
        raise ValueError(
            "cannot infer device from a policy with no parameters; pass device explicitly"
        ) from exc


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
    device = _infer_device(policy, device)

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
        # Act-1 clear rate: fraction whose terminal act reached Act 2 (boss beaten).
        act1_clear_rate=sum(r.act >= ACT2_INDEX for r in results) / n,
    )


def _paired_mean_and_se(diffs: Sequence[float]) -> tuple[float, float]:
    """Mean of the per-seed differences and the standard error of that mean.

    The standard error uses the sample variance (``ddof=1``): ``sqrt(var / n)``.
    It is the *paired* SE - built from the per-seed differences, so seeds where
    both policies agree contribute a zero difference and shrink it - which is why
    a paired comparison resolves a smaller true gap than two independent evals.
    ``n < 2`` has no spread to estimate, so the SE is reported as ``0.0``.
    ``diffs`` must be non-empty (the public entry point guards the empty-seed
    case, so this private helper is never called with an empty sequence).
    """
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return mean, 0.0
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, math.sqrt(var / n)


def _mcnemar_exact_p(b01: int, b10: int) -> float:
    """Two-sided exact McNemar p-value for discordant win/loss counts.

    ``b01`` and ``b10`` are the two discordant tallies (one policy won a seed the
    other lost, and vice versa). Under the null that each discordant seed is a
    fair coin, the smaller tally follows ``Binomial(b01 + b10, 0.5)``; the
    two-sided p-value doubles the lower tail, capped at ``1.0``. With no
    discordant pairs there is no evidence of a difference, so the p-value is
    ``1.0``. Exact (via :func:`math.comb`) so it needs no SciPy dependency and is
    valid for the small discordant counts a holdout set produces.
    """
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) * (_MCNEMAR_NULL_PROB**n)
    return min(1.0, 2.0 * lower_tail)


@dataclass(frozen=True)
class PairedEvalReport:
    """A/B comparison of two policies over one shared holdout seed set.

    Both policies are run over the SAME seeds, so the comparison is paired: a
    seed where they agree cancels out of every delta, leaving a much smaller
    standard error than differencing two independent :class:`EvalReport` win
    rates - the property that lets a real gap show through the ~0.03-0.04 per-run
    win-rate noise.

    Every ``*_delta`` is ``policy_a`` minus ``policy_b`` (positive => ``a`` is
    better). ``*_delta_se`` is the paired standard error of that delta.
    ``win_a_only`` / ``win_b_only`` are the discordant full-run-victory counts
    (one policy won the run, the other did not), and ``mcnemar_p`` is the
    two-sided exact McNemar p-value over those discordant seeds.

    ``win_rate_delta_se`` is a normal-approximation standard error while
    ``mcnemar_p`` is the exact test of the same win outcome; at the small
    discordant counts a holdout set produces the two uncertainty views can
    disagree, so prefer ``mcnemar_p`` for a significance call on the win rate.
    """

    n_episodes: int
    win_rate_a: float
    win_rate_b: float
    win_rate_delta: float
    win_rate_delta_se: float
    act1_clear_rate_a: float
    act1_clear_rate_b: float
    act1_clear_rate_delta: float
    act1_clear_rate_delta_se: float
    mean_return_a: float
    mean_return_b: float
    return_delta: float
    return_delta_se: float
    win_a_only: int
    win_b_only: int
    mcnemar_p: float

    def as_dict(self) -> dict[str, float]:
        """Flat ``{metric: value}`` view for experiment logging (all floats)."""
        return {key: float(value) for key, value in asdict(self).items()}


def paired_evaluate(
    policy_a: ActorCritic,
    policy_b: ActorCritic,
    env: gym.Env,
    seeds: Sequence[int],
    *,
    device: torch.device | None = None,
    deterministic: bool = True,
) -> PairedEvalReport:
    """Compare two policies over one shared holdout seed set (a paired A/B eval).

    Runs :func:`run_episode` for ``policy_a`` over every seed, then ``policy_b``
    over the same seeds, and pairs the outcomes seed-by-seed into a
    :class:`PairedEvalReport`. Running both on identical seeds is what makes the
    win-rate difference low-variance (see :class:`PairedEvalReport`).

    ``policy_a`` is evaluated fully before ``policy_b``, so a stateful env double
    can key on reset order; a real env simply resets per seed and is reused for
    both. ``device`` is inferred per policy when omitted. ``seeds`` must be
    non-empty.
    """
    if len(seeds) == 0:
        raise InterfaceError("seeds is empty; a paired comparison needs at least one seed")
    device_a = _infer_device(policy_a, device)
    device_b = _infer_device(policy_b, device)

    results_a = [
        run_episode(policy_a, env, seed, device=device_a, deterministic=deterministic)
        for seed in seeds
    ]
    results_b = [
        run_episode(policy_b, env, seed, device=device_b, deterministic=deterministic)
        for seed in seeds
    ]

    n = len(seeds)
    pairs = list(zip(results_a, results_b, strict=True))
    won_diffs = [float(a.won) - float(b.won) for a, b in pairs]
    clear_diffs = [float(a.act >= ACT2_INDEX) - float(b.act >= ACT2_INDEX) for a, b in pairs]
    return_diffs = [a.ret - b.ret for a, b in pairs]

    win_rate_delta, win_rate_delta_se = _paired_mean_and_se(won_diffs)
    clear_delta, clear_delta_se = _paired_mean_and_se(clear_diffs)
    return_delta, return_delta_se = _paired_mean_and_se(return_diffs)

    # Discordant full-run victories: seeds exactly one policy won. These drive
    # McNemar's test; concordant seeds (both won or both lost) carry no signal.
    win_a_only = sum(1 for a, b in pairs if a.won and not b.won)
    win_b_only = sum(1 for a, b in pairs if b.won and not a.won)

    return PairedEvalReport(
        n_episodes=n,
        win_rate_a=sum(r.won for r in results_a) / n,
        win_rate_b=sum(r.won for r in results_b) / n,
        win_rate_delta=win_rate_delta,
        win_rate_delta_se=win_rate_delta_se,
        act1_clear_rate_a=sum(r.act >= ACT2_INDEX for r in results_a) / n,
        act1_clear_rate_b=sum(r.act >= ACT2_INDEX for r in results_b) / n,
        act1_clear_rate_delta=clear_delta,
        act1_clear_rate_delta_se=clear_delta_se,
        mean_return_a=sum(r.ret for r in results_a) / n,
        mean_return_b=sum(r.ret for r in results_b) / n,
        return_delta=return_delta,
        return_delta_se=return_delta_se,
        win_a_only=win_a_only,
        win_b_only=win_b_only,
        mcnemar_p=_mcnemar_exact_p(win_a_only, win_b_only),
    )
