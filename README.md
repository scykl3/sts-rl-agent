# Slay the Spire RL

A reinforcement-learning agent that learns to play **Slay the Spire** (Ironclad,
Acts 1-3), built from scratch as a portfolio project.

The learning stack - observation encoding, action masking, reward design, the
neural network, a self-implemented PPO, and the training loop - is our own work.
We use the [`daniel-ziegler/sts_lightspeed`](https://github.com/daniel-ziegler/sts_lightspeed)
C++ engine **as the environment only** (a fast, headless, RNG-accurate Ironclad
simulator with Python bindings). We do **not** use its bundled RL code
(`silverbot`). That boundary is the core of the "from scratch" claim: the
environment is a third-party dependency, like a game engine; the agent is ours.

## Why RL

Playing a full Ironclad run is sequential decision-making under stochasticity
with delayed, sparse reward (win or lose the run) - the canonical RL setting.
Heuristics can't self-improve, imitation caps the agent at its teacher, and pure
search trains no model. RL is the only approach that trains our own model end to
end, and a fast headless simulator supplies the one hard prerequisite: millions
of cheap episodes.

## Goal

Train an Ironclad agent that plays competently across ascension levels, with a
reproducible pipeline and a demonstrable result (training curves plus a sample
playthrough).

- **Primary target:** 50-70% win rate at Ascension 0 through Act 3.
- **Stretch:** climb the ascension ladder (A1-A10+), reported as a
  win-rate-vs-ascension curve.

Scope is Ironclad only, Acts 1-3 (no Act 4 / Corrupt Heart, which the engine does
not implement).

## Status

Implementation is just beginning. The next step is **Phase 0**: repo scaffold,
engine build, enum validation, and interface sign-off.

## Roadmap

| Phase | Milestone |
|---|---|
| 0 | Repo scaffold, engine build, enum validation, interface sign-off |
| 1 | Observation encoder + action masking against the live engine |
| 2 | Self-implemented PPO on single Act 1 combats (>90% on sampled combats) |
| 3 | Full Act 1 clear (>80% clear rate) |
| 4 | Full-run training, Acts 1-3, toward the 50-70% primary band |
| 5 | RL + MCTS search extension (deferred; needs a trained policy/value first) |

## Design decisions

The approaches we considered, why we rejected the alternatives, and the empirical
evidence behind each choice are recorded as
[Architecture Decision Records](https://adr.github.io/) (ADRs). The decision log
will be published under `decisions/` as the project matures - each record's
evidence section filled in from real results (benchmarks, learning curves) as the
corresponding phase lands. It is not in the repository yet because implementation
is just starting.

## License and attribution

The `sts_lightspeed` engine is MIT licensed and used as an external dependency.
All RL code in this repository is our own. Slay the Spire is a trademark of
Mega Crit; this is a non-commercial, educational project.
