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

Phases 0-2 are implemented. The engine build, interface, observation encoder, and
action masking are in place, and the from-scratch PPO stack trains against the
live engine on single Act 1 combats. On sampled Act 1 elites (Gremlin Nob,
Lagavulin, Three Sentries) the greedy win rate climbs from near the random-legal
floor (0-3.5%) to ~99%. Phase 3 (full Act 1 clear) is next.

## Roadmap

| Phase | Milestone | Status |
|---|---|---|
| 0 | Repo scaffold, engine build, enum validation, interface sign-off | Done |
| 1 | Observation encoder + action masking against the live engine | Done |
| 2 | Self-implemented PPO on single Act 1 combats (>90% on sampled combats) | Done |
| 3 | Full Act 1 clear (>80% clear rate) | Next |
| 4 | Full-run training, Acts 1-3, toward the 50-70% primary band | Planned |
| 5 | RL + MCTS search extension (deferred; needs a trained policy/value first) | Planned |

## Design decisions

The approaches we considered, why we rejected the alternatives, and the empirical
evidence behind each choice are recorded as
[Architecture Decision Records](https://adr.github.io/) (ADRs). The decision log
will be published under `decisions/` as the project matures - each record's
evidence section filled in from real results (benchmarks, learning curves) as the
corresponding phase lands. It is not in the repository yet; it will be published
once its evidence sections are backfilled from the landed phases.

## Engine dependency and fork

The engine originates from
[`daniel-ziegler/sts_lightspeed`](https://github.com/daniel-ziegler/sts_lightspeed).
The submodule under `engine/sts_lightspeed` does not point at upstream directly:
it tracks a light fork
([`maxy1991991/sts_lightspeed`](https://github.com/maxy1991991/sts_lightspeed),
branch `master`) that carries a build-portability patch and a few
Python-binding additions. No simulation logic is changed, so runs stay
RNG-accurate against upstream. Relative to the upstream `heart1` tag, the pinned
commit adds:

- **Portable macOS SDK path.** Upstream hardcodes an absolute SDK path
  (`/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk`) before the
  `project()` command, which fails to configure on any machine without that exact
  SDK. The fork pins that path only when it exists and otherwise lets CMake
  auto-detect the active SDK.
- **Read-only `BattleContext` bindings** exposing the potion belt and the
  card-select selected bits for observation encoding.
- **Optional encounter selection.** `create_battle_context` accepts an optional
  `MonsterEncounter`, so a caller can spawn a chosen combat by reusing the
  engine's existing `BattleContext::init(gc, encounter)` path. It defaults to the
  rolled encounter, so existing callers are unchanged.

The fork carries these changes so the exact vendored source is a single submodule
checkout, rather than pinning upstream and re-applying a patch at build time. If
they land upstream, the submodule can be repointed at
`daniel-ziegler/sts_lightspeed` with no other change.

## License and attribution

The `sts_lightspeed` engine is MIT licensed and used as an external dependency.
All RL code in this repository is our own. Slay the Spire is a trademark of
Mega Crit; this is a non-commercial, educational project.
