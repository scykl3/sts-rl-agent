# Slay the Spire RL

A reinforcement learning agent for playing **Slay the Spire** as Ironclad across Acts 1–3.

The project uses a custom state encoder, legal-action masking, reward shaping, policy/value networks, and PPO. Training runs on the [`daniel-ziegler/sts_lightspeed`](https://github.com/daniel-ziegler/sts_lightspeed) C++ engine, which provides a fast headless Slay the Spire simulator with Python bindings.

We use `sts_lightspeed` only for the game simulation. The engine's included RL agent, `silverbot`, is not used.

## Why RL

Slay the Spire has a large number of decisions whose effects can show up much later in a run. A card picked early in Act 1 can change how a deck handles elites, bosses, shops, and later acts. The same applies to routing, upgrades, purchases, and combat decisions.

Running the game through a headless C++ simulator makes it possible to train on far more runs than would be practical through the normal game client.

## Architecture

The project is split into a few main pieces:

* **Environment interface** — passes actions and game state between the Python training code and C++ simulator.
* **Observation encoder** — turns the current game state into fixed-size numerical input for the network.
* **Action masking** — removes illegal actions before sampling from the policy.
* **Policy/value network** — predicts action probabilities and the expected value of the current state.
* **PPO trainer** — collects rollouts and updates the policy using Proximal Policy Optimization.
* **Reward system** — assigns rewards for run outcomes and intermediate progress.
* **Evaluation pipeline** — runs the policy without training-time exploration and records its performance.

The action space includes both combat actions and run-level choices, including card rewards, path selection, shops, and rest sites.

## Results

We first tested the training setup on sampled Act 1 elite fights against **Gremlin Nob, Lagavulin, and Three Sentries**.

On these encounters, the greedy policy went from roughly random legal-action performance to about a **99% win rate** after training.

The current full-run setup trains one policy across Acts 1–3. We also use it to measure how performance changes as ascension increases.

## Training

Training uses **Proximal Policy Optimization (PPO)** with action masking.

For each rollout:

1. The simulator returns the current game state.
2. The encoder converts that state into the model input.
3. Illegal actions are removed from the policy distribution.
4. The policy samples an action.
5. The simulator executes the action and returns the next state.
6. The transition, reward, and value estimate are stored.
7. Generalized Advantage Estimation (GAE) computes the advantages for the rollout.
8. PPO updates the policy and value networks using the collected data.

During evaluation, the agent takes the highest-probability legal action instead of sampling. This removes training-time exploration from the reported results.

## License and Attribution

[`sts_lightspeed`](https://github.com/daniel-ziegler/sts_lightspeed) is MIT licensed and used as an external dependency.

Slay the Spire is a trademark of Mega Crit. This project is non-commercial and educational.
