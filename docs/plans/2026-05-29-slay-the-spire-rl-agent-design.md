# Slay the Spire RL Agent - Design

## Goal

Build a reinforcement learning agent that empirically proves every Slay the Spire seed is beatable, starting with Ironclad at Ascension 0 and scaling up to all four characters at Ascension 20.

## Architecture

Three layers:

1. **Game Interface** - Communicates with Slay the Spire via CommunicationMod. Receives game state as JSON, sends commands back. Handles the game loop (receive state, decide action, send action, repeat).

2. **Agent** - Decides what action to take given a game state. Initially a simple rule-based agent to validate the infrastructure. Later replaced by an RL agent.

3. **RL Training** - Train an RL agent using a reward signal (win = positive, lose = negative, with shaping rewards like HP remaining, floor reached). Uses Stable Baselines3 with PyTorch.

## Game Loop

```
CommunicationMod -> JSON game state -> Agent -> action command -> CommunicationMod -> next state
```

The agent handles all decision points: playing cards, choosing card rewards, navigating the map, shop purchases, rest site choices, boss relic swaps, etc.

## State Representation

CommunicationMod JSON includes: current HP, hand, draw pile, discard pile, relics, potions, enemy info (HP, intent), map, gold, etc. For RL, this gets encoded into a numerical feature vector.

## Phased Approach

### Phase 1: Project Setup + CommunicationMod Integration
Get a working game loop that can receive state and send commands.

### Phase 2: Rule-Based Agent
Simple heuristics that can play through a full run (play highest damage card, etc.).

### Phase 3: RL Agent
Define state/action spaces, reward function, and train with Stable Baselines3.

### Phase 4: Scale Up
Other characters (Silent, Defect, Watcher), higher ascensions, systematic seed testing.

### Future: Monitoring Dashboard
A way to monitor training progress and display results (win rates per seed, per ascension, etc.). Out of scope for initial implementation.

## Tech Stack

- Python 3.11+
- CommunicationMod (Slay the Spire mod)
- Stable Baselines3 + PyTorch (for RL)
- Standard lib for JSON/socket communication with the mod

## Decisions

- **Character**: Start with Ironclad, expand to all four.
- **Ascension**: Start at 0, work up to 20.
- **Success criteria**: Empirical - run the bot on thousands of seeds and demonstrate a very high (ideally 100%) win rate.
- **Game interaction**: CommunicationMod (no computer vision needed).
