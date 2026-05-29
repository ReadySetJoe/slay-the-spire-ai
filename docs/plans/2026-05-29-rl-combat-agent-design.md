# RL Combat Agent with PPO - Design

## Goal

Replace rule-based combat decision-making with a PPO-based RL agent that learns which cards to play through experience. Non-combat decisions stay rule-based.

## Architecture

### Gym Environment (SlayTheSpireCombatEnv)
Wraps combat interactions as a Gym environment. Each "episode" is one combat encounter.

- **Observation space:** Fixed-size numerical vector encoding combat state
- **Action space:** 61 discrete actions with masking for invalid actions
- **Reward:** HP remaining / max HP after fight (+0 to +1), -1 for dying, +5 bonus for full run win
- **Step flow:** Agent picks action -> send command -> receive new state -> return obs, reward, done

### Hybrid Agent
Uses PPO for combat decisions, falls back to SimpleAgent logic for everything else (card rewards, map, rest, events).

### Training
MaskablePPO from sb3-contrib. Trains live as the bot plays runs. Model saved periodically.

## State Encoding (observation vector ~54 floats)

- Player: HP, max HP, block, energy (4)
- Hand: 10 slots x 3 features (cost, type, is_playable) (30)
- Monsters: 5 slots x 4 features (HP, max HP, block, intent) (20)
- Padded with zeros for empty slots

## Action Space (61 discrete actions)

- 0-9: play card in slot 0-9 (no target)
- 10-59: play card in slot 0-9 on target 0-4
- 60: end turn
- Invalid actions masked each step

## Reward Signal

- Per-fight: HP remaining / max HP after winning (+0 to +1)
- Death: -1
- Full run win bonus: +5

## Files

- `src/combat_env.py` - Gym environment wrapping combat
- `src/state_encoder.py` - encodes GameState into observation vector
- `src/rl_agent.py` - hybrid agent using PPO for combat, rules for rest
- `main.py` - updated to use RL agent + training
- New deps: stable-baselines3, sb3-contrib, gymnasium
