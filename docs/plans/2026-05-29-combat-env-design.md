# CombatEnv — Gymnasium Wrapper Design

## Goal

Replace the stub `RLAgent` with a proper `gymnasium.Env` so SB3's `MaskablePPO.learn()` drives the game loop and actually trains the model.

## Architecture

One episode = one combat encounter. `CombatEnv` owns the communicator and handles all game interaction. `main.py --rl` creates the env, loads or creates `MaskablePPO`, and calls `model.learn()`. SB3 drives everything from there.

```
main.py --rl
  └── CombatEnv(communicator, run_tracker)
        ├── reset() → advance to next combat, return obs
        └── step(action) → send command, receive state, return (obs, reward, done, ...)
              non-combat screens handled transparently by SimpleAgent
```

`src/rl_agent.py` and `tests/test_rl_agent.py` are deleted — the `DummyCombatEnv` workaround is no longer needed.

## Spaces

- `observation_space`: `Box(0, 1, shape=(54,), dtype=float32)` — from `StateEncoder.OBS_SIZE`
- `action_space`: `Discrete(61)` — from `ActionSpace.TOTAL_ACTIONS`

## step() / reset() Boundary

After combat ends, the game sends a non-combat screen (e.g. COMBAT_REWARD) that requires a response before the next combat can start. `step()` stores this in `_buffered_state` and returns `done=True`. `reset()` checks the buffer before calling `receive_state()`, so the communicator stays in sync.

## Key Methods

**`reset()`**
1. Check `_buffered_state`; if set, use it as the starting state (don't call `receive_state()`)
2. Call `_advance_to_combat()` which loops:
   - Receive state (or use buffered)
   - If not in game: send `START IRONCLAD 0`
   - If GAME_OVER: record run, send `PROCEED`
   - If non-combat screen: SimpleAgent handles it, send response, loop
   - If `is_in_combat`: return state
3. Store `_combat_start_hp`, return encoded obs

**`step(action)`**
1. Translate action → command via `ActionSpace.action_to_command()`
2. Send command, receive next state
3. If `state.is_in_combat`: return `(obs, 0.0, False, False, {})`
4. If GAME_OVER: reward = `-1.0`, send `PROCEED`, `_buffered_state = None`, return `(last_obs, reward, True, False, {})`
5. Otherwise (combat ended): reward = `current_hp / max_hp`, `_buffered_state = state`, return `(last_obs, reward, True, False, {})`

**`_compute_reward(state)`**
- Death / GAME_OVER: `-1.0`
- Survival: `current_hp / max_hp`

**`action_masks()`**
- Delegates to `ActionSpace.get_action_mask(self._current_state)`
- Returns all-ones mask if `_current_state` is None

## Training Parameters

```python
MaskablePPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1,
)
model.learn(
    total_timesteps=10_000_000,
    callback=EpisodeLoggerCallback(save_path="data/checkpoints/", save_freq=1000),
)
```

## Logging

**Per-step** (DEBUG): floor, HP, action command

**Per-episode** (INFO):
```
[Episode 42] reward=0.81 | steps=7 | hp=65/80 | floor=3 | total_steps=294
```

**Every 100 episodes** (INFO):
```
[Summary ep 100-200] avg_reward=0.54 | avg_steps=9.2 | win_rate=0.0% | avg_floor=4.1
```

**SB3 updates**: `verbose=1` prints loss/entropy/value_loss after each policy update

**Checkpoints**:
```
[Checkpoint] Saved data/checkpoints/combat_1000_steps.zip (episode 47)
```

## Callback

A single `EpisodeLoggerCallback(BaseCallback)` handles both logging and saving:
- Tracks per-episode stats via `infos` from SB3
- Logs summary every 100 episodes
- Saves checkpoint every 1000 timesteps via `CheckpointCallback` (composed or subclassed)

## Files Changed

| Action | File |
|--------|------|
| Create | `src/combat_env.py` |
| Create | `tests/test_combat_env.py` |
| Update | `main.py` |
| Delete | `src/rl_agent.py` |
| Delete | `tests/test_rl_agent.py` |

## Tests

Using a mock `Communicator`:
- `action_masks()` returns correct shape/dtype
- `_compute_reward()` — survival gives hp ratio, death gives -1.0
- `step()` when combat continues — reward=0, done=False
- `step()` when combat ends — correct reward, done=True, state buffered
- `reset()` consumes buffered state without calling `receive_state()`
