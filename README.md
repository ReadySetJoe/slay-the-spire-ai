# Slay the Spire AI

Reinforcement learning agent that plays Slay the Spire. The long-term goal is to empirically prove every seed is beatable — starting with Ironclad at Ascension 0, scaling to all characters and A20.

The agent communicates with the game via [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod), which exposes a stdin/stdout JSON protocol.

---

## How the Learning Algorithm Works

The agent uses a **hybrid approach**: RL for combat, hand-crafted rules for everything else.

### Architecture

```
main.py --rl
  └─ CombatEnv (Gymnasium)
       ├─ reset()  → SimpleAgent drives non-combat screens until a fight starts
       ├─ step()   → MaskablePPO picks a card/end-turn action
       └─ masks    → invalid actions (wrong energy, dead targets) are blocked
```

### Combat as an RL Episode

One episode = one combat encounter. The environment resets between fights, letting `MaskablePPO` (from `sb3-contrib`) learn purely combat tactics without being confused by map navigation or card rewards.

**Observation space** — 104 floats, all normalized to [0, 1]:
- Player: HP ratio, max HP, block, energy (4 values)
- Hand: 10 card slots × 7 features each (cost, type, playable, status flags like vulnerable/weak/draw/block)
- Enemies: 5 enemy slots × 6 features each (HP ratio, block, intent, debuff stacks)

**Action space** — 61 discrete actions:
- Actions 0–9: play an untargeted card in hand slot N
- Actions 10–59: play a targeted card (10 slots × 5 target positions)
- Action 60: end turn

**Action masking** is applied every step so the model never sees invalid actions — unplayable cards, dead targets, and out-of-energy plays are masked out before the policy samples. This keeps training stable and eliminates the need for penalizing illegal moves.

### Reward Shaping

Rewards are shaped at each step to give the model a learning signal within long combats:

```
step reward  = (damage_dealt - damage_taken + 0.1 * kills) / player_max_hp
final reward = current_hp / max_hp   (if survived)
             = -1.0                  (if died)
```

The terminal reward anchors the episode to actual health outcome; step rewards guide the policy during the fight.

### Training Setup

```python
MaskablePPO(
    "MlpPolicy",        # 2-layer MLP
    env,
    learning_rate=3e-4,
    n_steps=2048,       # steps per rollout
    batch_size=64,
    n_epochs=10,
)
model.learn(total_timesteps=10_000_000)
```

Checkpoints are saved every 1000 steps to `data/checkpoints/`. Training resumes from the latest checkpoint automatically.

### Non-Combat Decisions (SimpleAgent)

Everything outside combat is handled by `SimpleAgent` with hand-crafted heuristics:

| Screen | Strategy |
|---|---|
| Card reward | Tier list + empirical EMA score blended via softmax |
| Map | Favour elites/bosses above 60% HP, avoid below |
| Rest | Rest if < 60% HP, otherwise upgrade a card |
| Shop | Tier-based purchases within gold budget |
| Events | Safe random choices |
| Potions | Used automatically before the RL agent acts each turn |

### Card Score Learning (non-RL)

`CardScorer` maintains a separate, slower learning loop for deck-building. After each run it updates an exponential moving average of run quality for every card picked:

```
quality = (current_hp / max_hp) × (floor / 55)
EMA     = α × quality + (1 − α) × prior_EMA
```

These scores are blended with the static tier list at card-selection time and persisted to `data/card_scores.json`. This gives the agent a data-driven card preference that improves independently of the combat RL model.

---

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure CommunicationMod to launch `python main.py` as its command, then start the game normally.

**Rule-based mode** (default, no training):
```bash
python main.py
```

**RL training mode:**
```bash
python main.py --rl
```

---

## Project Layout

```
src/
  agent.py          # SimpleAgent: all non-combat decisions
  combat_env.py     # Gymnasium environment (one episode = one fight)
  state_encoder.py  # GameState → 104-float observation
  action_space.py   # Action discretization + masking
  card_scorer.py    # EMA-based empirical card scoring
  card_tier_list.py # Static Ironclad tier rankings
  card_properties.py# Card attributes (vulnerable, draws, block, etc.)
  callbacks.py      # SB3 logging + checkpoint callbacks
  communicator.py   # stdin/stdout JSON bridge to game
  game_state.py     # GameState dataclass
  game_loop.py      # Loop for rule-based mode
  run_tracker.py    # Records per-run outcomes
  live_state.py     # Writes real-time state for dashboards
  grapher.py        # Performance graphs from run log
main.py             # Entry point
dashboard.py        # Web dashboard
```

### Data outputs

| Path | Contents |
|---|---|
| `data/run_log.jsonl` | One JSON object per completed run |
| `data/combat_model.zip` | Latest trained RL model |
| `data/checkpoints/` | Periodic model snapshots |
| `data/card_scores.json` | Learned card quality EMA scores |
| `data/graphs/` | Performance visualizations |
| `game.log` | Detailed action/decision log |

---

## Dependencies

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) + [sb3-contrib](https://github.com/Stable-Baselines3/stable-baselines3-contrib) — MaskablePPO
- [gymnasium](https://gymnasium.farama.org/) — environment interface
- [PyTorch](https://pytorch.org/) — neural network backend
- numpy, pytest
