# Slay the Spire AI

Reinforcement learning agent that plays Slay the Spire. The long-term goal is to empirically prove every seed is beatable — starting with Ironclad at Ascension 0, scaling to all characters and A20.

The agent communicates with the game via [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod), which exposes a stdin/stdout JSON protocol.

---

## Versioning

Two learning algorithm versions are in active development:

| Version | Episode scope | Decisions | Flag |
|---|---|---|---|
| **v1** (`--rl`) | One combat | RL for combat; `SimpleAgent` for everything else | `--rl` |
| **v2** (`--v2`) | One full run | RL for all decisions (combat + map + cards + shop + rest) | `--v2` |

v1 is the original approach. v2 is the current research direction — it replaces `SimpleAgent` entirely and lets the RL agent learn the full game loop.

---

## v2: Full-Run RL

### Architecture

```
main.py --v2
  └─ RunEnv (Gymnasium)
       ├─ reset()  → sends START, waits for first actionable state
       ├─ step()   → MaskablePPO picks from 104-action space (combat + non-combat)
       └─ masks    → only legal actions for the current screen are unmasked
```

One episode = one complete run, from character select through game over. There is no `SimpleAgent` fallback — the RL model handles every screen.

### Observation Space — 227 floats

The observation vector is split into three blocks, all normalized to [0, 1]:

**Global block (55 floats) — always present:**
- Player stats: HP ratio, max HP scale, floor progress, gold, act, ascension, energy, block
- Deck composition: size, attack/skill/power/curse ratios, exhaust/strength/draw card counts
- Potions: 5 slots × 4 features (present, healing, attack, requires target)
- Relics: count + 6 high-impact relic flags (Akabeko, Burning Blood, etc.)
- Screen type: 12-class one-hot

**Combat block (112 floats) — active during combat screens:**
- Hand: 10 card slots × 7 features (cost, type, playable, vulnerable/weak/draw/block flags)
- Enemies: 5 slots × 6 features (HP ratio, max HP scale, block ratio, intent, vuln stacks, weak stacks)
- Player powers: Strength, Dexterity, Weak, Vulnerable, Barricade
- Turn metadata: draw pile size, discard pile size, turn number
- Debuff signal: fraction of hand cards that apply Vulnerable/Weak

**Non-combat block (60 floats) — active during non-combat screens:**
- Choices: 8 slots × 4 features (tier value, synergy score, cost ratio, availability)
- Deck synergy context: exhaust/strength/draw/block/curse card counts
- Screen metadata: shop affordability, rest heal amount, map node availability (elite/rest/shop/event/monster)

### Action Space — 104 discrete actions

| Range | Action |
|---|---|
| 0–9 | Play untargeted card in hand slot N |
| 10–59 | Play targeted card in slot N on target M (`10 + slot×5 + target`) |
| 60 | End turn |
| 61–65 | Use potion (no target) |
| 66–90 | Use potion on target |
| 91–98 | CHOOSE option N (card reward, shop item, map node, event option…) |
| 99 | PROCEED |
| 100 | PURGE (card removal at shop) |
| 101 | CHOOSE rest |
| 102 | CHOOSE smith (upgrade at campfire) |
| 103 | OPEN (chest) |

Action masking is applied every step: only actions valid for the current screen and game state are unmasked.

### Reward Shaping

**Combat steps:**
```
reward = damage_dealt / max_hp
       − damage_taken / max_hp
       + 0.1 × kills
       + 0.05 × new_debuff_stacks
       + 0.03  [if attack after debuffing this turn]
       − 0.3 × (energy_remaining / max_energy)  [on END TURN only]
```

**Non-combat steps:**
- Card reward pick: `tier_value × 0.05 + synergy_score × 0.05`
- Shop purchase: same as card reward, minus `cost_ratio × 0.02`
- Shop relic: +0.05
- Card removal (purge): +0.03 for D-tier or curse/status
- Rest heal: `(hp_gained / max_hp) × 0.2`
- Campfire upgrade: +0.05

**Terminal reward (game over):**
```
terminal = (floor / 55) × 3.0 − 1.0
```
Ranges from −1.0 (die on floor 0) to +2.0 (clear the final boss).

### Training Setup

```python
MaskablePPO(
    "MlpPolicy",        # 2-layer MLP
    env,                # RunEnv — one episode per full run
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)
model.learn(total_timesteps=10_000_000)
```

Model is saved to `data/v2_run_model.zip`. Checkpoints are written to `data/v2_checkpoints/` every 100 steps with the prefix `v2_run`.

---

## v1: Combat-Only RL

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
- Player: HP ratio, max HP (always 1.0, kept for index stability), block ratio, energy ratio (4 values)
- Hand: 10 card slots × 7 features each (cost, type, playable, vulnerable/weak/draw/block flags)
- Enemies: 5 enemy slots × 6 features each (HP ratio, max HP scale, block ratio, intent, vulnerable stacks, weak stacks)

**Action space** — 61 discrete actions:
- Actions 0–9: play an untargeted card in hand slot N
- Actions 10–59: play a targeted card in hand slot N on target M (`10 + slot*5 + target`)
- Action 60: end turn

**Action masking** is applied every step so the model never sees invalid actions — unplayable cards, dead targets, and already-targeted untargetable cards are all masked out before the policy samples.

**Mid-combat screens** (GRID/HAND_SELECT triggered by cards like Armaments or Dual Wield) are treated as in-combat states — the RL episode continues and `SimpleAgent` resolves the screen automatically.

### Reward Shaping

Rewards are shaped at each step to give the model a learning signal within long combats:

```
step reward   = (damage_dealt − damage_taken + 0.1 × kills) / player_max_hp
               + energy_efficiency_bonus × (energy_spent / max_energy)  [on END TURN]

final reward  = 1.0                   (game victory — beat the final boss)
              = current_hp / max_hp   (survived this combat, run continues)
              = −1.0                  (died)
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

Checkpoints are saved every 1000 steps to `data/checkpoints/`. Training resumes from the latest checkpoint or final model automatically (preferring whichever has the newer modification time).

### Non-Combat Decisions (SimpleAgent)

Everything outside combat is handled by `SimpleAgent` with hand-crafted heuristics:

| Screen | Strategy |
|---|---|
| Card reward | Tier list + empirical EMA score blended via softmax |
| Map | Favour elites/bosses above 40% HP; prefer rest/events below |
| Rest | Rest if < 60% HP, otherwise upgrade a card |
| Shop | Purge D-tier cards first; then buy S/A cards and relics within gold budget |
| Events | Random choice among enabled options |
| Potions | Applied automatically before the RL agent acts each turn |
| GAME_OVER | Sends PROCEED and resets for next run |

`SimpleAgent` is wrapped in `StuckDetectorAgent`, which fingerprints `(screen_type, available_commands, screen_state)` and cycles through a fallback command sequence if the same state repeats too many times.

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

Configure CommunicationMod to launch `python main.py` (or `python main.py --rl`) as its command, then start the game normally. The bot runs from the game's working directory — all data files are written there.

**Rule-based mode** (no training, uses SimpleAgent for all decisions):
```bash
python main.py
```

**v1 RL training** (MaskablePPO for combat, SimpleAgent for everything else):
```bash
python main.py --rl
```

**v2 RL training** (MaskablePPO for all decisions — full-run episodes, no SimpleAgent):
```bash
python main.py --v2
```

### Web dashboard

A live dashboard is available while the bot is running:

```bash
python dashboard.py
```

Navigate to `http://localhost:5000` to see current game state, run stats, and the live training reward chart.

---

## Project Layout

```
src/
  agent.py           # SimpleAgent (non-combat) + StuckDetectorAgent wrapper
  combat_env.py      # v1 Gymnasium env (one episode = one combat)
  state_encoder.py   # v1 GameState → 104-float observation vector
  action_space.py    # v1 action discretization + mask generation
  card_scorer.py     # EMA-based empirical card scoring
  card_tier_list.py  # Static Ironclad tier rankings (S/A/B/C/D)
  card_properties.py # Per-card binary flags (vulnerable, weak, draws, block)
  callbacks.py       # SB3 training callback: logging + live dashboard updates
  communicator.py    # stdin/stdout JSON bridge to CommunicationMod
  game_state.py      # GameState dataclass parsed from CommunicationMod JSON
  game_loop.py       # Main loop for rule-based (non-RL) mode
  run_tracker.py     # Records per-run outcomes to JSONL + generates graphs
  live_state.py      # Writes real-time state JSON for the dashboard
  grapher.py         # Performance graphs from run log (matplotlib)
  v2/
    run_env.py         # v2 Gymnasium env (one episode = one full run)
    run_encoder.py     # v2 GameState → 227-float observation vector
    run_action_space.py # v2 action discretization + mask generation (104 actions)
    run_reward.py      # v2 reward shaping (combat + non-combat + terminal)
main.py              # Entry point (launched by CommunicationMod)
dashboard.py         # Flask web dashboard
```

### Data outputs

All paths are relative to the working directory at launch time. When CommunicationMod launches the bot, that is the STS game directory.

| Path | Contents |
|---|---|
| `data/run_log.jsonl` | One JSON object per completed run |
| `data/combat_model.zip` | v1 final trained model (saved on interrupt/completion) |
| `data/checkpoints/` | v1 periodic model snapshots (`combat_N_steps.zip`) |
| `data/v2_run_model.zip` | v2 final trained model |
| `data/v2_checkpoints/` | v2 periodic model snapshots (`v2_run_N_steps.zip`) |
| `data/card_scores.json` | Learned card quality EMA scores |
| `data/graphs/performance.png` | Performance visualizations (regenerated every 10 runs) |
| `data/live_state.json` | Real-time state for the dashboard (overwritten each step) |
| `game.log` | Detailed action/decision log |

---

## Dependencies

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) + [sb3-contrib](https://github.com/Stable-Baselines3/stable-baselines3-contrib) — MaskablePPO
- [gymnasium](https://gymnasium.farama.org/) — environment interface
- [PyTorch](https://pytorch.org/) — neural network backend
- matplotlib — performance graphs
- numpy, pytest
