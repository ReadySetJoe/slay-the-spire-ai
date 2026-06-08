# V2 Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified RL agent that controls every game decision across a full run (combat, shop, card rewards, map, rest, events) as a single Gymnasium episode.

**Architecture:** `RunEnv(gym.Env)` wraps one complete run as one episode. A screen-conditioned observation vector (227 features split into global + combat + non-combat blocks) feeds a single `MaskablePPO` model. A 104-action fixed space covers all game decisions; screen-specific masking zeroes invalid actions before the policy samples.

**Tech Stack:** Python 3.11, gymnasium, stable-baselines3, sb3-contrib (MaskablePPO), numpy, pytest

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/v2/__init__.py` | Package marker |
| Create | `src/v2/run_action_space.py` | 104-action space, action→command, screen masks |
| Create | `src/v2/run_encoder.py` | 227-feature observation encoder |
| Create | `src/v2/run_reward.py` | Per-decision reward shaping + synergy scoring |
| Create | `src/v2/run_env.py` | RunEnv — full-run Gymnasium episode |
| Create | `tests/v2/__init__.py` | Package marker |
| Create | `tests/v2/helpers.py` | GameState factory helpers shared across v2 tests |
| Create | `tests/v2/test_run_action_space.py` | Action space tests |
| Create | `tests/v2/test_run_encoder.py` | Encoder tests |
| Create | `tests/v2/test_run_reward.py` | Reward shaper tests |
| Create | `tests/v2/test_run_env.py` | RunEnv integration tests |
| Modify | `main.py` | Add `--v2` entry point |

---

## Task 1: Branch and package skeleton

**Files:**
- Create: `src/v2/__init__.py`
- Create: `tests/v2/__init__.py`
- Create: `tests/v2/helpers.py`

- [ ] **Step 1: Create the v2 branch**

```bash
git checkout -b v2
```

- [ ] **Step 2: Create package markers and shared test helpers**

`src/v2/__init__.py` — empty file.

`tests/v2/__init__.py` — empty file.

`tests/v2/helpers.py`:

```python
import json
from src.game_state import GameState


def make_state(
    screen_type="NONE",
    available_commands=None,
    hp=70, max_hp=80, floor=1, gold=99, energy=3, block=0, act=1,
    hand=None, monsters=None, draw_pile=None, discard_pile=None,
    potions=None, relics=None, deck=None,
    screen_state=None,
    combat=True,
    powers=None,
    turn=1,
) -> GameState:
    if available_commands is None:
        available_commands = ["PLAY", "END"] if combat and screen_type == "NONE" else ["PROCEED"]
    if hand is None:
        hand = [{"id": "Strike_R", "name": "Strike", "cost": 1,
                 "type": "ATTACK", "is_playable": True, "has_target": True, "uuid": "a1"}]
    if monsters is None:
        monsters = [{"name": "Jaw Worm", "current_hp": 42, "max_hp": 42,
                     "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}]
    if draw_pile is None:
        draw_pile = []
    if discard_pile is None:
        discard_pile = []
    if potions is None:
        potions = []
    if relics is None:
        relics = []
    if deck is None:
        deck = []
    if powers is None:
        powers = []

    combat_state = None
    if combat and screen_type == "NONE":
        combat_state = {
            "hand": hand,
            "draw_pile": draw_pile,
            "discard_pile": discard_pile,
            "exhaust_pile": [],
            "monsters": monsters,
            "player": {"current_hp": hp, "max_hp": max_hp, "block": block,
                       "energy": energy, "powers": powers},
            "turn": turn,
        }

    return GameState.from_json(json.dumps({
        "available_commands": available_commands,
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "screen_type": screen_type,
            "screen_state": screen_state,
            "seed": 1, "floor": floor, "ascension_level": 0,
            "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": gold, "act": act,
            "deck": deck, "relics": relics, "potions": potions, "map": [],
            "combat_state": combat_state,
        }
    }))


def make_game_over(floor=3, hp=0, max_hp=80) -> GameState:
    return GameState.from_json(json.dumps({
        "available_commands": ["PROCEED"],
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": "GAME_OVER",
            "seed": 1, "floor": floor, "ascension_level": 0,
            "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": 99, "act": 1,
            "deck": [], "relics": [], "potions": [], "map": [],
            "combat_state": None,
        }
    }))


def make_card_reward(cards=None, hp=70, max_hp=80, floor=1) -> GameState:
    if cards is None:
        cards = [
            {"id": "Shrug It Off", "name": "Shrug It Off", "type": "SKILL"},
            {"id": "Inflame", "name": "Inflame", "type": "POWER"},
            {"id": "Strike_R", "name": "Strike", "type": "ATTACK"},
        ]
    return make_state(
        screen_type="CARD_REWARD",
        available_commands=["CHOOSE", "PROCEED"],
        hp=hp, max_hp=max_hp, floor=floor,
        screen_state={"cards": cards},
        combat=False,
    )


def make_shop(cards=None, relics=None, gold=200, purge_available=True, hp=70, max_hp=80) -> GameState:
    if cards is None:
        cards = [{"id": "Inflame", "price": 75, "is_in_stock": True, "type": "POWER"}]
    if relics is None:
        relics = [{"id": "Anchor", "price": 150, "is_in_stock": True}]
    cmds = ["CHOOSE", "PROCEED"]
    if purge_available:
        cmds.append("PURGE")
    return make_state(
        screen_type="SHOP_SCREEN",
        available_commands=cmds,
        gold=gold, hp=hp, max_hp=max_hp,
        screen_state={"cards": cards, "relics": relics,
                      "purge_cost": 75, "purge_available": purge_available},
        combat=False,
    )


def make_rest(hp=50, max_hp=80) -> GameState:
    return make_state(
        screen_type="REST",
        available_commands=["CHOOSE"],
        hp=hp, max_hp=max_hp,
        combat=False,
    )


def make_map(nodes=None, hp=70, max_hp=80, floor=1) -> GameState:
    if nodes is None:
        nodes = [{"symbol": "M"}, {"symbol": "E"}, {"symbol": "?"}]
    return make_state(
        screen_type="MAP",
        available_commands=["CHOOSE"],
        hp=hp, max_hp=max_hp, floor=floor,
        screen_state={"next_nodes": nodes},
        combat=False,
    )
```

- [ ] **Step 3: Verify the helpers work**

```bash
python -c "from tests.v2.helpers import make_state, make_game_over; s = make_state(); print(s.screen_type, s.is_in_combat)"
```

Expected output: `NONE True`

- [ ] **Step 4: Commit**

```bash
git add src/v2/__init__.py tests/v2/__init__.py tests/v2/helpers.py
git commit -m "feat(v2): branch + package skeleton and test helpers"
```

---

## Task 2: RunActionSpace — action_to_command

**Files:**
- Create: `src/v2/run_action_space.py`
- Create: `tests/v2/test_run_action_space.py`

- [ ] **Step 1: Write failing tests for action_to_command**

`tests/v2/test_run_action_space.py`:

```python
import pytest
from src.v2.run_action_space import RunActionSpace

@pytest.fixture
def space():
    return RunActionSpace()

def test_play_no_target_slot_0(space, combat_state):
    assert space.action_to_command(0, combat_state) == "PLAY 1"

def test_play_no_target_slot_9(space, combat_state):
    assert space.action_to_command(9, combat_state) == "PLAY 10"

def test_play_targeted_slot_0_target_0(space, combat_state):
    assert space.action_to_command(10, combat_state) == "PLAY 1 0"

def test_play_targeted_slot_0_target_4(space, combat_state) :
    assert space.action_to_command(14, combat_state) == "PLAY 1 4"

def test_play_targeted_slot_1_target_0(space, combat_state):
    assert space.action_to_command(15, combat_state) == "PLAY 2 0"

def test_play_targeted_slot_9_target_4(space, combat_state):
    assert space.action_to_command(59, combat_state) == "PLAY 10 4"

def test_end_turn(space, combat_state):
    assert space.action_to_command(60, combat_state) == "END"

def test_potion_no_target_slot_0(space, combat_state):
    assert space.action_to_command(61, combat_state) == "POTION Use 0"

def test_potion_no_target_slot_4(space, combat_state):
    assert space.action_to_command(65, combat_state) == "POTION Use 4"

def test_potion_targeted_slot_0_target_0(space, combat_state):
    assert space.action_to_command(66, combat_state) == "POTION Use 0 0"

def test_potion_targeted_slot_1_target_2(space, combat_state):
    assert space.action_to_command(73, combat_state) == "POTION Use 1 2"

def test_potion_targeted_slot_4_target_4(space, combat_state):
    assert space.action_to_command(90, combat_state) == "POTION Use 4 4"

def test_choose_0(space, combat_state):
    assert space.action_to_command(91, combat_state) == "CHOOSE 0"

def test_choose_7(space, combat_state):
    assert space.action_to_command(98, combat_state) == "CHOOSE 7"

def test_proceed(space, combat_state):
    assert space.action_to_command(99, combat_state) == "PROCEED"

def test_purge(space, combat_state):
    assert space.action_to_command(100, combat_state) == "PURGE"

def test_choose_rest(space, combat_state):
    assert space.action_to_command(101, combat_state) == "CHOOSE rest"

def test_choose_smith(space, combat_state):
    assert space.action_to_command(102, combat_state) == "CHOOSE smith"

def test_open(space, combat_state):
    assert space.action_to_command(103, combat_state) == "OPEN"

def test_invalid_action_raises(space, combat_state):
    with pytest.raises(ValueError):
        space.action_to_command(104, combat_state)

def test_total_actions_constant():
    assert RunActionSpace.TOTAL_ACTIONS == 104

# Shared fixture
import pytest
from tests.v2.helpers import make_state

@pytest.fixture
def combat_state():
    return make_state()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_action_space.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'src.v2.run_action_space'`

- [ ] **Step 3: Implement RunActionSpace with action_to_command**

`src/v2/run_action_space.py`:

```python
import numpy as np
from src.game_state import GameState

MAX_HAND     = 10
MAX_TARGETS  = 5
MAX_POTIONS  = 5
MAX_CHOICES  = 8

_PLAY_NT_START = 0    # 0-9:   PLAY slot (no target)
_PLAY_T_START  = 10   # 10-59: PLAY slot target
_END_TURN      = 60   # 60:    END
_POT_NT_START  = 61   # 61-65: POTION Use slot
_POT_T_START   = 66   # 66-90: POTION Use slot target
_CHOOSE_START  = 91   # 91-98: CHOOSE 0-7
_PROCEED       = 99   # 99:    PROCEED
_PURGE         = 100  # 100:   PURGE
_CHOOSE_REST   = 101  # 101:   CHOOSE rest
_CHOOSE_SMITH  = 102  # 102:   CHOOSE smith
_OPEN          = 103  # 103:   OPEN

TOTAL_ACTIONS  = 104


class RunActionSpace:
    TOTAL_ACTIONS = TOTAL_ACTIONS
    END_TURN      = _END_TURN

    def action_to_command(self, action: int, state: GameState) -> str:
        if action < _PLAY_T_START:           # 0-9
            return f"PLAY {action + 1}"
        if action < _END_TURN:               # 10-59
            slot, target = divmod(action - _PLAY_T_START, MAX_TARGETS)
            return f"PLAY {slot + 1} {target}"
        if action == _END_TURN:              # 60
            return "END"
        if action < _POT_T_START:            # 61-65
            return f"POTION Use {action - _POT_NT_START}"
        if action < _CHOOSE_START:           # 66-90
            slot, target = divmod(action - _POT_T_START, MAX_TARGETS)
            return f"POTION Use {slot} {target}"
        if action < _PROCEED:                # 91-98
            return f"CHOOSE {action - _CHOOSE_START}"
        if action == _PROCEED:               # 99
            return "PROCEED"
        if action == _PURGE:                 # 100
            return "PURGE"
        if action == _CHOOSE_REST:           # 101
            return "CHOOSE rest"
        if action == _CHOOSE_SMITH:          # 102
            return "CHOOSE smith"
        if action == _OPEN:                  # 103
            return "OPEN"
        raise ValueError(f"Invalid action index {action}")

    def get_action_mask(self, state: GameState) -> np.ndarray:
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify action_to_command passes**

```bash
pytest tests/v2/test_run_action_space.py -v -k "not mask"
```

Expected: All `test_play_*`, `test_end_turn`, `test_potion_*`, `test_choose_*`, `test_proceed`, `test_purge`, `test_open`, `test_invalid_action_raises`, `test_total_actions_constant` pass.

- [ ] **Step 5: Commit**

```bash
git add src/v2/run_action_space.py tests/v2/test_run_action_space.py
git commit -m "feat(v2): RunActionSpace action_to_command"
```

---

## Task 3: RunActionSpace — get_action_mask

**Files:**
- Modify: `src/v2/run_action_space.py`
- Modify: `tests/v2/test_run_action_space.py`

- [ ] **Step 1: Add failing mask tests**

Append to `tests/v2/test_run_action_space.py`:

```python
import numpy as np
from tests.v2.helpers import (
    make_state, make_card_reward, make_shop, make_rest, make_map, make_game_over
)

# ---- mask shape/dtype ----

def test_mask_shape_and_dtype(space):
    mask = space.get_action_mask(make_state())
    assert mask.shape == (RunActionSpace.TOTAL_ACTIONS,)
    assert mask.dtype == np.bool_

# ---- combat masks ----

def test_combat_end_turn_always_enabled(space):
    mask = space.get_action_mask(make_state())
    assert mask[60] is np.bool_(True)

def test_combat_playable_targeted_card_enables_targeted_actions(space):
    # hand[0] = Strike (has_target=True), monster at index 0
    state = make_state(
        hand=[{"id": "Strike_R", "cost": 1, "type": "ATTACK",
               "is_playable": True, "has_target": True}],
        monsters=[{"name": "Worm", "current_hp": 40, "max_hp": 40,
                   "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}],
    )
    mask = space.get_action_mask(state)
    assert mask[10] is np.bool_(True)   # PLAY 1 0 (slot 0, target 0)
    assert mask[0] is np.bool_(False)   # no-target slot 0 disabled

def test_combat_playable_untargeted_card_enables_no_target_action(space):
    state = make_state(
        hand=[{"id": "Shrug It Off", "cost": 1, "type": "SKILL",
               "is_playable": True, "has_target": False}],
        monsters=[{"name": "Worm", "current_hp": 40, "max_hp": 40,
                   "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}],
    )
    mask = space.get_action_mask(state)
    assert mask[0] is np.bool_(True)    # PLAY 1 (slot 0, no target)
    assert mask[10] is np.bool_(False)  # targeted slot 0 target 0 disabled

def test_combat_unplayable_card_disabled(space):
    state = make_state(
        hand=[{"id": "Strike_R", "cost": 1, "type": "ATTACK",
               "is_playable": False, "has_target": True}],
    )
    mask = space.get_action_mask(state)
    assert mask[10] is np.bool_(False)

def test_combat_usable_no_target_potion_enabled(space):
    state = make_state(
        potions=[{"id": "Fire Potion", "can_use": True, "requires_target": False}],
    )
    mask = space.get_action_mask(state)
    assert mask[61] is np.bool_(True)   # POTION Use 0

def test_combat_usable_targeted_potion_enabled(space):
    state = make_state(
        potions=[{"id": "Poison Potion", "can_use": True, "requires_target": True}],
        monsters=[{"name": "Worm", "current_hp": 40, "max_hp": 40,
                   "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}],
    )
    mask = space.get_action_mask(state)
    assert mask[66] is np.bool_(True)   # POTION Use 0 0

def test_combat_non_combat_actions_disabled(space):
    mask = space.get_action_mask(make_state())
    assert mask[91] is np.bool_(False)  # CHOOSE 0
    assert mask[99] is np.bool_(False)  # PROCEED
    assert mask[103] is np.bool_(False) # OPEN

# ---- non-combat masks ----

def test_card_reward_enables_choose_and_proceed(space):
    state = make_card_reward()
    mask = space.get_action_mask(state)
    assert mask[91] is np.bool_(True)   # CHOOSE 0
    assert mask[92] is np.bool_(True)   # CHOOSE 1
    assert mask[93] is np.bool_(True)   # CHOOSE 2
    assert mask[94] is np.bool_(False)  # CHOOSE 3 (only 3 cards)
    assert mask[99] is np.bool_(True)   # PROCEED
    assert mask[60] is np.bool_(False)  # END disabled

def test_shop_enables_choose_and_proceed_and_purge(space):
    state = make_shop()  # 1 card, 1 relic
    mask = space.get_action_mask(state)
    assert mask[91] is np.bool_(True)   # CHOOSE 0 (card)
    assert mask[92] is np.bool_(True)   # CHOOSE 1 (relic)
    assert mask[93] is np.bool_(False)  # CHOOSE 2 (nothing at index 2)
    assert mask[99] is np.bool_(True)   # PROCEED
    assert mask[100] is np.bool_(True)  # PURGE

def test_rest_enables_only_rest_and_smith(space):
    mask = space.get_action_mask(make_rest())
    assert mask[101] is np.bool_(True)  # CHOOSE rest
    assert mask[102] is np.bool_(True)  # CHOOSE smith
    assert mask[91] is np.bool_(False)  # CHOOSE 0 disabled
    assert mask[60] is np.bool_(False)  # END disabled

def test_map_enables_node_choices(space):
    state = make_map(nodes=[{"symbol": "M"}, {"symbol": "E"}, {"symbol": "?"}])
    mask = space.get_action_mask(state)
    assert mask[91] is np.bool_(True)   # CHOOSE 0
    assert mask[92] is np.bool_(True)   # CHOOSE 1
    assert mask[93] is np.bool_(True)   # CHOOSE 2
    assert mask[94] is np.bool_(False)  # CHOOSE 3 — no 4th node
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_action_space.py -v -k "mask" 2>&1 | head -20
```

Expected: `NotImplementedError`

- [ ] **Step 3: Implement get_action_mask**

Replace the `get_action_mask` stub in `src/v2/run_action_space.py`:

```python
    def get_action_mask(self, state: GameState) -> np.ndarray:
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.bool_)

        if state.is_in_combat:
            self._mask_combat(mask, state)
            return mask

        screen = state.screen_type
        cmds   = state.available_commands
        ss     = state.screen_state or {}

        if screen == "CARD_REWARD":
            n = len(ss.get("cards", []))
            for i in range(min(n, MAX_CHOICES)):
                mask[_CHOOSE_START + i] = True
            if "PROCEED" in cmds:
                mask[_PROCEED] = True

        elif screen == "SHOP_SCREEN":
            n = len(ss.get("cards", [])) + len(ss.get("relics", []))
            for i in range(min(n, MAX_CHOICES)):
                mask[_CHOOSE_START + i] = True
            if "PURGE" in cmds:
                mask[_PURGE] = True
            mask[_PROCEED] = True

        elif screen == "MAP":
            n = len(ss.get("next_nodes", []))
            for i in range(min(n, MAX_CHOICES)):
                mask[_CHOOSE_START + i] = True

        elif screen == "REST":
            if "CHOOSE" in cmds:
                mask[_CHOOSE_REST]  = True
                mask[_CHOOSE_SMITH] = True

        elif screen == "CHEST":
            if "OPEN"    in cmds: mask[_OPEN]    = True
            if "PROCEED" in cmds: mask[_PROCEED] = True

        elif screen in ("EVENT", "GRID", "HAND_SELECT",
                        "COMBAT_REWARD", "BOSS_REWARD"):
            if "CHOOSE" in cmds:
                options = (ss.get("options") or ss.get("cards") or
                           ss.get("rewards") or [])
                n = min(len(options), MAX_CHOICES) if options else MAX_CHOICES
                for i in range(n):
                    mask[_CHOOSE_START + i] = True
            if "PROCEED" in cmds:
                mask[_PROCEED] = True

        elif screen == "SHOP_ROOM":
            mask[_PROCEED] = True

        else:
            if "PROCEED" in cmds:
                mask[_PROCEED] = True

        return mask

    def _mask_combat(self, mask: np.ndarray, state: GameState) -> None:
        mask[_END_TURN] = True

        living = {i for i, m in enumerate(state.monsters[:MAX_TARGETS])
                  if not m.get("is_gone", False)}

        for slot, card in enumerate(state.hand[:MAX_HAND]):
            if not card.get("is_playable", False):
                continue
            if card.get("has_target", False):
                for t in living:
                    mask[_PLAY_T_START + slot * MAX_TARGETS + t] = True
            else:
                mask[_PLAY_NT_START + slot] = True

        for slot, potion in enumerate(state.potions[:MAX_POTIONS]):
            if not potion.get("can_use", False):
                continue
            if potion.get("requires_target", False):
                for t in living:
                    mask[_POT_T_START + slot * MAX_TARGETS + t] = True
            else:
                mask[_POT_NT_START + slot] = True
```

- [ ] **Step 4: Run all action space tests**

```bash
pytest tests/v2/test_run_action_space.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/v2/run_action_space.py tests/v2/test_run_action_space.py
git commit -m "feat(v2): RunActionSpace get_action_mask"
```

---

## Task 4: RunEncoder — global block

**Files:**
- Create: `src/v2/run_encoder.py`
- Create: `tests/v2/test_run_encoder.py`

- [ ] **Step 1: Write failing tests for global block**

`tests/v2/test_run_encoder.py`:

```python
import numpy as np
import pytest
from src.v2.run_encoder import RunEncoder
from tests.v2.helpers import make_state, make_card_reward, make_shop, make_rest, make_map

GLOBAL_SIZE    = 55
COMBAT_SIZE    = 112
NONCOMBAT_SIZE = 60
OBS_SIZE       = GLOBAL_SIZE + COMBAT_SIZE + NONCOMBAT_SIZE  # 227


@pytest.fixture
def enc():
    return RunEncoder()


def test_obs_shape(enc):
    obs = enc.encode(make_state())
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32


def test_obs_values_in_range(enc):
    obs = enc.encode(make_state())
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_hp_ratio_global(enc):
    obs = enc.encode(make_state(hp=40, max_hp=80))
    assert obs[0] == pytest.approx(0.5)


def test_floor_global(enc):
    obs = enc.encode(make_state(floor=11))
    assert obs[2] == pytest.approx(11 / 55)


def test_gold_global(enc):
    obs = enc.encode(make_state(gold=300))
    assert obs[3] == pytest.approx(300 / 999)


def test_energy_global(enc):
    obs = enc.encode(make_state(energy=2))
    assert obs[6] == pytest.approx(2 / 4)


def test_screen_onehot_combat(enc):
    obs = enc.encode(make_state(screen_type="NONE"))
    # Screen one-hot starts at index 43, NONE=combat is index 0
    assert obs[43] == pytest.approx(1.0)
    assert obs[44] == pytest.approx(0.0)


def test_screen_onehot_card_reward(enc):
    obs = enc.encode(make_card_reward())
    # CARD_REWARD is screen index 1 → global offset 44
    assert obs[43] == pytest.approx(0.0)
    assert obs[44] == pytest.approx(1.0)


def test_potion_slot_has_potion(enc):
    state = make_state(
        potions=[{"id": "Fire Potion", "can_use": True, "requires_target": False}]
    )
    obs = enc.encode(state)
    # Potion block starts at index 16; slot 0 = indices 16-19
    assert obs[16] == pytest.approx(1.0)  # has_potion


def test_potion_slot_empty(enc):
    obs = enc.encode(make_state(potions=[]))
    assert obs[16] == pytest.approx(0.0)


def test_global_block_size(enc):
    # Combat and non-combat blocks are zeroed during combat — only global non-zero in range 0:55
    # (combat block starts at 55 and will have non-zero values in combat)
    obs = enc.encode(make_state())
    assert obs.shape[0] == OBS_SIZE
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_encoder.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.v2.run_encoder'`

- [ ] **Step 3: Implement RunEncoder with global block**

`src/v2/run_encoder.py`:

```python
import numpy as np
from src.game_state import GameState
from src.card_properties import get_card_properties

GLOBAL_SIZE    = 55
COMBAT_SIZE    = 112
NONCOMBAT_SIZE = 60
OBS_SIZE       = GLOBAL_SIZE + COMBAT_SIZE + NONCOMBAT_SIZE  # 227

_SCREEN_ORDER = [
    "NONE", "CARD_REWARD", "REST", "MAP", "CHEST", "EVENT",
    "SHOP_ROOM", "SHOP_SCREEN", "GRID", "HAND_SELECT",
    "COMBAT_REWARD", "BOSS_REWARD",
]
_SCREEN_IDX = {s: i for i, s in enumerate(_SCREEN_ORDER)}

_HEALING_POTIONS = {
    "Fruit Juice", "Blood Potion", "Fairy in a Bottle",
    "Regen Potion", "Ancient Potion",
}
_ATTACK_POTIONS = {
    "Fire Potion", "Explosive Potion", "Poison Potion", "Fear Potion",
    "Strength Potion", "Dexterity Potion", "Speed Potion", "Weak Potion",
    "Energy Potion", "Swift Potion", "Flex Potion", "Steroid Potion",
    "Focus Potion", "Cultist Potion", "Liquid Bronze", "Essence of Steel",
    "Heart of Iron", "Ghost In A Jar", "Ambrosia", "Liquid Memories",
    "Distilled Chaos", "Duplication Potion", "Blessing of the Forge",
    "Elixir", "Gambler's Brew", "Entropic Brew", "Block Potion",
}

_HIGH_IMPACT_RELICS = [
    "Akabeko", "Anchor", "Burning Blood",
    "Centennial Puzzle", "Philosopher's Stone", "Astrolabe",
]

_STRENGTH_CARDS = {
    "Inflame", "Spot Weakness", "Demon Form", "Flex",
    "Limit Break", "Berserk",
}
_DRAW_CARDS = {
    "Battle Trance", "Pommel Strike", "Warcry", "Burning Pact",
    "Headbutt", "Exhume",
}
_EXHAUST_CARDS = {
    "True Grit", "Second Wind", "Corruption", "Fiend Fire",
    "Feel No Pain", "Dark Embrace", "Burning Pact", "Sentinel",
    "Exhume",
}


class RunEncoder:
    OBS_SIZE = OBS_SIZE

    def encode(self, state: GameState) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._encode_global(obs, state)
        if state.is_in_combat:
            self._encode_combat(obs, state)
        else:
            self._encode_noncombat(obs, state)
        return obs

    def _encode_global(self, obs: np.ndarray, state: GameState) -> None:
        max_hp = max(state.max_hp, 1)

        # [0:8] player stats
        obs[0] = state.current_hp / max_hp
        obs[1] = min(max_hp / 400, 1.0)
        obs[2] = state.floor / 55
        obs[3] = min(state.gold / 999, 1.0)
        obs[4] = state.act / 3
        obs[5] = state.ascension_level / 20
        obs[6] = state.energy / 4
        obs[7] = state.player_block / max_hp

        # [8:16] deck composition
        deck = state.deck
        n = max(len(deck), 1)
        n_attack  = sum(1 for c in deck if c.get("type") == "ATTACK")
        n_skill   = sum(1 for c in deck if c.get("type") == "SKILL")
        n_power   = sum(1 for c in deck if c.get("type") == "POWER")
        n_curse   = sum(1 for c in deck if c.get("type") in ("STATUS", "CURSE"))
        n_exhaust = sum(1 for c in deck if c.get("id", "") in _EXHAUST_CARDS)
        n_strength = sum(1 for c in deck if c.get("id", "") in _STRENGTH_CARDS)
        n_draw    = sum(1 for c in deck if c.get("id", "") in _DRAW_CARDS)

        obs[8]  = len(deck) / 60
        obs[9]  = n_attack / n
        obs[10] = n_skill / n
        obs[11] = n_power / n
        obs[12] = n_curse / n
        obs[13] = min(n_exhaust / 10, 1.0)
        obs[14] = min(n_strength / 5, 1.0)
        obs[15] = min(n_draw / 5, 1.0)

        # [16:36] potions: 5 slots × 4 features
        for i, potion in enumerate(state.potions[:5]):
            base = 16 + i * 4
            pid = potion.get("id", "")
            obs[base]     = 1.0 if pid else 0.0
            obs[base + 1] = 1.0 if pid in _HEALING_POTIONS else 0.0
            obs[base + 2] = 1.0 if pid in _ATTACK_POTIONS else 0.0
            obs[base + 3] = 1.0 if potion.get("requires_target", False) else 0.0

        # [36:43] relics
        relic_ids = {r.get("id", "") for r in state.relics}
        obs[36] = min(len(state.relics) / 20, 1.0)
        for j, relic_name in enumerate(_HIGH_IMPACT_RELICS):
            obs[37 + j] = 1.0 if relic_name in relic_ids else 0.0

        # [43:55] screen type one-hot
        idx = _SCREEN_IDX.get(state.screen_type, 0)
        obs[43 + idx] = 1.0

    def _encode_combat(self, obs: np.ndarray, state: GameState) -> None:
        raise NotImplementedError

    def _encode_noncombat(self, obs: np.ndarray, state: GameState) -> None:
        raise NotImplementedError
```

- [ ] **Step 4: Run global block tests**

```bash
pytest tests/v2/test_run_encoder.py -v -k "hp_ratio or floor or gold or energy or screen or potion or global_block or obs_shape or obs_values"
```

Expected: All selected tests pass. (`_encode_combat` and `_encode_noncombat` raise `NotImplementedError` but those tests are not selected yet.)

- [ ] **Step 5: Commit**

```bash
git add src/v2/run_encoder.py tests/v2/test_run_encoder.py
git commit -m "feat(v2): RunEncoder global block (55 features)"
```

---

## Task 5: RunEncoder — combat block

**Files:**
- Modify: `src/v2/run_encoder.py`
- Modify: `tests/v2/test_run_encoder.py`

- [ ] **Step 1: Add failing combat block tests**

Append to `tests/v2/test_run_encoder.py`:

```python
INTENT_MAP = {
    "ATTACK": 0.2, "ATTACK_BUFF": 0.3, "ATTACK_DEBUFF": 0.35,
    "ATTACK_DEFEND": 0.4, "BUFF": 0.5, "DEBUFF": 0.6,
    "STRONG_DEBUFF": 0.65, "DEFEND": 0.7, "DEFEND_BUFF": 0.75,
    "ESCAPE": 0.8, "MAGIC": 0.85, "SLEEP": 0.1, "STUN": 0.9,
    "UNKNOWN": 0.5, "NONE": 0.0,
}
CARD_TYPE_MAP = {"ATTACK": 0.25, "SKILL": 0.5, "POWER": 0.75,
                 "STATUS": 0.9, "CURSE": 1.0}

COMBAT_BASE = GLOBAL_SIZE  # 55


def test_combat_block_zeroed_on_noncombat(enc):
    obs = enc.encode(make_card_reward())
    assert obs[COMBAT_BASE:COMBAT_BASE + COMBAT_SIZE].sum() == pytest.approx(0.0)


def test_hand_card_cost_encoded(enc):
    state = make_state(
        hand=[{"id": "Strike_R", "cost": 1, "type": "ATTACK",
               "is_playable": True, "has_target": True}]
    )
    obs = enc.encode(state)
    # Hand starts at COMBAT_BASE (55); card 0 feature 0 = cost/5
    assert obs[COMBAT_BASE] == pytest.approx(1 / 5)


def test_hand_card_type_encoded(enc):
    state = make_state(
        hand=[{"id": "Inflame", "cost": 1, "type": "POWER",
               "is_playable": True, "has_target": False}]
    )
    obs = enc.encode(state)
    assert obs[COMBAT_BASE + 1] == pytest.approx(CARD_TYPE_MAP["POWER"])


def test_hand_card_playable_flag(enc):
    state = make_state(
        hand=[{"id": "Strike_R", "cost": 1, "type": "ATTACK",
               "is_playable": True, "has_target": True}]
    )
    obs = enc.encode(state)
    assert obs[COMBAT_BASE + 2] == pytest.approx(1.0)


def test_hand_card_applies_vulnerable(enc):
    state = make_state(
        hand=[{"id": "Bash", "cost": 2, "type": "ATTACK",
               "is_playable": True, "has_target": True}]
    )
    obs = enc.encode(state)
    # Bash applies_vulnerable → feature index 3 of card 0
    assert obs[COMBAT_BASE + 3] == pytest.approx(1.0)


def test_monster_hp_ratio(enc):
    monsters = [{"name": "Worm", "current_hp": 21, "max_hp": 42,
                 "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}]
    obs = enc.encode(make_state(monsters=monsters))
    # Monster block starts at COMBAT_BASE + 70 (hand) = 125; monster 0 feature 0 = hp_ratio
    assert obs[125] == pytest.approx(0.5)


def test_monster_intent_encoded(enc):
    monsters = [{"name": "Worm", "current_hp": 42, "max_hp": 42,
                 "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}]
    obs = enc.encode(make_state(monsters=monsters))
    # Monster 0 feature 3 = intent, starts at 125+3=128
    assert obs[128] == pytest.approx(INTENT_MAP["ATTACK"])


def test_gone_monster_zeroed(enc):
    monsters = [
        {"name": "Dead", "current_hp": 0, "max_hp": 42,
         "block": 0, "intent": "NONE", "is_gone": True, "powers": []},
        {"name": "Alive", "current_hp": 30, "max_hp": 42,
         "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []},
    ]
    obs = enc.encode(make_state(monsters=monsters))
    # Monster 0 (gone) should be zeroed
    assert obs[125] == pytest.approx(0.0)
    # Monster 1 (alive) should have hp_ratio
    assert obs[125 + 6] == pytest.approx(30 / 42)


def test_player_power_strength(enc):
    state = make_state(
        powers=[{"id": "Strength", "amount": 3}]
    )
    obs = enc.encode(state)
    # Player powers start at COMBAT_BASE + 70 + 30 = 155
    assert obs[155] == pytest.approx(3 / 10)


def test_turn_metadata(enc):
    state = make_state(
        draw_pile=[{"id": "Strike_R"}] * 5,
        discard_pile=[{"id": "Defend_R"}] * 3,
        turn=2,
    )
    obs = enc.encode(state)
    # Turn metadata at 155 + 5 = 160
    assert obs[160] == pytest.approx(5 / 60)
    assert obs[161] == pytest.approx(3 / 60)
    assert obs[162] == pytest.approx(2 / 20)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_encoder.py -v -k "combat" 2>&1 | head -15
```

Expected: `NotImplementedError` on `_encode_combat`.

- [ ] **Step 3: Implement _encode_combat**

Replace the `_encode_combat` stub in `src/v2/run_encoder.py`:

```python
    INTENT_MAP = {
        "ATTACK": 0.2, "ATTACK_BUFF": 0.3, "ATTACK_DEBUFF": 0.35,
        "ATTACK_DEFEND": 0.4, "BUFF": 0.5, "DEBUFF": 0.6,
        "STRONG_DEBUFF": 0.65, "DEFEND": 0.7, "DEFEND_BUFF": 0.75,
        "ESCAPE": 0.8, "MAGIC": 0.85, "SLEEP": 0.1, "STUN": 0.9,
        "UNKNOWN": 0.5, "NONE": 0.0,
    }
    CARD_TYPE_MAP = {
        "ATTACK": 0.25, "SKILL": 0.5, "POWER": 0.75,
        "STATUS": 0.9, "CURSE": 1.0,
    }

    def _encode_combat(self, obs: np.ndarray, state: GameState) -> None:
        base = GLOBAL_SIZE  # 55

        # Hand: 10 cards × 7 features [55:125]
        for i, card in enumerate(state.hand[:10]):
            b = base + i * 7
            props = get_card_properties(card.get("id", ""))
            obs[b]     = min(card.get("cost", 0), 5) / 5
            obs[b + 1] = self.CARD_TYPE_MAP.get(card.get("type", ""), 0.5)
            obs[b + 2] = 1.0 if card.get("is_playable", False) else 0.0
            obs[b + 3] = 1.0 if props["applies_vulnerable"] else 0.0
            obs[b + 4] = 1.0 if props["applies_weak"]       else 0.0
            obs[b + 5] = 1.0 if props["draws_cards"]        else 0.0
            obs[b + 6] = 1.0 if props["gains_block"]        else 0.0

        # Monsters: 5 × 6 features [125:155]
        monster_base = base + 70
        for i, m in enumerate(state.monsters[:5]):
            if m.get("is_gone", False):
                continue
            b = monster_base + i * 6
            m_max  = max(m.get("max_hp", 1), 1)
            powers = m.get("powers", [])
            vuln   = next((p.get("amount", 0) for p in powers if p.get("id") == "Vulnerable"), 0)
            weak   = next((p.get("amount", 0) for p in powers if p.get("id") == "Weak"), 0)
            obs[b]     = m.get("current_hp", 0) / m_max
            obs[b + 1] = min(m_max / 400, 1.0)
            obs[b + 2] = m.get("block", 0) / m_max
            obs[b + 3] = self.INTENT_MAP.get(m.get("intent", "UNKNOWN"), 0.5)
            obs[b + 4] = min(vuln / 10, 1.0)
            obs[b + 5] = min(weak / 10, 1.0)

        # Player powers [155:160]
        power_base = monster_base + 30  # 155
        player_powers = state.combat_state.get("player", {}).get("powers", []) if state.combat_state else []
        def _pwr(name):
            return next((p.get("amount", 0) for p in player_powers if p.get("id") == name), 0)
        obs[power_base]     = min(_pwr("Strength") / 10, 1.0)
        obs[power_base + 1] = min(_pwr("Dexterity") / 10, 1.0)
        obs[power_base + 2] = min(_pwr("Weak") / 5, 1.0)
        obs[power_base + 3] = min(_pwr("Vulnerable") / 5, 1.0)
        obs[power_base + 4] = 1.0 if any(p.get("id") == "Barricade" for p in player_powers) else 0.0

        # Turn metadata [160:163]
        meta_base = power_base + 5  # 160
        obs[meta_base]     = min(len(state.draw_pile) / 60, 1.0)
        obs[meta_base + 1] = min(len(state.discard_pile) / 60, 1.0)
        obs[meta_base + 2] = min(state.turn / 20, 1.0)

        # Debuff signal [163:166] — computed as fraction of hand that applies debuffs
        debuff_base = meta_base + 3  # 163
        n_hand = max(len(state.hand), 1)
        n_debuff_cards = sum(
            1 for c in state.hand
            if get_card_properties(c.get("id", "")).get("applies_vulnerable") or
               get_card_properties(c.get("id", "")).get("applies_weak")
        )
        obs[debuff_base] = n_debuff_cards / n_hand
        # obs[debuff_base+1] and obs[debuff_base+2] reserved for damage potential (future)
```

- [ ] **Step 4: Run combat block tests**

```bash
pytest tests/v2/test_run_encoder.py -v -k "combat or hand or monster or player_power or turn_meta"
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/v2/run_encoder.py tests/v2/test_run_encoder.py
git commit -m "feat(v2): RunEncoder combat block (112 features)"
```

---

## Task 6: RunEncoder — non-combat block

**Files:**
- Modify: `src/v2/run_encoder.py`
- Modify: `tests/v2/test_run_encoder.py`

- [ ] **Step 1: Add failing non-combat block tests**

Append to `tests/v2/test_run_encoder.py`:

```python
from tests.v2.helpers import make_shop, make_rest, make_map

NONCOMBAT_BASE = GLOBAL_SIZE + COMBAT_SIZE  # 167


def test_noncombat_block_zeroed_during_combat(enc):
    obs = enc.encode(make_state())
    assert obs[NONCOMBAT_BASE:].sum() == pytest.approx(0.0)


def test_card_reward_choice_tier_value(enc):
    # Inflame is A-tier → tier_value = 0.8
    state = make_card_reward(
        cards=[{"id": "Inflame", "name": "Inflame", "type": "POWER"}]
    )
    obs = enc.encode(state)
    # Choice 0 feature 0 = tier_value at NONCOMBAT_BASE
    assert obs[NONCOMBAT_BASE] == pytest.approx(0.8)


def test_card_reward_is_available_flag(enc):
    state = make_card_reward(
        cards=[{"id": "Inflame", "name": "Inflame", "type": "POWER"}]
    )
    obs = enc.encode(state)
    # Choice 0 feature 3 = is_available = 1.0
    assert obs[NONCOMBAT_BASE + 3] == pytest.approx(1.0)


def test_shop_cost_ratio(enc):
    # 1 card priced 75, gold 150 → cost_ratio = 75/150 = 0.5
    state = make_shop(
        cards=[{"id": "Inflame", "price": 75, "is_in_stock": True, "type": "POWER"}],
        relics=[],
        gold=150,
    )
    obs = enc.encode(state)
    # Choice 0 feature 2 = cost_ratio
    assert obs[NONCOMBAT_BASE + 2] == pytest.approx(0.5)


def test_rest_heal_ratio_in_metadata(enc):
    # REST: hp=50, max_hp=80 → heal = 80*0.3=24, capped at 80-50=30 → 24
    state = make_rest(hp=50, max_hp=80)
    obs = enc.encode(state)
    # Screen metadata starts at NONCOMBAT_BASE + 37 (32 choices + 5 synergy)
    hp_heal_idx = NONCOMBAT_BASE + 32 + 5 + 1
    assert obs[hp_heal_idx] == pytest.approx(24 / 80)


def test_map_node_elite_flag(enc):
    state = make_map(nodes=[{"symbol": "E"}, {"symbol": "M"}])
    obs = enc.encode(state)
    # Screen metadata feature 2 = node_elite_flag
    node_elite_idx = NONCOMBAT_BASE + 32 + 5 + 2
    assert obs[node_elite_idx] == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_encoder.py -v -k "noncombat or card_reward_choice or shop_cost or rest_heal or map_node" 2>&1 | head -15
```

Expected: `NotImplementedError`.

- [ ] **Step 3: Implement _encode_noncombat**

Replace the `_encode_noncombat` stub in `src/v2/run_encoder.py`. Also add the import at the top: `from src.card_tier_list import get_card_tier`.

```python
_TIER_VALUE = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}

    def _encode_noncombat(self, obs: np.ndarray, state: GameState) -> None:
        base = GLOBAL_SIZE + COMBAT_SIZE  # 167
        ss   = state.screen_state or {}
        screen = state.screen_type

        # Choices block [167:199]: 8 × 4 features
        choices = self._get_choices(screen, ss, state)
        for i, choice in enumerate(choices[:8]):
            b = base + i * 4
            obs[b]     = choice.get("tier_value",    0.0)
            obs[b + 1] = choice.get("synergy_score", 0.0)
            obs[b + 2] = choice.get("cost_ratio",    0.0)
            obs[b + 3] = choice.get("is_available",  1.0)

        # Deck synergy context [199:204]
        syn_base = base + 32
        deck_ids = [c.get("id", "") for c in state.deck]
        obs[syn_base]     = min(sum(1 for d in deck_ids if d in _EXHAUST_CARDS) / 10, 1.0)
        obs[syn_base + 1] = min(sum(1 for d in deck_ids if d in _STRENGTH_CARDS) / 5, 1.0)
        obs[syn_base + 2] = min(sum(1 for d in deck_ids if d in _DRAW_CARDS) / 5, 1.0)
        n_block = sum(1 for d in deck_ids
                      if get_card_properties(d).get("gains_block"))
        obs[syn_base + 3] = min(n_block / 10, 1.0)
        n_curse = sum(1 for c in state.deck if c.get("type") in ("STATUS", "CURSE"))
        obs[syn_base + 4] = min(n_curse / 10, 1.0)

        # Screen metadata [204:212]
        meta_base = syn_base + 5  # 204
        max_hp = max(state.max_hp, 1)

        if screen == "SHOP_SCREEN":
            min_price = min(
                (c.get("price", 9999) for c in ss.get("cards", []) + ss.get("relics", [])
                 if c.get("is_in_stock", True)),
                default=0,
            )
            obs[meta_base] = min(min_price / max(state.gold, 1), 1.0)

        if screen == "REST":
            heal = min(int(max_hp * 0.3), max_hp - state.current_hp)
            obs[meta_base + 1] = max(heal, 0) / max_hp

        if screen == "MAP":
            nodes = ss.get("next_nodes", [])
            symbols = [n.get("symbol", "") for n in nodes]
            obs[meta_base + 2] = 1.0 if "E" in symbols else 0.0
            obs[meta_base + 3] = 1.0 if "R" in symbols else 0.0
            obs[meta_base + 4] = 1.0 if "$" in symbols else 0.0
            obs[meta_base + 5] = 1.0 if "T" in symbols else 0.0
            obs[meta_base + 6] = 1.0 if "?" in symbols else 0.0
            obs[meta_base + 7] = 1.0 if "M" in symbols else 0.0

    def _get_choices(self, screen: str, ss: dict, state: GameState) -> list:
        gold = max(state.gold, 1)
        deck = state.deck

        if screen == "CARD_REWARD":
            return [
                {
                    "tier_value":    _TIER_VALUE.get(get_card_tier(c.get("id", "")), 0.4),
                    "synergy_score": self._synergy(c.get("id", ""), deck),
                    "cost_ratio":    0.0,
                    "is_available":  1.0,
                }
                for c in ss.get("cards", [])
            ]

        if screen == "SHOP_SCREEN":
            items = []
            for c in ss.get("cards", []):
                items.append({
                    "tier_value":    _TIER_VALUE.get(get_card_tier(c.get("id", "")), 0.4),
                    "synergy_score": self._synergy(c.get("id", ""), deck),
                    "cost_ratio":    min(c.get("price", 0) / gold, 1.0),
                    "is_available":  1.0 if c.get("is_in_stock", True) else 0.0,
                })
            for r in ss.get("relics", []):
                items.append({
                    "tier_value":    0.6,
                    "synergy_score": 0.0,
                    "cost_ratio":    min(r.get("price", 0) / gold, 1.0),
                    "is_available":  1.0 if r.get("is_in_stock", True) else 0.0,
                })
            return items

        return []

    def _synergy(self, card_id: str, deck: list) -> float:
        props    = get_card_properties(card_id)
        deck_ids = [c.get("id", "") for c in deck]
        score    = 0.0

        if card_id in _STRENGTH_CARDS:
            if sum(1 for d in deck_ids if d in _STRENGTH_CARDS) >= 2:
                score += 0.3

        if card_id in _EXHAUST_CARDS:
            if sum(1 for d in deck_ids if d in _EXHAUST_CARDS) >= 2:
                score += 0.3

        if props.get("applies_vulnerable"):
            high_damage = {"Carnage", "Whirlwind", "Reaper", "Immolate",
                           "Fiend Fire", "Hemokinesis", "Blood for Blood"}
            if sum(1 for d in deck_ids if d in high_damage) >= 2:
                score += 0.3

        if card_id in _DRAW_CARDS:
            score += 0.2

        return min(score, 1.0)
```

- [ ] **Step 4: Run all encoder tests**

```bash
pytest tests/v2/test_run_encoder.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/v2/run_encoder.py tests/v2/test_run_encoder.py
git commit -m "feat(v2): RunEncoder non-combat block (60 features)"
```

---

## Task 7: RunRewardShaper — terminal and combat rewards

**Files:**
- Create: `src/v2/run_reward.py`
- Create: `tests/v2/test_run_reward.py`

- [ ] **Step 1: Write failing tests**

`tests/v2/test_run_reward.py`:

```python
import pytest
from src.v2.run_reward import RunRewardShaper


@pytest.fixture
def shaper():
    return RunRewardShaper()


# ---- terminal reward ----

def test_terminal_floor_0(shaper):
    assert shaper.terminal_reward(0) == pytest.approx(-1.0)


def test_terminal_floor_55_win(shaper):
    assert shaper.terminal_reward(55) == pytest.approx(2.0)


def test_terminal_floor_27_midpoint(shaper):
    expected = (27 / 55) * 3.0 - 1.0
    assert shaper.terminal_reward(27) == pytest.approx(expected)


def test_terminal_increases_with_floor(shaper):
    r10 = shaper.terminal_reward(10)
    r20 = shaper.terminal_reward(20)
    r40 = shaper.terminal_reward(40)
    assert r10 < r20 < r40


# ---- combat step reward ----

def _combat_reward(shaper, **kwargs):
    defaults = dict(
        prev_hp=70, new_hp=70,
        prev_monster_hp=42, new_monster_hp=42,
        prev_living=1, new_living=1,
        prev_debuffs=0, new_debuffs=0,
        max_hp=80, is_end_action=False,
        energy_remaining=3, max_energy=3,
        card_is_attack=False,
        debuff_applied_this_turn=False,
    )
    defaults.update(kwargs)
    return shaper.combat_step_reward(**defaults)


def test_no_change_no_reward(shaper):
    assert _combat_reward(shaper) == pytest.approx(0.0)


def test_damage_dealt_positive(shaper):
    r = _combat_reward(shaper, prev_monster_hp=42, new_monster_hp=30)
    assert r == pytest.approx(12 / 80)


def test_damage_taken_negative(shaper):
    r = _combat_reward(shaper, prev_hp=70, new_hp=60)
    assert r == pytest.approx(-10 / 80)


def test_kill_bonus(shaper):
    r = _combat_reward(shaper, prev_living=2, new_living=1,
                       prev_monster_hp=42, new_monster_hp=0)
    assert r == pytest.approx(42 / 80 + 0.1)


def test_debuff_gain_reward(shaper):
    r = _combat_reward(shaper, prev_debuffs=0, new_debuffs=2)
    assert r == pytest.approx(0.05 * 2)


def test_energy_waste_penalty_on_end(shaper):
    # End with 2 energy remaining out of 3 → -0.3 * (2/3)
    r = _combat_reward(shaper, is_end_action=True, energy_remaining=2, max_energy=3)
    assert r == pytest.approx(-0.3 * (2 / 3))


def test_no_energy_penalty_on_card_play(shaper):
    # Playing a card, not ending — no energy penalty
    r = _combat_reward(shaper, is_end_action=False, energy_remaining=2, max_energy=3)
    assert r == pytest.approx(0.0)


def test_debuff_before_damage_bonus(shaper):
    # Attack card, debuff was applied earlier this turn
    r = _combat_reward(shaper,
                       prev_monster_hp=42, new_monster_hp=36,
                       card_is_attack=True,
                       debuff_applied_this_turn=True)
    assert r == pytest.approx(6 / 80 + 0.03)


def test_no_debuff_bonus_without_prior_debuff(shaper):
    r = _combat_reward(shaper,
                       prev_monster_hp=42, new_monster_hp=36,
                       card_is_attack=True,
                       debuff_applied_this_turn=False)
    assert r == pytest.approx(6 / 80)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_reward.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.v2.run_reward'`

- [ ] **Step 3: Implement terminal and combat rewards**

`src/v2/run_reward.py`:

```python
class RunRewardShaper:

    def terminal_reward(self, floor: int) -> float:
        return (floor / 55) * 3.0 - 1.0

    def combat_step_reward(
        self,
        prev_hp: int, new_hp: int,
        prev_monster_hp: int, new_monster_hp: int,
        prev_living: int, new_living: int,
        prev_debuffs: int, new_debuffs: int,
        max_hp: int,
        is_end_action: bool,
        energy_remaining: int, max_energy: int,
        card_is_attack: bool,
        debuff_applied_this_turn: bool,
    ) -> float:
        max_hp = max(max_hp, 1)

        damage_dealt = max(prev_monster_hp - new_monster_hp, 0) / max_hp
        damage_taken = max(prev_hp - new_hp, 0) / max_hp
        kills        = max(prev_living - new_living, 0)
        debuff_gain  = max(new_debuffs - prev_debuffs, 0)

        reward = (
            damage_dealt
            - damage_taken
            + 0.1 * kills
            + 0.05 * debuff_gain
        )

        if is_end_action:
            reward -= 0.3 * (energy_remaining / max(max_energy, 1))

        if card_is_attack and debuff_applied_this_turn:
            reward += 0.03

        return reward
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/v2/test_run_reward.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/v2/run_reward.py tests/v2/test_run_reward.py
git commit -m "feat(v2): RunRewardShaper terminal and combat rewards"
```

---

## Task 8: RunRewardShaper — non-combat rewards

**Files:**
- Modify: `src/v2/run_reward.py`
- Modify: `tests/v2/test_run_reward.py`

- [ ] **Step 1: Add failing non-combat reward tests**

Append to `tests/v2/test_run_reward.py`:

```python
# ---- non-combat rewards ----

def test_card_reward_s_tier(shaper):
    card = {"id": "Offering"}  # S-tier
    r = shaper.card_reward(card, deck=[])
    assert r == pytest.approx(1.0 * 0.05)  # tier_value=1.0, synergy=0


def test_card_reward_d_tier(shaper):
    card = {"id": "Strike_R"}  # D-tier
    r = shaper.card_reward(card, deck=[])
    assert r == pytest.approx(0.2 * 0.05)


def test_shop_card_reward_includes_cost_penalty(shaper):
    card = {"id": "Inflame", "price": 75}  # A-tier
    r = shaper.shop_card_reward(card, gold=150, deck=[])
    # tier=0.8, synergy=0, cost_ratio=0.5
    assert r == pytest.approx(0.8 * 0.05 + 0.0 * 0.05 - 0.5 * 0.02)


def test_shop_relic_reward(shaper):
    assert shaper.shop_relic_reward() == pytest.approx(0.05)


def test_purge_d_tier_card(shaper):
    card = {"id": "Strike_R", "type": "ATTACK"}  # D-tier
    assert shaper.purge_reward(card) == pytest.approx(0.03)


def test_purge_b_tier_card_no_reward(shaper):
    card = {"id": "Thunderclap", "type": "ATTACK"}  # B-tier
    assert shaper.purge_reward(card) == pytest.approx(0.0)


def test_purge_curse_card(shaper):
    card = {"id": "Curse of the Bell", "type": "CURSE"}
    assert shaper.purge_reward(card) == pytest.approx(0.03)


def test_rest_heal_reward(shaper):
    r = shaper.rest_heal_reward(hp_gained=24, max_hp=80)
    assert r == pytest.approx(24 / 80 * 0.2)


def test_rest_smith_reward(shaper):
    assert shaper.rest_smith_reward() == pytest.approx(0.05)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_reward.py -v -k "card_reward or shop or purge or rest" 2>&1 | head -10
```

Expected: `AttributeError: 'RunRewardShaper' object has no attribute 'card_reward'`

- [ ] **Step 3: Implement non-combat rewards**

Append to `src/v2/run_reward.py`:

```python
from src.card_tier_list import get_card_tier

_TIER_VALUE = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}


class RunRewardShaper:
    # ... existing methods above ...

    def card_reward(self, card: dict, deck: list) -> float:
        tier_val = _TIER_VALUE.get(get_card_tier(card.get("id", "")), 0.4)
        syn      = self._synergy(card.get("id", ""), deck)
        return tier_val * 0.05 + syn * 0.05

    def shop_card_reward(self, card: dict, gold: int, deck: list) -> float:
        tier_val   = _TIER_VALUE.get(get_card_tier(card.get("id", "")), 0.4)
        syn        = self._synergy(card.get("id", ""), deck)
        cost_ratio = min(card.get("price", 0) / max(gold, 1), 1.0)
        return tier_val * 0.05 + syn * 0.05 - cost_ratio * 0.02

    def shop_relic_reward(self) -> float:
        return 0.05

    def purge_reward(self, card: dict) -> float:
        tier = get_card_tier(card.get("id", ""))
        if tier == "D" or card.get("type") in ("STATUS", "CURSE"):
            return 0.03
        return 0.0

    def rest_heal_reward(self, hp_gained: int, max_hp: int) -> float:
        return (hp_gained / max(max_hp, 1)) * 0.2

    def rest_smith_reward(self) -> float:
        return 0.05

    def _synergy(self, card_id: str, deck: list) -> float:
        from src.card_properties import get_card_properties
        _STRENGTH_CARDS = {"Inflame", "Spot Weakness", "Demon Form", "Flex", "Limit Break", "Berserk"}
        _EXHAUST_CARDS  = {"True Grit", "Second Wind", "Corruption", "Fiend Fire",
                           "Feel No Pain", "Dark Embrace", "Burning Pact", "Sentinel", "Exhume"}
        _DRAW_CARDS     = {"Battle Trance", "Pommel Strike", "Warcry", "Burning Pact", "Headbutt", "Exhume"}
        _HIGH_DAMAGE    = {"Carnage", "Whirlwind", "Reaper", "Immolate",
                           "Fiend Fire", "Hemokinesis", "Blood for Blood"}

        props    = get_card_properties(card_id)
        deck_ids = [c.get("id", "") for c in deck]
        score    = 0.0

        if card_id in _STRENGTH_CARDS:
            if sum(1 for d in deck_ids if d in _STRENGTH_CARDS) >= 2:
                score += 0.3
        if card_id in _EXHAUST_CARDS:
            if sum(1 for d in deck_ids if d in _EXHAUST_CARDS) >= 2:
                score += 0.3
        if props.get("applies_vulnerable"):
            if sum(1 for d in deck_ids if d in _HIGH_DAMAGE) >= 2:
                score += 0.3
        if card_id in _DRAW_CARDS:
            score += 0.2

        return min(score, 1.0)
```

- [ ] **Step 4: Run all reward tests**

```bash
pytest tests/v2/test_run_reward.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/v2/run_reward.py tests/v2/test_run_reward.py
git commit -m "feat(v2): RunRewardShaper non-combat rewards and synergy scoring"
```

---

## Task 9: RunEnv — reset() and _next_actionable_state()

**Files:**
- Create: `src/v2/run_env.py`
- Create: `tests/v2/test_run_env.py`

- [ ] **Step 1: Write failing tests for reset()**

`tests/v2/test_run_env.py`:

```python
import pytest
from unittest.mock import MagicMock, call
from src.v2.run_env import RunEnv
from src.v2.run_encoder import RunEncoder
from src.v2.run_action_space import RunActionSpace
from tests.v2.helpers import make_state, make_game_over, make_card_reward


@pytest.fixture
def env():
    return RunEnv(communicator=MagicMock())


def test_observation_space_shape(env):
    assert env.observation_space.shape == (RunEncoder.OBS_SIZE,)


def test_action_space_size(env):
    assert env.action_space.n == RunActionSpace.TOTAL_ACTIONS


def test_action_masks_all_ones_before_reset(env):
    import numpy as np
    assert env.action_masks().all()


def test_reset_sends_ready_on_first_call(env):
    env.communicator.receive_state.return_value = make_state()
    env.reset()
    env.communicator.send_ready.assert_called_once()


def test_reset_does_not_send_ready_on_second_call(env):
    env.communicator.receive_state.return_value = make_state()
    env.reset()
    env.reset()
    env.communicator.send_ready.assert_called_once()


def test_reset_returns_obs_of_correct_shape(env):
    env.communicator.receive_state.return_value = make_state()
    obs, info = env.reset()
    assert obs.shape == (RunEncoder.OBS_SIZE,)
    assert isinstance(info, dict)


def test_reset_skips_not_ready_states(env):
    import json
    from src.game_state import GameState
    not_ready = GameState.from_json(json.dumps({
        "available_commands": [],
        "ready_for_command": False,
        "in_game": True,
        "game_state": {
            "screen_type": "NONE", "seed": 1, "floor": 1,
            "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": 70, "max_hp": 80, "gold": 99, "act": 1,
            "deck": [], "relics": [], "potions": [], "map": [],
            "combat_state": None,
        }
    }))
    env.communicator.receive_state.side_effect = [not_ready, make_state()]
    obs, _ = env.reset()
    assert env.communicator.receive_state.call_count == 2


def test_reset_sends_start_when_not_in_game(env):
    import json
    from src.game_state import GameState
    main_menu = GameState.from_json(json.dumps({
        "available_commands": ["START"],
        "ready_for_command": True,
        "in_game": False,
        "game_state": None,
    }))
    env.communicator.receive_state.side_effect = [main_menu, make_state()]
    env.reset()
    env.communicator.send_command.assert_any_call("START IRONCLAD 0")
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_env.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.v2.run_env'`

- [ ] **Step 3: Implement RunEnv with reset()**

`src/v2/run_env.py`:

```python
import logging
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.communicator import Communicator
from src.game_state import GameState
from src.run_tracker import RunTracker
from src.v2.run_encoder import RunEncoder
from src.v2.run_action_space import RunActionSpace
from src.v2.run_reward import RunRewardShaper

logger = logging.getLogger(__name__)


class RunEnv(gym.Env):
    """Gymnasium env for STS. One episode = one complete run."""
    metadata = {"render_modes": []}

    def __init__(self, communicator: Communicator,
                 run_tracker: Optional[RunTracker] = None):
        super().__init__()
        self.communicator = communicator
        self.run_tracker  = run_tracker or RunTracker()
        self.encoder      = RunEncoder()
        self._action_space_helper = RunActionSpace()
        self.reward_shaper = RunRewardShaper()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(RunEncoder.OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(RunActionSpace.TOTAL_ACTIONS)

        self._current_state: Optional[GameState] = None
        self._initialized: bool = False
        self._debuff_applied_this_turn: bool = False
        self._current_turn: int = 0

    def action_masks(self) -> np.ndarray:
        if self._current_state is None:
            return np.ones(RunActionSpace.TOTAL_ACTIONS, dtype=np.bool_)
        return self._action_space_helper.get_action_mask(self._current_state)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._debuff_applied_this_turn = False
        self._current_turn = 0

        if not self._initialized:
            self.communicator.send_ready()
            self._initialized = True

        state = self._next_actionable_state()
        self._current_state = state
        return self.encoder.encode(state), {}

    def _next_actionable_state(self) -> GameState:
        while True:
            state = self.communicator.receive_state()
            if state is None:
                raise RuntimeError("Connection closed while waiting for actionable state")
            if state.error:
                logger.warning("Error from game: %s", state.error)
                continue
            if not state.ready_for_command:
                continue
            if not state.in_game:
                if "START" in state.available_commands:
                    self.communicator.send_command("START IRONCLAD 0")
                continue
            return state

    def step(self, action: int):
        raise NotImplementedError
```

- [ ] **Step 4: Run reset tests**

```bash
pytest tests/v2/test_run_env.py -v -k "reset or obs_shape or action_space or action_masks"
```

Expected: All selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/v2/run_env.py tests/v2/test_run_env.py
git commit -m "feat(v2): RunEnv skeleton + reset()"
```

---

## Task 10: RunEnv — step() combat path

**Files:**
- Modify: `src/v2/run_env.py`
- Modify: `tests/v2/test_run_env.py`

- [ ] **Step 1: Add failing combat step tests**

Append to `tests/v2/test_run_env.py`:

```python
from tests.v2.helpers import make_state, make_card_reward, make_game_over


def _make_env(next_states):
    env = RunEnv(communicator=MagicMock())
    env.communicator.receive_state.side_effect = next_states
    return env


def test_step_combat_continues(env):
    before = make_state(hp=70, max_hp=80, energy=3,
                        monsters=[{"name": "Worm", "current_hp": 42, "max_hp": 42,
                                   "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}])
    after  = make_state(hp=65, max_hp=80, energy=2,
                        monsters=[{"name": "Worm", "current_hp": 36, "max_hp": 42,
                                   "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}])
    env.communicator.receive_state.return_value = after
    env._current_state = before

    obs, reward, done, truncated, info = env.step(10)  # PLAY 1 0

    assert done is False
    assert truncated is False
    assert obs.shape == (RunEncoder.OBS_SIZE,)
    # damage_dealt = 6/80, damage_taken = 5/80
    assert reward == pytest.approx(6 / 80 - 5 / 80)


def test_step_end_with_energy_waste_penalty(env):
    before = make_state(hp=70, max_hp=80, energy=3)
    after  = make_state(hp=70, max_hp=80, energy=0)
    env.communicator.receive_state.return_value = after
    env._current_state = before

    _, reward, done, _, _ = env.step(60)  # END

    # energy_waste = 3/3 → penalty = -0.3 * 1.0
    assert reward == pytest.approx(-0.3)


def test_step_sends_correct_command(env):
    env.communicator.receive_state.return_value = make_state()
    env._current_state = make_state(
        hand=[{"id": "Strike_R", "cost": 1, "type": "ATTACK",
               "is_playable": True, "has_target": True}]
    )
    env.step(10)  # PLAY 1 0
    env.communicator.send_command.assert_called_with("PLAY 1 0")


def test_step_debuff_tracking_reset_on_new_turn(env):
    env._debuff_applied_this_turn = True
    env._current_turn = 1
    # New state has turn=2 → tracker should reset
    before = make_state(turn=2)
    after  = make_state(turn=2)
    env.communicator.receive_state.return_value = after
    env._current_state = before

    env.step(60)  # END

    assert env._debuff_applied_this_turn is False


def test_step_debuff_flag_set_when_debuff_card_played(env):
    before = make_state(
        hand=[{"id": "Bash", "cost": 2, "type": "ATTACK",
               "is_playable": True, "has_target": True}],
        monsters=[{"name": "Worm", "current_hp": 42, "max_hp": 42,
                   "block": 0, "intent": "ATTACK", "is_gone": False, "powers": []}],
    )
    after = make_state(
        monsters=[{"name": "Worm", "current_hp": 34, "max_hp": 42,
                   "block": 0, "intent": "ATTACK", "is_gone": False,
                   "powers": [{"id": "Vulnerable", "amount": 2}]}],
    )
    env.communicator.receive_state.return_value = after
    env._current_state = before
    env._debuff_applied_this_turn = False

    env.step(10)  # PLAY 1 0 (Bash)

    assert env._debuff_applied_this_turn is True
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_env.py -v -k "step_combat or step_end or step_sends or step_debuff" 2>&1 | head -10
```

Expected: `NotImplementedError` from step().

- [ ] **Step 3: Implement step() for combat**

Replace the `step` stub in `src/v2/run_env.py`:

```python
    def step(self, action: int):
        assert self._current_state is not None, "Call reset() first"

        # Reset debuff tracking when turn advances
        if (self._current_state.is_in_combat and
                self._current_state.turn != self._current_turn):
            self._debuff_applied_this_turn = False
            self._current_turn = self._current_state.turn

        prev = self._current_state
        command = self._action_space_helper.action_to_command(action, prev)
        logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                    prev.floor, prev.current_hp, prev.max_hp,
                    prev.screen_type, command)
        self.communicator.send_command(command)

        state = self.communicator.receive_state()
        if state is None:
            return self.encoder.encode(prev), 0.0, True, False, {}

        if state.screen_type == "GAME_OVER":
            return self._handle_game_over(prev, state)

        reward = self._compute_reward(action, prev, state)

        # Update debuff tracking after computing reward (so bonus fires this step)
        if prev.is_in_combat:
            self._update_debuff_tracking(action, prev)

        self._current_state = state
        return self.encoder.encode(state), reward, False, False, {}

    def _compute_reward(self, action: int, prev: GameState, new: GameState) -> float:
        if prev.is_in_combat:
            return self._combat_reward(action, prev, new)
        return self._noncombat_reward(action, prev, new)

    def _combat_reward(self, action: int, prev: GameState, new: GameState) -> float:
        def debuff_stacks(monsters):
            total = 0
            for m in monsters:
                if m.get("is_gone"):
                    continue
                for p in m.get("powers", []):
                    if p.get("id") in ("Vulnerable", "Weak"):
                        total += p.get("amount", 0)
            return total

        new_monsters = new.monsters if new.is_in_combat else []
        card_is_attack = self._played_card_is_attack(action, prev)

        return self.reward_shaper.combat_step_reward(
            prev_hp         = prev.current_hp,
            new_hp          = new.current_hp,
            prev_monster_hp = sum(m.get("current_hp", 0) for m in prev.monsters if not m.get("is_gone")),
            new_monster_hp  = sum(m.get("current_hp", 0) for m in new_monsters if not m.get("is_gone")),
            prev_living     = sum(1 for m in prev.monsters if not m.get("is_gone")),
            new_living      = sum(1 for m in new_monsters if not m.get("is_gone")),
            prev_debuffs    = debuff_stacks(prev.monsters),
            new_debuffs     = debuff_stacks(new_monsters),
            max_hp          = prev.max_hp,
            is_end_action   = (action == RunActionSpace.END_TURN),
            energy_remaining= prev.energy,
            max_energy      = 3,
            card_is_attack  = card_is_attack,
            debuff_applied_this_turn = self._debuff_applied_this_turn,
        )

    def _played_card_is_attack(self, action: int, state: GameState) -> bool:
        if action < 10:
            slot = action
        elif 10 <= action < 60:
            slot = (action - 10) // 5
        else:
            return False
        if slot < len(state.hand):
            return state.hand[slot].get("type") == "ATTACK"
        return False

    def _update_debuff_tracking(self, action: int, state: GameState) -> None:
        from src.card_properties import get_card_properties
        if action < 10:
            slot = action
        elif 10 <= action < 60:
            slot = (action - 10) // 5
        else:
            return
        if slot < len(state.hand):
            props = get_card_properties(state.hand[slot].get("id", ""))
            if props.get("applies_vulnerable") or props.get("applies_weak"):
                self._debuff_applied_this_turn = True

    def _noncombat_reward(self, action: int, prev: GameState, new: GameState) -> float:
        raise NotImplementedError

    def _handle_game_over(self, prev: GameState, state: GameState):
        raise NotImplementedError
```

- [ ] **Step 4: Run combat step tests**

```bash
pytest tests/v2/test_run_env.py -v -k "step_combat or step_end or step_sends or step_debuff"
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/v2/run_env.py tests/v2/test_run_env.py
git commit -m "feat(v2): RunEnv step() combat path"
```

---

## Task 11: RunEnv — step() non-combat and GAME_OVER

**Files:**
- Modify: `src/v2/run_env.py`
- Modify: `tests/v2/test_run_env.py`

- [ ] **Step 1: Add failing non-combat and GAME_OVER tests**

Append to `tests/v2/test_run_env.py`:

```python
from tests.v2.helpers import make_card_reward, make_shop, make_rest


def test_step_card_reward_pick_returns_shaped_reward(env):
    before = make_card_reward(
        cards=[{"id": "Inflame", "name": "Inflame", "type": "POWER"}]
    )
    after = make_state(screen_type="MAP", available_commands=["CHOOSE"],
                       screen_state={"next_nodes": [{"symbol": "M"}]}, combat=False)
    env.communicator.receive_state.return_value = after
    env._current_state = before

    _, reward, done, _, _ = env.step(91)  # CHOOSE 0 → pick Inflame

    # Inflame is A-tier (0.8), synergy=0
    assert reward == pytest.approx(0.8 * 0.05)
    assert done is False


def test_step_card_reward_proceed_zero_reward(env):
    before = make_card_reward()
    after  = make_state(screen_type="MAP", available_commands=["CHOOSE"],
                        screen_state={"next_nodes": [{"symbol": "M"}]}, combat=False)
    env.communicator.receive_state.return_value = after
    env._current_state = before

    _, reward, _, _, _ = env.step(99)  # PROCEED

    assert reward == pytest.approx(0.0)


def test_step_rest_heal_reward(env):
    before = make_rest(hp=50, max_hp=80)
    after  = make_state(hp=74, max_hp=80, combat=False,
                        screen_type="MAP", available_commands=["CHOOSE"],
                        screen_state={"next_nodes": [{"symbol": "M"}]})
    env.communicator.receive_state.return_value = after
    env._current_state = before

    _, reward, _, _, _ = env.step(101)  # CHOOSE rest

    # Heal = min(80*0.3=24, 80-50=30) = 24 → reward = 24/80 * 0.2
    assert reward == pytest.approx(24 / 80 * 0.2)


def test_step_game_over_terminal_reward(env):
    before = make_state()
    game_over = make_game_over(floor=10)
    env.communicator.receive_state.return_value = game_over
    env._current_state = before

    _, reward, done, _, info = env.step(60)  # END

    assert done is True
    assert reward == pytest.approx((10 / 55) * 3.0 - 1.0)
    assert "episode" in info
    assert info["episode"]["floor"] == 10


def test_step_game_over_sends_proceed(env):
    env.communicator.receive_state.return_value = make_game_over(floor=5)
    env._current_state = make_state()

    env.step(60)

    env.communicator.send_command.assert_called_with("PROCEED")


def test_step_game_over_records_run(env):
    env.communicator.receive_state.return_value = make_game_over(floor=5)
    env._current_state = make_state()

    env.step(60)

    assert env.run_tracker.summary()["total_runs"] == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v2/test_run_env.py -v -k "noncombat or card_reward or rest_heal or game_over" 2>&1 | head -10
```

Expected: `NotImplementedError` from `_noncombat_reward` or `_handle_game_over`.

- [ ] **Step 3: Implement _noncombat_reward and _handle_game_over**

Replace the two stubs in `src/v2/run_env.py`:

```python
    def _noncombat_reward(self, action: int, prev: GameState, new: GameState) -> float:
        screen = prev.screen_type
        ss     = prev.screen_state or {}

        if screen == "CARD_REWARD" and 91 <= action <= 98:
            idx   = action - 91
            cards = ss.get("cards", [])
            if idx < len(cards):
                return self.reward_shaper.card_reward(cards[idx], prev.deck)
            return 0.0

        if screen == "SHOP_SCREEN":
            if 91 <= action <= 98:
                idx     = action - 91
                cards   = ss.get("cards", [])
                relics  = ss.get("relics", [])
                if idx < len(cards):
                    return self.reward_shaper.shop_card_reward(
                        cards[idx], prev.gold, prev.deck)
                relic_idx = idx - len(cards)
                if relic_idx < len(relics):
                    return self.reward_shaper.shop_relic_reward()
            if action == 100:  # PURGE
                d_cards = [c for c in prev.deck
                           if c.get("type") in ("STATUS", "CURSE")]
                if d_cards:
                    return self.reward_shaper.purge_reward(d_cards[0])
                from src.card_tier_list import get_card_tier
                d_tier = [c for c in prev.deck if get_card_tier(c.get("id", "")) == "D"]
                if d_tier:
                    return self.reward_shaper.purge_reward(d_tier[0])
            return 0.0

        if screen == "REST":
            if action == 101:  # CHOOSE rest
                max_hp   = max(prev.max_hp, 1)
                hp_gained = min(int(max_hp * 0.3), max_hp - prev.current_hp)
                return self.reward_shaper.rest_heal_reward(hp_gained, max_hp)
            if action == 102:  # CHOOSE smith
                return self.reward_shaper.rest_smith_reward()

        return 0.0

    def _handle_game_over(self, prev: GameState, state: GameState):
        reward = self.reward_shaper.terminal_reward(state.floor)
        self.run_tracker.record_run(state)
        summary = self.run_tracker.summary()
        logger.info(
            "GAME_OVER | floor=%d | runs=%d | win_rate=%.1f%%",
            state.floor, summary["total_runs"], summary["win_rate"] * 100,
        )
        self.communicator.send_command("PROCEED")
        obs = self.encoder.encode(prev)
        self._current_state = None
        info = {"episode": {"r": reward, "floor": state.floor}}
        return obs, reward, True, False, info
```

- [ ] **Step 4: Run all env tests**

```bash
pytest tests/v2/test_run_env.py -v
```

Expected: All pass.

- [ ] **Step 5: Run full test suite to check no v1 regressions**

```bash
pytest tests/ -v --ignore=tests/v2 2>&1 | tail -20
```

Expected: All v1 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/v2/run_env.py tests/v2/test_run_env.py
git commit -m "feat(v2): RunEnv step() non-combat path and GAME_OVER"
```

---

## Task 12: Wire up main.py --v2 entry point

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add the --v2 branch to main()**

In `main.py`, after the existing `if use_rl:` block and before the `else:` block, insert the new branch. The existing `else:` block stays unchanged.

The full updated `main()` function body (replacing only the conditional logic, keeping imports and helpers unchanged):

```python
def main():
    use_rl = "--rl" in sys.argv
    use_v2 = "--v2" in sys.argv

    from src.live_state import LiveStateWriter
    live_writer = LiveStateWriter(path="data/live_state.json")

    communicator = Communicator()
    tracker = RunTracker(log_path="data/run_log.jsonl", live_state_writer=live_writer)

    from src.card_scorer import CardScorer
    scorer = CardScorer(path="data/card_scores.json")

    if use_v2:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from src.v2.run_env import RunEnv
        from src.callbacks import EpisodeLoggerCallback

        env = RunEnv(communicator=communicator, run_tracker=tracker)
        model_path     = "data/v2_run_model.zip"
        checkpoint_dir = "data/v2_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = _load_model(model_path, checkpoint_dir, env)
        if model is None:
            model = MaskablePPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                verbose=1,
            )
            logger.info("Created new v2 MaskablePPO model")

        callbacks = CallbackList([
            EpisodeLoggerCallback(summary_freq=10),
            CheckpointCallback(
                save_freq=100,
                save_path=checkpoint_dir,
                name_prefix="v2_run",
                verbose=1,
            ),
        ])

        logger.info("Starting v2 RL training (MaskablePPO, full-run episodes)...")
        try:
            model.learn(total_timesteps=10_000_000, callback=callbacks)
            logger.info("Training complete.")
        except KeyboardInterrupt:
            logger.info("Training interrupted.")
        finally:
            model.save(model_path)
            logger.info("Model saved to %s", model_path)

    elif use_rl:
        # ... existing --rl block unchanged ...
```

Note: the `_load_model` helper reuses the existing function — it already handles arbitrary `model_path` and `checkpoint_dir` strings. The `EpisodeLoggerCallback` is already imported from `src.callbacks`.

- [ ] **Step 2: Verify the import resolves without a running game**

```bash
python -c "from src.v2.run_env import RunEnv; print('RunEnv OK')"
```

Expected: `RunEnv OK`

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v 2>&1 | tail -20
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(v2): wire --v2 entry point in main.py"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Action space (104 actions) ✓ | Observation (227 features, 3 blocks) ✓ | Terminal reward floor-scaled ✓ | Energy waste penalty -0.3 ✓ | Debuff-before-damage bonus ✓ | Synergy scoring ✓ | Non-combat rewards (card pick, shop, purge, rest) ✓ | `--v2` entry point ✓ | V1 preserved ✓
- [x] **Placeholder scan:** No TBDs. All code blocks are complete.
- [x] **Type consistency:** `RunActionSpace.END_TURN` used in RunEnv matches constant defined in Task 2. `RunEncoder.OBS_SIZE` used in RunEnv matches class attribute defined in Task 4. `RunRewardShaper` method signatures match calls in RunEnv Task 10/11.
