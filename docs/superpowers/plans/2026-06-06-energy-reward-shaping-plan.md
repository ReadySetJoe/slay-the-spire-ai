# Energy Efficiency Reward Shaping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small proportional bonus to `_compute_step_reward` when the agent ends its turn, rewarding full energy usage.

**Architecture:** On `END_TURN_ACTION`, append `energy_efficiency_bonus * (1 - prev_energy / 3)` to the step reward. `prev_energy` is snapshotted in `step()` before the command is sent — it represents energy remaining when the agent chose to end the turn. The bonus is non-negative, so it never fights the damage/kill signals.

**Tech Stack:** Python, Gymnasium, pytest

---

## File Map

- **Modify:** `src/combat_env.py` — only file that changes
- **Modify:** `tests/test_combat_env.py` — fix one pre-existing broken assertion, add new energy bonus tests

---

### Task 1: Fix pre-existing broken test and add failing energy bonus tests

**Files:**
- Modify: `tests/test_combat_env.py`

One existing test (`test_step_continues_combat`) currently asserts `reward == 0.0` but actually gets `-0.0625` (5 HP taken / max HP 80). Fix it, then add new failing tests for the energy bonus.

- [ ] **Step 1: Fix the broken assertion**

In `tests/test_combat_env.py`, find `test_step_continues_combat` and replace:

```python
assert reward == 0.0
```

with:

```python
assert reward == pytest.approx(-5 / 80)
```

- [ ] **Step 2: Run the fixed test to verify it now passes**

```
python -m pytest tests/test_combat_env.py::test_step_continues_combat -v
```

Expected: PASS

- [ ] **Step 3: Add failing tests for the energy bonus**

Append these tests to the bottom of `tests/test_combat_env.py`:

```python
def _combat_energy(energy: int, hp=70, max_hp=80):
    """Combat state fixture with configurable remaining energy."""
    return GameState.from_json(json.dumps({
        "available_commands": ["PLAY", "END"],
        "ready_for_command": True, "in_game": True,
        "game_state": {
            "screen_type": "NONE",
            "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": hp, "max_hp": max_hp, "gold": 99,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": {
                "hand": [],
                "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
                "monsters": [{"name": "Jaw Worm", "current_hp": 42, "max_hp": 42,
                               "block": 0, "intent": "ATTACK", "is_gone": False}],
                "player": {"current_hp": hp, "max_hp": max_hp, "block": 0,
                            "energy": energy, "powers": []},
                "turn": 1,
            },
        }
    }))


def test_compute_step_reward_end_turn_full_energy_used():
    """Using all energy (leftover=0) gives the full bonus."""
    env = CombatEnv(communicator=MagicMock())
    post_state = _combat_energy(energy=3, hp=70)  # same hp, no damage exchanged
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=ActionSpace.END_TURN_ACTION, prev_energy=0,
    )
    assert reward == pytest.approx(0.1)


def test_compute_step_reward_end_turn_partial_energy_used():
    """Using 2 of 3 energy gives a proportionally smaller bonus."""
    env = CombatEnv(communicator=MagicMock())
    post_state = _combat_energy(energy=3, hp=70)
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=ActionSpace.END_TURN_ACTION, prev_energy=1,
    )
    assert reward == pytest.approx(0.1 * (1 - 1 / 3))


def test_compute_step_reward_end_turn_no_energy_used():
    """Wasting all energy gives no bonus."""
    env = CombatEnv(communicator=MagicMock())
    post_state = _combat_energy(energy=3, hp=70)
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=ActionSpace.END_TURN_ACTION, prev_energy=3,
    )
    assert reward == pytest.approx(0.0)


def test_compute_step_reward_card_play_no_energy_bonus():
    """Energy bonus is NOT applied on card play, only on END_TURN."""
    env = CombatEnv(communicator=MagicMock())
    post_state = _combat_energy(energy=0, hp=70)
    # action=0 is a card-play action (not END_TURN_ACTION=60)
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=0, prev_energy=0,
    )
    assert reward == pytest.approx(0.0)


def test_energy_efficiency_bonus_configurable():
    """energy_efficiency_bonus constructor param overrides the default weight."""
    env = CombatEnv(communicator=MagicMock(), energy_efficiency_bonus=0.2)
    post_state = _combat_energy(energy=3, hp=70)
    reward = env._compute_step_reward(
        prev_hp=70, prev_monster_hp=42, prev_living=1,
        max_hp=80, state=post_state,
        action=ActionSpace.END_TURN_ACTION, prev_energy=0,
    )
    assert reward == pytest.approx(0.2)


def test_step_end_turn_with_zero_energy_applies_bonus():
    """Integration: step() captures pre-command energy and applies bonus."""
    comm = MagicMock()
    comm.receive_state.return_value = _combat_energy(energy=3, hp=70)  # post-turn energy reset
    env = CombatEnv(communicator=comm)
    env._current_state = _combat_energy(energy=0, hp=70)  # agent spent all energy

    _, reward, done, _, _ = env.step(ActionSpace.END_TURN_ACTION)

    assert done is False
    assert reward == pytest.approx(0.1)  # no damage delta + full energy bonus
```

- [ ] **Step 4: Run new tests to confirm they all fail**

```
python -m pytest tests/test_combat_env.py -k "energy" -v
```

Expected: all 6 new tests FAIL (various errors — method signature mismatch, missing param, etc.)

- [ ] **Step 5: Commit the tests**

```
git add tests/test_combat_env.py
git commit -m "test: add failing tests for energy efficiency reward bonus"
```

---

### Task 2: Implement the energy efficiency bonus

**Files:**
- Modify: `src/combat_env.py`

Three small changes: constructor param, snapshot in `step()`, bonus in `_compute_step_reward`.

- [ ] **Step 1: Add `energy_efficiency_bonus` parameter to `__init__`**

In `src/combat_env.py`, find the `__init__` signature:

```python
def __init__(self, communicator: Communicator,
             run_tracker: Optional[RunTracker] = None,
             scorer=None):
```

Replace with:

```python
def __init__(self, communicator: Communicator,
             run_tracker: Optional[RunTracker] = None,
             scorer=None,
             energy_efficiency_bonus: float = 0.1):
```

Then find the last line of `__init__` that sets an instance variable (currently `self._initialized: bool = False`) and add after it:

```python
        self._energy_efficiency_bonus = energy_efficiency_bonus
```

- [ ] **Step 2: Snapshot `prev_energy` in `step()`**

In `step()`, find the block:

```python
        max_hp = self._current_state.max_hp
```

Add one line immediately after it:

```python
        prev_energy = self._current_state.energy
```

- [ ] **Step 3: Pass `action` and `prev_energy` to `_compute_step_reward`**

Still in `step()`, find:

```python
            step_reward = self._compute_step_reward(
                prev_hp, prev_monster_hp, prev_living, max_hp, state
            )
```

Replace with:

```python
            step_reward = self._compute_step_reward(
                prev_hp, prev_monster_hp, prev_living, max_hp, state, action, prev_energy
            )
```

- [ ] **Step 4: Update `_compute_step_reward` to accept the new params and apply the bonus**

Find the current method:

```python
    def _compute_step_reward(self, prev_hp: int, prev_monster_hp: int,
                              prev_living: int, max_hp: int,
                              state: GameState) -> float:
        new_hp = state.current_hp
        new_monster_hp = sum(
            m.get("current_hp", 0) for m in state.monsters
            if not m.get("is_gone", False)
        )
        new_living = sum(1 for m in state.monsters if not m.get("is_gone", False))

        damage_dealt = max(prev_monster_hp - new_monster_hp, 0) / max(max_hp, 1)
        damage_taken = max(prev_hp - new_hp, 0) / max(max_hp, 1)
        kills = max(prev_living - new_living, 0)
        return damage_dealt - damage_taken + 0.1 * kills
```

Replace with:

```python
    def _compute_step_reward(self, prev_hp: int, prev_monster_hp: int,
                              prev_living: int, max_hp: int,
                              state: GameState, action: int = 0,
                              prev_energy: int = 0) -> float:
        new_hp = state.current_hp
        new_monster_hp = sum(
            m.get("current_hp", 0) for m in state.monsters
            if not m.get("is_gone", False)
        )
        new_living = sum(1 for m in state.monsters if not m.get("is_gone", False))

        damage_dealt = max(prev_monster_hp - new_monster_hp, 0) / max(max_hp, 1)
        damage_taken = max(prev_hp - new_hp, 0) / max(max_hp, 1)
        kills = max(prev_living - new_living, 0)
        base = damage_dealt - damage_taken + 0.1 * kills
        if action == ActionSpace.END_TURN_ACTION:
            base += self._energy_efficiency_bonus * (1 - prev_energy / 3)
        return base
```

Note: `3` is the base max energy for Ironclad A0. If future relics that modify max energy need to be accounted for, this constant can be replaced with a dynamic lookup.

- [ ] **Step 5: Run all energy tests to confirm they pass**

```
python -m pytest tests/test_combat_env.py -k "energy" -v
```

Expected: all 6 tests PASS

- [ ] **Step 6: Run the full test suite**

```
python -m pytest tests/test_combat_env.py -v
```

Expected: all 9 original + 6 new = 15 tests PASS (the previously-broken `test_step_continues_combat` now passes with the corrected assertion from Task 1)

- [ ] **Step 7: Commit**

```
git add src/combat_env.py
git commit -m "feat: reward energy efficiency on END_TURN in combat step reward"
```
