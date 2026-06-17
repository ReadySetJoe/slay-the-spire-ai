# V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the STS RL agent to MaskableRecurrentPPO (LSTM) with a 250-feature observation space, dynamic card synergy scoring, stronger relic rewards, and a hung-episode watchdog.

**Architecture:** `V3RunEnv(RunEnv)` overrides `step()`, `reset()`, and `_next_actionable_state()` to add turn-state tracking, timeout-protected receives, and CardScorer integration. `V3RunEncoder(RunEncoder)` produces a 250-feature vector (global 55 + combat 123 + non-combat 60 + turn context 12). `V3RunRewardShaper(RunRewardShaper)` overrides relic rewards and the energy-waste penalty. `CardScorer` maintains EMA-updated synergy scores persisted to `data/card_scores.json`.

**Tech Stack:** Python 3.11, gymnasium, stable-baselines3, sb3-contrib (MaskableRecurrentPPO), numpy, pytest, threading

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/v3/__init__.py` | Package marker |
| Create | `src/v3/run_reward.py` | `V3RunRewardShaper` — overrides relic + energy rewards |
| Create | `src/v3/card_scorer.py` | `CardScorer` — EMA synergy scores, JSON persistence |
| Create | `src/v3/run_encoder.py` | `V3RunEncoder` — 250-feature obs (intent flags + turn context) |
| Create | `src/v3/run_env.py` | `V3RunEnv` — turn tracking, watchdog, CardScorer integration |
| Create | `tests/v3/__init__.py` | Package marker |
| Create | `tests/v3/helpers.py` | Re-exports v2 helpers + turn-state factories |
| Create | `tests/v3/test_run_reward.py` | V3RunRewardShaper tests |
| Create | `tests/v3/test_card_scorer.py` | CardScorer EMA + persistence tests |
| Create | `tests/v3/test_run_encoder.py` | Encoder tests (intent flags, turn context) |
| Create | `tests/v3/test_run_env.py` | Turn tracking + watchdog tests |
| Modify | `src/run_tracker.py` | Add `hung_count`, `record_hung()` |
| Modify | `dashboard.py` | Add Hung stat chip |
| Modify | `main.py` | Add `--v3` entry point |

---

## Task 1: Branch + package skeleton

**Files:**
- Create: `src/v3/__init__.py`
- Create: `tests/v3/__init__.py`
- Create: `tests/v3/helpers.py`

- [ ] **Step 1: Create the v3 branch**

```bash
git checkout -b v3
```

- [ ] **Step 2: Create package markers**

`src/v3/__init__.py` — empty file.

`tests/v3/__init__.py` — empty file.

- [ ] **Step 3: Create test helpers**

`tests/v3/helpers.py`:

```python
from tests.v2.helpers import (
    make_state, make_game_over, make_card_reward,
    make_shop, make_rest, make_map,
)


def empty_turn_state() -> dict:
    return {
        "actions_taken": 0, "energy_spent": 0,
        "attacks_played": 0, "skills_played": 0, "powers_played": 0,
        "strength_gained": 0, "vulnerable_applied": False, "weak_applied": False,
        "damage_dealt": 0.0, "block_gained": 0.0,
        "last_card_was_buff": False, "last_card_was_debuff": False,
    }


def flex_turn_state() -> dict:
    """Turn state after playing Flex (power, buff, +2 strength)."""
    return {**empty_turn_state(),
            "actions_taken": 1, "energy_spent": 1, "powers_played": 1,
            "strength_gained": 2, "last_card_was_buff": True}


def bash_turn_state() -> dict:
    """Turn state after playing Bash (attack, applies vulnerable)."""
    return {**empty_turn_state(),
            "actions_taken": 1, "energy_spent": 2, "attacks_played": 1,
            "vulnerable_applied": True, "last_card_was_debuff": True}
```

- [ ] **Step 4: Verify helpers import**

```bash
python -c "from tests.v3.helpers import make_state, flex_turn_state; print('ok')"
```

Expected output: `ok`

- [ ] **Step 5: Commit**

```bash
git add src/v3/__init__.py tests/v3/__init__.py tests/v3/helpers.py
git commit -m "feat(v3): branch + package skeleton and test helpers"
```

---

## Task 2: V3RunRewardShaper

**Files:**
- Create: `src/v3/run_reward.py`
- Create: `tests/v3/test_run_reward.py`

- [ ] **Step 1: Write failing tests**

`tests/v3/test_run_reward.py`:

```python
import pytest
from src.v3.run_reward import V3RunRewardShaper


@pytest.fixture
def shaper():
    return V3RunRewardShaper()


# --- relic reward overrides ---

def test_open_chest_reward(shaper):
    assert shaper.open_chest_reward() == pytest.approx(0.25)


def test_combat_relic_reward(shaper):
    assert shaper.combat_relic_reward() == pytest.approx(0.15)


def test_boss_relic_reward(shaper):
    assert shaper.boss_relic_reward() == pytest.approx(0.20)


def test_shop_relic_reward(shaper):
    assert shaper.shop_relic_reward() == pytest.approx(0.15)


# --- energy waste penalty override ---

def test_energy_penalty_full_waste(shaper):
    # End with 3 energy remaining (all wasted) → -0.5 * (3/3) = -0.5
    r = shaper.combat_step_reward(
        prev_hp=70, new_hp=70,
        prev_monster_hp=42, new_monster_hp=42,
        prev_living=1, new_living=1,
        prev_debuffs=0, new_debuffs=0,
        max_hp=80, is_end_action=True,
        energy_remaining=3, max_energy=4,  # max_energy arg ignored; v3 uses 3
        card_is_attack=False, debuff_applied_this_turn=False,
    )
    assert r == pytest.approx(-0.5)


def test_energy_penalty_partial_waste(shaper):
    # End with 1 energy remaining → -0.5 * (1/3) ≈ -0.167
    r = shaper.combat_step_reward(
        prev_hp=70, new_hp=70,
        prev_monster_hp=42, new_monster_hp=42,
        prev_living=1, new_living=1,
        prev_debuffs=0, new_debuffs=0,
        max_hp=80, is_end_action=True,
        energy_remaining=1, max_energy=4,
        card_is_attack=False, debuff_applied_this_turn=False,
    )
    assert r == pytest.approx(-0.5 / 3)


def test_energy_penalty_no_waste(shaper):
    # End with 0 energy remaining → no penalty
    r = shaper.combat_step_reward(
        prev_hp=70, new_hp=70,
        prev_monster_hp=42, new_monster_hp=42,
        prev_living=1, new_living=1,
        prev_debuffs=0, new_debuffs=0,
        max_hp=80, is_end_action=True,
        energy_remaining=0, max_energy=4,
        card_is_attack=False, debuff_applied_this_turn=False,
    )
    assert r == pytest.approx(0.0)


def test_damage_reward_still_works(shaper):
    # Base damage reward unchanged
    r = shaper.combat_step_reward(
        prev_hp=70, new_hp=70,
        prev_monster_hp=42, new_monster_hp=30,
        prev_living=1, new_living=1,
        prev_debuffs=0, new_debuffs=0,
        max_hp=80, is_end_action=False,
        energy_remaining=2, max_energy=4,
        card_is_attack=False, debuff_applied_this_turn=False,
    )
    assert r == pytest.approx(12 / 80)


# --- inherited rewards still work ---

def test_terminal_reward_unchanged(shaper):
    assert shaper.terminal_reward(55) == pytest.approx(2.0)
    assert shaper.terminal_reward(0)  == pytest.approx(-1.0)


def test_shop_card_reward_unchanged(shaper):
    card = {"id": "Inflame", "price": 75}
    r = shaper.shop_card_reward(card, gold=150, deck=[])
    assert r == pytest.approx(0.8 * 0.05 + 0.0 * 0.05 - 0.5 * 0.02)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v3/test_run_reward.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.v3.run_reward'`

- [ ] **Step 3: Implement V3RunRewardShaper**

`src/v3/run_reward.py`:

```python
from src.v2.run_reward import RunRewardShaper

_MAX_ENERGY = 3  # Ironclad A0 base max energy


class V3RunRewardShaper(RunRewardShaper):
    """Stronger relic rewards and tighter energy-waste penalty than v2."""

    def open_chest_reward(self) -> float:
        return 0.25

    def combat_relic_reward(self) -> float:
        return 0.15

    def boss_relic_reward(self) -> float:
        return 0.20

    def shop_relic_reward(self) -> float:
        return 0.15

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
        # Call parent with is_end_action=False to skip v2's -0.3 penalty
        reward = super().combat_step_reward(
            prev_hp=prev_hp, new_hp=new_hp,
            prev_monster_hp=prev_monster_hp, new_monster_hp=new_monster_hp,
            prev_living=prev_living, new_living=new_living,
            prev_debuffs=prev_debuffs, new_debuffs=new_debuffs,
            max_hp=max_hp,
            is_end_action=False,
            energy_remaining=energy_remaining,
            max_energy=max_energy,
            card_is_attack=card_is_attack,
            debuff_applied_this_turn=debuff_applied_this_turn,
        )
        if is_end_action:
            reward -= 0.5 * (energy_remaining / _MAX_ENERGY)
        return reward
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/v3/test_run_reward.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/v3/run_reward.py tests/v3/test_run_reward.py
git commit -m "feat(v3): V3RunRewardShaper — stronger relic rewards and -0.5 energy penalty"
```

---

## Task 3: CardScorer

**Files:**
- Create: `src/v3/card_scorer.py`
- Create: `tests/v3/test_card_scorer.py`

- [ ] **Step 1: Write failing tests**

`tests/v3/test_card_scorer.py`:

```python
import json
import os
import pytest
from src.v3.card_scorer import CardScorer


@pytest.fixture
def scorer(tmp_path):
    return CardScorer(path=str(tmp_path / "scores.json"))


def test_unseen_card_returns_default(scorer):
    assert scorer.score("Inflame") == pytest.approx(0.5)


def test_update_moves_toward_signal(scorer):
    scorer.update(["Inflame"], performance_signal=1.0)
    assert scorer.score("Inflame") > 0.5


def test_update_moves_down_on_low_signal(scorer):
    scorer.update(["Strike_R"], performance_signal=0.0)
    assert scorer.score("Strike_R") < 0.5


def test_alpha_controls_update_rate(scorer):
    scorer_fast = CardScorer(path=scorer._path + ".fast", alpha=0.5)
    scorer_slow = CardScorer(path=scorer._path + ".slow", alpha=0.01)
    scorer_fast.update(["X"], 1.0)
    scorer_slow.update(["X"], 1.0)
    assert scorer_fast.score("X") > scorer_slow.score("X")


def test_update_clamps_performance_signal(scorer):
    scorer.update(["Bash"], performance_signal=5.0)  # should clamp to 1.0
    assert scorer.score("Bash") <= 1.0
    scorer.update(["Bash"], performance_signal=-3.0)  # should clamp to 0.0
    assert scorer.score("Bash") >= 0.0


def test_update_increments_total_combats(scorer):
    assert scorer._total_combats == 0
    scorer.update(["Bash"], 0.8)
    assert scorer._total_combats == 1


def test_save_and_load_round_trip(scorer, tmp_path):
    path = str(tmp_path / "scores.json")
    s1 = CardScorer(path=path)
    s1.update(["Inflame"], 0.9)
    s1.save()

    s2 = CardScorer(path=path)
    assert s2.score("Inflame") == pytest.approx(s1.score("Inflame"))
    assert s2._total_combats == 1


def test_save_is_atomic(scorer, tmp_path):
    """Save writes to .tmp then renames — no partial file visible."""
    path = str(tmp_path / "scores.json")
    scorer2 = CardScorer(path=path)
    scorer2.update(["Bash"], 0.7)
    scorer2.save()
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_load_missing_file_is_noop(tmp_path):
    s = CardScorer(path=str(tmp_path / "missing.json"))
    assert s.score("anything") == pytest.approx(0.5)
    assert s._total_combats == 0


def test_multiple_cards_updated(scorer):
    scorer.update(["Bash", "Strike_R", "Inflame"], 1.0)
    for card in ["Bash", "Strike_R", "Inflame"]:
        assert scorer.score(card) > 0.5
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v3/test_card_scorer.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.v3.card_scorer'`

- [ ] **Step 3: Implement CardScorer**

`src/v3/card_scorer.py`:

```python
import json
import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_SCORE = 0.5


class CardScorer:
    def __init__(self, path: str = "data/card_scores.json", alpha: float = 0.05):
        self._path = path
        self._alpha = alpha
        self._scores: dict[str, float] = {}
        self._total_combats: int = 0
        self.load()

    def score(self, card_id: str) -> float:
        return self._scores.get(card_id, _DEFAULT_SCORE)

    def update(self, cards_played: list[str], performance_signal: float) -> None:
        performance_signal = max(0.0, min(1.0, performance_signal))
        for card_id in cards_played:
            current = self._scores.get(card_id, _DEFAULT_SCORE)
            self._scores[card_id] = (1 - self._alpha) * current + self._alpha * performance_signal
        self._total_combats += 1

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        data = {"card_scores": self._scores, "total_combats_scored": self._total_combats}
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except OSError as e:
            logger.warning("CardScorer save failed: %s", e)

    def load(self) -> None:
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._scores = data.get("card_scores", {})
            self._total_combats = data.get("total_combats_scored", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            self._scores = {}
            self._total_combats = 0
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/v3/test_card_scorer.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/v3/card_scorer.py tests/v3/test_card_scorer.py
git commit -m "feat(v3): CardScorer — EMA-updated card synergy scores with JSON persistence"
```

---

## Task 4: V3RunEncoder

**Files:**
- Create: `src/v3/run_encoder.py`
- Create: `tests/v3/test_run_encoder.py`

- [ ] **Step 1: Write failing tests**

`tests/v3/test_run_encoder.py`:

```python
import numpy as np
import pytest
from src.v3.run_encoder import V3RunEncoder, V3_OBS_SIZE, V3_GLOBAL_SIZE, V3_COMBAT_SIZE, V3_NONCOMBAT_SIZE
from src.v3.card_scorer import CardScorer
from tests.v3.helpers import (
    make_state, make_card_reward, make_shop, make_rest, make_map,
    empty_turn_state, flex_turn_state, bash_turn_state,
)

TURN_CTX_BASE = V3_GLOBAL_SIZE + V3_COMBAT_SIZE + V3_NONCOMBAT_SIZE  # 238


@pytest.fixture
def enc():
    return V3RunEncoder()


# --- shape and dtype ---

def test_obs_shape(enc):
    obs = enc.encode(make_state())
    assert obs.shape == (V3_OBS_SIZE,)
    assert obs.dtype == np.float32


def test_obs_size_is_250(enc):
    assert V3_OBS_SIZE == 250


def test_obs_values_in_range(enc):
    obs = enc.encode(make_state())
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


# --- global block unchanged ---

def test_hp_ratio_global(enc):
    obs = enc.encode(make_state(hp=40, max_hp=80))
    assert obs[0] == pytest.approx(0.5)


# --- combat block: intent flags ---

def test_attacking_monster_sets_is_attacking(enc):
    state = make_state(monsters=[{
        "name": "Worm", "current_hp": 42, "max_hp": 42,
        "block": 0, "intent": "ATTACK", "is_gone": False, "powers": [],
    }])
    obs = enc.encode(state)
    monster_base = V3_GLOBAL_SIZE + 70  # 125
    assert obs[monster_base + 3] == pytest.approx(1.0)   # is_attacking
    assert obs[monster_base + 4] == pytest.approx(0.0)   # is_buffing
    assert obs[monster_base + 5] == pytest.approx(0.0)   # is_debuffing


def test_buffing_monster_sets_is_buffing(enc):
    state = make_state(monsters=[{
        "name": "Worm", "current_hp": 42, "max_hp": 42,
        "block": 0, "intent": "BUFF", "is_gone": False, "powers": [],
    }])
    obs = enc.encode(state)
    monster_base = V3_GLOBAL_SIZE + 70
    assert obs[monster_base + 3] == pytest.approx(0.0)   # is_attacking
    assert obs[monster_base + 4] == pytest.approx(1.0)   # is_buffing


def test_debuffing_monster_sets_is_debuffing(enc):
    state = make_state(monsters=[{
        "name": "Worm", "current_hp": 42, "max_hp": 42,
        "block": 0, "intent": "DEBUFF", "is_gone": False, "powers": [],
    }])
    obs = enc.encode(state)
    monster_base = V3_GLOBAL_SIZE + 70
    assert obs[monster_base + 5] == pytest.approx(1.0)   # is_debuffing


def test_attack_buff_intent_sets_both_flags(enc):
    state = make_state(monsters=[{
        "name": "Worm", "current_hp": 42, "max_hp": 42,
        "block": 0, "intent": "ATTACK_BUFF", "is_gone": False, "powers": [],
    }])
    obs = enc.encode(state)
    monster_base = V3_GLOBAL_SIZE + 70
    assert obs[monster_base + 3] == pytest.approx(1.0)   # is_attacking
    assert obs[monster_base + 4] == pytest.approx(1.0)   # is_buffing


# --- aggregate intent features ---

def test_any_enemy_attacking_true(enc):
    state = make_state(monsters=[{
        "name": "Worm", "current_hp": 42, "max_hp": 42,
        "block": 0, "intent": "ATTACK", "is_gone": False, "powers": [],
    }])
    obs = enc.encode(state)
    agg_base = V3_GLOBAL_SIZE + 70 + 40  # 165
    assert obs[agg_base] == pytest.approx(1.0)           # any_enemy_attacking
    assert obs[agg_base + 1] == pytest.approx(1 / 5)     # attacking_count/5


def test_any_enemy_attacking_false_when_buffing(enc):
    state = make_state(monsters=[{
        "name": "Worm", "current_hp": 42, "max_hp": 42,
        "block": 0, "intent": "BUFF", "is_gone": False, "powers": [],
    }])
    obs = enc.encode(state)
    agg_base = V3_GLOBAL_SIZE + 70 + 40
    assert obs[agg_base] == pytest.approx(0.0)


# --- combat block size ---

def test_combat_block_size(enc):
    obs = enc.encode(make_state())
    # Non-combat block should start at index 178
    assert V3_GLOBAL_SIZE + V3_COMBAT_SIZE == 178


# --- turn context block (Block 4) ---

def test_turn_context_zeroed_without_turn_state(enc):
    obs = enc.encode(make_state())
    assert obs[TURN_CTX_BASE:].sum() == pytest.approx(0.0)


def test_turn_context_zeroed_on_noncombat(enc):
    obs = enc.encode(make_card_reward(), turn_state=flex_turn_state())
    assert obs[TURN_CTX_BASE:].sum() == pytest.approx(0.0)


def test_turn_context_actions_taken(enc):
    ts = {**empty_turn_state(), "actions_taken": 5}
    obs = enc.encode(make_state(), turn_state=ts)
    assert obs[TURN_CTX_BASE + 0] == pytest.approx(5 / 10)


def test_turn_context_energy_spent(enc):
    ts = {**empty_turn_state(), "energy_spent": 2}
    obs = enc.encode(make_state(), turn_state=ts)
    assert obs[TURN_CTX_BASE + 1] == pytest.approx(2 / 4)


def test_turn_context_strength_gained(enc):
    ts = {**empty_turn_state(), "strength_gained": 3}
    obs = enc.encode(make_state(), turn_state=ts)
    assert obs[TURN_CTX_BASE + 5] == pytest.approx(3 / 10)


def test_turn_context_vulnerable_applied(enc):
    ts = {**empty_turn_state(), "vulnerable_applied": True}
    obs = enc.encode(make_state(), turn_state=ts)
    assert obs[TURN_CTX_BASE + 6] == pytest.approx(1.0)


def test_turn_context_last_card_was_buff(enc):
    obs = enc.encode(make_state(), turn_state=flex_turn_state())
    assert obs[TURN_CTX_BASE + 10] == pytest.approx(1.0)


def test_turn_context_last_card_was_debuff(enc):
    obs = enc.encode(make_state(), turn_state=bash_turn_state())
    assert obs[TURN_CTX_BASE + 11] == pytest.approx(1.0)


# --- CardScorer integration ---

def test_card_scorer_synergy_replaces_static_heuristic(enc, tmp_path):
    scorer = CardScorer(path=str(tmp_path / "s.json"))
    scorer.update(["Inflame"], 1.0)  # push score above default 0.5

    state = make_card_reward(cards=[{"id": "Inflame", "name": "Inflame", "type": "POWER"}])
    obs_with    = enc.encode(state, card_scorer=scorer)
    obs_without = enc.encode(state)
    noncombat_base = V3_GLOBAL_SIZE + V3_COMBAT_SIZE  # 178
    # synergy_score is feature index 1 of choice 0 → index 179
    assert obs_with[noncombat_base + 1] != obs_without[noncombat_base + 1]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v3/test_run_encoder.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.v3.run_encoder'`

- [ ] **Step 3: Implement V3RunEncoder**

`src/v3/run_encoder.py`:

```python
import numpy as np
from src.game_state import GameState
from src.card_properties import get_card_properties
from src.card_tier_list import get_card_tier
from src.v2.run_encoder import (
    RunEncoder,
    _HEALING_POTIONS, _ATTACK_POTIONS, _HIGH_IMPACT_RELICS,
    _STRENGTH_CARDS, _DRAW_CARDS, _EXHAUST_CARDS,
    _SCREEN_IDX, _TIER_VALUE,
)

V3_GLOBAL_SIZE       = 55
V3_COMBAT_SIZE       = 123
V3_NONCOMBAT_SIZE    = 60
V3_TURN_CONTEXT_SIZE = 12
V3_OBS_SIZE          = 250  # 55 + 123 + 60 + 12

_ATTACKING_INTENTS = {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}
_BUFFING_INTENTS   = {"BUFF",   "DEFEND_BUFF", "ATTACK_BUFF"}
_DEBUFFING_INTENTS = {"DEBUFF", "STRONG_DEBUFF", "ATTACK_DEBUFF"}


class V3RunEncoder(RunEncoder):
    OBS_SIZE = V3_OBS_SIZE

    def encode(self, state: GameState, turn_state: dict | None = None,
               card_scorer=None) -> np.ndarray:
        obs = np.zeros(V3_OBS_SIZE, dtype=np.float32)
        self._encode_global(obs, state)              # 0–54 (inherited, unchanged)
        if state.is_in_combat:
            self._encode_combat(obs, state)          # 55–177 (overridden)
            if turn_state:
                self._encode_turn_context(obs, turn_state, state.max_hp)  # 238–249
        else:
            self._encode_noncombat(obs, state, card_scorer)  # 178–237 (overridden)
        return obs

    def _encode_combat(self, obs: np.ndarray, state: GameState) -> None:
        base = V3_GLOBAL_SIZE  # 55

        # Hand: 10 × 7 features [55:125] — identical to v2
        for i, card in enumerate(state.hand[:10]):
            b     = base + i * 7
            props = get_card_properties(card.get("id", ""))
            obs[b]     = min(card.get("cost", 0), 5) / 5
            obs[b + 1] = self.CARD_TYPE_MAP.get(card.get("type", ""), 0.5)
            obs[b + 2] = 1.0 if card.get("is_playable", False) else 0.0
            obs[b + 3] = 1.0 if props["applies_vulnerable"] else 0.0
            obs[b + 4] = 1.0 if props["applies_weak"]       else 0.0
            obs[b + 5] = 1.0 if props["draws_cards"]        else 0.0
            obs[b + 6] = 1.0 if props["gains_block"]        else 0.0

        # Monsters: 5 × 8 features [125:165]
        monster_base  = base + 70   # 125
        any_attacking = False
        attack_count  = 0
        for i, m in enumerate(state.monsters[:5]):
            if m.get("is_gone", False):
                continue
            b      = monster_base + i * 8
            m_max  = max(m.get("max_hp", 1), 1)
            intent = m.get("intent", "UNKNOWN")
            powers = m.get("powers", [])
            vuln   = next((p.get("amount", 0) for p in powers if p.get("id") == "Vulnerable"), 0)
            weak   = next((p.get("amount", 0) for p in powers if p.get("id") == "Weak"), 0)
            is_atk = intent in _ATTACKING_INTENTS
            is_buf = intent in _BUFFING_INTENTS
            is_deb = intent in _DEBUFFING_INTENTS
            if is_atk:
                any_attacking = True
                attack_count += 1
            obs[b]     = m.get("current_hp", 0) / m_max
            obs[b + 1] = min(m_max / 400, 1.0)
            obs[b + 2] = m.get("block", 0) / m_max
            obs[b + 3] = 1.0 if is_atk else 0.0
            obs[b + 4] = 1.0 if is_buf else 0.0
            obs[b + 5] = 1.0 if is_deb else 0.0
            obs[b + 6] = min(vuln / 10, 1.0)
            obs[b + 7] = min(weak / 10, 1.0)

        # Aggregate intent [165:167]
        agg_base          = monster_base + 40   # 165
        obs[agg_base]     = 1.0 if any_attacking else 0.0
        obs[agg_base + 1] = attack_count / 5

        # Player powers [167:172]
        power_base    = agg_base + 2            # 167
        player_powers = (state.combat_state or {}).get("player", {}).get("powers", [])

        def _pwr(name):
            return next((p.get("amount", 0) for p in player_powers if p.get("id") == name), 0)

        obs[power_base]     = min(_pwr("Strength")  / 10, 1.0)
        obs[power_base + 1] = min(_pwr("Dexterity") / 10, 1.0)
        obs[power_base + 2] = min(_pwr("Weak")      / 5,  1.0)
        obs[power_base + 3] = min(_pwr("Vulnerable") / 5, 1.0)
        obs[power_base + 4] = 1.0 if any(p.get("id") == "Barricade" for p in player_powers) else 0.0

        # Turn metadata [172:175]
        meta_base          = power_base + 5     # 172
        obs[meta_base]     = min(len(state.draw_pile)    / 60, 1.0)
        obs[meta_base + 1] = min(len(state.discard_pile) / 60, 1.0)
        obs[meta_base + 2] = min(state.turn              / 20, 1.0)

        # Debuff signal [175:178]
        debuff_base    = meta_base + 3          # 175
        n_hand         = max(len(state.hand), 1)
        n_debuff_cards = sum(
            1 for c in state.hand
            if get_card_properties(c.get("id", "")).get("applies_vulnerable") or
               get_card_properties(c.get("id", "")).get("applies_weak")
        )
        obs[debuff_base] = n_debuff_cards / n_hand
        # 176–177 reserved (stay zero)

    def _encode_noncombat(self, obs: np.ndarray, state: GameState,
                          card_scorer=None) -> None:
        base   = V3_GLOBAL_SIZE + V3_COMBAT_SIZE  # 178
        ss     = state.screen_state or {}
        screen = state.screen_type
        gold   = max(state.gold, 1)
        deck   = state.deck

        # Choices block [178:210]: 8 × 4 features
        choices = self._get_choices(screen, ss, state, card_scorer)
        for i, choice in enumerate(choices[:8]):
            b          = base + i * 4
            obs[b]     = choice.get("tier_value",    0.0)
            obs[b + 1] = choice.get("synergy_score", 0.0)
            obs[b + 2] = choice.get("cost_ratio",    0.0)
            obs[b + 3] = choice.get("is_available",  1.0)

        # Deck synergy context [210:215]
        syn_base  = base + 32  # 210
        deck_ids  = [c.get("id", "") for c in deck]
        obs[syn_base]     = min(sum(1 for d in deck_ids if d in _EXHAUST_CARDS)  / 10, 1.0)
        obs[syn_base + 1] = min(sum(1 for d in deck_ids if d in _STRENGTH_CARDS) / 5,  1.0)
        obs[syn_base + 2] = min(sum(1 for d in deck_ids if d in _DRAW_CARDS)     / 5,  1.0)
        n_block           = sum(1 for d in deck_ids if get_card_properties(d).get("gains_block"))
        obs[syn_base + 3] = min(n_block / 10, 1.0)
        n_curse           = sum(1 for c in deck if c.get("type") in ("STATUS", "CURSE"))
        obs[syn_base + 4] = min(n_curse / 10, 1.0)

        # Screen metadata [215:223]
        meta_base = syn_base + 5  # 215
        max_hp    = max(state.max_hp, 1)
        if screen == "SHOP_SCREEN":
            items     = ss.get("cards", []) + ss.get("relics", [])
            min_price = min((c.get("price", 9999) for c in items if c.get("is_in_stock", True)),
                           default=0)
            obs[meta_base] = min(min_price / gold, 1.0)
        if screen == "REST":
            heal               = min(int(max_hp * 0.3), max_hp - state.current_hp)
            obs[meta_base + 1] = max(heal, 0) / max_hp
        if screen == "MAP":
            nodes   = ss.get("next_nodes", [])
            symbols = [n.get("symbol", "") for n in nodes]
            obs[meta_base + 2] = 1.0 if "E" in symbols else 0.0
            obs[meta_base + 3] = 1.0 if "R" in symbols else 0.0
            obs[meta_base + 4] = 1.0 if "$" in symbols else 0.0
            obs[meta_base + 5] = 1.0 if "T" in symbols else 0.0
            obs[meta_base + 6] = 1.0 if "?" in symbols else 0.0
            obs[meta_base + 7] = 1.0 if "M" in symbols else 0.0
        # padding [223:238] stays 0

    def _encode_turn_context(self, obs: np.ndarray, turn_state: dict,
                             max_hp: int) -> None:
        base   = V3_GLOBAL_SIZE + V3_COMBAT_SIZE + V3_NONCOMBAT_SIZE  # 238
        max_hp = max(max_hp, 1)
        obs[base]      = min(turn_state.get("actions_taken",   0)   / 10,   1.0)
        obs[base + 1]  = min(turn_state.get("energy_spent",    0)   / 4,    1.0)
        obs[base + 2]  = min(turn_state.get("attacks_played",  0)   / 5,    1.0)
        obs[base + 3]  = min(turn_state.get("skills_played",   0)   / 5,    1.0)
        obs[base + 4]  = min(turn_state.get("powers_played",   0)   / 3,    1.0)
        obs[base + 5]  = min(turn_state.get("strength_gained", 0)   / 10,   1.0)
        obs[base + 6]  = 1.0 if turn_state.get("vulnerable_applied",  False) else 0.0
        obs[base + 7]  = 1.0 if turn_state.get("weak_applied",        False) else 0.0
        obs[base + 8]  = min(turn_state.get("damage_dealt",    0.0) / max_hp, 1.0)
        obs[base + 9]  = min(turn_state.get("block_gained",    0.0) / max_hp, 1.0)
        obs[base + 10] = 1.0 if turn_state.get("last_card_was_buff",  False) else 0.0
        obs[base + 11] = 1.0 if turn_state.get("last_card_was_debuff", False) else 0.0

    def _get_choices(self, screen: str, ss: dict, state: GameState,
                     card_scorer=None) -> list:
        gold = max(state.gold, 1)
        deck = state.deck

        def _syn(card_id):
            if card_scorer is not None:
                return card_scorer.score(card_id)
            return self._synergy(card_id, deck)

        if screen == "CARD_REWARD":
            return [
                {
                    "tier_value":    _TIER_VALUE.get(get_card_tier(c.get("id", "")), 0.4),
                    "synergy_score": _syn(c.get("id", "")),
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
                    "synergy_score": _syn(c.get("id", "")),
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/v3/test_run_encoder.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/v3/run_encoder.py tests/v3/test_run_encoder.py
git commit -m "feat(v3): V3RunEncoder — 250-feature obs with intent flags and turn context block"
```

---

## Task 5: V3RunEnv

**Files:**
- Create: `src/v3/run_env.py`
- Create: `tests/v3/test_run_env.py`

- [ ] **Step 1: Write failing tests**

`tests/v3/test_run_env.py`:

```python
import time
import pytest
from unittest.mock import MagicMock, patch
from src.v3.run_env import V3RunEnv, HungEpisodeError
from src.v3.run_encoder import V3RunEncoder
from src.v3.card_scorer import CardScorer
from src.v2.run_action_space import RunActionSpace
from tests.v2.helpers import make_state, make_game_over, make_card_reward


def make_env(timeout=20.0, scorer=None):
    comm = MagicMock()
    scorer = scorer or CardScorer(path="data/card_scores_test.json")
    env = V3RunEnv(communicator=comm, card_scorer=scorer, timeout_seconds=timeout)
    return env, comm


# --- observation space ---

def test_obs_space_shape():
    env, _ = make_env()
    assert env.observation_space.shape == (V3RunEncoder.OBS_SIZE,)


def test_action_space_size():
    env, _ = make_env()
    assert env.action_space.n == RunActionSpace.TOTAL_ACTIONS


# --- reset ---

def test_reset_returns_correct_obs_shape():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    obs, info = env.reset()
    assert obs.shape == (V3RunEncoder.OBS_SIZE,)


def test_reset_clears_turn_state():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env._turn_state["actions_taken"] = 5
    env.reset()
    assert env._turn_state["actions_taken"] == 0


# --- turn state tracking ---

def test_turn_state_updates_actions_taken_on_card_play():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env.reset()
    comm.receive_state.return_value = make_state()
    env.step(0)  # PLAY 1 (no-target slot 0)
    assert env._turn_state["actions_taken"] == 1


def test_turn_state_resets_on_end_action():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env.reset()
    env._turn_state["actions_taken"] = 3
    comm.receive_state.return_value = make_state()
    env.step(60)  # END
    assert env._turn_state["actions_taken"] == 0


def test_turn_state_resets_on_new_combat():
    """Transitioning from non-combat → combat resets turn state."""
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env.reset()
    env._current_state = make_card_reward()  # simulate non-combat
    env._turn_state["attacks_played"] = 3
    comm.receive_state.return_value = make_state()  # back to combat
    env.step(99)  # PROCEED
    assert env._turn_state["attacks_played"] == 0


# --- hung watchdog ---

def test_hung_episode_returns_truncated():
    env, comm = make_env(timeout=0.1)

    comm.receive_state.return_value = make_state()
    env.reset()

    def _hang():
        time.sleep(10)
        return make_state()

    comm.receive_state.side_effect = _hang
    obs, reward, terminated, truncated, info = env.step(60)

    assert truncated is True
    assert terminated is False
    assert reward == 0.0
    assert info.get("hung") is True


def test_hung_episode_not_counted_as_death(tmp_path):
    from src.run_tracker import RunTracker
    tracker = RunTracker(log_path=str(tmp_path / "runs.jsonl"))
    env, comm = make_env(timeout=0.1)
    env.run_tracker = tracker

    comm.receive_state.return_value = make_state()
    env.reset()

    def _hang():
        time.sleep(10)
    comm.receive_state.side_effect = _hang

    env.step(60)
    assert tracker.hung_count == 1
    assert tracker.run_number == 0  # no normal run recorded


# --- CardScorer integration ---

def test_card_scorer_updated_on_combat_end(tmp_path):
    scorer = CardScorer(path=str(tmp_path / "scores.json"))
    env, comm = make_env(scorer=scorer)
    comm.receive_state.return_value = make_state()
    env.reset()

    # Simulate a card play that adds to _combat_cards_played
    env._combat_cards_played = ["Bash"]
    env._combat_total_damage = 20.0
    env._combat_total_enemy_max_hp = 42.0

    # Simulate combat→non-combat transition (triggers _on_combat_end)
    env._on_combat_end()

    assert scorer.score("Bash") != pytest.approx(0.5)


# --- normal game over ---

def test_normal_game_over_returns_terminated():
    env, comm = make_env()
    comm.receive_state.return_value = make_state()
    env.reset()
    comm.receive_state.return_value = make_game_over(floor=10)
    obs, reward, terminated, truncated, info = env.step(60)
    assert terminated is True
    assert truncated is False
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/v3/test_run_env.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.v3.run_env'`

- [ ] **Step 3: Implement V3RunEnv**

`src/v3/run_env.py`:

```python
import logging
import threading
from typing import Optional

import numpy as np
from gymnasium import spaces

from src.communicator import Communicator
from src.game_state import GameState
from src.run_tracker import RunTracker
from src.card_properties import get_card_properties
from src.v2.run_env import RunEnv
from src.v2.run_action_space import RunActionSpace
from src.v3.run_encoder import V3RunEncoder
from src.v3.run_reward import V3RunRewardShaper
from src.v3.card_scorer import CardScorer

logger = logging.getLogger(__name__)


class HungEpisodeError(Exception):
    pass


class V3RunEnv(RunEnv):
    def __init__(
        self,
        communicator: Communicator,
        run_tracker: Optional[RunTracker] = None,
        card_scorer: Optional[CardScorer] = None,
        timeout_seconds: float = 20.0,
    ):
        super().__init__(communicator, run_tracker)
        self._timeout_seconds = timeout_seconds
        self.card_scorer      = card_scorer or CardScorer()

        # Override encoder, reward shaper, and observation space
        self.encoder       = V3RunEncoder()
        self.reward_shaper = V3RunRewardShaper()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(V3RunEncoder.OBS_SIZE,),
            dtype=np.float32,
        )

        self._turn_state: dict = self._empty_turn_state()
        self._combat_cards_played: list[str] = []
        self._combat_total_damage: float = 0.0
        self._combat_total_enemy_max_hp: float = 0.0

    @staticmethod
    def _empty_turn_state() -> dict:
        return {
            "actions_taken": 0, "energy_spent": 0,
            "attacks_played": 0, "skills_played": 0, "powers_played": 0,
            "strength_gained": 0, "vulnerable_applied": False, "weak_applied": False,
            "damage_dealt": 0.0, "block_gained": 0.0,
            "last_card_was_buff": False, "last_card_was_debuff": False,
        }

    def _obs(self, state: Optional[GameState] = None) -> np.ndarray:
        s = state or self._current_state
        return self.encoder.encode(s, self._turn_state, self.card_scorer)

    def _reset_combat_tracking(self) -> None:
        self._combat_cards_played = []
        self._combat_total_damage = 0.0
        self._combat_total_enemy_max_hp = 0.0

    # --- timeout-protected receive ---

    def _receive_with_timeout(self) -> GameState:
        result: list = [None]
        error: list  = [None]

        def _recv():
            try:
                result[0] = self.communicator.receive_state()
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_recv, daemon=True)
        t.start()
        t.join(self._timeout_seconds)
        if t.is_alive():
            raise HungEpisodeError(f"No game response after {self._timeout_seconds}s")
        if error[0]:
            raise error[0]
        return result[0]

    def _next_actionable_state(self) -> GameState:
        while True:
            state = self._receive_with_timeout()
            if state is None:
                raise RuntimeError("Connection closed")
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

    # --- reset ---

    def reset(self, seed=None, options=None):
        self._turn_state = self._empty_turn_state()
        self._reset_combat_tracking()
        # super().reset() calls self.encoder.encode(state) via V3RunEncoder with
        # turn_state=None → turn context block = zeros. Shape is correct (250).
        return super().reset(seed=seed, options=options)

    # --- step ---

    def step(self, action: int):
        assert self._current_state is not None, "Call reset() first"

        # Inherited debuff-turn tracking
        if (self._current_state.is_in_combat and
                self._current_state.turn != self._current_turn):
            self._debuff_applied_this_turn = False
            self._current_turn = self._current_state.turn

        prev    = self._current_state
        command = self._action_space_helper.action_to_command(action, prev)
        logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                    prev.floor, prev.current_hp, prev.max_hp, prev.screen_type, command)

        writer = self.run_tracker.live_state_writer
        if writer:
            writer.write(prev, command)

        self.communicator.send_command(command)

        # Receive with timeout
        try:
            state = self._receive_with_timeout()
        except HungEpisodeError:
            logger.warning("Hung episode at floor %d", prev.floor)
            self.run_tracker.record_hung()
            return self._obs(), 0.0, False, True, {"hung": True, "floor": prev.floor}

        if state is None:
            return self._obs(), 0.0, True, False, {}

        if state.screen_type == "GAME_OVER":
            return self._handle_game_over(prev, state)

        # Combat transition detection
        prev_in_combat = prev.is_in_combat
        new_in_combat  = state.is_in_combat
        if prev_in_combat and not new_in_combat:
            self._on_combat_end()
        if not prev_in_combat and new_in_combat:
            self._reset_combat_tracking()
            self._turn_state = self._empty_turn_state()

        # Reward
        reward = self._compute_reward(action, prev, state)

        # Turn state + combat tracking updates
        if prev_in_combat:
            self._update_combat_tracking(action, prev, state)
            if action == RunActionSpace.END_TURN:
                self._turn_state = self._empty_turn_state()

        # Inherited per-episode metrics
        self._episode_reward_total += reward
        bucket = ("play" if action <= 59 else "end" if action == 60
                  else "potion" if action <= 90 else "noncombat")
        self._action_buckets[bucket] += 1
        if action == RunActionSpace.END_TURN and prev.is_in_combat:
            self._turn_energy_remaining.append(prev.energy)
        if prev.screen_type == "CARD_REWARD" and 91 <= action <= 98:
            self._track_card_pick(action, prev)
        if prev.is_in_combat:
            self._update_debuff_tracking(action, prev)

        self._current_state = state
        return self._obs(state), reward, False, False, {}

    # --- combat tracking ---

    def _update_combat_tracking(self, action: int, prev: GameState,
                                new: GameState) -> None:
        ts = self._turn_state

        # Energy spent this action
        if new.is_in_combat:
            ts["energy_spent"] += max(prev.energy - new.energy, 0)

        # Damage dealt
        prev_mon_hp = sum(m.get("current_hp", 0) for m in prev.monsters
                          if not m.get("is_gone"))
        new_mon_hp  = sum(m.get("current_hp", 0) for m in (new.monsters if new.is_in_combat else [])
                          if not m.get("is_gone"))
        damage = max(prev_mon_hp - new_mon_hp, 0)
        ts["damage_dealt"]          += damage
        self._combat_total_damage   += damage

        # Seed total enemy max HP once per combat
        if not self._combat_total_enemy_max_hp and prev.is_in_combat:
            self._combat_total_enemy_max_hp = sum(
                m.get("max_hp", 0) for m in prev.monsters if not m.get("is_gone")
            )

        # Block gained
        new_block = new.player_block if new.is_in_combat else 0
        ts["block_gained"] += max(new_block - prev.player_block, 0)

        # Strength gained
        def _strength(s):
            return next((p.get("amount", 0) for p in (s.combat_state or {})
                         .get("player", {}).get("powers", [])
                         if p.get("id") == "Strength"), 0)
        ts["strength_gained"] += max(_strength(new) - _strength(prev), 0)

        # Debuffs applied
        if new.is_in_combat:
            for m_new, m_prev in zip(new.monsters[:5], prev.monsters[:5]):
                if m_new.get("is_gone"):
                    continue
                def _stacks(m, pid):
                    return next((p.get("amount", 0) for p in m.get("powers", [])
                                 if p.get("id") == pid), 0)
                if _stacks(m_new, "Vulnerable") > _stacks(m_prev, "Vulnerable"):
                    ts["vulnerable_applied"] = True
                if _stacks(m_new, "Weak") > _stacks(m_prev, "Weak"):
                    ts["weak_applied"] = True

        # Card-specific tracking (actions 0–59 = card plays)
        if action < 60:
            ts["actions_taken"] += 1
            slot = action if action < 10 else (action - 10) // 5
            if slot < len(prev.hand):
                card      = prev.hand[slot]
                card_id   = card.get("id", "")
                card_type = card.get("type", "")
                if card_type == "ATTACK":
                    ts["attacks_played"] += 1
                elif card_type == "SKILL":
                    ts["skills_played"] += 1
                elif card_type == "POWER":
                    ts["powers_played"] += 1
                props                    = get_card_properties(card_id)
                ts["last_card_was_buff"] = (card_type == "POWER")
                ts["last_card_was_debuff"] = bool(
                    props.get("applies_vulnerable") or props.get("applies_weak")
                )
                self._combat_cards_played.append(card_id)

    def _on_combat_end(self) -> None:
        if not self._combat_cards_played:
            self._reset_combat_tracking()
            self._turn_state = self._empty_turn_state()
            return
        denom       = max(self._combat_total_enemy_max_hp, 1.0)
        performance = min(self._combat_total_damage / denom, 1.0)
        self.card_scorer.update(self._combat_cards_played, performance)
        self.card_scorer.save()
        self._reset_combat_tracking()
        self._turn_state = self._empty_turn_state()

    # --- game over ---

    def _handle_game_over(self, prev: GameState, state: GameState):
        reward = self.reward_shaper.terminal_reward(state.floor)
        self._episode_reward_total += reward

        energy_efficiency = 1.0
        if self._turn_energy_remaining:
            energy_efficiency = 1.0 - sum(self._turn_energy_remaining) / (
                len(self._turn_energy_remaining) * 3  # Ironclad A0 max = 3
            )

        self.run_tracker.record_run(
            state, version="v3",
            episode_reward=round(self._episode_reward_total, 4),
            energy_efficiency=round(energy_efficiency, 4),
        )

        writer = self.run_tracker.live_state_writer
        if writer:
            writer.write_v2_metrics(
                action_counts=dict(self._action_buckets),
                card_picks=list(self._card_picks_this_run),
                episode_reward=round(self._episode_reward_total, 4),
                energy_efficiency=round(energy_efficiency, 4),
            )

        summary = self.run_tracker.summary()
        logger.info(
            "GAME_OVER | floor=%d | runs=%d | win_rate=%.1f%% | reward=%.3f | energy=%.0f%%",
            state.floor, summary["total_runs"], summary["win_rate"] * 100,
            self._episode_reward_total, energy_efficiency * 100,
        )
        self.communicator.send_command("PROCEED")
        obs = self.encoder.encode(prev, self._turn_state, self.card_scorer)
        self._current_state = None
        return obs, reward, True, False, {"episode": {"r": reward, "floor": state.floor}}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/v3/test_run_env.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/v3/run_env.py tests/v3/test_run_env.py
git commit -m "feat(v3): V3RunEnv — turn tracking, hung watchdog, CardScorer integration"
```

---

## Task 6: RunTracker + dashboard

**Files:**
- Modify: `src/run_tracker.py`
- Modify: `dashboard.py`

- [ ] **Step 1: Add `hung_count` and `record_hung()` to RunTracker**

In `src/run_tracker.py`, add `hung_count: int = 0` to `__init__` and add `record_hung()`:

```python
# In __init__, after self.runs: list[dict] = []:
self.hung_count: int = 0
```

Add method after `record_run`:

```python
def record_hung(self) -> None:
    self.hung_count += 1
    logger.info("Hung episode #%d recorded (program failure, not a death)", self.hung_count)
    if self.live_state_writer:
        self.live_state_writer.write_run_summary(self.summary())

def summary(self) -> dict:
    wins   = sum(1 for r in self.runs if r["result"] == "win")
    losses = sum(1 for r in self.runs if r["result"] == "loss")
    total  = len(self.runs)
    return {
        "total_runs":  total,
        "wins":        wins,
        "losses":      losses,
        "hung":        self.hung_count,
        "win_rate":    wins / total if total > 0 else 0,
        "avg_floor":   sum(r["floor_reached"] for r in self.runs) / total if total > 0 else 0,
    }
```

- [ ] **Step 2: Verify RunTracker change**

```bash
python -c "from src.run_tracker import RunTracker; t = RunTracker(); t.record_hung(); print(t.hung_count)"
```

Expected output: `1`

- [ ] **Step 3: Add Hung stat chip to dashboard**

Open `dashboard.py` and find the stat chips section (search for `win_rate` or `total_runs` in the HTML string). Add a "Hung" chip alongside the existing Wins/Deaths chips. The exact HTML depends on the dashboard implementation, but the pattern is:

Find the existing chip for deaths/losses (e.g., `data.stats.losses` or similar) and add after it:

```html
<div class="chip">
  <div class="chip-label">HUNG</div>
  <div class="chip-value" id="hung-count">—</div>
</div>
```

And in the JavaScript that populates chips, add:

```javascript
document.getElementById('hung-count').textContent =
    data.stats?.hung ?? data.stats?.hung_count ?? '—';
```

The exact location depends on which HTML block in dashboard.py handles the run stats. Search for `total_runs` in dashboard.py to find the right section.

- [ ] **Step 4: Run existing dashboard tests**

```bash
pytest tests/test_dashboard.py -v
```

Expected: Existing tests pass (the Hung chip is additive, shouldn't break existing routes).

- [ ] **Step 5: Commit**

```bash
git add src/run_tracker.py dashboard.py
git commit -m "feat(v3): add hung_count to RunTracker and Hung stat chip to dashboard"
```

---

## Task 7: main.py — `--v3` entry point

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Verify MaskableRecurrentPPO is available**

```bash
python -c "from sb3_contrib import MaskableRecurrentPPO; print('ok')"
```

If this fails with `ImportError`, run:

```bash
pip install sb3-contrib --upgrade
```

Then retry. If `MaskableRecurrentPPO` still isn't available (older sb3-contrib), install the latest:

```bash
pip install "sb3-contrib>=2.3.0"
```

- [ ] **Step 2: Find the `--v2` block in main.py**

Read `main.py` and locate the `elif "--v2" in sys.argv:` block. The `--v3` entry point goes directly after it, following the same pattern.

- [ ] **Step 3: Add `--v3` block to main.py**

After the `elif "--v2" in sys.argv:` block, add:

```python
elif "--v3" in sys.argv:
    from sb3_contrib import MaskableRecurrentPPO
    from src.v3.run_env import V3RunEnv
    from src.v3.card_scorer import CardScorer

    card_scorer = CardScorer(path="data/card_scores.json")
    env = V3RunEnv(
        communicator=communicator,
        run_tracker=tracker,
        card_scorer=card_scorer,
        timeout_seconds=20.0,
    )

    checkpoint_path = "data/v3_checkpoints/model"
    model_path      = "data/v3_run_model.zip"

    if os.path.exists(model_path):
        logger.info("Loading existing v3 model from %s", model_path)
        model = MaskableRecurrentPPO.load(model_path, env=env)
    else:
        logger.info("Creating new v3 MaskableRecurrentPPO model")
        model = MaskableRecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            lstm_hidden_size=256,
            tensorboard_log="data/v3_tensorboard/",
        )

    os.makedirs("data/v3_checkpoints", exist_ok=True)
    callbacks = build_callbacks(
        checkpoint_path=checkpoint_path,
        live_writer=live_writer,
    )
    model.learn(total_timesteps=10_000_000, callback=callbacks)
    model.save(model_path)
```

Note: `build_callbacks`, `communicator`, `tracker`, and `live_writer` are assumed to already exist in main.py from the `--v2` entry point. Use exactly the same variable names that the `--v2` block uses.

- [ ] **Step 4: Verify import chain**

```bash
python -c "
from src.v3.run_env import V3RunEnv
from src.v3.run_encoder import V3RunEncoder
from src.v3.run_reward import V3RunRewardShaper
from src.v3.card_scorer import CardScorer
from sb3_contrib import MaskableRecurrentPPO
print('all imports ok')
"
```

Expected: `all imports ok`

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -q --ignore=tests/test_dashboard.py
```

Expected: No new failures. (Existing `test_dashboard.py` failures are pre-existing from v2.)

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat(v3): wire --v3 entry point with MaskableRecurrentPPO and V3RunEnv"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] MaskableRecurrentPPO (LSTM) — Task 7
- [x] 250-feature obs (intent flags + turn context) — Task 4
- [x] Turn context block 12 features — Task 4
- [x] Intent redesign (is_attacking/is_buffing/is_debuffing + aggregates) — Task 4
- [x] Relic rewards strengthened — Task 2
- [x] Energy waste penalty -0.5 — Task 2
- [x] CardScorer EMA — Task 3
- [x] CardScorer feeds into non-combat encoder synergy_score — Task 4
- [x] Hung watchdog 20s timeout — Task 5
- [x] Hung returns truncated=True, info["hung"]=True — Task 5
- [x] Hung tracked separately in RunTracker — Task 6
- [x] Hung shown in dashboard — Task 6
- [x] V2 untouched — all v3 code in src/v3/, tests/v3/
- [x] Entry point main.py --v3 — Task 7
