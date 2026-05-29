# Improved Rule-Based Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make SimpleAgent collect rewards, use potions, and pick cards intelligently via an Ironclad tier list.

**Architecture:** Add a card tier list module, update the combat reward handler to iterate through rewards, add potion usage logic in combat, and improve card selection with tier-based ranking.

**Tech Stack:** Python 3.11+, pytest

---

### Task 1: Ironclad Card Tier List

**Files:**
- Create: `src/card_tier_list.py`
- Create: `tests/test_card_tier_list.py`

**Step 1: Write failing tests**

```python
# tests/test_card_tier_list.py
from src.card_tier_list import IRONCLAD_TIERS, get_card_tier, pick_best_card


def test_known_card_has_tier():
    assert get_card_tier("Offering") == "S"
    assert get_card_tier("Shrug It Off") == "A"
    assert get_card_tier("Strike_R") == "D"


def test_unknown_card_defaults_to_c():
    assert get_card_tier("SomeModdedCard") == "C"


def test_pick_best_card_chooses_highest_tier():
    cards = [
        {"id": "Strike_R", "name": "Strike"},
        {"id": "Offering", "name": "Offering"},
        {"id": "Shrug It Off", "name": "Shrug It Off"},
    ]
    best = pick_best_card(cards)
    assert best == 1  # index of Offering (S tier)


def test_pick_best_card_breaks_ties_by_order():
    cards = [
        {"id": "Shrug It Off", "name": "Shrug It Off"},
        {"id": "Inflame", "name": "Inflame"},
    ]
    best = pick_best_card(cards)
    assert best == 0  # both A tier, first wins


def test_pick_best_card_empty_returns_none():
    assert pick_best_card([]) is None
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_card_tier_list.py -v`
Expected: FAIL with import errors

**Step 3: Implement card tier list**

```python
# src/card_tier_list.py

# Ironclad card tiers: S (best) > A > B > C > D (worst)
# Cards not listed default to C tier
IRONCLAD_TIERS = {
    "S": [
        "Offering", "Impervious", "Feed", "Reaper",
        "Demon Form", "Barricade", "Limit Break",
    ],
    "A": [
        "Shrug It Off", "Inflame", "Battle Trance", "Pommel Strike",
        "Flame Barrier", "Metallicize", "Disarm", "Clothesline",
        "Uppercut", "Shockwave", "Spot Weakness", "True Grit",
        "Body Slam", "Carnage", "Hemokinesis", "Blood for Blood",
        "Fiend Fire", "Brutality", "Dark Embrace", "Feel No Pain",
        "Corruption", "Berserk", "Juggernaut",
    ],
    "B": [
        "Armaments", "Thunderclap", "Iron Wave", "Power Through",
        "Ghostly Armor", "Rage", "Evolve", "Fire Breathing",
        "Combust", "Rupture", "Dual Wield", "Exhume",
        "Second Wind", "Entrench", "Whirlwind", "Immolate",
        "Seeing Red", "Burning Pact", "Sentinel", "Headbutt",
    ],
    "C": [
        "Warcry", "Flex", "Havoc", "Rampage",
        "Searing Blow", "Bloodletting", "Intimidate",
        "Dropkick", "Sever Soul", "Wild Strike",
        "Reckless Charge", "Cleave", "Twin Strike",
    ],
    "D": [
        "Strike_R", "Defend_R",
    ],
}

# Build reverse lookup: card_id -> tier
_CARD_TO_TIER = {}
for tier, cards in IRONCLAD_TIERS.items():
    for card in cards:
        _CARD_TO_TIER[card] = tier

_TIER_RANK = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}


def get_card_tier(card_id: str) -> str:
    return _CARD_TO_TIER.get(card_id, "C")


def pick_best_card(cards: list) -> int | None:
    if not cards:
        return None
    best_idx = 0
    best_rank = _TIER_RANK.get(get_card_tier(cards[0].get("id", "")), 3)
    for i, card in enumerate(cards[1:], start=1):
        rank = _TIER_RANK.get(get_card_tier(card.get("id", "")), 3)
        if rank < best_rank:
            best_rank = rank
            best_idx = i
    return best_idx
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_card_tier_list.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/card_tier_list.py tests/test_card_tier_list.py
git commit -m "feat: Ironclad card tier list with pick_best_card"
```

---

### Task 2: Combat Reward Handling

The COMBAT_REWARD screen has a `screen_state.rewards` list. Each reward has a `reward_type` (GOLD, POTION, RELIC, CARD, STOLEN_GOLD, SAPPHIRE_KEY). We need to CHOOSE each reward in order, then PROCEED when done. The tricky part: after choosing a CARD reward, the screen changes to CARD_REWARD — so the card selection happens there, not here.

**Files:**
- Modify: `src/agent.py`
- Modify: `tests/test_agent.py`

**Step 1: Write failing tests**

Add to `tests/test_agent.py`:

```python
COMBAT_REWARD_WITH_REWARDS = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "COMBAT_REWARD",
        "screen_state": {
            "rewards": [
                {"reward_type": "GOLD", "gold": 25},
                {"reward_type": "POTION", "potion": {"id": "Fire Potion", "name": "Fire Potion"}},
                {"reward_type": "CARD"},
            ]
        },
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Potion Slot", "name": "Potion Slot"},
            {"id": "Potion Slot", "name": "Potion Slot"},
        ],
        "map": [], "act": 1,
        "combat_state": None,
    }
})

COMBAT_REWARD_POTIONS_FULL = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "COMBAT_REWARD",
        "screen_state": {
            "rewards": [
                {"reward_type": "GOLD", "gold": 25},
                {"reward_type": "POTION", "potion": {"id": "Fire Potion", "name": "Fire Potion"}},
            ]
        },
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Fire Potion", "name": "Fire Potion"},
            {"id": "Block Potion", "name": "Block Potion"},
        ],
        "map": [], "act": 1,
        "combat_state": None,
    }
})

COMBAT_REWARD_NO_REWARDS = json.dumps({
    "available_commands": ["PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "COMBAT_REWARD",
        "screen_state": {"rewards": []},
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    }
})


def test_combat_reward_chooses_first_reward():
    state = GameState.from_json(COMBAT_REWARD_WITH_REWARDS)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action == "CHOOSE 0"


def test_combat_reward_no_rewards_proceeds():
    state = GameState.from_json(COMBAT_REWARD_NO_REWARDS)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action == "PROCEED"


def test_combat_reward_skips_potion_when_full():
    state = GameState.from_json(COMBAT_REWARD_POTIONS_FULL)
    agent = SimpleAgent()
    action = agent.act(state)
    # Should choose gold (index 0), not potion (index 1) since potions are full
    assert action == "CHOOSE 0"
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_agent.py -v`
Expected: new tests FAIL

**Step 3: Implement combat reward handling**

In `src/agent.py`, replace the COMBAT_REWARD line and add the helper:

```python
    # In act() method, replace:
    #     if state.screen_type == "COMBAT_REWARD":
    #         return "PROCEED"
    # with:
        if state.screen_type == "COMBAT_REWARD":
            return self._handle_combat_reward(state)
```

Add new method:

```python
    def _handle_combat_reward(self, state: GameState) -> str:
        rewards = []
        if state.screen_state:
            rewards = state.screen_state.get("rewards", [])

        if not rewards or "CHOOSE" not in state.available_commands:
            return "PROCEED"

        potion_slots_full = all(
            p.get("id") != "Potion Slot" for p in state.potions
        )

        for i, reward in enumerate(rewards):
            rtype = reward.get("reward_type", "")
            if rtype == "POTION" and potion_slots_full:
                continue
            return f"CHOOSE {i}"

        return "PROCEED"
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_agent.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: combat reward handling - collect gold, relics, potions, cards"
```

---

### Task 3: Card Selection with Tier List

Update `_handle_card_reward` to use `pick_best_card` instead of always picking index 0.

**Files:**
- Modify: `src/agent.py`
- Modify: `tests/test_agent.py`

**Step 1: Write failing test**

Add to `tests/test_agent.py`:

```python
CARD_REWARD_WITH_TIERS = json.dumps({
    "available_commands": ["CHOOSE", "PROCEED"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "CARD_REWARD",
        "screen_state": {
            "cards": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK"},
                {"id": "Offering", "name": "Offering", "cost": 0, "type": "SKILL"},
                {"id": "Inflame", "name": "Inflame", "cost": 1, "type": "POWER"},
            ]
        },
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
        "combat_state": None,
    }
})


def test_card_reward_picks_best_tier():
    state = GameState.from_json(CARD_REWARD_WITH_TIERS)
    agent = SimpleAgent()
    action = agent.act(state)
    # Offering is S tier (index 1), should pick it
    assert action == "CHOOSE 1"
```

**Step 2: Run tests to verify it fails**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_agent.py::test_card_reward_picks_best_tier -v`
Expected: FAIL (currently returns CHOOSE 0)

**Step 3: Update _handle_card_reward**

In `src/agent.py`, add import and update the method:

```python
# At top of file, add:
from src.card_tier_list import pick_best_card

# Replace _handle_card_reward:
    def _handle_card_reward(self, state: GameState) -> str:
        if "CHOOSE" not in state.available_commands:
            return "PROCEED"
        cards = []
        if state.screen_state:
            cards = state.screen_state.get("cards", [])
        best = pick_best_card(cards)
        if best is not None:
            return f"CHOOSE {best}"
        return "PROCEED"
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/Scripts/activate && python -m pytest -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: card selection using Ironclad tier list"
```

---

### Task 4: Potion Usage in Combat

Use potions during combat: healing potions when below 40% HP, damage/buff potions against elites and bosses (detected by monster max_hp > 100 as a rough heuristic, since CommunicationMod doesn't directly expose fight type).

**Files:**
- Modify: `src/agent.py`
- Modify: `tests/test_agent.py`

**Step 1: Write failing tests**

Add to `tests/test_agent.py`:

```python
COMBAT_LOW_HP_WITH_POTION = json.dumps({
    "available_commands": ["PLAY", "END", "POTION"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 20, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Fruit Juice", "name": "Fruit Juice", "can_use": True, "can_discard": True, "requires_target": False},
        ],
        "map": [], "act": 1,
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a1"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42, "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 20, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
    }
})

COMBAT_FULL_HP_WITH_POTION = json.dumps({
    "available_commands": ["PLAY", "END", "POTION"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 80, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Fruit Juice", "name": "Fruit Juice", "can_use": True, "can_discard": True, "requires_target": False},
        ],
        "map": [], "act": 1,
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a1"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42, "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
    }
})

COMBAT_ELITE_WITH_ATTACK_POTION = json.dumps({
    "available_commands": ["PLAY", "END", "POTION"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
        "current_hp": 70, "max_hp": 80, "gold": 99,
        "deck": [], "relics": [],
        "potions": [
            {"id": "Fire Potion", "name": "Fire Potion", "can_use": True, "can_discard": True, "requires_target": True},
        ],
        "map": [], "act": 1,
        "combat_state": {
            "hand": [
                {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                 "is_playable": True, "has_target": True, "uuid": "a1"},
            ],
            "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
            "monsters": [
                {"name": "Gremlin Nob", "current_hp": 106, "max_hp": 106, "block": 0, "intent": "ATTACK", "is_gone": False},
            ],
            "player": {"current_hp": 70, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
    }
})


def test_use_potion_when_low_hp():
    state = GameState.from_json(COMBAT_LOW_HP_WITH_POTION)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action == "POTION Use 0"


def test_no_potion_when_full_hp():
    state = GameState.from_json(COMBAT_FULL_HP_WITH_POTION)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action.startswith("PLAY")


def test_use_attack_potion_on_elite():
    state = GameState.from_json(COMBAT_ELITE_WITH_ATTACK_POTION)
    agent = SimpleAgent()
    action = agent.act(state)
    assert action == "POTION Use 0 0"
```

**Step 2: Run tests to verify they fail**

Run: `source .venv/Scripts/activate && python -m pytest tests/test_agent.py -v`
Expected: New tests FAIL

**Step 3: Implement potion usage**

Update `_handle_combat` in `src/agent.py`. Add potion check before card play:

```python
    # Known healing potions
    HEALING_POTIONS = {"Fruit Juice", "Blood Potion", "Fairy in a Bottle",
                       "Regen Potion", "Ancient Potion"}

    # Known attack/buff potions (non-healing, usable offensively)
    ATTACK_POTIONS = {"Fire Potion", "Explosive Potion", "Poison Potion",
                      "Fear Potion", "Strength Potion", "Dexterity Potion",
                      "Speed Potion", "Weak Potion", "Energy Potion",
                      "Swift Potion", "Flex Potion", "Steroid Potion",
                      "Focus Potion", "Cultist Potion", "Liquid Bronze",
                      "Essence of Steel", "Heart of Iron", "Ghost In A Jar",
                      "Ambrosia", "Liquid Memories", "Distilled Chaos",
                      "Duplication Potion", "Blessing of the Forge",
                      "Elixir", "Gambler's Brew", "Entropic Brew",
                      "Smoke Bomb", "Snecko Oil", "Block Potion"}

    def _handle_combat(self, state: GameState) -> str:
        if "POTION" in state.available_commands:
            potion_action = self._check_potions(state)
            if potion_action:
                return potion_action

        playable = [
            (i, card) for i, card in enumerate(state.hand)
            if card.get("is_playable", False)
        ]

        if not playable:
            return "END"

        # Find first living monster for targeting
        target = 0
        for i, m in enumerate(state.monsters):
            if not m.get("is_gone", False):
                target = i
                break

        # Play first playable card (1-indexed)
        idx, card = playable[0]
        card_index = idx + 1  # CommunicationMod uses 1-indexed cards
        if card.get("has_target", False):
            return f"PLAY {card_index} {target}"
        return f"PLAY {card_index}"

    def _check_potions(self, state: GameState) -> str | None:
        hp_ratio = state.current_hp / max(state.max_hp, 1)
        is_tough_fight = any(
            m.get("max_hp", 0) > 100 for m in state.monsters
            if not m.get("is_gone", False)
        )

        target = 0
        for i, m in enumerate(state.monsters):
            if not m.get("is_gone", False):
                target = i
                break

        for i, potion in enumerate(state.potions):
            if not potion.get("can_use", False):
                continue
            pid = potion.get("id", "")

            # Use healing potions when low
            if pid in self.HEALING_POTIONS and hp_ratio < 0.4:
                if potion.get("requires_target", False):
                    return f"POTION Use {i} {target}"
                return f"POTION Use {i}"

            # Use attack/buff potions on tough fights (turn 1 only for buffs)
            if pid in self.ATTACK_POTIONS and is_tough_fight:
                if potion.get("requires_target", False):
                    return f"POTION Use {i} {target}"
                return f"POTION Use {i}"

        return None
```

**Step 4: Run tests to verify they pass**

Run: `source .venv/Scripts/activate && python -m pytest -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: potion usage in combat - healing when low, attack on elites"
```

---

### Task 5: Full Test Suite Run + Integration Commit

**Step 1: Run full test suite**

Run: `source .venv/Scripts/activate && python -m pytest -v`
Expected: All tests PASS

**Step 2: Manual integration test**

Launch Slay the Spire and start a run. Verify:
- Bot picks up gold, relics, and potions from combat rewards
- Bot selects cards using tier list (not always index 0)
- Bot uses healing potions when low HP
- Bot uses attack potions on tough fights
- Bot skips potions when slots are full

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes from improved agent testing"
```
