import numpy as np
import pytest
from src.v2.run_encoder import RunEncoder
from tests.v2.helpers import make_state, make_card_reward, make_shop, make_rest, make_map

GLOBAL_SIZE    = 55
COMBAT_SIZE    = 112
NONCOMBAT_SIZE = 60
OBS_SIZE       = GLOBAL_SIZE + COMBAT_SIZE + NONCOMBAT_SIZE  # 227

INTENT_MAP = {
    "ATTACK": 0.2, "ATTACK_BUFF": 0.3, "ATTACK_DEBUFF": 0.35,
    "ATTACK_DEFEND": 0.4, "BUFF": 0.5, "DEBUFF": 0.6,
    "STRONG_DEBUFF": 0.65, "DEFEND": 0.7, "DEFEND_BUFF": 0.75,
    "ESCAPE": 0.8, "MAGIC": 0.85, "SLEEP": 0.1, "STUN": 0.9,
    "UNKNOWN": 0.5, "NONE": 0.0,
}
CARD_TYPE_MAP = {"ATTACK": 0.25, "SKILL": 0.5, "POWER": 0.75,
                 "STATUS": 0.9, "CURSE": 1.0}

COMBAT_BASE    = GLOBAL_SIZE           # 55
NONCOMBAT_BASE = GLOBAL_SIZE + COMBAT_SIZE  # 167


@pytest.fixture
def enc():
    return RunEncoder()


# ---- shape and range ----

def test_obs_shape(enc):
    obs = enc.encode(make_state())
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32


def test_obs_values_in_range(enc):
    obs = enc.encode(make_state())
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


# ---- global block ----

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


# ---- combat block ----

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


# ---- non-combat block ----

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
    # REST: hp=50, max_hp=80 → heal = min(80*0.3=24, 80-50=30) = 24
    state = make_rest(hp=50, max_hp=80)
    obs = enc.encode(state)
    # Screen metadata starts at NONCOMBAT_BASE + 32 (choices) + 5 (synergy) = NONCOMBAT_BASE + 37
    hp_heal_idx = NONCOMBAT_BASE + 32 + 5 + 1
    assert obs[hp_heal_idx] == pytest.approx(24 / 80)


def test_map_node_elite_flag(enc):
    state = make_map(nodes=[{"symbol": "E"}, {"symbol": "M"}])
    obs = enc.encode(state)
    # Screen metadata feature 2 = node_elite_flag
    node_elite_idx = NONCOMBAT_BASE + 32 + 5 + 2
    assert obs[node_elite_idx] == pytest.approx(1.0)
