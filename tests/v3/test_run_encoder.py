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
