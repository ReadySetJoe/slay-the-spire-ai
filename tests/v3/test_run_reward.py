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
