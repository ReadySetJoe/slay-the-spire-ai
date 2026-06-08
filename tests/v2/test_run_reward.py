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
    r = _combat_reward(shaper, is_end_action=False, energy_remaining=2, max_energy=3)
    assert r == pytest.approx(0.0)


def test_debuff_before_damage_bonus(shaper):
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


# ---- non-combat rewards ----

def test_card_reward_s_tier(shaper):
    card = {"id": "Offering"}  # S-tier
    r = shaper.card_reward(card, deck=[])
    assert r == pytest.approx(1.0 * 0.05)


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
