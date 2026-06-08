import pytest
import numpy as np
from src.v2.run_action_space import RunActionSpace
from tests.v2.helpers import (
    make_state, make_card_reward, make_shop, make_rest, make_map, make_game_over
)


@pytest.fixture
def space():
    return RunActionSpace()


@pytest.fixture
def combat_state():
    return make_state()


def test_play_no_target_slot_0(space, combat_state):
    assert space.action_to_command(0, combat_state) == "PLAY 1"


def test_play_no_target_slot_9(space, combat_state):
    assert space.action_to_command(9, combat_state) == "PLAY 10"


def test_play_targeted_slot_0_target_0(space, combat_state):
    assert space.action_to_command(10, combat_state) == "PLAY 1 0"


def test_play_targeted_slot_0_target_4(space, combat_state):
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
