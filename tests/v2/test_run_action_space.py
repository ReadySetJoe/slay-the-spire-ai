import pytest
from src.v2.run_action_space import RunActionSpace
from tests.v2.helpers import make_state


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
