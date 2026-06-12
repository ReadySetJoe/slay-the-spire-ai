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
