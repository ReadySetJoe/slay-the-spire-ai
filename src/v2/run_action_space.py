import numpy as np
from src.game_state import GameState

MAX_HAND     = 10
MAX_TARGETS  = 5
MAX_POTIONS  = 5
MAX_CHOICES  = 8

_PLAY_NT_START = 0    # 0-9:   PLAY slot (no target)
_PLAY_T_START  = 10   # 10-59: PLAY slot target
_END_TURN      = 60   # 60:    END
_POT_NT_START  = 61   # 61-65: POTION Use slot
_POT_T_START   = 66   # 66-90: POTION Use slot target
_CHOOSE_START  = 91   # 91-98: CHOOSE 0-7
_PROCEED       = 99   # 99:    PROCEED
_PURGE         = 100  # 100:   PURGE
_CHOOSE_REST   = 101  # 101:   CHOOSE rest
_CHOOSE_SMITH  = 102  # 102:   CHOOSE smith
_OPEN          = 103  # 103:   OPEN

TOTAL_ACTIONS  = 104


class RunActionSpace:
    TOTAL_ACTIONS = TOTAL_ACTIONS
    END_TURN      = _END_TURN

    def action_to_command(self, action: int, state: GameState) -> str:
        if action < _PLAY_T_START:           # 0-9
            return f"PLAY {action + 1}"
        if action < _END_TURN:               # 10-59
            slot, target = divmod(action - _PLAY_T_START, MAX_TARGETS)
            return f"PLAY {slot + 1} {target}"
        if action == _END_TURN:              # 60
            return "END"
        if action < _POT_T_START:            # 61-65
            return f"POTION Use {action - _POT_NT_START}"
        if action < _CHOOSE_START:           # 66-90
            slot, target = divmod(action - _POT_T_START, MAX_TARGETS)
            return f"POTION Use {slot} {target}"
        if action < _PROCEED:                # 91-98
            return f"CHOOSE {action - _CHOOSE_START}"
        if action == _PROCEED:               # 99
            return "PROCEED"
        if action == _PURGE:                 # 100
            return "PURGE"
        if action == _CHOOSE_REST:           # 101
            return "CHOOSE rest"
        if action == _CHOOSE_SMITH:          # 102
            return "CHOOSE smith"
        if action == _OPEN:                  # 103
            return "OPEN"
        raise ValueError(f"Invalid action index {action}")

    def get_action_mask(self, state: GameState) -> np.ndarray:
        raise NotImplementedError
