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
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.bool_)

        if state.is_in_combat:
            self._mask_combat(mask, state)
            return mask

        screen = state.screen_type
        cmds   = state.available_commands
        ss     = state.screen_state or {}

        if screen == "CARD_REWARD":
            n = len(ss.get("cards", []))
            for i in range(min(n, MAX_CHOICES)):
                mask[_CHOOSE_START + i] = True
            if "PROCEED" in cmds:
                mask[_PROCEED] = True

        elif screen == "SHOP_SCREEN":
            n = len(ss.get("cards", [])) + len(ss.get("relics", []))
            for i in range(min(n, MAX_CHOICES)):
                mask[_CHOOSE_START + i] = True
            if "PURGE" in cmds:
                mask[_PURGE] = True
            mask[_PROCEED] = True

        elif screen == "MAP":
            n = len(ss.get("next_nodes", []))
            for i in range(min(n, MAX_CHOICES)):
                mask[_CHOOSE_START + i] = True

        elif screen == "REST":
            if "CHOOSE" in cmds:
                mask[_CHOOSE_REST]  = True
                mask[_CHOOSE_SMITH] = True

        elif screen == "CHEST":
            if "OPEN"    in cmds: mask[_OPEN]    = True
            if "PROCEED" in cmds: mask[_PROCEED] = True

        elif screen == "COMBAT_REWARD":
            if "PROCEED" in cmds:
                mask[_PROCEED] = True
            if "CHOOSE" in cmds:
                rewards = ss.get("rewards", [])
                potion_full = all(p.get("id") != "Potion Slot"
                                  for p in state.potions)
                for i, reward in enumerate(rewards[:MAX_CHOICES]):
                    if reward.get("reward_type") == "POTION" and potion_full:
                        continue
                    mask[_CHOOSE_START + i] = True

        elif screen in ("EVENT", "GRID", "HAND_SELECT", "BOSS_REWARD"):
            if "CHOOSE" in cmds:
                options = (ss.get("options") or ss.get("cards") or
                           ss.get("rewards") or [])
                n = min(len(options), MAX_CHOICES) if options else MAX_CHOICES
                for i in range(n):
                    mask[_CHOOSE_START + i] = True
            if "PROCEED" in cmds:
                mask[_PROCEED] = True

        elif screen == "SHOP_ROOM":
            mask[_PROCEED] = True

        else:
            if "PROCEED" in cmds:
                mask[_PROCEED] = True

        return mask

    def _mask_combat(self, mask: np.ndarray, state: GameState) -> None:
        mask[_END_TURN] = True

        living = {i for i, m in enumerate(state.monsters[:MAX_TARGETS])
                  if not m.get("is_gone", False)}

        for slot, card in enumerate(state.hand[:MAX_HAND]):
            if not card.get("is_playable", False):
                continue
            if card.get("has_target", False):
                for t in living:
                    mask[_PLAY_T_START + slot * MAX_TARGETS + t] = True
            else:
                mask[_PLAY_NT_START + slot] = True

        for slot, potion in enumerate(state.potions[:MAX_POTIONS]):
            if not potion.get("can_use", False):
                continue
            if potion.get("requires_target", False):
                for t in living:
                    mask[_POT_T_START + slot * MAX_TARGETS + t] = True
            else:
                mask[_POT_NT_START + slot] = True
