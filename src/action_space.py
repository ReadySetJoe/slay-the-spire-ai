import numpy as np
from src.game_state import GameState

MAX_HAND = 10
MAX_TARGETS = 5


class ActionSpace:
    # Actions 0-9: play card in slot 0-9 (no target)
    # Actions 10-59: play card in slot 0-9 on target 0-4
    #   action = 10 + slot * MAX_TARGETS + target
    # Action 60: end turn
    TOTAL_ACTIONS = MAX_HAND + MAX_HAND * MAX_TARGETS + 1  # 61
    END_TURN_ACTION = TOTAL_ACTIONS - 1  # 60

    def get_action_mask(self, state: GameState) -> np.ndarray:
        mask = np.zeros(self.TOTAL_ACTIONS, dtype=np.bool_)

        # End turn is always valid in combat
        mask[self.END_TURN_ACTION] = True

        # Find living monster indices
        living_targets = set()
        for i, m in enumerate(state.monsters[:MAX_TARGETS]):
            if not m.get("is_gone", False):
                living_targets.add(i)

        # Card actions
        for slot, card in enumerate(state.hand[:MAX_HAND]):
            if not card.get("is_playable", False):
                continue

            if card.get("has_target", False):
                # Targeted card: only targeted actions valid
                for target in living_targets:
                    action = MAX_HAND + slot * MAX_TARGETS + target
                    mask[action] = True
            else:
                # Untargeted card: only no-target action valid
                mask[slot] = True

        return mask

    def action_to_command(self, action: int, state: GameState) -> str:
        if action == self.END_TURN_ACTION:
            return "END"

        if action < MAX_HAND:
            # No-target card play
            card_index = action + 1  # 1-indexed
            return f"PLAY {card_index}"

        # Targeted card play
        action -= MAX_HAND
        slot = action // MAX_TARGETS
        target = action % MAX_TARGETS
        card_index = slot + 1  # 1-indexed
        return f"PLAY {card_index} {target}"
