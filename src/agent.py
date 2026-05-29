# src/agent.py
import logging
from abc import ABC, abstractmethod

from src.game_state import GameState

logger = logging.getLogger(__name__)


class Agent(ABC):
    @abstractmethod
    def act(self, state: GameState) -> str:
        """Given a game state, return a command string."""
        pass


class SimpleAgent(Agent):
    """Rule-based agent that plays Ironclad with simple heuristics."""

    def act(self, state: GameState) -> str:
        if state.is_in_combat:
            return self._handle_combat(state)

        if state.screen_type == "CARD_REWARD":
            return self._handle_card_reward(state)

        if state.screen_type == "REST":
            return self._handle_rest(state)

        if state.screen_type == "MAP":
            return self._handle_map(state)

        if state.screen_type == "COMBAT_REWARD":
            return "PROCEED"

        if state.screen_type == "BOSS_REWARD":
            return "CHOOSE 0"

        if "PROCEED" in state.available_commands:
            return "PROCEED"

        if "CHOOSE" in state.available_commands:
            return "CHOOSE 0"

        return "STATE"

    def _handle_combat(self, state: GameState) -> str:
        playable = [
            (i, card) for i, card in enumerate(state.hand)
            if card.get("is_playable", False)
        ]

        if not playable:
            return "END"

        # Find first living monster for targeting
        target = 0
        for i, m in enumerate(state.monsters):
            if not m.get("is_gone", False):
                target = i
                break

        # Play first playable card (1-indexed)
        idx, card = playable[0]
        card_index = idx + 1  # CommunicationMod uses 1-indexed cards
        if card.get("has_target", False):
            return f"PLAY {card_index} {target}"
        return f"PLAY {card_index}"

    def _handle_card_reward(self, state: GameState) -> str:
        # Always pick first offered card for now
        if "CHOOSE" in state.available_commands:
            return "CHOOSE 0"
        return "PROCEED"

    def _handle_rest(self, state: GameState) -> str:
        # Rest if below 60% HP, otherwise smith
        hp_ratio = state.current_hp / max(state.max_hp, 1)
        if hp_ratio < 0.6:
            return "CHOOSE rest"
        return "CHOOSE smith"

    def _handle_map(self, state: GameState) -> str:
        return "CHOOSE 0"
