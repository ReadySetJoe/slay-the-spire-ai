# src/agent.py
import logging
from abc import ABC, abstractmethod
from typing import Optional

from src.card_tier_list import pick_best_card
from src.game_state import GameState

logger = logging.getLogger(__name__)


class Agent(ABC):
    @abstractmethod
    def act(self, state: GameState) -> str:
        """Given a game state, return a command string."""
        pass


class SimpleAgent(Agent):
    """Rule-based agent that plays Ironclad with simple heuristics."""

    HEALING_POTIONS = {"Fruit Juice", "Blood Potion", "Fairy in a Bottle",
                       "Regen Potion", "Ancient Potion"}

    ATTACK_POTIONS = {"Fire Potion", "Explosive Potion", "Poison Potion",
                      "Fear Potion", "Strength Potion", "Dexterity Potion",
                      "Speed Potion", "Weak Potion", "Energy Potion",
                      "Swift Potion", "Flex Potion", "Steroid Potion",
                      "Focus Potion", "Cultist Potion", "Liquid Bronze",
                      "Essence of Steel", "Heart of Iron", "Ghost In A Jar",
                      "Ambrosia", "Liquid Memories", "Distilled Chaos",
                      "Duplication Potion", "Blessing of the Forge",
                      "Elixir", "Gambler's Brew", "Entropic Brew",
                      "Smoke Bomb", "Snecko Oil", "Block Potion"}

    def __init__(self, scorer=None):
        self.scorer = scorer
        self._run_picks: list[str] = []

    def on_game_over(self, state: GameState):
        """Update card scores from this run's picks, then reset for next run."""
        if self.scorer and self._run_picks:
            quality = (state.current_hp / max(state.max_hp, 1)) * (state.floor / 55)
            logger.info(
                "Run quality=%.3f | %d cards picked: %s",
                quality, len(self._run_picks), self._run_picks,
            )
            self.scorer.update_run(self._run_picks, quality)
        self._run_picks.clear()

    def act(self, state: GameState) -> str:
        if state.is_in_combat:
            return self._handle_combat(state)

        if state.screen_type == "CARD_REWARD":
            return self._handle_card_reward(state)

        if state.screen_type == "REST":
            return self._handle_rest(state)

        if state.screen_type == "MAP":
            return self._handle_map(state)

        if state.screen_type == "CHEST":
            if "OPEN" in state.available_commands:
                return "OPEN"
            if "PROCEED" in state.available_commands:
                return "PROCEED"

        if state.screen_type == "EVENT":
            return "CHOOSE 0"

        if state.screen_type in ("SHOP_ROOM", "SHOP_SCREEN"):
            return "PROCEED"

        if state.screen_type in ("GRID", "HAND_SELECT"):
            if "CHOOSE" in state.available_commands:
                cards = state.screen_state.get("cards", []) if state.screen_state else []
                best = pick_best_card(cards) if cards else None
                return f"CHOOSE {best if best is not None else 0}"
            if "CONFIRM" in state.available_commands:
                return "CONFIRM"
            return "CANCEL"

        if state.screen_type == "COMBAT_REWARD":
            return self._handle_combat_reward(state)

        if state.screen_type == "BOSS_REWARD":
            return "CHOOSE 0"

        if "PROCEED" in state.available_commands:
            return "PROCEED"

        if "CHOOSE" in state.available_commands:
            return "CHOOSE 0"

        if "CONFIRM" in state.available_commands:
            return "CONFIRM"

        logger.warning("Unhandled screen type: %s | Commands: %s", state.screen_type, state.available_commands)
        return "STATE"

    def _handle_combat(self, state: GameState) -> str:
        if "POTION" in state.available_commands:
            potion_action = self._check_potions(state)
            if potion_action:
                return potion_action

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

    def _check_potions(self, state: GameState) -> str | None:
        hp_ratio = state.current_hp / max(state.max_hp, 1)
        is_tough_fight = any(
            m.get("max_hp", 0) > 100 for m in state.monsters
            if not m.get("is_gone", False)
        )

        target = 0
        for i, m in enumerate(state.monsters):
            if not m.get("is_gone", False):
                target = i
                break

        for i, potion in enumerate(state.potions):
            if not potion.get("can_use", False):
                continue
            pid = potion.get("id", "")

            # Use healing potions when low
            if pid in self.HEALING_POTIONS and hp_ratio < 0.4:
                if potion.get("requires_target", False):
                    return f"POTION Use {i} {target}"
                return f"POTION Use {i}"

            # Use attack/buff potions on tough fights
            if pid in self.ATTACK_POTIONS and is_tough_fight:
                if potion.get("requires_target", False):
                    return f"POTION Use {i} {target}"
                return f"POTION Use {i}"

        return None

    def _handle_card_reward(self, state: GameState) -> str:
        if "CHOOSE" not in state.available_commands:
            return "PROCEED"
        cards = state.screen_state.get("cards", []) if state.screen_state else []
        if not cards:
            return "PROCEED"

        if self.scorer:
            idx = self.scorer.softmax_pick(cards)
        else:
            idx = pick_best_card(cards) or 0

        self._run_picks.append(cards[idx].get("id", ""))
        return f"CHOOSE {idx}"

    def _handle_rest(self, state: GameState) -> str:
        if "CHOOSE" not in state.available_commands:
            return "PROCEED"
        # Rest if below 60% HP, otherwise smith
        hp_ratio = state.current_hp / max(state.max_hp, 1)
        if hp_ratio < 0.6:
            return "CHOOSE rest"
        return "CHOOSE smith"

    def _handle_combat_reward(self, state: GameState) -> str:
        rewards = []
        if state.screen_state:
            rewards = state.screen_state.get("rewards", [])

        if not rewards or "CHOOSE" not in state.available_commands:
            return "PROCEED"

        potion_slots_full = all(
            p.get("id") != "Potion Slot" for p in state.potions
        )

        for i, reward in enumerate(rewards):
            rtype = reward.get("reward_type", "")
            if rtype == "POTION" and potion_slots_full:
                continue
            return f"CHOOSE {i}"

        return "PROCEED"

    def _handle_map(self, state: GameState) -> str:
        return "CHOOSE 0"
