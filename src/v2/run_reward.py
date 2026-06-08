from src.card_tier_list import get_card_tier
from src.card_properties import get_card_properties

_TIER_VALUE = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}

_STRENGTH_CARDS = {"Inflame", "Spot Weakness", "Demon Form", "Flex", "Limit Break", "Berserk"}
_EXHAUST_CARDS  = {"True Grit", "Second Wind", "Corruption", "Fiend Fire",
                   "Feel No Pain", "Dark Embrace", "Burning Pact", "Sentinel", "Exhume"}
_DRAW_CARDS     = {"Battle Trance", "Pommel Strike", "Warcry", "Burning Pact", "Headbutt", "Exhume"}
_HIGH_DAMAGE    = {"Carnage", "Whirlwind", "Reaper", "Immolate",
                   "Fiend Fire", "Hemokinesis", "Blood for Blood"}


class RunRewardShaper:

    def terminal_reward(self, floor: int) -> float:
        return (floor / 55) * 3.0 - 1.0

    def combat_step_reward(
        self,
        prev_hp: int, new_hp: int,
        prev_monster_hp: int, new_monster_hp: int,
        prev_living: int, new_living: int,
        prev_debuffs: int, new_debuffs: int,
        max_hp: int,
        is_end_action: bool,
        energy_remaining: int, max_energy: int,
        card_is_attack: bool,
        debuff_applied_this_turn: bool,
    ) -> float:
        max_hp = max(max_hp, 1)

        damage_dealt = max(prev_monster_hp - new_monster_hp, 0) / max_hp
        damage_taken = max(prev_hp - new_hp, 0) / max_hp
        kills        = max(prev_living - new_living, 0)
        debuff_gain  = max(new_debuffs - prev_debuffs, 0)

        reward = (
            damage_dealt
            - damage_taken
            + 0.1 * kills
            + 0.05 * debuff_gain
        )

        if is_end_action:
            reward -= 0.3 * (energy_remaining / max(max_energy, 1))

        if card_is_attack and debuff_applied_this_turn:
            reward += 0.03

        return reward

    def card_reward(self, card: dict, deck: list) -> float:
        tier_val = _TIER_VALUE.get(get_card_tier(card.get("id", "")), 0.4)
        syn      = self._synergy(card.get("id", ""), deck)
        return tier_val * 0.05 + syn * 0.05

    def shop_card_reward(self, card: dict, gold: int, deck: list) -> float:
        tier_val   = _TIER_VALUE.get(get_card_tier(card.get("id", "")), 0.4)
        syn        = self._synergy(card.get("id", ""), deck)
        cost_ratio = min(card.get("price", 0) / max(gold, 1), 1.0)
        return tier_val * 0.05 + syn * 0.05 - cost_ratio * 0.02

    def shop_relic_reward(self) -> float:
        return 0.05

    def purge_reward(self, card: dict) -> float:
        tier = get_card_tier(card.get("id", ""))
        if tier == "D" or card.get("type") in ("STATUS", "CURSE"):
            return 0.03
        return 0.0

    def rest_heal_reward(self, hp_gained: int, max_hp: int) -> float:
        return (hp_gained / max(max_hp, 1)) * 0.2

    def rest_smith_reward(self) -> float:
        return 0.05

    def _synergy(self, card_id: str, deck: list) -> float:
        props    = get_card_properties(card_id)
        deck_ids = [c.get("id", "") for c in deck]
        score    = 0.0

        if card_id in _STRENGTH_CARDS:
            if sum(1 for d in deck_ids if d in _STRENGTH_CARDS) >= 2:
                score += 0.3
        if card_id in _EXHAUST_CARDS:
            if sum(1 for d in deck_ids if d in _EXHAUST_CARDS) >= 2:
                score += 0.3
        if props.get("applies_vulnerable"):
            if sum(1 for d in deck_ids if d in _HIGH_DAMAGE) >= 2:
                score += 0.3
        if card_id in _DRAW_CARDS:
            score += 0.2

        return min(score, 1.0)
