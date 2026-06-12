import numpy as np
from src.game_state import GameState
from src.card_properties import get_card_properties
from src.card_tier_list import get_card_tier

GLOBAL_SIZE    = 55
COMBAT_SIZE    = 112
NONCOMBAT_SIZE = 60
OBS_SIZE       = GLOBAL_SIZE + COMBAT_SIZE + NONCOMBAT_SIZE  # 227

_SCREEN_ORDER = [
    "NONE", "CARD_REWARD", "REST", "MAP", "CHEST", "EVENT",
    "SHOP_ROOM", "SHOP_SCREEN", "GRID", "HAND_SELECT",
    "COMBAT_REWARD", "BOSS_REWARD",
]
_SCREEN_IDX = {s: i for i, s in enumerate(_SCREEN_ORDER)}

_HEALING_POTIONS = {
    "Fruit Juice", "Blood Potion", "Fairy in a Bottle",
    "Regen Potion", "Ancient Potion",
}
_ATTACK_POTIONS = {
    "Fire Potion", "Explosive Potion", "Poison Potion", "Fear Potion",
    "Strength Potion", "Dexterity Potion", "Speed Potion", "Weak Potion",
    "Energy Potion", "Swift Potion", "Flex Potion", "Steroid Potion",
    "Focus Potion", "Cultist Potion", "Liquid Bronze", "Essence of Steel",
    "Heart of Iron", "Ghost In A Jar", "Ambrosia", "Liquid Memories",
    "Distilled Chaos", "Duplication Potion", "Blessing of the Forge",
    "Elixir", "Gambler's Brew", "Entropic Brew", "Block Potion",
}

_HIGH_IMPACT_RELICS = [
    "Akabeko", "Anchor", "Burning Blood",
    "Centennial Puzzle", "Philosopher's Stone", "Astrolabe",
]

_STRENGTH_CARDS = {
    "Inflame", "Spot Weakness", "Demon Form", "Flex",
    "Limit Break", "Berserk",
}
_DRAW_CARDS = {
    "Battle Trance", "Pommel Strike", "Warcry", "Burning Pact",
    "Headbutt", "Exhume",
}
_EXHAUST_CARDS = {
    "True Grit", "Second Wind", "Corruption", "Fiend Fire",
    "Feel No Pain", "Dark Embrace", "Burning Pact", "Sentinel",
    "Exhume",
}

_TIER_VALUE = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}


class RunEncoder:
    OBS_SIZE = OBS_SIZE

    INTENT_MAP = {
        "ATTACK": 0.2, "ATTACK_BUFF": 0.3, "ATTACK_DEBUFF": 0.35,
        "ATTACK_DEFEND": 0.4, "BUFF": 0.5, "DEBUFF": 0.6,
        "STRONG_DEBUFF": 0.65, "DEFEND": 0.7, "DEFEND_BUFF": 0.75,
        "ESCAPE": 0.8, "MAGIC": 0.85, "SLEEP": 0.1, "STUN": 0.9,
        "UNKNOWN": 0.5, "NONE": 0.0,
    }
    CARD_TYPE_MAP = {
        "ATTACK": 0.25, "SKILL": 0.5, "POWER": 0.75,
        "STATUS": 0.9, "CURSE": 1.0,
    }

    def encode(self, state: GameState) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._encode_global(obs, state)
        if state.is_in_combat:
            self._encode_combat(obs, state)
        else:
            self._encode_noncombat(obs, state)
        return obs

    def _encode_global(self, obs: np.ndarray, state: GameState) -> None:
        max_hp = max(state.max_hp, 1)

        # [0:8] player stats
        obs[0] = state.current_hp / max_hp
        obs[1] = min(max_hp / 400, 1.0)
        obs[2] = min(state.floor / 55, 1.0)
        obs[3] = min(state.gold / 999, 1.0)
        obs[4] = min(state.act / 3, 1.0)
        obs[5] = state.ascension_level / 20
        obs[6] = min(state.energy / 4, 1.0)
        obs[7] = min(state.player_block / max_hp, 1.0)

        # [8:16] deck composition
        deck = state.deck
        n = max(len(deck), 1)
        n_attack   = sum(1 for c in deck if c.get("type") == "ATTACK")
        n_skill    = sum(1 for c in deck if c.get("type") == "SKILL")
        n_power    = sum(1 for c in deck if c.get("type") == "POWER")
        n_curse    = sum(1 for c in deck if c.get("type") in ("STATUS", "CURSE"))
        n_exhaust  = sum(1 for c in deck if c.get("id", "") in _EXHAUST_CARDS)
        n_strength = sum(1 for c in deck if c.get("id", "") in _STRENGTH_CARDS)
        n_draw     = sum(1 for c in deck if c.get("id", "") in _DRAW_CARDS)

        obs[8]  = len(deck) / 60
        obs[9]  = n_attack / n
        obs[10] = n_skill / n
        obs[11] = n_power / n
        obs[12] = n_curse / n
        obs[13] = min(n_exhaust / 10, 1.0)
        obs[14] = min(n_strength / 5, 1.0)
        obs[15] = min(n_draw / 5, 1.0)

        # [16:36] potions: 5 slots × 4 features
        for i, potion in enumerate(state.potions[:5]):
            base = 16 + i * 4
            pid = potion.get("id", "")
            obs[base]     = 1.0 if pid else 0.0
            obs[base + 1] = 1.0 if pid in _HEALING_POTIONS else 0.0
            obs[base + 2] = 1.0 if pid in _ATTACK_POTIONS else 0.0
            obs[base + 3] = 1.0 if potion.get("requires_target", False) else 0.0

        # [36:43] relics: count + 6 high-impact flags
        relic_ids = {r.get("id", "") for r in state.relics}
        obs[36] = min(len(state.relics) / 20, 1.0)
        for j, relic_name in enumerate(_HIGH_IMPACT_RELICS):
            obs[37 + j] = 1.0 if relic_name in relic_ids else 0.0

        # [43:55] screen type one-hot (12 types)
        idx = _SCREEN_IDX.get(state.screen_type, 0)
        obs[43 + idx] = 1.0

    def _encode_combat(self, obs: np.ndarray, state: GameState) -> None:
        base = GLOBAL_SIZE  # 55

        # Hand: 10 cards × 7 features [55:125]
        for i, card in enumerate(state.hand[:10]):
            b = base + i * 7
            props = get_card_properties(card.get("id", ""))
            obs[b]     = min(card.get("cost", 0), 5) / 5
            obs[b + 1] = self.CARD_TYPE_MAP.get(card.get("type", ""), 0.5)
            obs[b + 2] = 1.0 if card.get("is_playable", False) else 0.0
            obs[b + 3] = 1.0 if props["applies_vulnerable"] else 0.0
            obs[b + 4] = 1.0 if props["applies_weak"]       else 0.0
            obs[b + 5] = 1.0 if props["draws_cards"]        else 0.0
            obs[b + 6] = 1.0 if props["gains_block"]        else 0.0

        # Monsters: 5 × 6 features [125:155]
        monster_base = base + 70
        for i, m in enumerate(state.monsters[:5]):
            if m.get("is_gone", False):
                continue
            b = monster_base + i * 6
            m_max  = max(m.get("max_hp", 1), 1)
            powers = m.get("powers", [])
            vuln   = next((p.get("amount", 0) for p in powers if p.get("id") == "Vulnerable"), 0)
            weak   = next((p.get("amount", 0) for p in powers if p.get("id") == "Weak"), 0)
            obs[b]     = m.get("current_hp", 0) / m_max
            obs[b + 1] = min(m_max / 400, 1.0)
            obs[b + 2] = m.get("block", 0) / m_max
            obs[b + 3] = self.INTENT_MAP.get(m.get("intent", "UNKNOWN"), 0.5)
            obs[b + 4] = min(vuln / 10, 1.0)
            obs[b + 5] = min(weak / 10, 1.0)

        # Player powers [155:160]
        power_base = monster_base + 30  # 155
        player_powers = (state.combat_state or {}).get("player", {}).get("powers", [])
        def _pwr(name):
            return next((p.get("amount", 0) for p in player_powers if p.get("id") == name), 0)
        obs[power_base]     = min(_pwr("Strength") / 10, 1.0)
        obs[power_base + 1] = min(_pwr("Dexterity") / 10, 1.0)
        obs[power_base + 2] = min(_pwr("Weak") / 5, 1.0)
        obs[power_base + 3] = min(_pwr("Vulnerable") / 5, 1.0)
        obs[power_base + 4] = 1.0 if any(p.get("id") == "Barricade" for p in player_powers) else 0.0

        # Turn metadata [160:163]
        meta_base = power_base + 5  # 160
        obs[meta_base]     = min(len(state.draw_pile) / 60, 1.0)
        obs[meta_base + 1] = min(len(state.discard_pile) / 60, 1.0)
        obs[meta_base + 2] = min(state.turn / 20, 1.0)

        # Debuff signal [163:166]
        debuff_base = meta_base + 3  # 163
        n_hand = max(len(state.hand), 1)
        n_debuff_cards = sum(
            1 for c in state.hand
            if get_card_properties(c.get("id", "")).get("applies_vulnerable") or
               get_card_properties(c.get("id", "")).get("applies_weak")
        )
        obs[debuff_base] = n_debuff_cards / n_hand

    def _encode_noncombat(self, obs: np.ndarray, state: GameState) -> None:
        base   = GLOBAL_SIZE + COMBAT_SIZE  # 167
        ss     = state.screen_state or {}
        screen = state.screen_type

        # Choices block [167:199]: 8 × 4 features
        choices = self._get_choices(screen, ss, state)
        for i, choice in enumerate(choices[:8]):
            b = base + i * 4
            obs[b]     = choice.get("tier_value",    0.0)
            obs[b + 1] = choice.get("synergy_score", 0.0)
            obs[b + 2] = choice.get("cost_ratio",    0.0)
            obs[b + 3] = choice.get("is_available",  1.0)

        # Deck synergy context [199:204]
        syn_base = base + 32
        deck_ids = [c.get("id", "") for c in state.deck]
        obs[syn_base]     = min(sum(1 for d in deck_ids if d in _EXHAUST_CARDS) / 10, 1.0)
        obs[syn_base + 1] = min(sum(1 for d in deck_ids if d in _STRENGTH_CARDS) / 5, 1.0)
        obs[syn_base + 2] = min(sum(1 for d in deck_ids if d in _DRAW_CARDS) / 5, 1.0)
        n_block = sum(1 for d in deck_ids if get_card_properties(d).get("gains_block"))
        obs[syn_base + 3] = min(n_block / 10, 1.0)
        n_curse = sum(1 for c in state.deck if c.get("type") in ("STATUS", "CURSE"))
        obs[syn_base + 4] = min(n_curse / 10, 1.0)

        # Screen metadata [204:212]
        meta_base = syn_base + 5  # 204
        max_hp = max(state.max_hp, 1)

        if screen == "SHOP_SCREEN":
            all_items = ss.get("cards", []) + ss.get("relics", [])
            prices = [item.get("price", 9999) for item in all_items if item.get("is_in_stock", True)]
            min_price = min(prices, default=0)
            obs[meta_base] = min(min_price / max(state.gold, 1), 1.0)

        if screen == "REST":
            heal = min(int(max_hp * 0.3), max_hp - state.current_hp)
            obs[meta_base + 1] = max(heal, 0) / max_hp

        if screen == "MAP":
            nodes = ss.get("next_nodes", [])
            symbols = [n.get("symbol", "") for n in nodes]
            obs[meta_base + 2] = 1.0 if "E" in symbols else 0.0
            obs[meta_base + 3] = 1.0 if "R" in symbols else 0.0
            obs[meta_base + 4] = 1.0 if "$" in symbols else 0.0
            obs[meta_base + 5] = 1.0 if "T" in symbols else 0.0
            obs[meta_base + 6] = 1.0 if "?" in symbols else 0.0
            obs[meta_base + 7] = 1.0 if "M" in symbols else 0.0

    def _get_choices(self, screen: str, ss: dict, state: GameState) -> list:
        gold = max(state.gold, 1)
        deck = state.deck

        if screen == "CARD_REWARD":
            return [
                {
                    "tier_value":    _TIER_VALUE.get(get_card_tier(c.get("id", "")), 0.4),
                    "synergy_score": self._synergy(c.get("id", ""), deck),
                    "cost_ratio":    0.0,
                    "is_available":  1.0,
                }
                for c in ss.get("cards", [])
            ]

        if screen == "SHOP_SCREEN":
            items = []
            for c in ss.get("cards", []):
                items.append({
                    "tier_value":    _TIER_VALUE.get(get_card_tier(c.get("id", "")), 0.4),
                    "synergy_score": self._synergy(c.get("id", ""), deck),
                    "cost_ratio":    min(c.get("price", 0) / gold, 1.0),
                    "is_available":  1.0 if c.get("is_in_stock", True) else 0.0,
                })
            for r in ss.get("relics", []):
                items.append({
                    "tier_value":    0.6,
                    "synergy_score": 0.0,
                    "cost_ratio":    min(r.get("price", 0) / gold, 1.0),
                    "is_available":  1.0 if r.get("is_in_stock", True) else 0.0,
                })
            return items

        return []

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
            high_damage = {"Carnage", "Whirlwind", "Reaper", "Immolate",
                           "Fiend Fire", "Hemokinesis", "Blood for Blood"}
            if sum(1 for d in deck_ids if d in high_damage) >= 2:
                score += 0.3

        if card_id in _DRAW_CARDS:
            score += 0.2

        return min(score, 1.0)
