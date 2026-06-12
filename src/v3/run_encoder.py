import numpy as np
from src.game_state import GameState
from src.card_properties import get_card_properties
from src.card_tier_list import get_card_tier
from src.v2.run_encoder import (
    RunEncoder,
    _HEALING_POTIONS, _ATTACK_POTIONS, _HIGH_IMPACT_RELICS,
    _STRENGTH_CARDS, _DRAW_CARDS, _EXHAUST_CARDS,
    _SCREEN_IDX, _TIER_VALUE,
)

V3_GLOBAL_SIZE       = 55
V3_COMBAT_SIZE       = 123
V3_NONCOMBAT_SIZE    = 60
V3_TURN_CONTEXT_SIZE = 12
V3_OBS_SIZE          = 250  # 55 + 123 + 60 + 12

_ATTACKING_INTENTS = {"ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND"}
_BUFFING_INTENTS   = {"BUFF",   "DEFEND_BUFF", "ATTACK_BUFF"}
_DEBUFFING_INTENTS = {"DEBUFF", "STRONG_DEBUFF", "ATTACK_DEBUFF"}


class V3RunEncoder(RunEncoder):
    OBS_SIZE = V3_OBS_SIZE

    def encode(self, state: GameState, turn_state: dict | None = None,
               card_scorer=None) -> np.ndarray:
        obs = np.zeros(V3_OBS_SIZE, dtype=np.float32)
        self._encode_global(obs, state)              # 0–54 (inherited, unchanged)
        if state.is_in_combat:
            self._encode_combat(obs, state)          # 55–177 (overridden)
            if turn_state:
                self._encode_turn_context(obs, turn_state, state.max_hp)  # 238–249
        else:
            self._encode_noncombat(obs, state, card_scorer)  # 178–237 (overridden)
        return obs

    def _encode_combat(self, obs: np.ndarray, state: GameState) -> None:
        base = V3_GLOBAL_SIZE  # 55

        # Hand: 10 × 7 features [55:125] — identical to v2
        for i, card in enumerate(state.hand[:10]):
            b     = base + i * 7
            props = get_card_properties(card.get("id", ""))
            obs[b]     = min(card.get("cost", 0), 5) / 5
            obs[b + 1] = self.CARD_TYPE_MAP.get(card.get("type", ""), 0.5)
            obs[b + 2] = 1.0 if card.get("is_playable", False) else 0.0
            obs[b + 3] = 1.0 if props["applies_vulnerable"] else 0.0
            obs[b + 4] = 1.0 if props["applies_weak"]       else 0.0
            obs[b + 5] = 1.0 if props["draws_cards"]        else 0.0
            obs[b + 6] = 1.0 if props["gains_block"]        else 0.0

        # Monsters: 5 × 8 features [125:165]
        monster_base  = base + 70   # 125
        any_attacking = False
        attack_count  = 0
        for i, m in enumerate(state.monsters[:5]):
            if m.get("is_gone", False):
                continue
            b      = monster_base + i * 8
            m_max  = max(m.get("max_hp", 1), 1)
            intent = m.get("intent", "UNKNOWN")
            powers = m.get("powers", [])
            vuln   = next((p.get("amount", 0) for p in powers if p.get("id") == "Vulnerable"), 0)
            weak   = next((p.get("amount", 0) for p in powers if p.get("id") == "Weak"), 0)
            is_atk = intent in _ATTACKING_INTENTS
            is_buf = intent in _BUFFING_INTENTS
            is_deb = intent in _DEBUFFING_INTENTS
            if is_atk:
                any_attacking = True
                attack_count += 1
            obs[b]     = m.get("current_hp", 0) / m_max
            obs[b + 1] = min(m_max / 400, 1.0)
            obs[b + 2] = m.get("block", 0) / m_max
            obs[b + 3] = 1.0 if is_atk else 0.0
            obs[b + 4] = 1.0 if is_buf else 0.0
            obs[b + 5] = 1.0 if is_deb else 0.0
            obs[b + 6] = min(vuln / 10, 1.0)
            obs[b + 7] = min(weak / 10, 1.0)

        # Aggregate intent [165:167]
        agg_base          = monster_base + 40   # 165
        obs[agg_base]     = 1.0 if any_attacking else 0.0
        obs[agg_base + 1] = attack_count / 5

        # Player powers [167:172]
        power_base    = agg_base + 2            # 167
        player_powers = (state.combat_state or {}).get("player", {}).get("powers", [])

        def _pwr(name):
            return next((p.get("amount", 0) for p in player_powers if p.get("id") == name), 0)

        obs[power_base]     = min(_pwr("Strength")  / 10, 1.0)
        obs[power_base + 1] = min(_pwr("Dexterity") / 10, 1.0)
        obs[power_base + 2] = min(_pwr("Weak")      / 5,  1.0)
        obs[power_base + 3] = min(_pwr("Vulnerable") / 5, 1.0)
        obs[power_base + 4] = 1.0 if any(p.get("id") == "Barricade" for p in player_powers) else 0.0

        # Turn metadata [172:175]
        meta_base          = power_base + 5     # 172
        obs[meta_base]     = min(len(state.draw_pile)    / 60, 1.0)
        obs[meta_base + 1] = min(len(state.discard_pile) / 60, 1.0)
        obs[meta_base + 2] = min(state.turn              / 20, 1.0)

        # Debuff signal [175:178]
        debuff_base    = meta_base + 3          # 175
        n_hand         = max(len(state.hand), 1)
        n_debuff_cards = sum(
            1 for c in state.hand
            if get_card_properties(c.get("id", "")).get("applies_vulnerable") or
               get_card_properties(c.get("id", "")).get("applies_weak")
        )
        obs[debuff_base] = n_debuff_cards / n_hand
        # 176–177 reserved (stay zero)

    def _encode_noncombat(self, obs: np.ndarray, state: GameState,
                          card_scorer=None) -> None:
        base   = V3_GLOBAL_SIZE + V3_COMBAT_SIZE  # 178
        ss     = state.screen_state or {}
        screen = state.screen_type
        gold   = max(state.gold, 1)
        deck   = state.deck

        # Choices block [178:210]: 8 × 4 features
        choices = self._get_choices(screen, ss, state, card_scorer)
        for i, choice in enumerate(choices[:8]):
            b          = base + i * 4
            obs[b]     = choice.get("tier_value",    0.0)
            obs[b + 1] = choice.get("synergy_score", 0.0)
            obs[b + 2] = choice.get("cost_ratio",    0.0)
            obs[b + 3] = choice.get("is_available",  1.0)

        # Deck synergy context [210:215]
        syn_base  = base + 32  # 210
        deck_ids  = [c.get("id", "") for c in deck]
        obs[syn_base]     = min(sum(1 for d in deck_ids if d in _EXHAUST_CARDS)  / 10, 1.0)
        obs[syn_base + 1] = min(sum(1 for d in deck_ids if d in _STRENGTH_CARDS) / 5,  1.0)
        obs[syn_base + 2] = min(sum(1 for d in deck_ids if d in _DRAW_CARDS)     / 5,  1.0)
        n_block           = sum(1 for d in deck_ids if get_card_properties(d).get("gains_block"))
        obs[syn_base + 3] = min(n_block / 10, 1.0)
        n_curse           = sum(1 for c in deck if c.get("type") in ("STATUS", "CURSE"))
        obs[syn_base + 4] = min(n_curse / 10, 1.0)

        # Screen metadata [215:223]
        meta_base = syn_base + 5  # 215
        max_hp    = max(state.max_hp, 1)
        if screen == "SHOP_SCREEN":
            items     = ss.get("cards", []) + ss.get("relics", [])
            min_price = min((c.get("price", 9999) for c in items if c.get("is_in_stock", True)),
                           default=0)
            obs[meta_base] = min(min_price / gold, 1.0)
        if screen == "REST":
            heal               = min(int(max_hp * 0.3), max_hp - state.current_hp)
            obs[meta_base + 1] = max(heal, 0) / max_hp
        if screen == "MAP":
            nodes   = ss.get("next_nodes", [])
            symbols = [n.get("symbol", "") for n in nodes]
            obs[meta_base + 2] = 1.0 if "E" in symbols else 0.0
            obs[meta_base + 3] = 1.0 if "R" in symbols else 0.0
            obs[meta_base + 4] = 1.0 if "$" in symbols else 0.0
            obs[meta_base + 5] = 1.0 if "T" in symbols else 0.0
            obs[meta_base + 6] = 1.0 if "?" in symbols else 0.0
            obs[meta_base + 7] = 1.0 if "M" in symbols else 0.0
        # padding [223:238] stays 0

    def _encode_turn_context(self, obs: np.ndarray, turn_state: dict,
                             max_hp: int) -> None:
        base   = V3_GLOBAL_SIZE + V3_COMBAT_SIZE + V3_NONCOMBAT_SIZE  # 238
        max_hp = max(max_hp, 1)
        obs[base]      = min(turn_state.get("actions_taken",   0)   / 10,   1.0)
        obs[base + 1]  = min(turn_state.get("energy_spent",    0)   / 4,    1.0)
        obs[base + 2]  = min(turn_state.get("attacks_played",  0)   / 5,    1.0)
        obs[base + 3]  = min(turn_state.get("skills_played",   0)   / 5,    1.0)
        obs[base + 4]  = min(turn_state.get("powers_played",   0)   / 3,    1.0)
        obs[base + 5]  = min(turn_state.get("strength_gained", 0)   / 10,   1.0)
        obs[base + 6]  = 1.0 if turn_state.get("vulnerable_applied",  False) else 0.0
        obs[base + 7]  = 1.0 if turn_state.get("weak_applied",        False) else 0.0
        obs[base + 8]  = min(turn_state.get("damage_dealt",    0.0) / max_hp, 1.0)
        obs[base + 9]  = min(turn_state.get("block_gained",    0.0) / max_hp, 1.0)
        obs[base + 10] = 1.0 if turn_state.get("last_card_was_buff",  False) else 0.0
        obs[base + 11] = 1.0 if turn_state.get("last_card_was_debuff", False) else 0.0

    def _get_choices(self, screen: str, ss: dict, state: GameState,
                     card_scorer=None) -> list:
        gold = max(state.gold, 1)
        deck = state.deck

        def _syn(card_id):
            if card_scorer is not None:
                return card_scorer.score(card_id)
            return self._synergy(card_id, deck)

        if screen == "CARD_REWARD":
            return [
                {
                    "tier_value":    _TIER_VALUE.get(get_card_tier(c.get("id", "")), 0.4),
                    "synergy_score": _syn(c.get("id", "")),
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
                    "synergy_score": _syn(c.get("id", "")),
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
