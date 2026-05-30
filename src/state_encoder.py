import numpy as np
from src.card_properties import get_card_properties
from src.game_state import GameState

INTENT_MAP = {
    "ATTACK": 0.2,
    "ATTACK_BUFF": 0.3,
    "ATTACK_DEBUFF": 0.35,
    "ATTACK_DEFEND": 0.4,
    "BUFF": 0.5,
    "DEBUFF": 0.6,
    "STRONG_DEBUFF": 0.65,
    "DEFEND": 0.7,
    "DEFEND_BUFF": 0.75,
    "ESCAPE": 0.8,
    "MAGIC": 0.85,
    "SLEEP": 0.1,
    "STUN": 0.9,
    "UNKNOWN": 0.5,
    "NONE": 0.0,
}

CARD_TYPE_MAP = {
    "ATTACK": 0.25,
    "SKILL": 0.5,
    "POWER": 0.75,
    "STATUS": 0.9,
    "CURSE": 1.0,
}

MAX_HP_SCALE   = 400.0  # reasonable ceiling for monster max HP
MAX_STACKS     = 10.0   # normalise status-effect stacks

MAX_HAND     = 10
MAX_MONSTERS = 5

PLAYER_FEATURES  = 4
# cost, type, is_playable, applies_vulnerable, applies_weak, draws_cards, gains_block
CARD_FEATURES    = 7
# hp_ratio, max_hp_scale, block_ratio, intent, vulnerable, weak
MONSTER_FEATURES = 6


class StateEncoder:
    OBS_SIZE = PLAYER_FEATURES + MAX_HAND * CARD_FEATURES + MAX_MONSTERS * MONSTER_FEATURES
    # = 4 + 70 + 30 = 104

    def encode(self, state: GameState) -> np.ndarray:
        obs = np.zeros(self.OBS_SIZE, dtype=np.float32)
        max_hp    = max(state.max_hp, 1)
        max_energy = 4

        # Player features [0:4]
        obs[0] = state.current_hp / max_hp
        obs[1] = max_hp / max_hp          # always 1.0 — kept for index stability
        obs[2] = state.player_block / max_hp
        obs[3] = state.energy / max_energy

        # Hand cards [4:74]
        for i, card in enumerate(state.hand[:MAX_HAND]):
            base  = PLAYER_FEATURES + i * CARD_FEATURES
            props = get_card_properties(card.get("id", ""))
            obs[base]     = min(card.get("cost", 0), 5) / 5
            obs[base + 1] = CARD_TYPE_MAP.get(card.get("type", ""), 0.5)
            obs[base + 2] = 1.0 if card.get("is_playable", False) else 0.0
            obs[base + 3] = 1.0 if props["applies_vulnerable"] else 0.0
            obs[base + 4] = 1.0 if props["applies_weak"]       else 0.0
            obs[base + 5] = 1.0 if props["draws_cards"]        else 0.0
            obs[base + 6] = 1.0 if props["gains_block"]        else 0.0

        # Monsters [74:104]
        for i, monster in enumerate(state.monsters[:MAX_MONSTERS]):
            if monster.get("is_gone", False):
                continue
            base    = PLAYER_FEATURES + MAX_HAND * CARD_FEATURES + i * MONSTER_FEATURES
            m_max   = max(monster.get("max_hp", 1), 1)
            powers  = monster.get("powers", [])
            vuln    = next((p.get("amount", 0) for p in powers if p.get("id") == "Vulnerable"), 0)
            weak    = next((p.get("amount", 0) for p in powers if p.get("id") == "Weak"),       0)

            obs[base]     = monster.get("current_hp", 0) / m_max
            obs[base + 1] = min(m_max / MAX_HP_SCALE, 1.0)
            obs[base + 2] = monster.get("block", 0) / max(m_max, 1)
            obs[base + 3] = INTENT_MAP.get(monster.get("intent", "UNKNOWN"), 0.5)
            obs[base + 4] = min(vuln, MAX_STACKS) / MAX_STACKS
            obs[base + 5] = min(weak, MAX_STACKS) / MAX_STACKS

        return obs
