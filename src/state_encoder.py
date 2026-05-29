import numpy as np
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

MAX_HAND = 10
MAX_MONSTERS = 5
PLAYER_FEATURES = 4
CARD_FEATURES = 3   # cost, type, is_playable
MONSTER_FEATURES = 4  # hp, max_hp, block, intent


class StateEncoder:
    OBS_SIZE = PLAYER_FEATURES + MAX_HAND * CARD_FEATURES + MAX_MONSTERS * MONSTER_FEATURES
    # = 4 + 30 + 20 = 54

    def encode(self, state: GameState) -> np.ndarray:
        obs = np.zeros(self.OBS_SIZE, dtype=np.float32)
        max_hp = max(state.max_hp, 1)
        max_energy = 4  # base max energy for Ironclad

        # Player features [0:4]
        obs[0] = state.current_hp / max_hp
        obs[1] = state.max_hp / max_hp
        obs[2] = state.player_block / max_hp
        obs[3] = state.energy / max_energy

        # Hand cards [4:34]
        for i, card in enumerate(state.hand[:MAX_HAND]):
            base = PLAYER_FEATURES + i * CARD_FEATURES
            obs[base] = min(card.get("cost", 0), 5) / 5
            obs[base + 1] = CARD_TYPE_MAP.get(card.get("type", ""), 0.5)
            obs[base + 2] = 1.0 if card.get("is_playable", False) else 0.0

        # Monsters [34:54]
        for i, monster in enumerate(state.monsters[:MAX_MONSTERS]):
            if monster.get("is_gone", False):
                continue
            base = PLAYER_FEATURES + MAX_HAND * CARD_FEATURES + i * MONSTER_FEATURES
            m_max_hp = max(monster.get("max_hp", 1), 1)
            obs[base] = monster.get("current_hp", 0) / m_max_hp
            obs[base + 1] = m_max_hp / m_max_hp
            obs[base + 2] = monster.get("block", 0) / max(m_max_hp, 1)
            obs[base + 3] = INTENT_MAP.get(monster.get("intent", "UNKNOWN"), 0.5)

        return obs
