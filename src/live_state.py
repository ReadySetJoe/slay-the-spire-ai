import json
import logging
import os
import re
from datetime import datetime, timezone

from src.game_state import GameState

logger = logging.getLogger(__name__)


class LiveStateWriter:
    def __init__(self, path: str = "data/live_state.json"):
        self.path = path

    def write(self, state: GameState, action: str) -> None:
        live = {
            "screen_type": state.screen_type,
            "current_hp": state.current_hp,
            "max_hp": state.max_hp,
            "last_action": self._enrich_action(state, action),
            "monsters": [
                {
                    "name": m.get("name", ""),
                    "current_hp": m.get("current_hp", 0),
                    "max_hp": m.get("max_hp", 0),
                    "intent": m.get("intent", ""),
                }
                for m in state.monsters
                if not m.get("is_gone", False)
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._update("live", live)

    def write_run_summary(self, summary: dict) -> None:
        stats = {
            "run_number": summary.get("total_runs", 0),
            "wins": summary.get("wins", 0),
            "losses": summary.get("losses", 0),
            "win_rate": round(summary.get("win_rate", 0.0), 4),
            "avg_floor": round(summary.get("avg_floor", 0.0), 1),
        }
        self._update("stats", stats)

    def _enrich_action(self, state: GameState, action: str) -> str:
        m = re.match(r"^PLAY\s+(\d+)(?:\s+(\d+))?", action)
        if not m:
            return action
        card_idx = int(m.group(1)) - 1
        target_idx = int(m.group(2)) if m.group(2) is not None else None
        card_name = ""
        if 0 <= card_idx < len(state.hand):
            card_name = state.hand[card_idx].get("name") or state.hand[card_idx].get("id", "")
        if target_idx is not None and 0 <= target_idx < len(state.monsters):
            monster_name = state.monsters[target_idx].get("name", "Enemy")
            return f"{action} ({card_name} → {monster_name})"
        if card_name:
            return f"{action} ({card_name})"
        return action

    def _update(self, key: str, value: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        existing = {}
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        existing[key] = value
        try:
            with open(self.path, "w") as f:
                json.dump(existing, f, indent=2)
        except OSError as e:
            logger.error("Failed to write live state: %s", e)
