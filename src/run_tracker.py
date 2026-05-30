import json
import logging
import os
from datetime import datetime, timezone

from src.game_state import GameState

logger = logging.getLogger(__name__)


class RunTracker:
    def __init__(self, log_path: str = "data/run_log.jsonl", live_state_writer=None):
        self.log_path = log_path
        self.live_state_writer = live_state_writer
        self.run_number = 0
        self.runs: list[dict] = []

    def record_run(self, state: GameState) -> dict:
        self.run_number += 1

        victory = False
        if state.screen_state:
            victory = state.screen_state.get("victory", False)

        record = {
            "run_number": self.run_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": "win" if victory else "loss",
            "seed": state.seed,
            "floor_reached": state.floor,
            "ascension_level": state.ascension_level,
            "player_class": state.player_class,
            "current_hp": state.current_hp,
            "max_hp": state.max_hp,
            "gold": state.gold,
            "deck_size": len(state.deck),
            "relic_count": len(state.relics),
            "act": state.act,
        }

        self.runs.append(record)
        self._write_record(record)
        self._update_graphs()
        if self.live_state_writer:
            self.live_state_writer.write_run_summary(self.summary())

        logger.info(
            "Run #%d complete: %s | Floor %d | HP %d/%d | Deck %d | Relics %d",
            record["run_number"], record["result"], record["floor_reached"],
            record["current_hp"], record["max_hp"],
            record["deck_size"], record["relic_count"],
        )

        return record

    def _write_record(self, record: dict):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _update_graphs(self):
        from src.grapher import generate_graphs
        data_dir = os.path.dirname(self.log_path) or "."
        generate_graphs(
            log_path=self.log_path,
            scores_path=os.path.join(data_dir, "card_scores.json"),
            output_dir=os.path.join(data_dir, "graphs"),
        )

    def summary(self) -> dict:
        wins = sum(1 for r in self.runs if r["result"] == "win")
        losses = sum(1 for r in self.runs if r["result"] == "loss")
        total = len(self.runs)
        return {
            "total_runs": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0,
            "avg_floor": sum(r["floor_reached"] for r in self.runs) / total if total > 0 else 0,
        }
