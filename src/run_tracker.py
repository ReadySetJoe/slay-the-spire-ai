import json
import logging
import os
from datetime import datetime, timezone

from src.game_state import GameState

logger = logging.getLogger(__name__)


_GRAPH_REGEN_INTERVAL = 10  # regenerate performance graphs every N runs


class RunTracker:
    def __init__(self, log_path: str = "data/run_log.jsonl", live_state_writer=None):
        self.log_path = log_path
        self.live_state_writer = live_state_writer
        self.run_number = self._load_last_run_number()
        self.runs: list[dict] = []

    def _load_last_run_number(self) -> int:
        try:
            last_line = None
            with open(self.log_path) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        last_line = stripped
            if last_line:
                return json.loads(last_line).get("run_number", 0)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        return 0

    def record_run(self, state: GameState, version: str = "v1",
                   episode_reward: "float | None" = None,
                   energy_efficiency: "float | None" = None) -> dict:
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
            "version": version,
            "episode_reward": episode_reward,
            "energy_efficiency": energy_efficiency,
        }

        self.runs.append(record)
        self._write_record(record)
        if self.run_number % _GRAPH_REGEN_INTERVAL == 0:
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
