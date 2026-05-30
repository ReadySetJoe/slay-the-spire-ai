# src/game_loop.py
import logging

from src.communicator import Communicator
from src.agent import Agent
from src.run_tracker import RunTracker

logger = logging.getLogger(__name__)


class GameLoop:
    def __init__(self, communicator: Communicator, agent: Agent,
                 run_tracker: RunTracker | None = None):
        self.communicator = communicator
        self.agent = agent
        self.run_tracker = run_tracker or RunTracker()
        self._ready_sent = False

    def step(self):
        """Send ready (if needed) and process one state."""
        if not self._ready_sent:
            self.communicator.send_ready()
            self._ready_sent = True

        state = self.communicator.receive_state()
        if state is None:
            logger.info("No more input, stopping.")
            return False

        if state.error:
            logger.warning("Received error: %s", state.error)
            return True

        if not state.ready_for_command:
            return True

        if not state.in_game:
            if "START" in state.available_commands:
                logger.info("Starting new run...")
                self.communicator.send_command("START IRONCLAD 0")
            return True

        if state.screen_type == "GAME_OVER":
            self.run_tracker.record_run(state)
            summary = self.run_tracker.summary()
            logger.info(
                "Stats: %d runs | %d wins | %d losses | %.1f%% win rate | avg floor %.1f",
                summary["total_runs"], summary["wins"], summary["losses"],
                summary["win_rate"] * 100, summary["avg_floor"],
            )
            if hasattr(self.agent, 'on_game_over'):
                self.agent.on_game_over(state)
            self.communicator.send_command("PROCEED")
            return True

        action = self.agent.act(state)
        logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                    state.floor, state.current_hp, state.max_hp, state.screen_type, action)
        self.communicator.send_command(action)
        return True

    def run(self):
        """Run the game loop until input is exhausted."""
        self.communicator.send_ready()
        self._ready_sent = True
        while True:
            state = self.communicator.receive_state()
            if state is None:
                logger.info("Connection closed.")
                break

            if state.error:
                logger.warning("Error: %s", state.error)
                continue

            if not state.ready_for_command:
                continue

            if not state.in_game:
                if "START" in state.available_commands:
                    logger.info("Starting new run...")
                    self.communicator.send_command("START IRONCLAD 0")
                continue

            if state.screen_type == "GAME_OVER":
                self.run_tracker.record_run(state)
                summary = self.run_tracker.summary()
                logger.info(
                    "Stats: %d runs | %d wins | %d losses | %.1f%% win rate | avg floor %.1f",
                    summary["total_runs"], summary["wins"], summary["losses"],
                    summary["win_rate"] * 100, summary["avg_floor"],
                )
                if hasattr(self.agent, 'on_game_over'):
                    self.agent.on_game_over(state)
                self.communicator.send_command("PROCEED")
                continue

            logger.debug("Screen: %s | Commands: %s", state.screen_type, state.available_commands)
            action = self.agent.act(state)
            logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                        state.floor, state.current_hp, state.max_hp, state.screen_type, action)
            if self.run_tracker.live_state_writer:
                self.run_tracker.live_state_writer.write(state, action)
            self.communicator.send_command(action)
