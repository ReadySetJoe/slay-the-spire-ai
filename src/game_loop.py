# src/game_loop.py
import logging

from src.communicator import Communicator
from src.agent import Agent

logger = logging.getLogger(__name__)


class GameLoop:
    def __init__(self, communicator: Communicator, agent: Agent):
        self.communicator = communicator
        self.agent = agent
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
            logger.debug("Not in game, waiting for game to start.")
            return True

        action = self.agent.act(state)
        logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                    state.floor, state.current_hp, state.max_hp, state.screen_type, action)
        self.communicator.send_command(action)
        return True

    def run(self):
        """Run the game loop until the game ends or input is exhausted."""
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
                logger.debug("Not in game, waiting for game to start.")
                continue

            logger.debug("Screen: %s | Commands: %s", state.screen_type, state.available_commands)
            action = self.agent.act(state)
            logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                        state.floor, state.current_hp, state.max_hp, state.screen_type, action)
            self.communicator.send_command(action)
