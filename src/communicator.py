# src/communicator.py
import json
import logging
import sys
import time
from typing import Optional, TextIO

from src.game_state import GameState

logger = logging.getLogger(__name__)


class Communicator:
    def __init__(
        self,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
        step_delay: float = 0.0,
    ):
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.step_delay = step_delay

    def send_ready(self):
        self.output_stream.write("ready\n")
        self.output_stream.flush()

    def send_command(self, command: str):
        if self.step_delay > 0:
            time.sleep(self.step_delay)
        self.output_stream.write(command + "\n")
        self.output_stream.flush()

    def receive_state(self) -> Optional[GameState]:
        while True:
            line = self.input_stream.readline()
            if not line:
                return None
            raw = line.strip()
            if not raw:
                continue
            try:
                return GameState.from_json(raw)
            except json.JSONDecodeError:
                logger.warning("Ignoring non-JSON line from game: %.120s", raw)
