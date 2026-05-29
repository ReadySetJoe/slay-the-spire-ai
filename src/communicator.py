# src/communicator.py
import sys
from typing import Optional, TextIO

from src.game_state import GameState


class Communicator:
    def __init__(
        self,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
    ):
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout

    def send_ready(self):
        self.output_stream.write("ready\n")
        self.output_stream.flush()

    def send_command(self, command: str):
        self.output_stream.write(command + "\n")
        self.output_stream.flush()

    def receive_state(self) -> Optional[GameState]:
        line = self.input_stream.readline()
        if not line:
            return None
        return GameState.from_json(line.strip())
