# main.py
import logging

from src.communicator import Communicator
from src.agent import SimpleAgent
from src.game_loop import GameLoop
from src.run_tracker import RunTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("game.log"), logging.StreamHandler()],
)


def main():
    communicator = Communicator()
    agent = SimpleAgent()
    tracker = RunTracker(log_path="data/run_log.jsonl")
    loop = GameLoop(communicator, agent, run_tracker=tracker)
    loop.run()


if __name__ == "__main__":
    main()
