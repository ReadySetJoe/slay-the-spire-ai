# main.py
import logging

from src.communicator import Communicator
from src.agent import SimpleAgent
from src.game_loop import GameLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("game.log"), logging.StreamHandler()],
)


def main():
    communicator = Communicator()
    agent = SimpleAgent()
    loop = GameLoop(communicator, agent)
    loop.run()


if __name__ == "__main__":
    main()
