# main.py
import logging
import sys

from src.communicator import Communicator
from src.game_loop import GameLoop
from src.run_tracker import RunTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("game.log"), logging.StreamHandler()],
)


def main():
    use_rl = "--rl" in sys.argv

    communicator = Communicator()
    tracker = RunTracker(log_path="data/run_log.jsonl")

    if use_rl:
        from src.rl_agent import RLAgent
        agent = RLAgent(model_path="data/combat_model.zip", train=True)
        logging.getLogger().info("Using RL agent (training mode)")
    else:
        from src.agent import SimpleAgent
        agent = SimpleAgent()
        logging.getLogger().info("Using rule-based agent")

    loop = GameLoop(communicator, agent, run_tracker=tracker)
    loop.run()


if __name__ == "__main__":
    main()
