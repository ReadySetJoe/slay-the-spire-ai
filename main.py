# main.py
import logging
import os
import sys

from src.communicator import Communicator
from src.run_tracker import RunTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("game.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    use_rl = "--rl" in sys.argv

    communicator = Communicator()
    tracker = RunTracker(log_path="data/run_log.jsonl")

    if use_rl:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from src.combat_env import CombatEnv
        from src.callbacks import EpisodeLoggerCallback

        env = CombatEnv(communicator=communicator, run_tracker=tracker)
        model_path = "data/combat_model.zip"

        if os.path.exists(model_path):
            model = MaskablePPO.load(model_path, env=env)
            logger.info("Loaded existing model from %s", model_path)
        else:
            model = MaskablePPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                verbose=1,
            )
            logger.info("Created new MaskablePPO model")

        os.makedirs("data/checkpoints", exist_ok=True)
        callbacks = CallbackList([
            EpisodeLoggerCallback(summary_freq=100),
            CheckpointCallback(
                save_freq=1000,
                save_path="data/checkpoints/",
                name_prefix="combat",
                verbose=1,
            ),
        ])

        logger.info("Starting RL training (MaskablePPO)...")
        model.learn(total_timesteps=10_000_000, callback=callbacks)
    else:
        from src.agent import SimpleAgent
        from src.game_loop import GameLoop

        agent = SimpleAgent()
        loop = GameLoop(communicator, agent, run_tracker=tracker)
        loop.run()


if __name__ == "__main__":
    main()
