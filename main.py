# main.py
import glob
import logging
import os
import re
import sys

from src.communicator import Communicator
from src.run_tracker import RunTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("game.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def _latest_checkpoint(checkpoint_dir: str) -> "tuple[str, int] | None":
    """Return (path, step_count) of the highest-step checkpoint file, or None."""
    files = glob.glob(os.path.join(checkpoint_dir, "combat_*_steps.zip"))
    best_path, best_steps = None, -1
    for f in files:
        m = re.search(r"combat_(\d+)_steps\.zip$", f)
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps, best_path = steps, f
    return (best_path, best_steps) if best_path else None


def _load_model(model_path: str, checkpoint_dir: str, env):
    """Load the most recent model: prefers whichever of the final save or
    latest checkpoint has the newer modification time."""
    from sb3_contrib import MaskablePPO

    latest = _latest_checkpoint(checkpoint_dir)
    has_final = os.path.exists(model_path)

    if latest and has_final:
        ckpt_path, ckpt_steps = latest
        if os.path.getmtime(ckpt_path) > os.path.getmtime(model_path):
            model = MaskablePPO.load(ckpt_path, env=env)
            logger.info("Resumed from checkpoint %s (%d steps)", ckpt_path, ckpt_steps)
        else:
            model = MaskablePPO.load(model_path, env=env)
            logger.info("Loaded final model from %s", model_path)
        return model

    if latest:
        ckpt_path, ckpt_steps = latest
        model = MaskablePPO.load(ckpt_path, env=env)
        logger.info("Resumed from checkpoint %s (%d steps)", ckpt_path, ckpt_steps)
        return model

    if has_final:
        model = MaskablePPO.load(model_path, env=env)
        logger.info("Loaded final model from %s", model_path)
        return model

    return None


def main():
    use_rl = "--rl" in sys.argv

    communicator = Communicator()
    tracker = RunTracker(log_path="data/run_log.jsonl")

    from src.card_scorer import CardScorer
    scorer = CardScorer(path="data/card_scores.json")

    if use_rl:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from src.combat_env import CombatEnv
        from src.callbacks import EpisodeLoggerCallback

        env = CombatEnv(communicator=communicator, run_tracker=tracker, scorer=scorer)
        model_path = "data/combat_model.zip"
        checkpoint_dir = "data/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = _load_model(model_path, checkpoint_dir, env)
        if model is None:
            model = MaskablePPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                verbose=1,
            )
            logger.info("Created new MaskablePPO model")
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
        try:
            model.learn(total_timesteps=10_000_000, callback=callbacks)
            logger.info("Training complete.")
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        finally:
            model.save(model_path)
            logger.info("Model saved to %s", model_path)
    else:
        from src.agent import SimpleAgent
        from src.game_loop import GameLoop

        agent = SimpleAgent(scorer=scorer)
        loop = GameLoop(communicator, agent, run_tracker=tracker)
        loop.run()


if __name__ == "__main__":
    main()
