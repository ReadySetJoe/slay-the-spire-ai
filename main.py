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


def _all_checkpoints(checkpoint_dir: str, prefix: str = "combat") -> "list[tuple[str, int]]":
    """Return (path, step_count) pairs for all valid checkpoint files, sorted descending by step."""
    files = glob.glob(os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip"))
    results = []
    for f in files:
        m = re.search(rf"{re.escape(prefix)}_(\d+)_steps\.zip$", f)
        if m:
            results.append((f, int(m.group(1))))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _load_model(model_path: str, checkpoint_dir: str, env, prefix: str = "combat"):
    """Load the most recent model: prefers whichever of the final save or
    latest valid checkpoint has the newer modification time.
    Skips checkpoints that are corrupt (truncated mid-write)."""
    import zipfile
    from sb3_contrib import MaskablePPO

    def _try_load_checkpoint(path, steps):
        try:
            zipfile.ZipFile(path).close()
        except zipfile.BadZipFile:
            logger.warning("Skipping corrupt checkpoint %s (bad zip)", path)
            return None
        model = MaskablePPO.load(path, env=env)
        logger.info("Resumed from checkpoint %s (%d steps)", path, steps)
        return model

    checkpoints = _all_checkpoints(checkpoint_dir, prefix=prefix)
    has_final = os.path.exists(model_path)

    if checkpoints and has_final:
        ckpt_path, ckpt_steps = checkpoints[0]
        if os.path.getmtime(ckpt_path) > os.path.getmtime(model_path):
            for ckpt_path, ckpt_steps in checkpoints:
                model = _try_load_checkpoint(ckpt_path, ckpt_steps)
                if model is not None:
                    return model
        model = MaskablePPO.load(model_path, env=env)
        logger.info("Loaded final model from %s", model_path)
        return model

    if checkpoints:
        for ckpt_path, ckpt_steps in checkpoints:
            model = _try_load_checkpoint(ckpt_path, ckpt_steps)
            if model is not None:
                return model

    if has_final:
        model = MaskablePPO.load(model_path, env=env)
        logger.info("Loaded final model from %s", model_path)
        return model

    return None


def main():
    use_v2 = "--v2" in sys.argv
    use_rl = "--rl" in sys.argv

    from src.live_state import LiveStateWriter
    live_writer = LiveStateWriter(path="data/live_state.json")

    communicator = Communicator()
    tracker = RunTracker(log_path="data/run_log.jsonl", live_state_writer=live_writer)

    from src.card_scorer import CardScorer
    scorer = CardScorer(path="data/card_scores.json")

    if use_v2:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from src.v2.run_env import RunEnv
        from src.callbacks import EpisodeLoggerCallback

        env = RunEnv(communicator=communicator, run_tracker=tracker)
        model_path     = "data/v2_run_model.zip"
        checkpoint_dir = "data/v2_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = _load_model(model_path, checkpoint_dir, env, prefix="v2_run")
        if model is None:
            model = MaskablePPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                verbose=1,
            )
            logger.info("Created new v2 MaskablePPO model")

        callbacks = CallbackList([
            EpisodeLoggerCallback(summary_freq=10),
            CheckpointCallback(
                save_freq=100,
                save_path=checkpoint_dir,
                name_prefix="v2_run",
                verbose=1,
            ),
        ])

        logger.info("Starting v2 RL training (MaskablePPO, full-run episodes)...")
        try:
            model.learn(total_timesteps=10_000_000, callback=callbacks)
            logger.info("Training complete.")
        except KeyboardInterrupt:
            logger.info("Training interrupted.")
        finally:
            model.save(model_path)
            logger.info("Model saved to %s", model_path)

    elif "--v3" in sys.argv:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from src.v3.run_env import V3RunEnv
        from src.v3.card_scorer import CardScorer as V3CardScorer
        from src.callbacks import EpisodeLoggerCallback

        card_scorer = V3CardScorer(path="data/card_scores.json")
        env = V3RunEnv(
            communicator=communicator,
            run_tracker=tracker,
            card_scorer=card_scorer,
            timeout_seconds=20.0,
        )

        model_path     = "data/v3_run_model.zip"
        checkpoint_dir = "data/v3_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        if os.path.exists(model_path):
            logger.info("Loading existing v3 model from %s", model_path)
            model = RecurrentPPO.load(model_path, env=env)
        else:
            logger.info("Creating new v3 RecurrentPPO model (MlpLstmPolicy)")
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                verbose=1,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=3e-4,
                policy_kwargs={"lstm_hidden_size": 256},
                tensorboard_log="data/v3_tensorboard/",
            )

        callbacks = CallbackList([
            EpisodeLoggerCallback(summary_freq=10),
            CheckpointCallback(
                save_freq=100,
                save_path=checkpoint_dir,
                name_prefix="v3_run",
                verbose=1,
            ),
        ])

        logger.info("Starting v3 RL training (RecurrentPPO, MlpLstmPolicy)...")
        try:
            model.learn(total_timesteps=10_000_000, callback=callbacks)
            logger.info("Training complete.")
        except KeyboardInterrupt:
            logger.info("Training interrupted.")
        finally:
            model.save(model_path)
            logger.info("Model saved to %s", model_path)

    elif use_rl:
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
            EpisodeLoggerCallback(summary_freq=100, live_state_writer=live_writer),
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
