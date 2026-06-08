# src/callbacks.py
import logging
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

_TRAINING_WINDOW = 500


class EpisodeLoggerCallback(BaseCallback):
    """Logs per-episode stats and periodic summaries to the game log."""

    def __init__(self, summary_freq: int = 100, live_state_writer=None):
        super().__init__(verbose=0)
        self.summary_freq = summary_freq
        self.live_state_writer = live_state_writer
        self._episode_count = 0
        self._episode_stats: deque = deque(maxlen=self.summary_freq)
        self._training_episodes: deque = deque(maxlen=_TRAINING_WINDOW)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done and "episode" in info:
                self._log_episode(info["episode"])
        return True

    def _log_episode(self, ep: dict):
        self._episode_count += 1
        self._episode_stats.append(ep)
        self._training_episodes.append({
            "ep": self._episode_count,
            "reward": ep["r"],
            "steps": ep["l"],
        })
        if self.live_state_writer is not None:
            self.live_state_writer.write_training_step(
                episodes=list(self._training_episodes),
                total_episodes=self._episode_count,
                total_timesteps=self.num_timesteps,
            )
        logger.info(
            "[Episode %d] reward=%.2f | steps=%d | hp=%d/%d | floor=%d | total_steps=%d",
            self._episode_count, ep["r"], ep["l"],
            ep.get("hp", 0), ep.get("max_hp", 0), ep.get("floor", 0),
            self.num_timesteps,
        )
        if self._episode_count % self.summary_freq == 0:
            recent = list(self._episode_stats)
            avg_r = sum(e["r"] for e in recent) / len(recent)
            avg_l = sum(e["l"] for e in recent) / len(recent)
            win_rate = sum(1 for e in recent if e["r"] >= 1.0) / len(recent)
            avg_floor = sum(e.get("floor", 0) for e in recent) / len(recent)
            logger.info(
                "[Summary ep %d-%d] avg_reward=%.2f | avg_steps=%.1f | win_rate=%.1f%% | avg_floor=%.1f",
                self._episode_count - self.summary_freq + 1, self._episode_count,
                avg_r, avg_l, win_rate * 100, avg_floor,
            )
