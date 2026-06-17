import json
import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_SCORE = 0.5


class CardScorer:
    def __init__(self, path: str = "data/card_scores.json", alpha: float = 0.05):
        self._path = path
        self._alpha = alpha
        self._scores: dict[str, float] = {}
        self._total_combats: int = 0
        self.load()

    def score(self, card_id: str) -> float:
        return self._scores.get(card_id, _DEFAULT_SCORE)

    def update(self, cards_played: list[str], performance_signal: float) -> None:
        performance_signal = max(0.0, min(1.0, performance_signal))
        for card_id in cards_played:
            current = self._scores.get(card_id, _DEFAULT_SCORE)
            self._scores[card_id] = (1 - self._alpha) * current + self._alpha * performance_signal
        self._total_combats += 1

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        data = {"card_scores": self._scores, "total_combats_scored": self._total_combats}
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except OSError as e:
            logger.warning("CardScorer save failed: %s", e)

    def load(self) -> None:
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._scores = data.get("card_scores", {})
            self._total_combats = data.get("total_combats_scored", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            self._scores = {}
            self._total_combats = 0
