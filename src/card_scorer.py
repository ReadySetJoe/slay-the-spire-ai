import json
import logging
import os

import numpy as np

from src.card_tier_list import get_card_tier

logger = logging.getLogger(__name__)

_TIER_VALUE = {"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}
_EMA_ALPHA = 0.1       # how quickly new run outcomes shift the score
_EMPIRICAL_WEIGHT = 0.5  # λ: how much the learned score weighs vs. static tier
_TEMPERATURE = 0.5     # softmax temperature; lower = more greedy
_DEFAULT_EMA = 0.5     # neutral starting score for unseen cards


class CardScorer:
    """
    Maintains an EMA run-quality score per card.

    Combined score = tier_value + λ * ema_score.  Card selection uses softmax
    over combined scores so high-tier cards are still preferred but exploration
    is non-zero and empirically strong cards can overtake their static rank.
    """

    def __init__(self, path: str = "data/card_scores.json"):
        self.path = path
        self._data: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def softmax_pick(self, cards: list) -> int:
        """Sample a card index proportionally to softmax(combined_score / T)."""
        if not cards:
            return 0
        scores = np.array(
            [self._combined(c.get("id", "")) for c in cards], dtype=np.float64
        )
        scores /= _TEMPERATURE
        scores -= scores.max()          # numerical stability
        probs = np.exp(scores)
        probs /= probs.sum()
        return int(np.random.choice(len(cards), p=probs))

    # ------------------------------------------------------------------
    # Updating
    # ------------------------------------------------------------------

    def update_run(self, card_ids: list[str], quality: float):
        """
        Update EMA scores for every card chosen during a run.

        quality = (current_hp / max_hp) * (floor / 55), in [0, 1].
        """
        for card_id in card_ids:
            entry = self._data.setdefault(card_id, {"picks": 0, "ema": _DEFAULT_EMA})
            entry["picks"] += 1
            entry["ema"] = _EMA_ALPHA * quality + (1 - _EMA_ALPHA) * entry["ema"]
        if card_ids:
            logger.debug(
                "CardScorer: updated %d cards | quality=%.3f", len(card_ids), quality
            )
            self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._data = json.load(f)
            logger.info(
                "CardScorer: loaded %d card entries from %s", len(self._data), self.path
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _combined(self, card_id: str) -> float:
        tier_val = _TIER_VALUE.get(get_card_tier(card_id), 0.4)
        ema = self._data.get(card_id, {}).get("ema", _DEFAULT_EMA)
        return tier_val + _EMPIRICAL_WEIGHT * ema
