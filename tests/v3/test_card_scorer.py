import json
import os
import pytest
from src.v3.card_scorer import CardScorer


@pytest.fixture
def scorer(tmp_path):
    return CardScorer(path=str(tmp_path / "scores.json"))


def test_unseen_card_returns_default(scorer):
    assert scorer.score("Inflame") == pytest.approx(0.5)


def test_update_moves_toward_signal(scorer):
    scorer.update(["Inflame"], performance_signal=1.0)
    assert scorer.score("Inflame") > 0.5


def test_update_moves_down_on_low_signal(scorer):
    scorer.update(["Strike_R"], performance_signal=0.0)
    assert scorer.score("Strike_R") < 0.5


def test_alpha_controls_update_rate(scorer):
    scorer_fast = CardScorer(path=scorer._path + ".fast", alpha=0.5)
    scorer_slow = CardScorer(path=scorer._path + ".slow", alpha=0.01)
    scorer_fast.update(["X"], 1.0)
    scorer_slow.update(["X"], 1.0)
    assert scorer_fast.score("X") > scorer_slow.score("X")


def test_update_clamps_performance_signal(scorer):
    scorer.update(["Bash"], performance_signal=5.0)  # should clamp to 1.0
    assert scorer.score("Bash") <= 1.0
    scorer.update(["Bash"], performance_signal=-3.0)  # should clamp to 0.0
    assert scorer.score("Bash") >= 0.0


def test_update_increments_total_combats(scorer):
    assert scorer._total_combats == 0
    scorer.update(["Bash"], 0.8)
    assert scorer._total_combats == 1


def test_save_and_load_round_trip(scorer, tmp_path):
    path = str(tmp_path / "scores.json")
    s1 = CardScorer(path=path)
    s1.update(["Inflame"], 0.9)
    s1.save()

    s2 = CardScorer(path=path)
    assert s2.score("Inflame") == pytest.approx(s1.score("Inflame"))
    assert s2._total_combats == 1


def test_save_is_atomic(scorer, tmp_path):
    """Save writes to .tmp then renames — no partial file visible."""
    path = str(tmp_path / "scores.json")
    scorer2 = CardScorer(path=path)
    scorer2.update(["Bash"], 0.7)
    scorer2.save()
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_load_missing_file_is_noop(tmp_path):
    s = CardScorer(path=str(tmp_path / "missing.json"))
    assert s.score("anything") == pytest.approx(0.5)
    assert s._total_combats == 0


def test_multiple_cards_updated(scorer):
    scorer.update(["Bash", "Strike_R", "Inflame"], 1.0)
    for card in ["Bash", "Strike_R", "Inflame"]:
        assert scorer.score(card) > 0.5
