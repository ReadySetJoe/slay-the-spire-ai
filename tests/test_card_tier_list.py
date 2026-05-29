from src.card_tier_list import IRONCLAD_TIERS, get_card_tier, pick_best_card


def test_known_card_has_tier():
    assert get_card_tier("Offering") == "S"
    assert get_card_tier("Shrug It Off") == "A"
    assert get_card_tier("Strike_R") == "D"


def test_unknown_card_defaults_to_c():
    assert get_card_tier("SomeModdedCard") == "C"


def test_pick_best_card_chooses_highest_tier():
    cards = [
        {"id": "Strike_R", "name": "Strike"},
        {"id": "Offering", "name": "Offering"},
        {"id": "Shrug It Off", "name": "Shrug It Off"},
    ]
    best = pick_best_card(cards)
    assert best == 1  # index of Offering (S tier)


def test_pick_best_card_breaks_ties_by_order():
    cards = [
        {"id": "Shrug It Off", "name": "Shrug It Off"},
        {"id": "Inflame", "name": "Inflame"},
    ]
    best = pick_best_card(cards)
    assert best == 0  # both A tier, first wins


def test_pick_best_card_empty_returns_none():
    assert pick_best_card([]) is None
