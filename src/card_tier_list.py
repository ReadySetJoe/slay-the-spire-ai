# Ironclad card tiers: S (best) > A > B > C > D (worst)
# Cards not listed default to C tier
IRONCLAD_TIERS = {
    "S": [
        "Offering", "Impervious", "Feed", "Reaper",
        "Demon Form", "Barricade", "Limit Break",
    ],
    "A": [
        "Shrug It Off", "Inflame", "Battle Trance", "Pommel Strike",
        "Flame Barrier", "Metallicize", "Disarm", "Clothesline",
        "Uppercut", "Shockwave", "Spot Weakness", "True Grit",
        "Body Slam", "Carnage", "Hemokinesis", "Blood for Blood",
        "Fiend Fire", "Brutality", "Dark Embrace", "Feel No Pain",
        "Corruption", "Berserk", "Juggernaut",
    ],
    "B": [
        "Armaments", "Thunderclap", "Iron Wave", "Power Through",
        "Ghostly Armor", "Rage", "Evolve", "Fire Breathing",
        "Combust", "Rupture", "Dual Wield", "Exhume",
        "Second Wind", "Entrench", "Whirlwind", "Immolate",
        "Seeing Red", "Burning Pact", "Sentinel", "Headbutt",
    ],
    "C": [
        "Warcry", "Flex", "Havoc", "Rampage",
        "Searing Blow", "Bloodletting", "Intimidate",
        "Dropkick", "Sever Soul", "Wild Strike",
        "Reckless Charge", "Cleave", "Twin Strike",
    ],
    "D": [
        "Strike_R", "Defend_R",
    ],
}

# Build reverse lookup: card_id -> tier
_CARD_TO_TIER = {}
for tier, cards in IRONCLAD_TIERS.items():
    for card in cards:
        _CARD_TO_TIER[card] = tier

_TIER_RANK = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}


def get_card_tier(card_id: str) -> str:
    return _CARD_TO_TIER.get(card_id, "C")


def pick_best_card(cards: list) -> int | None:
    if not cards:
        return None
    best_idx = 0
    best_rank = _TIER_RANK.get(get_card_tier(cards[0].get("id", "")), 3)
    for i, card in enumerate(cards[1:], start=1):
        rank = _TIER_RANK.get(get_card_tier(card.get("id", "")), 3)
        if rank < best_rank:
            best_rank = rank
            best_idx = i
    return best_idx
