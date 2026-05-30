# Binary property flags for Ironclad cards.
# Card IDs match the "id" field CommunicationMod puts in hand/deck objects.
# Upgraded cards share the same base ID (upgrade is a separate "upgrades" field).

_PROPS: dict[str, dict[str, bool]] = {
    # --- applies_vulnerable (enemy takes 50% more damage) ---
    "Bash":      {"applies_vulnerable": True},
    "Uppercut":  {"applies_vulnerable": True, "applies_weak": True},
    "Shockwave": {"applies_vulnerable": True, "applies_weak": True},

    # --- applies_weak only (enemy deals 25% less damage) ---
    "Clothesline": {"applies_weak": True},
    "Intimidate":  {"applies_weak": True},

    # --- draws_cards (play early to see more options this turn) ---
    "Battle Trance": {"draws_cards": True},
    "Pommel Strike": {"draws_cards": True},
    "Warcry":        {"draws_cards": True},
    "Burning Pact":  {"draws_cards": True},
    "Headbutt":      {"draws_cards": True},
    "Exhume":        {"draws_cards": True},

    # --- gains_block (play when you need to survive) ---
    "Defend_R":      {"gains_block": True},
    "Shrug It Off":  {"gains_block": True},
    "Iron Wave":     {"gains_block": True},
    "Ghostly Armor": {"gains_block": True},
    "Power Through": {"gains_block": True},
    "Flame Barrier": {"gains_block": True},
    "Sentinel":      {"gains_block": True},
    "True Grit":     {"gains_block": True},
    "Second Wind":   {"gains_block": True},
    "Entrench":      {"gains_block": True},
    "Impervious":    {"gains_block": True},

    # --- multi-flag ---
    "Feel No Pain": {"gains_block": True},  # gains block on exhaust (passive)
}

_DEFAULTS: dict[str, bool] = {
    "applies_vulnerable": False,
    "applies_weak":       False,
    "draws_cards":        False,
    "gains_block":        False,
}


def get_card_properties(card_id: str) -> dict[str, bool]:
    props = _DEFAULTS.copy()
    props.update(_PROPS.get(card_id, {}))
    return props
