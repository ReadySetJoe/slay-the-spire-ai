# src/agent.py
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Optional

from src.card_tier_list import get_card_tier, pick_best_card
from src.game_state import GameState

_FALLBACK_SEQUENCE = ("CONFIRM", "CANCEL", "PROCEED", "LEAVE", "SKIP", "STATE")

logger = logging.getLogger(__name__)


class Agent(ABC):
    @abstractmethod
    def act(self, state: GameState) -> str:
        """Given a game state, return a command string."""
        pass


class SimpleAgent(Agent):
    """Rule-based agent that plays Ironclad with simple heuristics."""

    HEALING_POTIONS = {"Fruit Juice", "Blood Potion", "Fairy in a Bottle",
                       "Regen Potion", "Ancient Potion"}

    ATTACK_POTIONS = {"Fire Potion", "Explosive Potion", "Poison Potion",
                      "Fear Potion", "Strength Potion", "Dexterity Potion",
                      "Speed Potion", "Weak Potion", "Energy Potion",
                      "Swift Potion", "Flex Potion", "Steroid Potion",
                      "Focus Potion", "Cultist Potion", "Liquid Bronze",
                      "Essence of Steel", "Heart of Iron", "Ghost In A Jar",
                      "Ambrosia", "Liquid Memories", "Distilled Chaos",
                      "Duplication Potion", "Blessing of the Forge",
                      "Elixir", "Gambler's Brew", "Entropic Brew",
                      "Smoke Bomb", "Snecko Oil", "Block Potion"}

    # Node symbol → priority rank (lower = more preferred)
    # Two tables: one for when HP is low, one for when HP is healthy.
    _MAP_PRIORITY_LOW_HP  = {"R": 0, "?": 1, "T": 2, "$": 3, "M": 4, "E": 5}
    _MAP_PRIORITY_HIGH_HP = {"E": 0, "T": 1, "?": 2, "$": 3, "R": 4, "M": 5}

    def __init__(self, scorer=None):
        self.scorer = scorer
        self._run_picks: list[str] = []
        self._last_shop_gold: int | None = None  # loop-guard for shop purchases

    def on_game_over(self, state: GameState):
        """Update card scores from this run's picks, then reset for next run."""
        if self.scorer and self._run_picks:
            quality = (state.current_hp / max(state.max_hp, 1)) * (state.floor / 55)
            logger.info(
                "Run quality=%.3f | %d cards picked: %s",
                quality, len(self._run_picks), self._run_picks,
            )
            self.scorer.update_run(self._run_picks, quality)
        self._run_picks.clear()
        self._last_shop_gold = None

    def act(self, state: GameState) -> str:
        # GRID/HAND_SELECT must be checked before is_in_combat: these screens can
        # appear mid-combat (Warcry → HAND_SELECT, Headbutt → GRID) and
        # is_in_combat returns True for them, which would otherwise route here to
        # _handle_combat and send "END" into a card-selection prompt.
        if state.screen_type in ("GRID", "HAND_SELECT"):
            return self._handle_grid_hand_select(state)

        if state.is_in_combat:
            return self._handle_combat(state)

        if state.screen_type == "CARD_REWARD":
            return self._handle_card_reward(state)

        if state.screen_type == "REST":
            return self._handle_rest(state)

        if state.screen_type == "MAP":
            return self._handle_map(state)

        if state.screen_type == "CHEST":
            if "OPEN" in state.available_commands:
                return "OPEN"
            if "CHOOSE" in state.available_commands:
                return "CHOOSE 0"
            if "PROCEED" in state.available_commands:
                return "PROCEED"

        if state.screen_type == "EVENT":
            return self._handle_event(state)

        if state.screen_type in ("SHOP_ROOM", "SHOP_SCREEN"):
            return self._handle_shop_screen(state)

        if state.screen_type == "COMBAT_REWARD":
            return self._handle_combat_reward(state)

        if state.screen_type == "BOSS_REWARD":
            return "CHOOSE 0"

        if "PROCEED" in state.available_commands:
            return "PROCEED"

        if "LEAVE" in state.available_commands:
            return "LEAVE"

        if "CHOOSE" in state.available_commands:
            return "CHOOSE 0"

        if "CONFIRM" in state.available_commands:
            return "CONFIRM"

        logger.warning("Unhandled screen type: %s | Commands: %s", state.screen_type, state.available_commands)
        return "STATE"

    def _handle_combat(self, state: GameState) -> str:
        if "POTION" in state.available_commands:
            potion_action = self._check_potions(state)
            if potion_action:
                return potion_action

        playable = [
            (i, card) for i, card in enumerate(state.hand)
            if card.get("is_playable", False)
        ]

        if not playable:
            return "END"

        # Find first living monster for targeting
        target = 0
        for i, m in enumerate(state.monsters):
            if not m.get("is_gone", False):
                target = i
                break

        # Play first playable card (1-indexed)
        idx, card = playable[0]
        card_index = idx + 1  # CommunicationMod uses 1-indexed cards
        if card.get("has_target", False):
            return f"PLAY {card_index} {target}"
        return f"PLAY {card_index}"

    def _check_potions(self, state: GameState) -> str | None:
        hp_ratio = state.current_hp / max(state.max_hp, 1)
        is_tough_fight = any(
            m.get("max_hp", 0) > 100 for m in state.monsters
            if not m.get("is_gone", False)
        )

        target = 0
        for i, m in enumerate(state.monsters):
            if not m.get("is_gone", False):
                target = i
                break

        for i, potion in enumerate(state.potions):
            if not potion.get("can_use", False):
                continue
            pid = potion.get("id", "")

            # Use healing potions when low
            if pid in self.HEALING_POTIONS and hp_ratio < 0.4:
                if potion.get("requires_target", False):
                    return f"POTION Use {i} {target}"
                return f"POTION Use {i}"

            # Use attack/buff potions on tough fights
            if pid in self.ATTACK_POTIONS and is_tough_fight:
                if potion.get("requires_target", False):
                    return f"POTION Use {i} {target}"
                return f"POTION Use {i}"

        return None

    def _handle_card_reward(self, state: GameState) -> str:
        if "CHOOSE" not in state.available_commands:
            return "PROCEED"
        cards = state.screen_state.get("cards", []) if state.screen_state else []
        if not cards:
            return "PROCEED"

        if self.scorer:
            idx = self.scorer.softmax_pick(cards)
        else:
            idx = pick_best_card(cards) or 0

        self._run_picks.append(cards[idx].get("id", ""))
        return f"CHOOSE {idx}"

    def _handle_rest(self, state: GameState) -> str:
        if "CHOOSE" not in state.available_commands:
            return "PROCEED"

        hp_ratio = state.current_hp / max(state.max_hp, 1)

        # Parse which options are actually enabled from screen_state.
        # CommunicationMod may use "rest_options" or "options"; try both.
        ss = state.screen_state or {}
        raw = ss.get("rest_options") or ss.get("options", [])
        available: set[str] = set()
        for opt in raw:
            if isinstance(opt, str):
                available.add(opt.lower())
            elif isinstance(opt, dict) and not opt.get("disabled", False):
                oid = opt.get("id", "").lower()
                if oid:
                    available.add(oid)

        priority = (
            ("rest", "smith", "lift", "toke", "dig", "recall")
            if hp_ratio < 0.6
            else ("smith", "rest", "lift", "toke", "dig", "recall")
        )

        if available:
            for option in priority:
                if option in available:
                    return f"CHOOSE {option}"
            return f"CHOOSE {next(iter(available))}"

        # No screen_state data — blind priority pick
        if hp_ratio < 0.6:
            return "CHOOSE rest"
        return "CHOOSE smith"

    def _handle_combat_reward(self, state: GameState) -> str:
        rewards = []
        if state.screen_state:
            rewards = state.screen_state.get("rewards", [])

        if not rewards or "CHOOSE" not in state.available_commands:
            return "PROCEED"

        potion_slots_full = all(
            p.get("id") != "Potion Slot" for p in state.potions
        )

        for i, reward in enumerate(rewards):
            rtype = reward.get("reward_type", "")
            if rtype == "POTION" and potion_slots_full:
                continue
            return f"CHOOSE {i}"

        return "PROCEED"

    def _handle_event(self, state: GameState) -> str:
        ss = state.screen_state or {}
        name = ss.get("event_name") or ss.get("name") or "unknown"
        options = ss.get("options", [])
        # Use choice_index from each option dict — that's what CommunicationMod
        # expects in the CHOOSE command. Filter out disabled options.
        enabled = [
            opt.get("choice_index", i)
            for i, opt in enumerate(options)
            if not opt.get("disabled", False)
        ]
        if not enabled:
            # All options disabled (shouldn't happen) — fall back to first
            enabled = [options[0].get("choice_index", 0)] if options else [0]
        choice = random.choice(enabled)
        logger.info("EVENT | %s | %d options (%d enabled) | chose %d | full_state=%s",
                    name, len(options), len(enabled), choice, ss)
        return f"CHOOSE {choice}"

    def _handle_grid_hand_select(self, state: GameState) -> str:
        ss = state.screen_state or {}
        cards = ss.get("cards", [])
        # CommunicationMod tracks selected cards in a separate array, not as a
        # boolean on individual card objects. HAND_SELECT uses "selected";
        # GRID uses "selected_cards".
        already_selected = {
            c.get("uuid")
            for c in ss.get("selected", []) + ss.get("selected_cards", [])
        }
        unselected = [
            (i, c) for i, c in enumerate(cards)
            if c.get("uuid") not in already_selected
        ]

        if ss.get("any_number", False):
            # Optional discard-and-redraw (Gambling Chip): swap D-tier cards only.
            if "CHOOSE" in state.available_commands:
                d_tier = [
                    (i, c) for i, c in unselected
                    if get_card_tier(c.get("id", "")) == "D"
                ]
                if d_tier:
                    return f"CHOOSE {d_tier[0][0]}"
            if "CONFIRM" in state.available_commands:
                return "CONFIRM"

        # Required selection (Warcry → HAND_SELECT, Headbutt → GRID, etc.)
        if "CONFIRM" in state.available_commands:
            return "CONFIRM"
        if "CHOOSE" in state.available_commands:
            if unselected:
                best_local = pick_best_card([c for _, c in unselected])
                best_idx = unselected[best_local if best_local is not None else 0][0]
                return f"CHOOSE {best_idx}"
            if "CANCEL" not in state.available_commands:
                # UUID tracking found no candidates but game still wants a CHOOSE.
                # (CANCEL unavailable means the game requires a selection.)
                return "CHOOSE 0"
        if "CANCEL" in state.available_commands:
            return "CANCEL"
        return "STATE"

    def _handle_map(self, state: GameState) -> str:
        nodes = state.screen_state.get("next_nodes", []) if state.screen_state else []
        if not nodes or "CHOOSE" not in state.available_commands:
            return "CHOOSE 0"

        hp_ratio = state.current_hp / max(state.max_hp, 1)
        priorities = self._MAP_PRIORITY_LOW_HP if hp_ratio < 0.4 else self._MAP_PRIORITY_HIGH_HP

        best_idx, best_rank = 0, float("inf")
        for i, node in enumerate(nodes):
            rank = priorities.get(node.get("symbol", "M"), 4)
            if rank < best_rank:
                best_rank, best_idx = rank, i

        chosen = nodes[best_idx]
        logger.info("MAP | hp=%.0f%% | options=%s | chose=%s",
                    hp_ratio * 100,
                    [n.get("symbol") for n in nodes],
                    chosen.get("symbol"))
        return f"CHOOSE {best_idx}"

    def _handle_shop_screen(self, state: GameState) -> str:
        ss = state.screen_state or {}
        gold = state.gold
        cards   = ss.get("cards",   [])
        relics  = ss.get("relics",  [])
        purge_cost      = ss.get("purge_cost",      9999)
        purge_available = ss.get("purge_available", False)

        logger.info(
            "SHOP | gold=%d | cards=%s | relics=%s | purge_cost=%s | purge_avail=%s | commands=%s",
            gold,
            [(c.get("id"), c.get("price")) for c in cards],
            [(r.get("id"), r.get("price")) for r in relics],
            purge_cost, purge_available,
            state.available_commands,
        )

        # Loop-guard: if gold didn't decrease since our last purchase attempt,
        # the command had no effect — leave rather than spinning.
        if self._last_shop_gold is not None and gold >= self._last_shop_gold:
            logger.warning("Shop purchase had no effect (gold %d >= %d), leaving shop.",
                           gold, self._last_shop_gold)
            self._last_shop_gold = None
            return "PROCEED"
        self._last_shop_gold = None

        BUFFER = 50  # always keep this much gold in reserve

        # Priority 1: remove a D-tier card (CommunicationMod exposes "PURGE" command)
        if purge_available and "PURGE" in state.available_commands:
            d_cards = [c for c in state.deck
                       if get_card_tier(c.get("id", "")) == "D"]
            if d_cards and gold >= purge_cost + BUFFER:
                logger.info("SHOP | removing D-tier card, cost=%d", purge_cost)
                self._last_shop_gold = gold
                return "PURGE"

        # Priority 2: buy the best S/A-tier card we can afford
        # CommunicationMod indexes shop cards 0-based in the cards list.
        for i, card in enumerate(cards):
            if not card.get("is_in_stock", True):
                continue
            price = card.get("price", 9999)
            tier  = get_card_tier(card.get("id", ""))
            if tier in ("S", "A") and gold >= price + BUFFER:
                logger.info("SHOP | buying %s (tier=%s, price=%d)", card.get("id"), tier, price)
                self._last_shop_gold = gold
                return f"CHOOSE {i}"

        # Priority 3: buy a relic (assumed to appear after cards in the flat index)
        for i, relic in enumerate(relics):
            if not relic.get("is_in_stock", True):
                continue
            price = relic.get("price", 9999)
            if gold >= price + BUFFER:
                flat_idx = len(cards) + i
                logger.info("SHOP | buying relic %s (price=%d, idx=%d)",
                            relic.get("id"), price, flat_idx)
                self._last_shop_gold = gold
                return f"CHOOSE {flat_idx}"

        self._last_shop_gold = None
        return "PROCEED"


class StuckDetectorAgent(Agent):
    """Wraps any Agent and prevents infinite oscillation.

    Tracks a fingerprint of (screen_type, frozenset(available_commands)).
    Once the same fingerprint has been seen `threshold` times in a row, the
    wrapper takes over and cycles through _FALLBACK_SEQUENCE instead of
    delegating to the wrapped agent. A CRITICAL log with full screen_state
    is emitted so the stuck state can be diagnosed from the log alone.

    Public attribute access falls through to the wrapped agent via __getattr__.
    Private (_-prefixed) names raise AttributeError intentionally — this keeps
    pickle/deepcopy safe. Callers needing private methods on the wrapped agent
    (e.g. _check_potions) must hold a direct reference to it.
    """

    def __init__(self, agent: Agent, threshold: int = 3, log_interval: int = 10):
        self._agent = agent
        self._threshold = threshold
        self._log_interval = log_interval
        self._last_fp: Optional[tuple] = None
        self._seen_count: int = 0
        self._action_history: list[str] = []

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._agent, name)

    def act(self, state: GameState) -> str:
        fp = (
            state.screen_type,
            frozenset(state.available_commands),
            json.dumps(state.screen_state, sort_keys=True, default=str),
        )

        if fp != self._last_fp:
            self._last_fp = fp
            self._seen_count = 1
            self._action_history.clear()
            action = self._agent.act(state)
            self._action_history.append(action)
            return action

        self._seen_count += 1

        if self._seen_count < self._threshold:
            action = self._agent.act(state)
            self._action_history.append(action)
            return action

        # Stuck — take over with fallback sequence
        stuck_step = self._seen_count - self._threshold  # 0-based

        if stuck_step == 0 or stuck_step % self._log_interval == 0:
            logger.critical(
                "STUCK DETECTED — floor=%d hp=%d/%d screen=%s commands=%s\n"
                "Actions sent (last 5): %s\n"
                "screen_state: %s\n"
                "Paste this block into Claude to diagnose.",
                state.floor, state.current_hp, state.max_hp,
                state.screen_type, sorted(state.available_commands),
                list(self._action_history[-5:]),
                json.dumps(state.screen_state, default=str),
            )

        start = min(stuck_step, len(_FALLBACK_SEQUENCE) - 1)
        cmd = "STATE"
        for i in range(start, len(_FALLBACK_SEQUENCE)):
            candidate = _FALLBACK_SEQUENCE[i]
            if candidate in state.available_commands or candidate == "STATE":
                cmd = candidate
                break

        # After emitting STATE, reset the stuck counter so the next state response
        # gets a fresh fingerprint comparison instead of immediately saturating again.
        if cmd == "STATE":
            self._last_fp = None
            self._seen_count = 0

        self._action_history.append(cmd)
        return cmd

    def on_game_over(self, state: GameState) -> None:
        if hasattr(self._agent, "on_game_over"):
            self._agent.on_game_over(state)
        self._last_fp = None
        self._seen_count = 0
        self._action_history.clear()
