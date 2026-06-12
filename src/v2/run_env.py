import logging
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.communicator import Communicator
from src.game_state import GameState
from src.run_tracker import RunTracker
from src.v2.run_encoder import RunEncoder
from src.v2.run_action_space import RunActionSpace
from src.v2.run_reward import RunRewardShaper

logger = logging.getLogger(__name__)


class RunEnv(gym.Env):
    """Gymnasium env for STS. One episode = one complete run."""
    metadata = {"render_modes": []}

    def __init__(self, communicator: Communicator,
                 run_tracker: Optional[RunTracker] = None):
        super().__init__()
        self.communicator  = communicator
        self.run_tracker   = run_tracker or RunTracker()
        self.encoder       = RunEncoder()
        self._action_space_helper = RunActionSpace()
        self.reward_shaper = RunRewardShaper()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(RunEncoder.OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(RunActionSpace.TOTAL_ACTIONS)

        self._current_state: Optional[GameState] = None
        self._initialized: bool = False
        self._debuff_applied_this_turn: bool = False
        self._current_turn: int = 0

        # Per-episode metric tracking (reset in reset())
        self._episode_reward_total: float = 0.0
        self._turn_energy_remaining: list = []   # energy left each time END is played
        self._action_buckets: dict = {"play": 0, "end": 0, "potion": 0, "noncombat": 0}
        self._card_picks_this_run: list = []

    def action_masks(self) -> np.ndarray:
        if self._current_state is None:
            return np.ones(RunActionSpace.TOTAL_ACTIONS, dtype=np.bool_)
        return self._action_space_helper.get_action_mask(self._current_state)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._debuff_applied_this_turn = False
        self._current_turn = 0
        self._episode_reward_total = 0.0
        self._turn_energy_remaining = []
        self._action_buckets = {"play": 0, "end": 0, "potion": 0, "noncombat": 0}
        self._card_picks_this_run = []

        if not self._initialized:
            self.communicator.send_ready()
            self._initialized = True

        state = self._next_actionable_state()
        self._current_state = state
        return self.encoder.encode(state), {}

    def _next_actionable_state(self) -> GameState:
        while True:
            state = self.communicator.receive_state()
            if state is None:
                raise RuntimeError("Connection closed while waiting for actionable state")
            if state.error:
                logger.warning("Error from game: %s", state.error)
                continue
            if not state.ready_for_command:
                continue
            if not state.in_game:
                if "START" in state.available_commands:
                    self.communicator.send_command("START IRONCLAD 0")
                continue
            return state

    def step(self, action: int):
        assert self._current_state is not None, "Call reset() first"

        # Reset debuff tracking when turn advances
        if (self._current_state.is_in_combat and
                self._current_state.turn != self._current_turn):
            self._debuff_applied_this_turn = False
            self._current_turn = self._current_state.turn

        prev = self._current_state
        command = self._action_space_helper.action_to_command(action, prev)
        logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                    prev.floor, prev.current_hp, prev.max_hp,
                    prev.screen_type, command)
        self.communicator.send_command(command)

        state = self.communicator.receive_state()
        if state is None:
            return self.encoder.encode(prev), 0.0, True, False, {}

        if state.screen_type == "GAME_OVER":
            return self._handle_game_over(prev, state)

        reward = self._compute_reward(action, prev, state)

        # Accumulate per-episode metrics
        self._episode_reward_total += reward
        bucket = ("play" if action <= 59 else "end" if action == 60
                  else "potion" if action <= 90 else "noncombat")
        self._action_buckets[bucket] += 1
        if action == RunActionSpace.END_TURN and prev.is_in_combat:
            self._turn_energy_remaining.append(prev.energy)
        if prev.screen_type == "CARD_REWARD" and 91 <= action <= 98:
            self._track_card_pick(action, prev)

        # Update debuff tracking after computing reward
        if prev.is_in_combat:
            self._update_debuff_tracking(action, prev)

        self._current_state = state
        return self.encoder.encode(state), reward, False, False, {}

    def _compute_reward(self, action: int, prev: GameState, new: GameState) -> float:
        if prev.is_in_combat:
            return self._combat_reward(action, prev, new)
        return self._noncombat_reward(action, prev, new)

    def _combat_reward(self, action: int, prev: GameState, new: GameState) -> float:
        def debuff_stacks(monsters):
            total = 0
            for m in monsters:
                if m.get("is_gone"):
                    continue
                for p in m.get("powers", []):
                    if p.get("id") in ("Vulnerable", "Weak"):
                        total += p.get("amount", 0)
            return total

        new_monsters = new.monsters if new.is_in_combat else []

        return self.reward_shaper.combat_step_reward(
            prev_hp          = prev.current_hp,
            new_hp           = new.current_hp,
            prev_monster_hp  = sum(m.get("current_hp", 0) for m in prev.monsters if not m.get("is_gone")),
            new_monster_hp   = sum(m.get("current_hp", 0) for m in new_monsters if not m.get("is_gone")),
            prev_living      = sum(1 for m in prev.monsters if not m.get("is_gone")),
            new_living       = sum(1 for m in new_monsters if not m.get("is_gone")),
            prev_debuffs     = debuff_stacks(prev.monsters),
            new_debuffs      = debuff_stacks(new_monsters),
            max_hp           = prev.max_hp,
            is_end_action    = (action == RunActionSpace.END_TURN),
            energy_remaining = prev.energy,
            max_energy       = 4,
            card_is_attack   = self._played_card_is_attack(action, prev),
            debuff_applied_this_turn = self._debuff_applied_this_turn,
        )

    def _played_card_is_attack(self, action: int, state: GameState) -> bool:
        if action < 10:
            slot = action
        elif 10 <= action < 60:
            slot = (action - 10) // 5
        else:
            return False
        if slot < len(state.hand):
            return state.hand[slot].get("type") == "ATTACK"
        return False

    def _update_debuff_tracking(self, action: int, state: GameState) -> None:
        from src.card_properties import get_card_properties
        if action < 10:
            slot = action
        elif 10 <= action < 60:
            slot = (action - 10) // 5
        else:
            return
        if slot < len(state.hand):
            props = get_card_properties(state.hand[slot].get("id", ""))
            if props.get("applies_vulnerable") or props.get("applies_weak"):
                self._debuff_applied_this_turn = True

    def _track_card_pick(self, action: int, prev: GameState) -> None:
        idx = action - 91
        cards = (prev.screen_state or {}).get("cards", [])
        if idx < len(cards):
            from src.card_tier_list import get_card_tier
            card = cards[idx]
            self._card_picks_this_run.append({
                "id": card.get("id", ""),
                "name": card.get("name") or card.get("id", ""),
                "tier": get_card_tier(card.get("id", "")) or "C",
                "run": self.run_tracker.run_number + 1,
            })

    def _noncombat_reward(self, action: int, prev: GameState, new: GameState) -> float:
        screen = prev.screen_type
        ss     = prev.screen_state or {}

        if screen == "CARD_REWARD" and 91 <= action <= 98:
            idx   = action - 91
            cards = ss.get("cards", [])
            if idx < len(cards):
                return self.reward_shaper.card_reward(cards[idx], prev.deck)
            return 0.0

        if screen == "SHOP_SCREEN":
            if 91 <= action <= 98:
                idx    = action - 91
                cards  = ss.get("cards", [])
                relics = ss.get("relics", [])
                if idx < len(cards):
                    return self.reward_shaper.shop_card_reward(cards[idx], prev.gold, prev.deck)
                relic_idx = idx - len(cards)
                if relic_idx < len(relics):
                    return self.reward_shaper.shop_relic_reward()
            if action == 100:  # PURGE
                from src.card_tier_list import get_card_tier
                d_cards = [c for c in prev.deck if c.get("type") in ("STATUS", "CURSE")]
                if d_cards:
                    return self.reward_shaper.purge_reward(d_cards[0])
                d_tier = [c for c in prev.deck if get_card_tier(c.get("id", "")) == "D"]
                if d_tier:
                    return self.reward_shaper.purge_reward(d_tier[0])
            return 0.0

        if screen == "REST":
            if action == 101:  # CHOOSE rest
                max_hp    = max(prev.max_hp, 1)
                hp_gained = min(int(max_hp * 0.3), max_hp - prev.current_hp)
                return self.reward_shaper.rest_heal_reward(hp_gained, max_hp)
            if action == 102:  # CHOOSE smith
                return self.reward_shaper.rest_smith_reward()

        if screen == "CHEST" and action == 103:  # OPEN
            return self.reward_shaper.open_chest_reward()

        if screen == "COMBAT_REWARD" and 91 <= action <= 98:
            idx = action - 91
            rewards = ss.get("rewards", [])
            if idx < len(rewards):
                reward_type = rewards[idx].get("reward_type", "")
                if reward_type == "RELIC":
                    return self.reward_shaper.combat_relic_reward()
                if reward_type == "POTION":
                    return self.reward_shaper.combat_potion_reward()
            return 0.0

        if screen == "BOSS_REWARD" and 91 <= action <= 98:
            return self.reward_shaper.boss_relic_reward()

        return 0.0

    def _handle_game_over(self, prev: GameState, state: GameState):
        reward = self.reward_shaper.terminal_reward(state.floor)
        self._episode_reward_total += reward

        if self._turn_energy_remaining:
            energy_efficiency = 1.0 - sum(self._turn_energy_remaining) / (
                len(self._turn_energy_remaining) * 4
            )
        else:
            energy_efficiency = 1.0

        self.run_tracker.record_run(
            state,
            version="v2",
            episode_reward=round(self._episode_reward_total, 4),
            energy_efficiency=round(energy_efficiency, 4),
        )

        writer = self.run_tracker.live_state_writer
        if writer:
            writer.write_v2_metrics(
                action_counts=dict(self._action_buckets),
                card_picks=list(self._card_picks_this_run),
                episode_reward=round(self._episode_reward_total, 4),
                energy_efficiency=round(energy_efficiency, 4),
            )

        summary = self.run_tracker.summary()
        logger.info(
            "GAME_OVER | floor=%d | runs=%d | win_rate=%.1f%% | reward=%.3f | energy=%.0f%%",
            state.floor, summary["total_runs"], summary["win_rate"] * 100,
            self._episode_reward_total, energy_efficiency * 100,
        )
        self.communicator.send_command("PROCEED")
        obs = self.encoder.encode(prev)
        self._current_state = None
        info = {"episode": {"r": reward, "floor": state.floor}}
        return obs, reward, True, False, info
