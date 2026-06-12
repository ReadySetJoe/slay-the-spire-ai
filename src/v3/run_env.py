import logging
import threading
from typing import Optional

import numpy as np
from gymnasium import spaces

from src.communicator import Communicator
from src.game_state import GameState
from src.run_tracker import RunTracker
from src.card_properties import get_card_properties
from src.v2.run_env import RunEnv
from src.v2.run_action_space import RunActionSpace
from src.v3.run_encoder import V3RunEncoder
from src.v3.run_reward import V3RunRewardShaper
from src.v3.card_scorer import CardScorer

logger = logging.getLogger(__name__)


class HungEpisodeError(Exception):
    pass


class V3RunEnv(RunEnv):
    def __init__(
        self,
        communicator: Communicator,
        run_tracker: Optional[RunTracker] = None,
        card_scorer: Optional[CardScorer] = None,
        timeout_seconds: float = 20.0,
    ):
        super().__init__(communicator, run_tracker)
        self._timeout_seconds = timeout_seconds
        self.card_scorer      = card_scorer or CardScorer()

        # Override encoder, reward shaper, and observation space
        self.encoder       = V3RunEncoder()
        self.reward_shaper = V3RunRewardShaper()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(V3RunEncoder.OBS_SIZE,),
            dtype=np.float32,
        )

        self._turn_state: dict = self._empty_turn_state()
        self._combat_cards_played: list[str] = []
        self._combat_total_damage: float = 0.0
        self._combat_total_enemy_max_hp: float = 0.0

    @staticmethod
    def _empty_turn_state() -> dict:
        return {
            "actions_taken": 0, "energy_spent": 0,
            "attacks_played": 0, "skills_played": 0, "powers_played": 0,
            "strength_gained": 0, "vulnerable_applied": False, "weak_applied": False,
            "damage_dealt": 0.0, "block_gained": 0.0,
            "last_card_was_buff": False, "last_card_was_debuff": False,
        }

    def _obs(self, state: Optional[GameState] = None) -> np.ndarray:
        s = state or self._current_state
        return self.encoder.encode(s, self._turn_state, self.card_scorer)

    def _reset_combat_tracking(self) -> None:
        self._combat_cards_played = []
        self._combat_total_damage = 0.0
        self._combat_total_enemy_max_hp = 0.0

    # --- timeout-protected receive ---

    def _receive_with_timeout(self) -> GameState:
        result: list = [None]
        error: list  = [None]

        def _recv():
            try:
                result[0] = self.communicator.receive_state()
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=_recv, daemon=True)
        t.start()
        t.join(self._timeout_seconds)
        if t.is_alive():
            raise HungEpisodeError(f"No game response after {self._timeout_seconds}s")
        if error[0]:
            raise error[0]
        return result[0]

    def _next_actionable_state(self) -> GameState:
        while True:
            state = self._receive_with_timeout()
            if state is None:
                raise RuntimeError("Connection closed")
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

    # --- reset ---

    def reset(self, seed=None, options=None):
        self._turn_state = self._empty_turn_state()
        self._reset_combat_tracking()
        # super().reset() sets self._current_state as a side effect; we then
        # re-encode with turn_state + card_scorer so the first obs is consistent
        # with step() observations (EMA scores instead of static heuristic).
        super().reset(seed=seed, options=options)
        return self._obs(), {}

    # --- step ---

    def step(self, action: int):
        assert self._current_state is not None, "Call reset() first"

        # Inherited debuff-turn tracking
        if (self._current_state.is_in_combat and
                self._current_state.turn != self._current_turn):
            self._debuff_applied_this_turn = False
            self._current_turn = self._current_state.turn

        prev    = self._current_state
        command = self._action_space_helper.action_to_command(action, prev)
        logger.info("Floor %d | HP %d/%d | Screen: %s | Action: %s",
                    prev.floor, prev.current_hp, prev.max_hp, prev.screen_type, command)

        writer = self.run_tracker.live_state_writer
        if writer:
            writer.write(prev, command)

        self.communicator.send_command(command)

        # Receive with timeout
        try:
            state = self._receive_with_timeout()
        except HungEpisodeError:
            logger.warning("Hung episode at floor %d", prev.floor)
            self.run_tracker.record_hung()
            return self._obs(), 0.0, False, True, {"hung": True, "floor": prev.floor}

        if state is None:
            return self._obs(), 0.0, True, False, {}

        if state.screen_type == "GAME_OVER":
            return self._handle_game_over(prev, state)

        # Combat transition detection
        prev_in_combat = prev.is_in_combat
        new_in_combat  = state.is_in_combat
        if prev_in_combat and not new_in_combat:
            self._on_combat_end()
        if not prev_in_combat and new_in_combat:
            self._reset_combat_tracking()
            self._turn_state = self._empty_turn_state()

        # Reward
        reward = self._compute_reward(action, prev, state)

        # Turn state + combat tracking updates
        if prev_in_combat:
            self._update_combat_tracking(action, prev, state)
            if action == RunActionSpace.END_TURN:
                self._turn_state = self._empty_turn_state()

        # Inherited per-episode metrics
        self._episode_reward_total += reward
        bucket = ("play" if action <= 59 else "end" if action == 60
                  else "potion" if action <= 90 else "noncombat")
        self._action_buckets[bucket] += 1
        if action == RunActionSpace.END_TURN and prev.is_in_combat:
            self._turn_energy_remaining.append(prev.energy)
        if prev.screen_type == "CARD_REWARD" and 91 <= action <= 98:
            self._track_card_pick(action, prev)
        if prev.is_in_combat:
            self._update_debuff_tracking(action, prev)

        self._current_state = state
        return self._obs(state), reward, False, False, {}

    # --- combat tracking ---

    def _update_combat_tracking(self, action: int, prev: GameState,
                                new: GameState) -> None:
        ts = self._turn_state

        # Energy spent this action
        if new.is_in_combat:
            ts["energy_spent"] += max(prev.energy - new.energy, 0)

        # Damage dealt
        prev_mon_hp = sum(m.get("current_hp", 0) for m in prev.monsters
                          if not m.get("is_gone"))
        new_mon_hp  = sum(m.get("current_hp", 0) for m in (new.monsters if new.is_in_combat else [])
                          if not m.get("is_gone"))
        damage = max(prev_mon_hp - new_mon_hp, 0)
        ts["damage_dealt"]          += damage
        self._combat_total_damage   += damage

        # Seed total enemy max HP once per combat
        if not self._combat_total_enemy_max_hp and prev.is_in_combat:
            self._combat_total_enemy_max_hp = sum(
                m.get("max_hp", 0) for m in prev.monsters if not m.get("is_gone")
            )

        # Block gained
        new_block = new.player_block if new.is_in_combat else 0
        ts["block_gained"] += max(new_block - prev.player_block, 0)

        # Strength gained
        def _strength(s):
            return next((p.get("amount", 0) for p in (s.combat_state or {})
                         .get("player", {}).get("powers", [])
                         if p.get("id") == "Strength"), 0)
        ts["strength_gained"] += max(_strength(new) - _strength(prev), 0)

        # Debuffs applied
        if new.is_in_combat:
            for m_new, m_prev in zip(new.monsters[:5], prev.monsters[:5]):
                if m_new.get("is_gone"):
                    continue
                def _stacks(m, pid):
                    return next((p.get("amount", 0) for p in m.get("powers", [])
                                 if p.get("id") == pid), 0)
                if _stacks(m_new, "Vulnerable") > _stacks(m_prev, "Vulnerable"):
                    ts["vulnerable_applied"] = True
                if _stacks(m_new, "Weak") > _stacks(m_prev, "Weak"):
                    ts["weak_applied"] = True

        # Card-specific tracking (actions 0–59 = card plays)
        if action < 60:
            ts["actions_taken"] += 1
            slot = action if action < 10 else (action - 10) // 5
            if slot < len(prev.hand):
                card      = prev.hand[slot]
                card_id   = card.get("id", "")
                card_type = card.get("type", "")
                if card_type == "ATTACK":
                    ts["attacks_played"] += 1
                elif card_type == "SKILL":
                    ts["skills_played"] += 1
                elif card_type == "POWER":
                    ts["powers_played"] += 1
                props                    = get_card_properties(card_id)
                ts["last_card_was_buff"] = (card_type == "POWER")
                ts["last_card_was_debuff"] = bool(
                    props.get("applies_vulnerable") or props.get("applies_weak")
                )
                self._combat_cards_played.append(card_id)

    def _on_combat_end(self) -> None:
        if not self._combat_cards_played:
            self._reset_combat_tracking()
            self._turn_state = self._empty_turn_state()
            return
        denom       = max(self._combat_total_enemy_max_hp, 1.0)
        performance = min(self._combat_total_damage / denom, 1.0)
        self.card_scorer.update(self._combat_cards_played, performance)
        self.card_scorer.save()
        self._reset_combat_tracking()
        self._turn_state = self._empty_turn_state()

    # --- game over ---

    def _handle_game_over(self, prev: GameState, state: GameState):
        # Feed final combat data to CardScorer before computing game-over reward.
        # When the last monster dies the game jumps straight to GAME_OVER, so
        # _on_combat_end() is never reached via the normal transition check.
        if prev.is_in_combat:
            self._on_combat_end()

        reward = self.reward_shaper.terminal_reward(state.floor)
        self._episode_reward_total += reward

        energy_efficiency = 1.0
        if self._turn_energy_remaining:
            energy_efficiency = 1.0 - sum(self._turn_energy_remaining) / (
                len(self._turn_energy_remaining) * 3  # Ironclad A0 max = 3
            )

        self.run_tracker.record_run(
            state, version="v3",
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
        obs = self.encoder.encode(prev, self._turn_state, self.card_scorer)
        self._current_state = None
        return obs, reward, True, False, {"episode": {"r": reward, "floor": state.floor}}
