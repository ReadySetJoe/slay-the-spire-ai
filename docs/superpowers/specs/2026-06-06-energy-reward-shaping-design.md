# Energy Efficiency Reward Shaping

**Date:** 2026-06-06
**Status:** Approved

## Problem

The RL agent currently receives reward signals only for damage dealt, damage taken, and kills. There is no incentive to spend energy efficiently within a turn. Leaving energy on the table at END_TURN is a common failure mode that limits how far runs progress.

## Goal

Add a small proportional bonus on END_TURN that rewards the agent for using all available energy each turn, without overwhelming the existing damage/kill signals.

## Design

### Reward formula

When the agent selects `END_TURN_ACTION` (action 60), append an energy efficiency bonus to the step reward:

```
bonus = energy_efficiency_bonus * (1 - prev_energy / max_energy)
```

- `prev_energy`: energy remaining when END_TURN was chosen (`self._current_state.energy` snapshotted before the command is sent)
- `max_energy`: constant `3` (Ironclad A0 base; not the encoder ceiling of 4)
- `energy_efficiency_bonus`: configurable weight, default `0.1`

Example values:
| Energy used | Leftover | Bonus |
|---|---|---|
| 3/3 | 0 | +0.10 |
| 2/3 | 1 | +0.067 |
| 1/3 | 2 | +0.033 |
| 0/3 | 3 | +0.00 |

The bonus is always non-negative, so it never fights against damage or kill signals.

### Affected code

**`src/combat_env.py`** — only file changed:

1. `__init__`: add `energy_efficiency_bonus: float = 0.1` parameter, store as `self._energy_efficiency_bonus`
2. `step()`: snapshot `prev_energy = self._current_state.energy` alongside the existing `prev_hp` / `prev_monster_hp` snapshots; pass `action` through to `_compute_step_reward`
3. `_compute_step_reward`: add `action: int` parameter; when `action == ActionSpace.END_TURN_ACTION`, compute and add the bonus

### What does not change

- `_compute_reward` (terminal reward) — unchanged, HP-based
- `StateEncoder`, `ActionSpace`, `GameState` — untouched
- Mid-turn card play steps receive no energy component

## Trade-offs considered

- **Proportional vs binary bonus:** Proportional chosen for smooth gradient signal; binary only rewards perfect turns and is sparse
- **Penalty vs bonus framing:** Bonus (non-negative) chosen to avoid adding negative pressure that could conflict with damage signals; mathematically equivalent relative gap
- **max_energy constant vs dynamic:** Using `3` (base Ironclad) rather than a dynamic lookup keeps the implementation simple; if relics that modify max energy become relevant this can be revisited

## Success criteria

Agent ends turns with 0 or 1 leftover energy more consistently than before the change. Run depth increases over baseline.
