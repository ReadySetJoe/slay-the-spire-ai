from src.v2.run_reward import RunRewardShaper

_MAX_ENERGY = 3  # Ironclad A0 base max energy


class V3RunRewardShaper(RunRewardShaper):
    """Stronger relic rewards and tighter energy-waste penalty than v2."""

    def open_chest_reward(self) -> float:
        return 0.25

    def combat_relic_reward(self) -> float:
        return 0.15

    def boss_relic_reward(self) -> float:
        return 0.20

    def shop_relic_reward(self) -> float:
        return 0.15

    def combat_step_reward(
        self,
        prev_hp: int, new_hp: int,
        prev_monster_hp: int, new_monster_hp: int,
        prev_living: int, new_living: int,
        prev_debuffs: int, new_debuffs: int,
        max_hp: int,
        is_end_action: bool,
        energy_remaining: int, max_energy: int,
        card_is_attack: bool,
        debuff_applied_this_turn: bool,
    ) -> float:
        # Call parent with is_end_action=False to skip v2's -0.3 penalty
        reward = super().combat_step_reward(
            prev_hp=prev_hp, new_hp=new_hp,
            prev_monster_hp=prev_monster_hp, new_monster_hp=new_monster_hp,
            prev_living=prev_living, new_living=new_living,
            prev_debuffs=prev_debuffs, new_debuffs=new_debuffs,
            max_hp=max_hp,
            is_end_action=False,
            energy_remaining=energy_remaining,
            max_energy=max_energy,
            card_is_attack=card_is_attack,
            debuff_applied_this_turn=debuff_applied_this_turn,
        )
        if is_end_action:
            reward -= 0.5 * (energy_remaining / _MAX_ENERGY)
        return reward
