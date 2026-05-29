# Improved Rule-Based Agent - Design

## Goal

Make the SimpleAgent interact with all game systems properly — collecting rewards, using potions, and making reasonable card choices via a tier list — to establish a competent baseline before layering on RL.

## Changes

### Combat Rewards
Currently the bot PROCEEDs past everything. Instead, it should pick up gold, take relics, and grab potions before proceeding. CommunicationMod's COMBAT_REWARD screen has a list of rewards to CHOOSE from.

### Card Selection
Add an Ironclad card tier list ranking cards into tiers (S/A/B/C/skip). When offered cards, pick the highest-tier one. Always take something for now (no skipping).

### Potions
Always pick up potions when available. In combat, use healing potions when below 40% HP, use damage/buff potions on elite and boss fights.

### Boss Relics
Always take the first option (already implemented as CHOOSE 0).

### Potion Slot Management
If potion slots are full, don't try to grab potions (will cause errors).

## Files Changed
- `src/agent.py` — update SimpleAgent with new reward/potion/card logic
- `src/card_tier_list.py` — new file, Ironclad card rankings
- Tests for both

## What This Doesn't Do (left for RL)
- Card skipping / deck thinning
- Card synergy evaluation
- Path planning (which map nodes to pick)
- Smart shop purchases
- Potion timing optimization
