# Auto-Restart Loop with Stats Tracking - Design

## Goal

Bot automatically starts new Ironclad runs after each game over, logging results to a JSONL file. Runs unattended overnight.

## Auto-Restart Flow

```
Game starts -> bot plays run -> GAME_OVER screen -> log results -> PROCEED -> main menu -> START IRONCLAD 0 -> repeat
```

## Stats Tracking

After each run, append a JSON line to `data/run_log.jsonl` with:
- seed, floor_reached, ascension_level
- result (win/loss)
- current_hp, max_hp at end
- gold, deck_size, relic_count
- timestamp, run_number

JSONL format (one JSON object per line) for easy appending.

## Files

- `src/run_tracker.py` - tracks run stats and writes to JSONL
- `src/agent.py` - add GAME_OVER handling, auto-start logic
- `src/game_loop.py` - pass run tracker through the loop
- `data/` - directory for run logs (gitignored)

## Error Resilience

If the bot hits an unknown state or error, log it and try to recover. The loop should not crash overnight.
