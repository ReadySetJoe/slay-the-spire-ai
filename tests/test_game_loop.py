# tests/test_game_loop.py
import io
import json
from src.game_loop import GameLoop
from src.agent import SimpleAgent
from src.communicator import Communicator
from src.run_tracker import RunTracker
import tempfile
import os


def make_state(screen_type="NONE", commands=None, in_game=True, combat=True):
    """Helper to build a JSON game state string."""
    if commands is None:
        commands = ["PLAY", "END"] if combat else ["CHOOSE", "PROCEED"]
    data = {
        "available_commands": commands,
        "ready_for_command": True,
        "in_game": in_game,
        "game_state": {
            "screen_type": screen_type,
            "seed": 1, "floor": 1, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": 80, "max_hp": 80, "gold": 99,
            "deck": [], "relics": [], "potions": [], "map": [], "act": 1,
            "combat_state": {
                "hand": [
                    {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
                     "is_playable": True, "has_target": True, "uuid": "a1"},
                ],
                "draw_pile": [], "discard_pile": [], "exhaust_pile": [],
                "monsters": [
                    {"name": "Jaw Worm", "current_hp": 42, "max_hp": 42,
                     "block": 0, "intent": "ATTACK", "is_gone": False},
                ],
                "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
                "turn": 1,
            } if combat else None,
        },
    }
    return json.dumps(data)


def test_game_loop_sends_ready():
    inp = io.StringIO(make_state() + "\n")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    loop = GameLoop(comm, agent)
    loop.step()  # sends ready + processes one state
    output = out.getvalue()
    assert output.startswith("ready\n")


def test_game_loop_processes_state_and_sends_action():
    inp = io.StringIO(make_state() + "\n")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    loop = GameLoop(comm, agent)
    loop.step()
    output_lines = out.getvalue().strip().split("\n")
    assert len(output_lines) == 2  # "ready" + one action
    assert output_lines[0] == "ready"
    assert output_lines[1].startswith("PLAY")


def test_game_loop_stops_on_eof():
    inp = io.StringIO("")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    loop = GameLoop(comm, agent)
    loop.step()
    output = out.getvalue()
    assert output == "ready\n"  # Only ready, no action (EOF)


def make_game_over(victory=False):
    data = {
        "available_commands": ["PROCEED"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "screen_type": "GAME_OVER",
            "screen_state": {"victory": victory},
            "seed": 1, "floor": 10, "ascension_level": 0, "class": "IRONCLAD",
            "current_hp": 0, "max_hp": 80, "gold": 50,
            "deck": [{"id": "Strike_R"}], "relics": [{"id": "Burning Blood"}],
            "potions": [], "map": [], "act": 1,
            "combat_state": None,
        },
    }
    return json.dumps(data)


def make_menu_state():
    return json.dumps({
        "available_commands": ["START"],
        "ready_for_command": True,
        "in_game": False,
    })


def test_game_loop_handles_game_over():
    """Game over should record stats and PROCEED."""
    inp = io.StringIO(make_game_over() + "\n")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = RunTracker(log_path=os.path.join(tmpdir, "log.jsonl"))
        loop = GameLoop(comm, agent, run_tracker=tracker)
        loop.step()
        output_lines = out.getvalue().strip().split("\n")
        assert output_lines[-1] == "PROCEED"
        assert tracker.run_number == 1


def test_game_loop_auto_starts_new_run():
    """When not in game, should send START command."""
    inp = io.StringIO(make_menu_state() + "\n")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    agent = SimpleAgent()
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = RunTracker(log_path=os.path.join(tmpdir, "log.jsonl"))
        loop = GameLoop(comm, agent, run_tracker=tracker)
        loop.step()
        output_lines = out.getvalue().strip().split("\n")
        assert output_lines[-1] == "START IRONCLAD 0"


def test_game_loop_wraps_agent_with_stuck_detector():
    from src.agent import StuckDetectorAgent
    from unittest.mock import MagicMock
    loop = GameLoop(communicator=MagicMock(), agent=SimpleAgent())
    assert isinstance(loop.agent, StuckDetectorAgent)


