# tests/test_game_loop.py
import io
import json
from src.game_loop import GameLoop
from src.agent import SimpleAgent
from src.communicator import Communicator


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
