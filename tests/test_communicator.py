# tests/test_communicator.py
import io
import json
from src.communicator import Communicator

SAMPLE_STATE = {
    "available_commands": ["PLAY", "END"],
    "ready_for_command": True,
    "in_game": True,
    "game_state": {
        "screen_type": "NONE",
        "seed": 1,
        "floor": 1,
        "ascension_level": 0,
        "class": "IRONCLAD",
        "current_hp": 80,
        "max_hp": 80,
        "gold": 99,
        "deck": [],
        "relics": [],
        "potions": [],
        "combat_state": {
            "hand": [],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "monsters": [],
            "player": {"current_hp": 80, "max_hp": 80, "block": 0, "energy": 3, "powers": []},
            "turn": 1,
        },
        "map": [],
        "act": 1,
    },
}


def test_send_ready():
    out = io.StringIO()
    comm = Communicator(input_stream=io.StringIO(), output_stream=out)
    comm.send_ready()
    assert out.getvalue() == "ready\n"


def test_send_command():
    out = io.StringIO()
    comm = Communicator(input_stream=io.StringIO(), output_stream=out)
    comm.send_command("PLAY 1 0")
    assert out.getvalue() == "PLAY 1 0\n"


def test_receive_state():
    json_line = json.dumps(SAMPLE_STATE) + "\n"
    inp = io.StringIO(json_line)
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    state = comm.receive_state()
    assert state.in_game is True
    assert state.ready_for_command is True


def test_receive_state_eof():
    inp = io.StringIO("")
    out = io.StringIO()
    comm = Communicator(input_stream=inp, output_stream=out)
    state = comm.receive_state()
    assert state is None
