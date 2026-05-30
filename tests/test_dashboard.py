import json
import os
import tempfile
import pytest


@pytest.fixture
def client(tmp_path):
    live_state_path = tmp_path / "live_state.json"
    import dashboard
    dashboard.LIVE_STATE_PATH = str(live_state_path)
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c, live_state_path


def test_api_state_returns_empty_when_no_file(client):
    c, _ = client
    resp = c.get("/api/state")
    assert resp.status_code == 200
    assert resp.get_json() == {}


def test_api_state_returns_file_contents(client):
    c, live_state_path = client
    data = {"live": {"screen_type": "COMBAT", "current_hp": 47, "max_hp": 80,
                     "last_action": "END", "monsters": [], "updated_at": "2026-01-01T00:00:00Z"},
            "stats": {"run_number": 5, "wins": 2, "losses": 3,
                      "win_rate": 0.4, "avg_floor": 12.0}}
    live_state_path.write_text(json.dumps(data))
    resp = c.get("/api/state")
    assert resp.status_code == 200
    assert resp.get_json()["live"]["current_hp"] == 47
    assert resp.get_json()["stats"]["run_number"] == 5


def test_live_route_returns_html(client):
    c, _ = client
    resp = c.get("/live")
    assert resp.status_code == 200
    assert b"text/html" in resp.content_type.encode()
    assert b"screen-type" in resp.data
    assert b"hp-bar" in resp.data
    assert b"last-action" in resp.data


def test_stats_route_returns_html(client):
    c, _ = client
    resp = c.get("/stats")
    assert resp.status_code == 200
    assert b"text/html" in resp.content_type.encode()
    assert b"run-number" in resp.data
    assert b"win-rate" in resp.data
