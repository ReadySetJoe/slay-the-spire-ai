import json
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
    data = resp.get_json()
    assert data["_exists"] is False
    assert "stats" not in data


def test_api_state_returns_file_contents(client):
    c, live_state_path = client
    data = {"stats": {"run_number": 5, "wins": 2, "losses": 3,
                      "win_rate": 0.4, "avg_floor": 12.0}}
    live_state_path.write_text(json.dumps(data))
    resp = c.get("/api/state")
    assert resp.status_code == 200
    assert resp.get_json()["stats"]["run_number"] == 5


def test_stats_route_returns_html(client):
    c, _ = client
    resp = c.get("/stats")
    assert resp.status_code == 200
    assert b"text/html" in resp.content_type.encode()
    assert b"run-number" in resp.data
    assert b"win-rate" in resp.data


def test_training_route_returns_html(client):
    c, _ = client
    resp = c.get("/training")
    assert resp.status_code == 200
    assert b"text/html" in resp.content_type.encode()
    assert b"FLOOR PROGRESS" in resp.data
    assert b"chart.js" in resp.data.lower()
