"""Unit tests for the web-app."""

# pylint: disable=redefined-outer-name,wrong-import-position
import sys

sys.path.append(".")

import pytest
import responses

from main import (
    app,
    get_random_thing,
    things,
    ML_CLIENT_PREDICT_URL,
    ML_CLIENT_HISTORY_URL,
)


@pytest.fixture
def client():
    """Test client for Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_get_random_thing():
    """Test that a random thing is strictly from the list."""
    thing = get_random_thing()
    assert thing in things


def test_index_get(client):
    """Test the main drawing page directly gets a 200."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"Draw" in res.data


def test_index_post_no_image(client):
    """Test post without image fails with 400."""
    res = client.post("/")
    assert res.status_code == 400
    assert b"no image bytes provided" in res.data


@responses.activate
def test_index_post_success(client):
    """Test standard valid prediction request logic."""
    responses.add(
        responses.POST,
        ML_CLIENT_PREDICT_URL,
        json={"predictions": [{"label": "cat", "confidence": 0.9}]},
        status=200,
    )
    res = client.post("/", data=b"fake-image", headers={"Draw-Instruction": "cat"})
    assert res.status_code == 200
    assert b"cat" in res.data
    assert b"90" in res.data  # 0.9 renders as 90.0%


@responses.activate
def test_index_post_connection_error(client):
    """Test handling of unreachable API prediction."""
    # don't add responses => will throw connection refused
    res = client.post("/", data=b"fake-image")
    assert res.status_code == 502
    assert b"unable to reach ml service" in res.data


@responses.activate
def test_history_get(client):
    """Test rendering the history layout list."""
    responses.add(
        responses.GET,
        ML_CLIENT_HISTORY_URL,
        json={
            "records": [
                {
                    "id": "1",
                    "predictions": [{"label": "cat", "confidence": 0.5}],
                    "metadata": {},
                }
            ]
        },
        status=200,
    )
    res = client.get("/history")
    assert res.status_code == 200
    assert b"cat" in res.data


@responses.activate
def test_history_delete(client):
    """Test successfully deleting history triggers redirect."""
    responses.add(
        responses.DELETE,
        f"{ML_CLIENT_HISTORY_URL}/event-123",
        status=200,
    )
    res = client.post("/history/event-123/delete")
    assert res.status_code == 302
    assert res.headers["Location"] == "/history"
