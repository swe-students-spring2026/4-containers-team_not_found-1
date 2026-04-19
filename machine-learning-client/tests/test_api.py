"""Tests for ML client HTTP API endpoints."""

from ml_client.api import create_app
from ml_client.config import Settings


class StubPredictor:
    """Predictor test double for API testing."""

    version = "stub-model-v1"

    def predict(self, raw_image: bytes, top_k: int = 3):
        del raw_image
        base_predictions = [
            type("Prediction", (), {"label": "cat", "confidence": 0.72})(),
            type("Prediction", (), {"label": "dog", "confidence": 0.21})(),
            type("Prediction", (), {"label": "car", "confidence": 0.07})(),
        ]
        return base_predictions[:top_k]


class StubRepository:
    """Repository test double for API testing."""

    def __init__(self):
        self.saved = []

    def save_prediction(self, source, model_version, predictions, metadata=None):
        self.saved.append(
            {
                "source": source,
                "model_version": model_version,
                "predictions": predictions,
                "metadata": metadata,
            }
        )
        return "event-123"

    def fetch_recent(self, limit: int = 10):
        return self.saved[:limit]

    def delete_prediction(self, record_id: str) -> bool:
        return record_id == "event-123"


def _settings() -> Settings:
    return Settings(
        mongo_uri="mongodb://unused",
        mongo_db="unused",
        mongo_collection="unused",
        top_k=3,
        labels=("cat", "dog", "car"),
        hf_model_id="stub/model",
        hf_task="image-classification",
        hf_device=-1,
    )


def test_health_endpoint_returns_ok():
    """Health endpoint should return healthy status and model version."""

    app = create_app(
        settings=_settings(),
        predictor=StubPredictor(),
        repository=StubRepository(),
    )
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model_version"] == "stub-model-v1"


def test_predict_endpoint_returns_predictions_and_event_id():
    """Predict endpoint returns persisted event payload for valid image bytes."""

    app = create_app(
        settings=_settings(),
        predictor=StubPredictor(),
        repository=StubRepository(),
    )
    client = app.test_client()

    response = client.post(
        "/predict?top_k=2&source=web-canvas",
        data=b"fake-image-bytes",
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["id"] == "event-123"
    assert payload["source"] == "web-canvas"
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][0]["label"] == "cat"


def test_predict_endpoint_rejects_empty_body():
    """Predict endpoint should reject requests that have no image bytes."""

    app = create_app(
        settings=_settings(),
        predictor=StubPredictor(),
        repository=StubRepository(),
    )
    client = app.test_client()

    response = client.post("/predict", data=b"")

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_endpoint_validates_top_k():
    """Predict endpoint should validate top_k query parameter."""

    app = create_app(
        settings=_settings(),
        predictor=StubPredictor(),
        repository=StubRepository(),
    )
    client = app.test_client()

    response = client.post("/predict?top_k=0", data=b"image")

    assert response.status_code == 400
    assert "top_k" in response.get_json()["error"]


def test_history_endpoint_returns_records():
    app = create_app(
        settings=_settings(),
        predictor=StubPredictor(),
        repository=StubRepository(),
    )
    client = app.test_client()
    response = client.get("/history")
    assert response.status_code == 200
    assert "records" in response.get_json()


def test_delete_history_endpoint_deletes():
    app = create_app(
        settings=_settings(),
        predictor=StubPredictor(),
        repository=StubRepository(),
    )
    client = app.test_client()
    response = client.delete("/history/event-123")
    assert response.status_code == 200
    assert response.get_json()["status"] == "deleted"


def test_delete_history_endpoint_not_found():
    app = create_app(
        settings=_settings(),
        predictor=StubPredictor(),
        repository=StubRepository(),
    )
    client = app.test_client()
    response = client.delete("/history/unknown")
    assert response.status_code == 404
