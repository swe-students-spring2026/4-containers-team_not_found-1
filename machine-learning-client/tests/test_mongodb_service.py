"""Tests for Mongo repository and inference service integration."""

import mongomock

from ml_client.mongodb import MongoPredictionRepository
from ml_client.service import DoodleInferenceService


class StubPredictor:
    """Predictor test double with deterministic output."""

    version = "test-model-v1"

    def predict(self, raw_image, top_k):
        del raw_image
        del top_k
        return [
            type("Prediction", (), {"label": "cat", "confidence": 0.77})(),
            type("Prediction", (), {"label": "dog", "confidence": 0.20})(),
        ]


def test_save_prediction_and_fetch_recent():
    """Repository persists and retrieves prediction records."""

    repo = MongoPredictionRepository(
        mongo_uri="mongodb://unused",
        database_name="db",
        collection_name="predictions",
        client=mongomock.MongoClient(),
    )

    event_id = repo.save_prediction(
        source="unit-test",
        model_version="v1",
        predictions=[{"label": "cat", "confidence": 0.8}],
        metadata={"origin": "test"},
    )

    records = repo.fetch_recent(limit=5)

    assert event_id
    assert len(records) == 1
    assert records[0]["source"] == "unit-test"


def test_service_process_image_writes_to_repository():
    """Service executes prediction and persists output."""

    repo = MongoPredictionRepository(
        mongo_uri="mongodb://unused",
        database_name="db",
        collection_name="predictions",
        client=mongomock.MongoClient(),
    )
    service = DoodleInferenceService(predictor=StubPredictor(), repository=repo)

    result = service.process_image(
        raw_image=b"not-an-image-needed",
        source="sensor-A",
        top_k=2,
        metadata={"frame": 1},
    )

    assert result["id"]
    assert result["source"] == "sensor-A"
    assert result["predictions"][0]["label"] == "cat"
    assert repo.fetch_recent(limit=1)[0]["metadata"]["frame"] == 1
