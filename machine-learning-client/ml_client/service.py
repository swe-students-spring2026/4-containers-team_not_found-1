"""Application service that runs inference and stores prediction metadata."""

from __future__ import annotations

from typing import Any, Protocol

from .mongodb import MongoPredictionRepository


class SupportsPrediction(Protocol):
    """Protocol for predictor objects used by the inference service."""

    version: str

    def predict(self, raw_image: bytes, top_k: int = 2) -> list[Any]:
        """Predict labels from image bytes."""


class DoodleInferenceService:
    """Coordinates prediction and persistence for doodle inference events."""

    def __init__(
        self,
        predictor: SupportsPrediction,
        repository: MongoPredictionRepository,
    ):
        self._predictor = predictor
        self._repository = repository

    def process_image(
        self,
        raw_image: bytes,
        source: str,
        top_k: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run inference on one image and persist the resulting event."""

        predictions = self._predictor.predict(raw_image, top_k=top_k)
        prediction_payload = [
            {"label": prediction.label, "confidence": prediction.confidence}
            for prediction in predictions
        ]
        event_id = self._repository.save_prediction(
            source=source,
            model_version=self._predictor.version,
            predictions=prediction_payload,
            metadata=metadata,
        )
        return {
            "id": event_id,
            "source": source,
            "model_version": self._predictor.version,
            "predictions": prediction_payload,
        }
