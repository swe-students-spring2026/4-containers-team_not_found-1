"""HTTP API server for pretrained doodle inference."""

from __future__ import annotations

import base64
import os
from typing import Any

from flask import Flask, jsonify, request

from .config import Settings, load_settings
from .pretrained import PretrainedDoodlePredictor
from .mongodb import MongoPredictionRepository
from .service import DoodleInferenceService


def _parse_top_k(raw_value: str | None, default_value: int) -> int:
    if raw_value is None or raw_value == "":
        return default_value

    parsed = int(raw_value)
    if parsed < 1:
        raise ValueError("top_k must be greater than 0.")
    return parsed


def create_app(
    settings: Settings | None = None,
    predictor: PretrainedDoodlePredictor | Any | None = None,
    repository: MongoPredictionRepository | Any | None = None,
) -> Flask:
    """Create and configure the Flask inference API."""

    runtime_settings = settings or load_settings()
    runtime_predictor = predictor or PretrainedDoodlePredictor.from_huggingface(
        model_id=runtime_settings.hf_model_id,
        labels=runtime_settings.labels,
        task=runtime_settings.hf_task,
        device=runtime_settings.hf_device,
    )
    runtime_repository = repository or MongoPredictionRepository(
        mongo_uri=runtime_settings.mongo_uri,
        database_name=runtime_settings.mongo_db,
        collection_name=runtime_settings.mongo_collection,
    )
    service = DoodleInferenceService(
        predictor=runtime_predictor,
        repository=runtime_repository,
    )

    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "status": "ok",
                    "model_version": runtime_predictor.version,
                }
            ),
            200,
        )

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        try:
            image_bytes = request.get_data()
            if not image_bytes:
                return jsonify({"error": "Request body must contain image bytes."}), 400

            top_k = _parse_top_k(request.args.get("top_k"), runtime_settings.top_k)
            source = request.args.get("source", "web-app")
            instruction = request.args.get("instruction", "unknown")
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            event = service.process_image(
                raw_image=image_bytes,
                source=source,
                top_k=top_k,
                metadata={
                    "content_type": request.content_type or "unknown",
                    "image_b64": image_b64,
                    "instruction": instruction,
                },
            )
            return jsonify(event), 200
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Exception:  # pylint: disable=broad-exception-caught
            return jsonify({"error": "Prediction failed."}), 500

    @app.get("/history")
    def history() -> tuple[Any, int]:
        try:
            records = runtime_repository.fetch_recent(limit=20)
            return jsonify({"records": records}), 200
        except Exception:  # pylint: disable=broad-exception-caught
            return jsonify({"error": "Failed to fetch history."}), 500

    @app.delete("/history/<record_id>")
    def delete_history(record_id: str) -> tuple[Any, int]:
        try:
            success = runtime_repository.delete_prediction(record_id)
            if success:
                return jsonify({"status": "deleted"}), 200
            return jsonify({"error": "Record not found"}), 404
        except Exception:  # pylint: disable=broad-exception-caught
            return jsonify({"error": "Failed to delete logic."}), 500

    return app


def main() -> int:
    app = create_app()
    host = os.getenv("ML_CLIENT_HOST", "0.0.0.0")
    port = int(os.getenv("ML_CLIENT_PORT", "8000"))
    debug = os.getenv("ML_CLIENT_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
