"""CLI entry points for pretrained doodle inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import Settings, load_settings
from .pretrained import PretrainedDoodlePredictor
from .repository import MongoPredictionRepository
from .service import DoodleInferenceService


def _parse_csv_labels(raw_labels: str) -> tuple[str, ...]:
    labels = tuple(label.strip() for label in raw_labels.split(",") if label.strip())
    if not labels:
        raise ValueError("At least one label must be provided.")
    return labels


def _resolve_top_k(candidate: int | None, default_value: int) -> int:
    top_k = default_value if candidate is None else candidate
    if top_k < 1:
        raise ValueError("top_k must be greater than 0.")
    return top_k


def _load_inference_predictor(
    args: argparse.Namespace,
) -> tuple[PretrainedDoodlePredictor, Settings]:
    settings = load_settings(args.env_file)
    labels = _parse_csv_labels(args.labels) if args.labels else settings.labels
    predictor = PretrainedDoodlePredictor.from_huggingface(
        model_id=args.hf_model_id or settings.hf_model_id,
        labels=labels,
        task=args.hf_task or settings.hf_task,
        device=settings.hf_device if args.hf_device is None else args.hf_device,
    )
    return predictor, settings


def _load_image_file(image_path: str) -> bytes:
    return Path(image_path).read_bytes()


def _handle_predict(args: argparse.Namespace) -> None:
    predictor, settings = _load_inference_predictor(args)
    image_bytes = _load_image_file(args.image_path)
    predictions = predictor.predict(
        image_bytes,
        top_k=_resolve_top_k(args.top_k, settings.top_k),
    )

    payload = {
        "model_version": predictor.version,
        "predictions": [
            {"label": prediction.label, "confidence": prediction.confidence}
            for prediction in predictions
        ],
    }
    print(json.dumps(payload, indent=2))


def _handle_run_once(args: argparse.Namespace) -> None:
    predictor, settings = _load_inference_predictor(args)
    repository = MongoPredictionRepository(
        mongo_uri=settings.mongo_uri,
        database_name=settings.mongo_db,
        collection_name=settings.mongo_collection,
    )
    service = DoodleInferenceService(predictor=predictor, repository=repository)

    image_bytes = _load_image_file(args.image_path)
    event = service.process_image(
        raw_image=image_bytes,
        source=args.source,
        top_k=_resolve_top_k(args.top_k, settings.top_k),
        metadata={"image_path": str(Path(args.image_path).name)},
    )
    print(json.dumps(event, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Doodle ML client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict doodle classes from a local image using a pretrained model.",
    )
    predict_parser.add_argument(
        "--env-file",
        default=None,
        help="Optional .env file for inference settings.",
    )
    predict_parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated candidate labels used in zero-shot mode.",
    )
    predict_parser.add_argument(
        "--hf-model-id",
        default=None,
        help="Hugging Face model ID for pretrained backend.",
    )
    predict_parser.add_argument(
        "--hf-task",
        choices=("zero-shot-image-classification", "image-classification"),
        default=None,
        help="Pretrained pipeline task. Defaults from settings.",
    )
    predict_parser.add_argument(
        "--hf-device",
        type=int,
        default=None,
        help="Device index for Hugging Face pipeline (-1 for CPU).",
    )
    predict_parser.add_argument("--image-path", required=True, help="Input image path.")
    predict_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top predictions to print. Defaults from settings.",
    )

    run_once_parser = subparsers.add_parser(
        "run-once",
        help="Run one pretrained inference, then persist metadata to MongoDB.",
    )
    run_once_parser.add_argument("--env-file", default=None, help="Optional .env file.")
    run_once_parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated candidate labels used in zero-shot mode.",
    )
    run_once_parser.add_argument(
        "--hf-model-id",
        default=None,
        help="Hugging Face model ID for pretrained backend.",
    )
    run_once_parser.add_argument(
        "--hf-task",
        choices=("zero-shot-image-classification", "image-classification"),
        default=None,
        help="Pretrained pipeline task. Defaults from settings.",
    )
    run_once_parser.add_argument(
        "--hf-device",
        type=int,
        default=None,
        help="Device index for Hugging Face pipeline (-1 for CPU).",
    )
    run_once_parser.add_argument("--image-path", required=True, help="Input image path.")
    run_once_parser.add_argument(
        "--source",
        default="file-sensor",
        help="Source identifier stored with prediction metadata.",
    )
    run_once_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k override; defaults to settings value.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "predict": _handle_predict,
        "run-once": _handle_run_once,
    }

    try:
        handlers[args.command](args)
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
