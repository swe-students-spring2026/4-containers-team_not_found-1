"""Pretrained model inference helpers for doodle recognition."""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class Prediction:
    """Single class prediction with confidence score."""

    label: str
    confidence: float


class PretrainedDoodlePredictor:
    """Runs doodle inference with a pretrained Hugging Face model."""

    def __init__(
        self,
        classifier: Any,
        labels: tuple[str, ...],
        model_id: str,
        task: str,
    ):
        self._classifier = classifier
        self._labels = labels
        self._task = task
        self._version = model_id

    @property
    def version(self) -> str:
        """Return a model identifier that can be stored as version metadata."""

        return self._version

    @classmethod
    def from_huggingface(
        cls,
        model_id: str,
        labels: tuple[str, ...],
        task: str = "image-classification",
        device: int = -1,
    ) -> "PretrainedDoodlePredictor":
        """Create a predictor from a Hugging Face model ID."""

        try:
            # pylint: disable=import-outside-toplevel
            from transformers import pipeline
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "transformers is required for pretrained inference. "
                "Install dependencies with 'pipenv install --dev'."
            ) from error

        classifier = pipeline(task=task, model=model_id, device=device)
        return cls(
            classifier=classifier,
            labels=labels,
            model_id=model_id,
            task=task,
        )

    def predict(self, raw_image: bytes, top_k: int = 2) -> list[Prediction]:
        """Predict the most likely doodle classes from an input image."""

        if top_k < 1:
            raise ValueError("top_k must be greater than 0.")

        with Image.open(io.BytesIO(raw_image)) as opened:
            if opened.mode in ("RGBA", "LA") or (
                opened.mode == "P" and "transparency" in opened.info
            ):
                background = Image.new("RGB", opened.size, (255, 255, 255))
                background.paste(opened, mask=opened.convert("RGBA"))
                image = background
            else:
                image = opened.convert("RGB")

        if self._task == "zero-shot-image-classification":
            results = self._classifier(
                image,
                candidate_labels=list(self._labels),
            )
            parsed = _parse_zero_shot_results(results)
        elif self._task == "image-classification":
            results = self._classifier(image, top_k=top_k)
            parsed = _parse_image_classification_results(results)
        else:
            raise ValueError(f"Unsupported pretrained task '{self._task}'.")

        ordered = sorted(parsed, key=lambda item: item.confidence, reverse=True)
        return ordered[:top_k]


def _parse_zero_shot_results(results: Any) -> list[Prediction]:
    if isinstance(results, dict) and "labels" in results and "scores" in results:
        labels = results["labels"]
        scores = results["scores"]
        return [
            Prediction(label=str(label), confidence=float(score))
            for label, score in zip(labels, scores)
        ]

    if isinstance(results, list):
        parsed: list[Prediction] = []
        for item in results:
            if isinstance(item, dict) and "label" in item and "score" in item:
                parsed.append(
                    Prediction(
                        label=str(item["label"]), confidence=float(item["score"])
                    )
                )
        if parsed:
            return parsed

    raise ValueError("Unexpected zero-shot prediction format.")


def _parse_image_classification_results(results: Any) -> list[Prediction]:
    if not isinstance(results, list):
        raise ValueError("Unexpected image-classification prediction format.")

    parsed: list[Prediction] = []
    for item in results:
        if isinstance(item, dict) and "label" in item and "score" in item:
            parsed.append(
                Prediction(label=str(item["label"]), confidence=float(item["score"]))
            )

    if not parsed:
        raise ValueError("No predictions returned by pretrained classifier.")
    return parsed
