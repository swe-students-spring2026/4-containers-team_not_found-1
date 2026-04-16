"""Tests for pretrained predictor behavior."""

from io import BytesIO

from PIL import Image
import pytest

from ml_client.pretrained import PretrainedDoodlePredictor


def _image_bytes() -> bytes:
    image = Image.new("RGB", (24, 24), color=(255, 255, 255))
    image.putpixel((12, 12), (0, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_zero_shot_predictions_sorted_and_trimmed():
    """Zero-shot result parsing should be confidence-sorted and top-k trimmed."""

    def classifier(image, candidate_labels):
        del image
        del candidate_labels
        return [
            {"label": "cat", "score": 0.4},
            {"label": "house", "score": 0.7},
            {"label": "tree", "score": 0.5},
        ]

    predictor = PretrainedDoodlePredictor(
        classifier=classifier,
        labels=("cat", "house", "tree"),
        model_id="demo/model",
        task="zero-shot-image-classification",
    )

    predictions = predictor.predict(_image_bytes(), top_k=2)

    assert [item.label for item in predictions] == ["house", "tree"]


def test_image_classification_predictions_work():
    """Image-classification result parsing returns expected labels."""

    def classifier(image, top_k):
        del image
        del top_k
        return [
            {"label": "car", "score": 0.8},
            {"label": "bicycle", "score": 0.2},
        ]

    predictor = PretrainedDoodlePredictor(
        classifier=classifier,
        labels=(),
        model_id="demo/model",
        task="image-classification",
    )

    predictions = predictor.predict(_image_bytes(), top_k=1)

    assert len(predictions) == 1
    assert predictions[0].label == "car"


def test_unsupported_task_raises_value_error():
    """Unsupported task names should fail fast."""

    predictor = PretrainedDoodlePredictor(
        classifier=lambda image: [],
        labels=("cat",),
        model_id="demo/model",
        task="unsupported-task",
    )

    with pytest.raises(ValueError):
        predictor.predict(_image_bytes(), top_k=1)
