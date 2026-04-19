"""Runtime configuration for the machine learning client."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class Settings:
    """Configuration values loaded from environment variables."""

    mongo_uri: str
    mongo_db: str
    mongo_collection: str
    top_k: int
    labels: tuple[str, ...]
    hf_model_id: str
    hf_task: str
    hf_device: int


DEFAULT_LABELS = (
    "airplane",
    "banana",
    "computer",
    "dog",
    "elephant",
    "fish",
    "garden",
    "helmet",
    "ice cream",
    "jail",
    "key",
    "lantern",
    "motorbike",
    "necklace",
    "onion",
    "penguin",
    "raccoon",
    "sandwich",
    "table",
    "underwear",
    "vase",
    "watermelon",
    "yoga",
    "zigzag",
)


def _parse_labels(raw_labels: str | None) -> tuple[str, ...]:
    if not raw_labels:
        return DEFAULT_LABELS

    labels = tuple(label.strip() for label in raw_labels.split(",") if label.strip())
    if not labels:
        raise ValueError("DOODLE_LABELS must contain at least one non-empty label.")
    return labels


def _parse_positive_int(raw_value: str | None, default_value: int, name: str) -> int:
    if raw_value is None:
        return default_value

    parsed = int(raw_value)
    if parsed < 1:
        raise ValueError(f"{name} must be greater than 0.")
    return parsed


def _parse_int(raw_value: str | None, default_value: int) -> int:
    if raw_value is None:
        return default_value
    return int(raw_value)


def load_settings(env_file: str | None = None) -> Settings:
    """Load runtime configuration from environment variables and optional .env file."""

    load_dotenv(dotenv_path=env_file, override=False)
    return Settings(
        mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        mongo_db=os.getenv("MONGO_DB", "doodle_game"),
        mongo_collection=os.getenv("MONGO_COLLECTION", "predictions"),
        top_k=_parse_positive_int(os.getenv("TOP_K"), 2, "TOP_K"),
        labels=_parse_labels(os.getenv("DOODLE_LABELS")),
        hf_model_id=os.getenv(
            "HF_MODEL_ID", "ilyesdjerfaf/vit-base-patch16-224-in21k-quickdraw"
        ),
        hf_task=os.getenv("HF_TASK", "image-classification"),
        hf_device=_parse_int(os.getenv("HF_DEVICE"), -1),
    )
