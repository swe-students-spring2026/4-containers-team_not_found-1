"""Tests for configuration loading."""

from ml_client.config import DEFAULT_LABELS, load_settings


def test_load_settings_defaults(monkeypatch):
    """Default settings are returned when env vars are absent."""

    monkeypatch.setattr("ml_client.config.load_dotenv", lambda **kwargs: False)
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("MONGO_DB", raising=False)
    monkeypatch.delenv("MONGO_COLLECTION", raising=False)
    monkeypatch.delenv("DOODLE_LABELS", raising=False)
    monkeypatch.delenv("TOP_K", raising=False)
    monkeypatch.delenv("HF_MODEL_ID", raising=False)
    monkeypatch.delenv("HF_TASK", raising=False)
    monkeypatch.delenv("HF_DEVICE", raising=False)

    settings = load_settings()

    assert settings.labels == DEFAULT_LABELS
    assert settings.top_k == 2
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.hf_model_id == "ilyesdjerfaf/vit-base-patch16-224-in21k-quickdraw"
    assert settings.hf_task == "image-classification"
    assert settings.hf_device == -1


def test_load_settings_from_env(monkeypatch):
    """Environment variables override defaults."""

    monkeypatch.setattr("ml_client.config.load_dotenv", lambda **kwargs: False)
    monkeypatch.setenv("DOODLE_LABELS", "cat,dog,bird")
    monkeypatch.setenv("TOP_K", "5")
    monkeypatch.setenv("MONGO_DB", "custom_db")
    monkeypatch.setenv("HF_MODEL_ID", "demo/custom-quickdraw-model")
    monkeypatch.setenv("HF_TASK", "image-classification")
    monkeypatch.setenv("HF_DEVICE", "0")

    settings = load_settings()

    assert settings.labels == ("cat", "dog", "bird")
    assert settings.top_k == 5
    assert settings.mongo_db == "custom_db"
    assert settings.hf_model_id == "demo/custom-quickdraw-model"
    assert settings.hf_task == "image-classification"
    assert settings.hf_device == 0
