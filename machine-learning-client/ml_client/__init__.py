"""Machine learning client package for doodle recognition."""

from .config import Settings, load_settings
from .pretrained import Prediction, PretrainedDoodlePredictor
from .service import DoodleInferenceService


def create_app(*args, **kwargs):
    """Create ML API app without eagerly importing the api module."""

    # pylint: disable=import-outside-toplevel
    from .api import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = [
    "Settings",
    "load_settings",
    "create_app",
    "Prediction",
    "PretrainedDoodlePredictor",
    "DoodleInferenceService",
]
