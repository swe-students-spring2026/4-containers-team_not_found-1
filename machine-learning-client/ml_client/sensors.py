"""Sensor adapters for obtaining raw image data."""

from pathlib import Path


class FileSensor:
    """Simple file-based sensor used for local testing and automation."""

    def __init__(self, image_path: str | Path):
        self._image_path = Path(image_path)

    def capture(self) -> bytes:
        """Capture the latest image bytes from disk."""

        return self._image_path.read_bytes()
