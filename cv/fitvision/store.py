import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .models import BodyMeasurements


@dataclass(frozen=True)
class SavedCapture:
    directory: Path
    json_path: Path
    image_path: Path | None
    measurements: BodyMeasurements


def slugify(label: str | None) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (label or "").lower()).strip("-")
    return cleaned or "capture"


class CaptureStore:
    def __init__(self, root: str | Path = "captures"):
        self._root = Path(root)

    def _new_directory(self, label: str | None) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(label)}"
        for attempt in range(1, 1000):
            name = stem if attempt == 1 else f"{stem}_{attempt}"
            try:
                (self._root / name).mkdir()
            except FileExistsError:
                continue
            return self._root / name
        raise FileExistsError(f"Could not create a unique folder for {stem}")

    def save(
        self,
        measurements: BodyMeasurements,
        image: np.ndarray | None = None,
        label: str | None = None,
    ) -> SavedCapture:
        directory = self._new_directory(label)
        json_path = directory / "measurements.json"
        json_path.write_text(measurements.to_json())
        image_path = None
        if image is not None:
            image_path = directory / "photo.jpg"
            cv2.imwrite(str(image_path), image)
        return SavedCapture(directory, json_path, image_path, measurements)
