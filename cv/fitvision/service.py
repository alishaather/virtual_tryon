import numpy as np

from .circumference import ModelLibrary
from .landmarks import PoseExtractor
from .models import BodyMeasurements
from .pipeline import MeasurementPipeline
from .request import MeasurementRequest


class FitVisionService:
    def __init__(self, library: ModelLibrary | None = None, model_complexity: int = 2):
        self._extractor = PoseExtractor(static_image=True, model_complexity=model_complexity)
        self._pipeline = MeasurementPipeline(self._extractor, library)

    def measure(self, image: np.ndarray, request: MeasurementRequest) -> BodyMeasurements:
        return self._pipeline.measure_image(image, request)

    def close(self) -> None:
        self._extractor.close()

    def __enter__(self) -> "FitVisionService":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
