from datetime import datetime, timezone

import numpy as np

from .calibration import calibrate
from .circumference import ModelLibrary
from .errors import MeasurementError
from .landmarks import PoseExtractor
from .measure import measure_silhouette
from .models import BodyMeasurements, PoseResult
from .request import MeasurementRequest

MIN_VISIBILITY = 0.5
MIN_CONFIDENCE = 0.5
UNEVEN_CONFIDENCE = 0.65
INSTABILITY_WEIGHT = 5.0
LOOSE_WARNING = (
    "Loose clothing hides the body outline, so circumferences may be overestimated. "
    "Fitted clothing gives the most accurate result."
)
UNEVEN_WARNING = "The body outline was uneven. Consider retaking in even light."
PLAUSIBLE_RANGES_CM = {
    "shoulder_width_cm": (25.0, 65.0),
    "chest_circumference_cm": (60.0, 180.0),
    "waist_circumference_cm": (50.0, 170.0),
    "hip_circumference_cm": (65.0, 190.0),
    "arm_length_cm": (40.0, 100.0),
    "torso_length_cm": (30.0, 80.0),
}


class MeasurementPipeline:
    def __init__(self, extractor: PoseExtractor | None, library: ModelLibrary | None = None):
        self._extractor = extractor
        self._library = library or ModelLibrary()

    def measure_image(self, bgr: np.ndarray, request: MeasurementRequest) -> BodyMeasurements:
        pose = self._extractor.extract(bgr)
        if pose is None:
            raise MeasurementError("No person detected in the image")
        return self.measure_pose(pose, request)

    def measure_pose(self, pose: PoseResult, request: MeasurementRequest) -> BodyMeasurements:
        if pose.min_visibility() < MIN_VISIBILITY:
            raise MeasurementError("The full body is not clearly visible")

        model, calibrated = self._library.resolve(request.sizing, request.clothing)
        calibration = calibrate(pose, request.height_cm)
        raw = measure_silhouette(pose, calibration, request.clothing)

        values = {
            "shoulder_width_cm": model.correct("shoulder", raw.shoulder_width_cm),
            "chest_circumference_cm": model.estimate("chest", raw.chest_width_cm),
            "waist_circumference_cm": model.estimate("waist", raw.waist_width_cm),
            "hip_circumference_cm": model.estimate("hip", raw.hip_width_cm),
            "arm_length_cm": raw.arm_length_cm,
            "torso_length_cm": raw.torso_length_cm,
        }
        for name, (low, high) in PLAUSIBLE_RANGES_CM.items():
            if not low <= values[name] <= high:
                raise MeasurementError(f"Implausible {name}: {values[name]:.1f}")

        confidence = max(
            0.0, min(raw.visibility, 1.0 - min(1.0, INSTABILITY_WEIGHT * raw.instability))
        )
        if confidence < MIN_CONFIDENCE:
            raise MeasurementError(
                "The body outline is too uneven to measure reliably. Retake in even light, "
                "against a plain wall"
            )

        warnings = []
        if request.clothing == "loose":
            warnings.append(LOOSE_WARNING)
        if confidence < UNEVEN_CONFIDENCE:
            warnings.append(UNEVEN_WARNING)

        return BodyMeasurements(
            height_cm=round(calibration.height_cm, 1),
            cm_per_pixel=round(calibration.cm_per_pixel, 5),
            confidence=round(confidence, 2),
            captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sizing=request.sizing,
            clothing=request.clothing,
            calibrated=calibrated,
            warnings=tuple(warnings),
            **{name: round(value, 1) for name, value in values.items()},
        )
