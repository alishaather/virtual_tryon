from dataclasses import dataclass

from .errors import CalibrationError
from .models import Joint, PoseResult

EYE_HEIGHT_RATIO = 0.936
MIN_HEIGHT_CM = 120.0
MAX_HEIGHT_CM = 220.0


@dataclass(frozen=True)
class Calibration:
    height_cm: float
    stature_px: float

    @property
    def cm_per_pixel(self) -> float:
        return self.height_cm / self.stature_px

    def to_cm(self, pixels: float) -> float:
        return pixels * self.cm_per_pixel


def stature_pixels(pose: PoseResult) -> float:
    eyes = pose.midpoint(Joint.LEFT_EYE, Joint.RIGHT_EYE)
    heels = pose.midpoint(Joint.LEFT_HEEL, Joint.RIGHT_HEEL)
    return float(heels[1] - eyes[1]) / EYE_HEIGHT_RATIO


def calibrate(pose: PoseResult, height_cm: float) -> Calibration:
    if not MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM:
        raise CalibrationError(
            f"Height must be between {MIN_HEIGHT_CM:.0f} and {MAX_HEIGHT_CM:.0f} cm"
        )
    stature_px = stature_pixels(pose)
    if stature_px <= 0:
        raise CalibrationError("Could not find a standing body in the frame")
    return Calibration(height_cm=float(height_cm), stature_px=stature_px)
