import re
from dataclasses import dataclass

from .calibration import MAX_HEIGHT_CM, MIN_HEIGHT_CM
from .errors import RequestError

SIZINGS = ("women", "men")
CLOTHINGS = ("fitted", "loose")
CM_PER_INCH = 2.54
FEET_AND_INCHES = re.compile(r"^(\d)\s*'\s*(\d{0,2})\s*\"?$")


def parse_height(text: str) -> float:
    cleaned = text.strip().lower().replace("cm", "").strip()
    if not cleaned:
        raise ValueError("Enter your height")
    match = FEET_AND_INCHES.match(cleaned)
    if match:
        feet, inches = int(match.group(1)), int(match.group(2) or 0)
        if inches >= 12:
            raise ValueError("Inches must be below 12")
        height = (feet * 12 + inches) * CM_PER_INCH
    elif cleaned.replace(".", "", 1).isdigit():
        height = float(cleaned)
    else:
        raise ValueError("Use centimetres (165) or feet and inches (5'4)")
    if not MIN_HEIGHT_CM <= height <= MAX_HEIGHT_CM:
        raise ValueError(f"Height must be {MIN_HEIGHT_CM:.0f} to {MAX_HEIGHT_CM:.0f} cm")
    return round(height, 1)


def cm_to_feet_inches(height_cm: float) -> tuple[int, int]:
    total = round(height_cm / CM_PER_INCH)
    return total // 12, total % 12


@dataclass(frozen=True)
class MeasurementRequest:
    height_cm: float
    sizing: str = "women"
    clothing: str = "fitted"
    label: str | None = None

    def __post_init__(self) -> None:
        if self.sizing not in SIZINGS:
            raise RequestError(f"sizing must be one of {SIZINGS}")
        if self.clothing not in CLOTHINGS:
            raise RequestError(f"clothing must be one of {CLOTHINGS}")
        if not MIN_HEIGHT_CM <= self.height_cm <= MAX_HEIGHT_CM:
            raise RequestError(f"height_cm must be {MIN_HEIGHT_CM:.0f} to {MAX_HEIGHT_CM:.0f}")

    @property
    def model_key(self) -> str:
        return f"{self.sizing}:{self.clothing}"
