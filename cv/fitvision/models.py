from dataclasses import MISSING, asdict, dataclass
from enum import IntEnum
import json

import numpy as np

from .errors import MeasurementError

SCHEMA_VERSION = 2


class Joint(IntEnum):
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


REQUIRED_JOINTS = (
    Joint.LEFT_EYE,
    Joint.RIGHT_EYE,
    Joint.LEFT_SHOULDER,
    Joint.RIGHT_SHOULDER,
    Joint.LEFT_ELBOW,
    Joint.RIGHT_ELBOW,
    Joint.LEFT_WRIST,
    Joint.RIGHT_WRIST,
    Joint.LEFT_HIP,
    Joint.RIGHT_HIP,
    Joint.LEFT_ANKLE,
    Joint.RIGHT_ANKLE,
    Joint.LEFT_HEEL,
    Joint.RIGHT_HEEL,
    Joint.LEFT_FOOT_INDEX,
    Joint.RIGHT_FOOT_INDEX,
)


@dataclass(frozen=True)
class PoseResult:
    landmarks: np.ndarray
    mask: np.ndarray | None
    width: int
    height: int

    def point(self, joint: Joint) -> np.ndarray:
        x, y = self.landmarks[int(joint), :2]
        return np.array([x * self.width, y * self.height], dtype=float)

    def visibility(self, joint: Joint) -> float:
        return float(self.landmarks[int(joint), 2])

    def midpoint(self, first: Joint, second: Joint) -> np.ndarray:
        return (self.point(first) + self.point(second)) / 2.0

    def distance(self, first: Joint, second: Joint) -> float:
        return float(np.linalg.norm(self.point(first) - self.point(second)))

    def min_visibility(self, joints=REQUIRED_JOINTS) -> float:
        return min(self.visibility(joint) for joint in joints)


@dataclass(frozen=True)
class BodyMeasurements:
    height_cm: float
    shoulder_width_cm: float
    chest_circumference_cm: float
    waist_circumference_cm: float
    hip_circumference_cm: float
    arm_length_cm: float
    torso_length_cm: float
    cm_per_pixel: float
    confidence: float
    captured_at: str
    sizing: str = "women"
    clothing: str = "fitted"
    calibrated: bool = False
    warnings: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "BodyMeasurements":
        required = {
            name
            for name, spec in cls.__dataclass_fields__.items()
            if spec.default is MISSING
        }
        missing = required - set(data)
        if missing:
            raise MeasurementError(f"Missing measurement fields: {sorted(missing)}")
        values = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        values["warnings"] = tuple(values.get("warnings", ()))
        return cls(**values)

    @classmethod
    def from_json(cls, text: str) -> "BodyMeasurements":
        return cls.from_dict(json.loads(text))
