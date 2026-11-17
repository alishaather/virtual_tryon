import math
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Calibration
from .errors import MeasurementError
from .models import Joint, PoseResult
from .silhouette import band_width, find_armpit_row

CHEST_FRACTION = 0.32
WAIST_WINDOW = (0.50, 0.85)
HIP_FRACTION = 1.00
HIP_SEARCH_DEPTH = 0.045
ARMPIT_OFFSET = 0.03
ARMPIT_SEARCH_TOP = 0.12
BAND_FRACTION = 0.004
WAIST_STEP = 2
MAX_CHEST_TO_FRAME = 1.35
SHOULDER_EDGE_ALLOWANCE_CM = 5.0
MIN_ARM_ANGLE_DEGREES = 8.0
ARM_BAND_FRACTION = {"fitted": 0.030, "loose": 0.042}

ARM_CHAINS = (
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
)


@dataclass(frozen=True)
class RawMeasurements:
    shoulder_width_cm: float
    chest_width_cm: float
    waist_width_cm: float
    hip_width_cm: float
    arm_length_cm: float
    torso_length_cm: float
    visibility: float
    instability: float


def arm_angle_degrees(pose: PoseResult, shoulder: Joint, wrist: Joint) -> float:
    delta = pose.point(wrist) - pose.point(shoulder)
    return abs(math.degrees(math.atan2(delta[0], abs(delta[1]) + 1e-9)))


def arm_length_px(pose: PoseResult) -> float:
    lengths = [
        pose.distance(shoulder, elbow) + pose.distance(elbow, wrist)
        for shoulder, elbow, wrist in ARM_CHAINS
    ]
    return sum(lengths) / len(lengths)


def require_arm_clearance(pose: PoseResult) -> None:
    for shoulder, _, wrist in ARM_CHAINS:
        if arm_angle_degrees(pose, shoulder, wrist) < MIN_ARM_ANGLE_DEGREES:
            raise MeasurementError(
                "Arms are too close to the body. Retake with your arms held "
                "about 25 to 30 degrees away from your sides"
            )


def remove_arms(pose: PoseResult, half_band_px: float) -> np.ndarray:
    canvas = pose.mask.astype(np.uint8)
    thickness = max(1, int(round(2 * half_band_px)))
    for chain in ARM_CHAINS:
        points = [tuple(int(round(v)) for v in pose.point(joint)) for joint in chain]
        cv2.line(canvas, points[0], points[1], 0, thickness)
        cv2.line(canvas, points[1], points[2], 0, thickness)
    return canvas.astype(bool)


def measure_silhouette(
    pose: PoseResult, calibration: Calibration, clothing: str = "fitted"
) -> RawMeasurements:
    if pose.mask is None:
        raise MeasurementError("Body silhouette is unavailable")
    require_arm_clearance(pose)

    shoulders = pose.midpoint(Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
    hips = pose.midpoint(Joint.LEFT_HIP, Joint.RIGHT_HIP)
    torso = float(hips[1] - shoulders[1])
    if torso <= 0:
        raise MeasurementError("Torso is not upright in the frame")

    stature = calibration.stature_px
    center = int(round((shoulders[0] + hips[0]) / 2.0))
    half_band = max(2, round(BAND_FRACTION * stature))
    mask = remove_arms(pose, ARM_BAND_FRACTION[clothing] * stature)

    def row_at(fraction: float) -> int:
        return int(round(shoulders[1] + fraction * torso))

    def sample(row: int) -> tuple[float, float] | None:
        return band_width(mask, row, center, half_band)

    hip_start = row_at(HIP_FRACTION)
    hip_samples = [
        s
        for row in range(hip_start, hip_start + int(HIP_SEARCH_DEPTH * stature) + 1, WAIST_STEP)
        if (s := sample(row)) is not None
    ]
    waist_samples = [
        s
        for row in range(row_at(WAIST_WINDOW[0]), row_at(WAIST_WINDOW[1]) + 1, WAIST_STEP)
        if (s := sample(row)) is not None
    ]

    armpit = find_armpit_row(
        mask, row_at(ARMPIT_SEARCH_TOP), row_at(WAIST_WINDOW[0]), center
    )
    chest_row = row_at(CHEST_FRACTION) if armpit is None else armpit + int(ARMPIT_OFFSET * stature)
    chest_row = min(max(chest_row, row_at(ARMPIT_SEARCH_TOP)), row_at(WAIST_WINDOW[0]))
    chest_sample = sample(chest_row)

    if not hip_samples or not waist_samples or chest_sample is None:
        raise MeasurementError("Could not trace the torso outline")

    hip = max(hip_samples, key=lambda s: s[0])
    waist = min(waist_samples, key=lambda s: s[0])

    shoulder_px = pose.distance(Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
    if chest_sample[0] > MAX_CHEST_TO_FRAME * max(shoulder_px, hip[0]):
        raise MeasurementError(
            "Arms or sleeves are merging with the torso outline. Retake with your arms "
            "held further from your body"
        )

    return RawMeasurements(
        shoulder_width_cm=calibration.to_cm(shoulder_px) + SHOULDER_EDGE_ALLOWANCE_CM,
        chest_width_cm=calibration.to_cm(chest_sample[0]),
        waist_width_cm=calibration.to_cm(waist[0]),
        hip_width_cm=calibration.to_cm(hip[0]),
        arm_length_cm=calibration.to_cm(arm_length_px(pose)),
        torso_length_cm=calibration.to_cm(torso),
        visibility=pose.min_visibility(),
        instability=max(chest_sample[1], waist[1], hip[1]),
    )
