import math
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import stature_pixels
from .models import REQUIRED_JOINTS, Joint, PoseResult

ANALYSIS_WIDTH = 640


@dataclass(frozen=True)
class GuideConfig:
    min_visibility: float = 0.5
    frame_margin: float = 0.02
    min_body_fraction: float = 0.65
    max_body_fraction: float = 0.95
    max_center_offset: float = 0.12
    min_shoulder_to_torso: float = 0.60
    max_tilt_degrees: float = 6.0
    min_arm_angle: float = 15.0
    max_arm_angle: float = 45.0
    min_brightness: float = 60.0
    max_brightness: float = 200.0
    min_sharpness: float = 25.0
    max_motion: float = 0.01
    hold_seconds: float = 2.0


FITTED_CONFIG = GuideConfig()
LOOSE_CONFIG = GuideConfig(min_arm_angle=25.0, max_arm_angle=50.0)


def guide_config_for(clothing: str) -> GuideConfig:
    return LOOSE_CONFIG if clothing == "loose" else FITTED_CONFIG


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class GuideState:
    checks: tuple[Check, ...]
    ready: bool
    hold_progress: float
    seconds_left: float
    shutter: bool

    @property
    def instruction(self) -> str:
        for check in self.checks:
            if not check.passed:
                return check.message
        return "Hold still"


def tilt_degrees(first: np.ndarray, second: np.ndarray, axis: str) -> float:
    delta = second - first
    if axis == "horizontal":
        return abs(math.degrees(math.atan2(delta[1], abs(delta[0]) + 1e-9)))
    return abs(math.degrees(math.atan2(delta[0], abs(delta[1]) + 1e-9)))


class CaptureGuide:
    def __init__(self, config: GuideConfig | None = None):
        self.config = config or GuideConfig()
        self._hold_start: float | None = None
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._hold_start = None
        self._previous = None

    def update(self, frame: np.ndarray, pose: PoseResult | None, now: float) -> GuideState:
        if pose is None:
            self.reset()
            return GuideState(
                (Check("person", False, "Step into the frame so your whole body is visible"),),
                False,
                0.0,
                self.config.hold_seconds,
                False,
            )

        checks = self.evaluate(frame, pose) + (self._stillness(pose),)
        self._previous = pose.landmarks[:, :2].copy()

        if not all(check.passed for check in checks):
            self._hold_start = None
            return GuideState(checks, False, 0.0, self.config.hold_seconds, False)

        if self._hold_start is None:
            self._hold_start = now
        elapsed = now - self._hold_start
        shutter = elapsed >= self.config.hold_seconds
        state = GuideState(
            checks,
            True,
            min(1.0, elapsed / self.config.hold_seconds),
            max(0.0, self.config.hold_seconds - elapsed),
            shutter,
        )
        if shutter:
            self.reset()
        return state

    def evaluate(self, frame: np.ndarray, pose: PoseResult) -> tuple[Check, ...]:
        return (
            self._distance(pose),
            self._centered(pose),
            self._visibility(pose),
            self._facing(pose),
            self._upright(pose),
            self._arms(pose),
            self._image_quality(frame),
        )

    def _visibility(self, pose: PoseResult) -> Check:
        config = self.config
        low = config.frame_margin
        high = 1.0 - config.frame_margin
        stature = stature_pixels(pose)
        head_top = pose.midpoint(Joint.LEFT_HEEL, Joint.RIGHT_HEEL)[1] - stature
        feet = max(
            pose.point(Joint.LEFT_FOOT_INDEX)[1], pose.point(Joint.RIGHT_FOOT_INDEX)[1]
        )
        inside = head_top >= low * pose.height and feet <= high * pose.height
        for joint in REQUIRED_JOINTS:
            x, y = pose.landmarks[int(joint), :2]
            inside = inside and low <= x <= high
        clear = pose.min_visibility() >= config.min_visibility
        passed = inside and clear
        return Check("visibility", passed, "Show your whole body from head to toe")

    def _distance(self, pose: PoseResult) -> Check:
        fraction = stature_pixels(pose) / pose.height
        if fraction < self.config.min_body_fraction:
            return Check("distance", False, "Step closer to the camera")
        if fraction > self.config.max_body_fraction:
            return Check("distance", False, "Step back from the camera")
        return Check("distance", True, "Distance is good")

    def _centered(self, pose: PoseResult) -> Check:
        hips = pose.midpoint(Joint.LEFT_HIP, Joint.RIGHT_HIP)
        offset = hips[0] / pose.width - 0.5
        if offset < -self.config.max_center_offset:
            return Check("centered", False, "Move a little to the right")
        if offset > self.config.max_center_offset:
            return Check("centered", False, "Move a little to the left")
        return Check("centered", True, "Centered")

    def _facing(self, pose: PoseResult) -> Check:
        shoulders = pose.distance(Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
        shoulder_mid = pose.midpoint(Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
        hip_mid = pose.midpoint(Joint.LEFT_HIP, Joint.RIGHT_HIP)
        torso = float(np.linalg.norm(hip_mid - shoulder_mid)) + 1e-9
        passed = shoulders / torso >= self.config.min_shoulder_to_torso
        return Check("facing", passed, "Face the camera squarely")

    def _upright(self, pose: PoseResult) -> Check:
        limit = self.config.max_tilt_degrees
        shoulder_tilt = tilt_degrees(
            pose.point(Joint.LEFT_SHOULDER), pose.point(Joint.RIGHT_SHOULDER), "horizontal"
        )
        hip_tilt = tilt_degrees(
            pose.point(Joint.LEFT_HIP), pose.point(Joint.RIGHT_HIP), "horizontal"
        )
        lean = tilt_degrees(
            pose.midpoint(Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
            pose.midpoint(Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE),
            "vertical",
        )
        passed = max(shoulder_tilt, hip_tilt, lean) <= limit
        return Check("upright", passed, "Stand tall with shoulders level")

    def _arms(self, pose: PoseResult) -> Check:
        shoulder_y = pose.midpoint(Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)[1]
        torso = pose.midpoint(Joint.LEFT_HIP, Joint.RIGHT_HIP)[1] - shoulder_y
        for shoulder, wrist in (
            (Joint.LEFT_SHOULDER, Joint.LEFT_WRIST),
            (Joint.RIGHT_SHOULDER, Joint.RIGHT_WRIST),
        ):
            wrist_point = pose.point(wrist)
            if wrist_point[1] <= shoulder_y + 0.5 * torso:
                return Check("arms", False, "Let your arms hang down by your sides")
            angle = tilt_degrees(pose.point(shoulder), wrist_point, "vertical")
            if angle < self.config.min_arm_angle:
                return Check("arms", False, "Move your arms a little away from your body")
            if angle > self.config.max_arm_angle:
                return Check("arms", False, "Bring your arms a little closer to your body")
        return Check("arms", True, "Arms are clear of the body")

    def _image_quality(self, frame: np.ndarray) -> Check:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = ANALYSIS_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if brightness < self.config.min_brightness:
            return Check("image", False, "Move to a brighter spot")
        if brightness > self.config.max_brightness:
            return Check("image", False, "Reduce glare or strong backlight")
        if sharpness < self.config.min_sharpness:
            return Check("image", False, "Image is blurry, hold the camera steady")
        return Check("image", True, "Image quality is good")

    def _stillness(self, pose: PoseResult) -> Check:
        if self._previous is None:
            return Check("still", True, "Hold still")
        indices = [int(joint) for joint in REQUIRED_JOINTS]
        motion = np.linalg.norm(pose.landmarks[indices, :2] - self._previous[indices], axis=1)
        passed = float(motion.mean()) <= self.config.max_motion
        return Check("still", passed, "Hold still")
