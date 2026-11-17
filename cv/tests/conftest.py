import cv2
import numpy as np
import pytest

from fitvision.models import Joint, PoseResult

WIDTH, HEIGHT = 1000, 1600
CENTER_X = 500
HEEL_Y = 1500
STATURE_PX = 1400
HEIGHT_CM = 170.0
PX_PER_CM = STATURE_PX / HEIGHT_CM

CHEST_WIDTH_CM = 34.0
WAIST_WIDTH_CM = 30.0
HIP_WIDTH_CM = 36.0
SHOULDER_WIDTH_CM = 36.0


def px(cm: float) -> int:
    return int(round(cm * PX_PER_CM))


def build_person(arms_angle: float = 25.0, offset_x: int = 0, arm_cm: float = 9.0) -> PoseResult:
    cx = CENTER_X + offset_x
    eye_y = HEEL_Y - 0.936 * STATURE_PX
    shoulder_y, hip_y = 355, 758
    shoulder_half = px(SHOULDER_WIDTH_CM) // 2
    landmarks = np.tile(np.array([cx / WIDTH, 800 / HEIGHT, 0.99], dtype=np.float32), (33, 1))

    def put(joint: Joint, x: float, y: float) -> None:
        landmarks[int(joint)] = [x / WIDTH, y / HEIGHT, 0.99]

    put(Joint.NOSE, cx, eye_y + 20)
    put(Joint.LEFT_EYE, cx + 15, eye_y)
    put(Joint.RIGHT_EYE, cx - 15, eye_y)
    put(Joint.LEFT_EAR, cx + 35, eye_y + 10)
    put(Joint.RIGHT_EAR, cx - 35, eye_y + 10)
    put(Joint.LEFT_SHOULDER, cx + shoulder_half, shoulder_y)
    put(Joint.RIGHT_SHOULDER, cx - shoulder_half, shoulder_y)
    direction = np.array([np.sin(np.radians(arms_angle)), np.cos(np.radians(arms_angle))])
    for sign, elbow, wrist, shoulder in (
        (1, Joint.LEFT_ELBOW, Joint.LEFT_WRIST, Joint.LEFT_SHOULDER),
        (-1, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST, Joint.RIGHT_SHOULDER),
    ):
        origin = np.array([cx + sign * shoulder_half, shoulder_y])
        put(elbow, origin[0] + sign * 300 * direction[0], origin[1] + 300 * direction[1])
        put(wrist, origin[0] + sign * 600 * direction[0], origin[1] + 600 * direction[1])
    put(Joint.LEFT_HIP, cx + 90, hip_y)
    put(Joint.RIGHT_HIP, cx - 90, hip_y)
    put(Joint.LEFT_KNEE, cx + 80, 1100)
    put(Joint.RIGHT_KNEE, cx - 80, 1100)
    put(Joint.LEFT_ANKLE, cx + 70, 1445)
    put(Joint.RIGHT_ANKLE, cx - 70, 1445)
    put(Joint.LEFT_HEEL, cx + 60, HEEL_Y)
    put(Joint.RIGHT_HEEL, cx - 60, HEEL_Y)
    put(Joint.LEFT_FOOT_INDEX, cx + 70, HEEL_Y + 20)
    put(Joint.RIGHT_FOOT_INDEX, cx - 70, HEEL_Y + 20)

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(mask, (cx, 160), 70, 1, -1)
    cv2.rectangle(mask, (cx - 25, 200), (cx + 25, 355), 1, -1)
    half = {
        "shoulder": px(SHOULDER_WIDTH_CM) // 2 + 2,
        "chest": px(CHEST_WIDTH_CM) // 2,
        "waist": px(WAIST_WIDTH_CM) // 2,
        "hip": px(HIP_WIDTH_CM) // 2,
    }
    torso = np.array(
        [
            (cx - half["shoulder"], 355),
            (cx + half["shoulder"], 355),
            (cx + half["chest"], 484),
            (cx + half["waist"], 629),
            (cx + half["hip"], 758),
            (cx + half["hip"], 842),
            (cx - half["hip"], 842),
            (cx - half["hip"], 758),
            (cx - half["waist"], 629),
            (cx - half["chest"], 484),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [torso], 1)
    cv2.rectangle(mask, (cx - half["hip"], 842), (cx - 10, HEEL_Y + 20), 1, -1)
    cv2.rectangle(mask, (cx + 10, 842), (cx + half["hip"], HEEL_Y + 20), 1, -1)
    for sign in (1, -1):
        start = (int(cx + sign * shoulder_half), shoulder_y + 5)
        end = (int(start[0] + sign * 600 * direction[0]), int(start[1] + 600 * direction[1]))
        cv2.line(mask, start, end, 1, px(arm_cm))
    return PoseResult(landmarks=landmarks, mask=mask.astype(bool), width=WIDTH, height=HEIGHT)


@pytest.fixture
def person() -> PoseResult:
    return build_person()


@pytest.fixture
def frame() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(90, 170, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
