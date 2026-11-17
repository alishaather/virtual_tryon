import cv2
import mediapipe as mp
import numpy as np

from .models import PoseResult
from .silhouette import clean_mask

POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)
MASK_THRESHOLD = 0.5


class PoseExtractor:
    def __init__(
        self,
        static_image: bool = False,
        model_complexity: int = 1,
        min_confidence: float = 0.6,
    ):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image,
            model_complexity=model_complexity,
            enable_segmentation=True,
            smooth_segmentation=not static_image,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

    def extract(self, bgr: np.ndarray) -> PoseResult | None:
        height, width = bgr.shape[:2]
        result = self._pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks is None:
            return None
        landmarks = np.array(
            [[p.x, p.y, p.visibility] for p in result.pose_landmarks.landmark],
            dtype=np.float32,
        )
        mask = None
        if result.segmentation_mask is not None:
            mask = clean_mask(result.segmentation_mask > MASK_THRESHOLD)
        return PoseResult(landmarks=landmarks, mask=mask, width=width, height=height)

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
