import time
from pathlib import Path

import cv2
import numpy as np

from .capture import CaptureGuide, guide_config_for
from .errors import CaptureError, FitVisionError
from .landmarks import PoseExtractor
from .request import MeasurementRequest
from .service import FitVisionService
from .store import CaptureStore, SavedCapture
from .ui import SetupState, render_busy, render_error, render_guide, render_results, render_setup

WINDOW_NAME = "FitVision"
QUIT_KEYS = (27, ord("q"))


def read_frame(camera: cv2.VideoCapture) -> np.ndarray:
    grabbed, frame = camera.read()
    if not grabbed:
        raise CaptureError("Camera stopped delivering frames")
    return frame


def window_closed() -> bool:
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def show(image: np.ndarray) -> int:
    cv2.imshow(WINDOW_NAME, image)
    return cv2.waitKey(1)


def run_setup(camera: cv2.VideoCapture, state: SetupState, label: str | None) -> MeasurementRequest | None:
    while not window_closed():
        frame = read_frame(camera)
        action = state.handle_key(show(render_setup(frame, state, time.time())))
        if action == "quit":
            return None
        if action == "start":
            return MeasurementRequest(state.height_cm, state.sizing, state.clothing, label)
    return None


def run_capture(camera: cv2.VideoCapture, live: PoseExtractor, request: MeasurementRequest) -> np.ndarray | None:
    guide = CaptureGuide(guide_config_for(request.clothing))
    while not window_closed():
        frame = read_frame(camera)
        pose = live.extract(frame)
        state = guide.update(frame, pose, time.monotonic())
        if state.shutter:
            return frame
        if show(render_guide(frame, pose, state)) & 0xFF in QUIT_KEYS:
            return None
    return None


def wait_for(image: np.ndarray, accepted: tuple[int, ...]) -> int | None:
    while not window_closed():
        key = show(image)
        if key != -1 and key & 0xFF in accepted:
            return key & 0xFF
    return None


def run_app(
    service: FitVisionService,
    store: CaptureStore,
    camera_index: int = 0,
    prefill: SetupState | None = None,
    label: str | None = None,
    save_photo: bool = True,
    width: int = 1280,
    height: int = 720,
) -> SavedCapture | None:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise CaptureError(f"Could not open camera {camera_index}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)

    retake = (ord("r"), ord("R"))
    try:
        with PoseExtractor(static_image=False, model_complexity=1) as live:
            show(read_frame(camera))
            request = run_setup(camera, prefill or SetupState(), label)
            while request is not None:
                frame = run_capture(camera, live, request)
                if frame is None:
                    return None
                show(render_busy(frame))
                try:
                    result = service.measure(frame, request)
                except FitVisionError as error:
                    if wait_for(render_error(frame, str(error)), retake + (27,)) in retake:
                        continue
                    return None
                saved = store.save(result, frame if save_photo else None, request.label)
                shown = render_results(frame, result, str(saved.directory))
                if wait_for(shown, retake + (27,)) in retake:
                    continue
                return saved
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return None
