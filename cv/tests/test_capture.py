import numpy as np

from fitvision.capture import CaptureGuide, GuideConfig, guide_config_for

from .conftest import build_person


def run(guide, frame, pose, times):
    return [guide.update(frame, pose, t) for t in times]


def test_good_pose_triggers_shutter_after_hold(frame, person):
    guide = CaptureGuide(GuideConfig(hold_seconds=2.0))
    states = run(guide, frame, person, [0.0, 0.5, 1.0, 1.5, 2.0])
    assert all(state.ready for state in states)
    assert [state.shutter for state in states] == [False, False, False, False, True]
    assert states[1].hold_progress == 0.25


def test_missing_person_prompts_to_step_in(frame):
    state = CaptureGuide().update(frame, None, 0.0)
    assert not state.ready
    assert "frame" in state.instruction


def test_off_center_pose_is_rejected(frame):
    state = CaptureGuide().update(frame, build_person(offset_x=200), 0.0)
    assert "centered" in [c.name for c in state.checks if not c.passed]
    assert state.instruction == "Move a little to the left"


def test_arms_pressed_to_body_are_rejected(frame):
    state = CaptureGuide().update(frame, build_person(arms_angle=0.0), 0.0)
    assert "arms" in [c.name for c in state.checks if not c.passed]


def test_dark_frame_is_rejected(person):
    dark = np.full((1600, 1000, 3), 10, dtype=np.uint8)
    state = CaptureGuide().update(dark, person, 0.0)
    assert state.instruction == "Move to a brighter spot"


def test_movement_resets_the_hold(frame, person):
    guide = CaptureGuide(GuideConfig(hold_seconds=2.0))
    guide.update(frame, person, 0.0)
    guide.update(frame, person, 1.0)
    moved = build_person(offset_x=40)
    assert not guide.update(frame, moved, 1.5).ready
    assert not guide.update(frame, person, 2.0).ready
    resumed = guide.update(frame, person, 2.1)
    assert resumed.ready
    assert resumed.hold_progress == 0.0
    assert not resumed.shutter


def test_distance_messages(frame, person):
    guide = CaptureGuide(GuideConfig(min_body_fraction=0.95, max_body_fraction=0.99))
    assert guide.update(frame, person, 0.0).instruction == "Step closer to the camera"
    guide = CaptureGuide(GuideConfig(min_body_fraction=0.3, max_body_fraction=0.5))
    assert guide.update(frame, person, 0.0).instruction == "Step back from the camera"


def test_loose_clothing_needs_wider_arms(frame):
    pose = build_person(arms_angle=18.0)
    assert CaptureGuide(guide_config_for("fitted")).update(frame, pose, 0.0).ready
    state = CaptureGuide(guide_config_for("loose")).update(frame, pose, 0.0)
    assert state.instruction == "Move your arms a little away from your body"


def test_long_kurta_hiding_knees_does_not_block_capture(frame):
    pose = build_person()
    pose.landmarks[25, 2] = 0.05
    pose.landmarks[26, 2] = 0.05
    assert CaptureGuide().update(frame, pose, 0.0).ready
