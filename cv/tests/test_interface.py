import json
import time

import numpy as np
import pytest

from fitvision.capture import CaptureGuide
from fitvision.models import BodyMeasurements
from fitvision.request import MeasurementRequest, cm_to_feet_inches, parse_height
from fitvision.errors import RequestError
from fitvision.store import CaptureStore, slugify
from fitvision.ui import SetupState, render_error, render_guide, render_results, render_setup

from .conftest import build_person


@pytest.mark.parametrize(
    "text,expected",
    [("165", 165.0), ("165cm", 165.0), ("5'4", 162.6), ("5'4\"", 162.6), ("5'", 152.4), ("5 ' 10", 177.8)],
)
def test_parse_height(text, expected):
    assert parse_height(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "5'13", "50", "400", "5.4.3"])
def test_parse_height_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_height(text)


def test_feet_inches_conversion():
    assert cm_to_feet_inches(165) == (5, 5)


def test_request_validation():
    with pytest.raises(RequestError):
        MeasurementRequest(165, sizing="kids")
    with pytest.raises(RequestError):
        MeasurementRequest(165, clothing="tight")
    with pytest.raises(RequestError):
        MeasurementRequest(30)
    assert MeasurementRequest(165, "men", "loose").model_key == "men:loose"


def type_keys(state, text):
    for character in text:
        state.handle_key(ord(character))


def test_setup_typing_and_selection():
    state = SetupState()
    type_keys(state, "5'4")
    assert state.height_cm == 162.6
    state.handle_key(8)
    assert state.text == "5'"
    state.handle_key(ord("m"))
    state.handle_key(ord("L"))
    assert (state.sizing, state.clothing) == ("men", "loose")
    state.handle_key(ord("w"))
    state.handle_key(ord("f"))
    assert (state.sizing, state.clothing) == ("women", "fitted")


def test_setup_enter_requires_valid_height():
    state = SetupState()
    assert state.handle_key(13) is None
    type_keys(state, "170")
    assert state.handle_key(13) == "start"
    assert state.handle_key(27) == "quit"
    assert state.handle_key(-1) is None


def test_setup_limits_length():
    state = SetupState()
    type_keys(state, "1234567890")
    assert len(state.text) == 6


def sample_result(**overrides):
    values = dict(
        height_cm=165.0, shoulder_width_cm=38.6, chest_circumference_cm=91.0,
        waist_circumference_cm=83.5, hip_circumference_cm=98.5, arm_length_cm=50.8,
        torso_length_cm=51.0, cm_per_pixel=0.3, confidence=0.93, captured_at="now",
        sizing="women", clothing="loose", warnings=("Loose clothing hides the body outline.",),
    )
    values.update(overrides)
    return BodyMeasurements(**values)


def test_screens_render_at_frame_size():
    small = np.zeros((720, 1280, 3), np.uint8)
    state = CaptureGuide().update(small, None, 0.0)
    assert render_setup(small, SetupState("165"), time.time()).shape == small.shape
    assert render_guide(small, None, state).shape == small.shape
    assert render_error(small, "Something went wrong with the photo").shape == small.shape
    assert render_results(small, sample_result(), "captures/x").shape == small.shape


def test_store_never_overwrites(tmp_path):
    store = CaptureStore(tmp_path)
    result = sample_result()
    image = np.zeros((20, 20, 3), np.uint8)
    saved = [store.save(result, image, "Ayesha Khan") for _ in range(3)]
    folders = {item.directory for item in saved}
    assert len(folders) == 3
    assert all(item.json_path.exists() and item.image_path.exists() for item in saved)
    assert BodyMeasurements.from_json(saved[0].json_path.read_text()) == result
    assert "ayesha-khan" in saved[0].directory.name


def test_store_can_skip_photo(tmp_path):
    saved = CaptureStore(tmp_path).save(sample_result())
    assert saved.image_path is None


def test_slugify():
    assert slugify(None) == "capture"
    assert slugify("  Sana / Malik!! ") == "sana-malik"


def test_schema_v2_json_contains_new_fields():
    data = json.loads(sample_result().to_json())
    assert data["schema_version"] == 2
    assert data["warnings"] == ["Loose clothing hides the body outline."]
    assert {"sizing", "clothing", "calibrated"} <= set(data)
