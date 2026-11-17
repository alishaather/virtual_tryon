import pytest

from fitvision.calibration import calibrate
from fitvision.circumference import CircumferenceModel, Correction, ellipse_circumference
from fitvision.errors import CalibrationError, MeasurementError
from fitvision.measure import SHOULDER_EDGE_ALLOWANCE_CM, measure_silhouette
from fitvision.models import BodyMeasurements
from fitvision.pipeline import MeasurementPipeline
from fitvision.request import MeasurementRequest

from .conftest import (
    CHEST_WIDTH_CM,
    HEIGHT_CM,
    HIP_WIDTH_CM,
    SHOULDER_WIDTH_CM,
    WAIST_WIDTH_CM,
    build_person,
)


def test_calibration_recovers_scale(person):
    calibration = calibrate(person, HEIGHT_CM)
    assert calibration.stature_px == pytest.approx(1400, abs=1)
    assert calibration.cm_per_pixel == pytest.approx(0.1214, abs=0.001)


@pytest.mark.parametrize("height", [50, 300])
def test_calibration_rejects_implausible_height(person, height):
    with pytest.raises(CalibrationError):
        calibrate(person, height)


@pytest.mark.parametrize("angle", [15.0, 25.0, 35.0])
def test_silhouette_widths_match_ground_truth(angle):
    pose = build_person(arms_angle=angle)
    raw = measure_silhouette(pose, calibrate(pose, HEIGHT_CM))
    assert raw.shoulder_width_cm == pytest.approx(SHOULDER_WIDTH_CM + SHOULDER_EDGE_ALLOWANCE_CM, abs=1.0)
    assert raw.chest_width_cm == pytest.approx(CHEST_WIDTH_CM, abs=1.5)
    assert raw.waist_width_cm == pytest.approx(WAIST_WIDTH_CM, abs=1.0)
    assert raw.hip_width_cm == pytest.approx(HIP_WIDTH_CM, abs=1.0)


def test_missing_mask_is_reported(person):
    pose = type(person)(person.landmarks, None, person.width, person.height)
    with pytest.raises(MeasurementError):
        measure_silhouette(pose, calibrate(pose, HEIGHT_CM))


def test_ellipse_circumference_reduces_to_circle():
    assert ellipse_circumference(20, 20) == pytest.approx(62.83, abs=0.01)


def test_depth_ratio_and_correction_apply():
    model = CircumferenceModel()
    base = model.estimate("chest", 34.0)
    assert 88 < base < 96
    corrected = model.with_corrections({"chest": Correction(1.0, 2.0)}).estimate("chest", 34.0)
    assert corrected == pytest.approx(base + 2.0)


def test_model_roundtrip(tmp_path):
    model = CircumferenceModel().with_corrections({"waist": Correction(0.95, 1.5)})
    model.save(tmp_path / "model.json")
    loaded = CircumferenceModel.load(tmp_path / "model.json")
    assert loaded.corrections["waist"] == Correction(0.95, 1.5)
    assert loaded.depth_ratios == model.depth_ratios


def test_pipeline_produces_clean_profile_object(person):
    result = MeasurementPipeline(extractor=None).measure_pose(person, MeasurementRequest(HEIGHT_CM))
    assert result.height_cm == HEIGHT_CM
    assert 88 < result.chest_circumference_cm < 96
    assert 78 < result.waist_circumference_cm < 84
    assert 93 < result.hip_circumference_cm < 99
    assert result.confidence > 0.9
    assert BodyMeasurements.from_json(result.to_json()) == result


def test_arms_pressed_to_the_body_are_rejected():
    pose = build_person(arms_angle=0.0)
    with pytest.raises(MeasurementError, match="Arms are too close"):
        measure_silhouette(pose, calibrate(pose, HEIGHT_CM))


@pytest.mark.parametrize("clothing,arm_cm", [("fitted", 9.0), ("loose", 13.0)])
def test_sleeves_do_not_inflate_the_chest(clothing, arm_cm):
    pose = build_person(arms_angle=28.0, arm_cm=arm_cm)
    raw = measure_silhouette(pose, calibrate(pose, HEIGHT_CM), clothing)
    assert raw.chest_width_cm == pytest.approx(CHEST_WIDTH_CM, abs=1.5)


def test_wide_chest_relative_to_shoulders_is_accepted():
    pose = build_person(arms_angle=28.0)
    raw = measure_silhouette(pose, calibrate(pose, HEIGHT_CM))
    assert raw.chest_width_cm / (raw.shoulder_width_cm - 5.0) < 1.15


def test_loose_clothing_adds_warning_and_calibration_flag(person):
    pipeline = MeasurementPipeline(extractor=None)
    loose = pipeline.measure_pose(person, MeasurementRequest(HEIGHT_CM, "women", "loose"))
    fitted = pipeline.measure_pose(person, MeasurementRequest(HEIGHT_CM, "men", "fitted"))
    assert loose.warnings and not fitted.warnings
    assert loose.calibrated is False
    assert (loose.sizing, loose.clothing) == ("women", "loose")


def test_calibrated_library_is_used(person):
    from fitvision.circumference import CircumferenceModel, Correction, ModelLibrary

    library = ModelLibrary().with_model(
        "women:fitted", CircumferenceModel().with_corrections({"chest": Correction(1.0, 3.0)})
    )
    base = MeasurementPipeline(None).measure_pose(person, MeasurementRequest(HEIGHT_CM))
    tuned = MeasurementPipeline(None, library).measure_pose(person, MeasurementRequest(HEIGHT_CM))
    assert tuned.calibrated and not base.calibrated
    assert tuned.chest_circumference_cm == pytest.approx(base.chest_circumference_cm + 3.0, abs=0.11)


def test_library_roundtrip(tmp_path):
    from fitvision.circumference import CircumferenceModel, Correction, ModelLibrary

    library = ModelLibrary().with_model("men:fitted", CircumferenceModel().with_corrections({"waist": Correction(0.9, 2.0)}))
    library.save(tmp_path / "library.json")
    loaded = ModelLibrary.load(tmp_path / "library.json")
    assert loaded.models["men:fitted"].corrections["waist"] == Correction(0.9, 2.0)
