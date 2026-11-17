import numpy as np
import pytest

from fitvision.circumference import Correction
from fitvision.errors import StudyError
from fitvision.study import (
    Observation,
    error_summary,
    fit_all_corrections,
    fit_correction,
    leave_one_out_mae,
    render_group,
    render_report,
    summarize,
)


def synthetic(metric="chest", n=20, slope=1.08, intercept=-4.0, noise=0.8):
    rng = np.random.default_rng(3)
    tape = rng.uniform(75, 115, n)
    predicted = (tape - intercept) / slope + rng.normal(0, noise, n)
    return tape, predicted


def test_error_summary_values():
    tape = np.array([90.0, 100.0, 110.0, 80.0])
    predicted = np.array([92.0, 99.0, 112.0, 78.0])
    summary = error_summary(tape, predicted)
    assert summary["bias_cm"] == pytest.approx(0.25)
    assert summary["mae_cm"] == pytest.approx(1.75)
    assert summary["rmse_cm"] == pytest.approx(np.sqrt((4 + 1 + 4 + 4) / 4))
    assert summary["within_2cm_percent"] == 100.0


def test_fit_recovers_known_bias():
    tape, predicted = synthetic()
    correction = fit_correction(tape, predicted)
    assert correction.slope == pytest.approx(1.08, abs=0.05)
    corrected = np.array([correction.apply(p) for p in predicted])
    assert np.abs(corrected - tape).mean() < np.abs(predicted - tape).mean() / 2


def test_cross_validation_improves_error():
    tape, predicted = synthetic()
    assert leave_one_out_mae(tape, predicted) < np.abs(predicted - tape).mean()


def test_fit_needs_enough_subjects():
    with pytest.raises(StudyError):
        fit_correction(np.arange(5.0), np.arange(5.0))


def test_report_lists_metrics_and_corrections():
    tape, predicted = synthetic()
    observations = [Observation(f"s{i}", "women:fitted", "chest", t, p) for i, (t, p) in enumerate(zip(tape, predicted))]
    corrections, validation = fit_all_corrections(observations)
    report = render_report([render_group("women:fitted", summarize(observations), validation, corrections)], {"s99": "No person detected"})
    assert "| chest |" in report
    assert "Fitted corrections" in report
    assert "## women:fitted" in report
    assert "s99: No person detected" in report
    assert isinstance(corrections["chest"], Correction)


def test_fit_with_no_spread_only_shifts():
    tape = np.full(10, 38.0)
    predicted = np.full(10, 36.0)
    correction = fit_correction(tape, predicted)
    assert correction.slope == 1.0
    assert correction.apply(36.0) == pytest.approx(38.0)
