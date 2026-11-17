import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .circumference import CircumferenceModel, Correction, ModelLibrary
from .errors import FitVisionError, StudyError
from .pipeline import MeasurementPipeline
from .request import MeasurementRequest

METRICS = {
    "shoulder": ("shoulder_width_cm", "shoulder_cm"),
    "chest": ("chest_circumference_cm", "chest_cm"),
    "waist": ("waist_circumference_cm", "waist_cm"),
    "hip": ("hip_circumference_cm", "hip_cm"),
}
ACCEPTABLE_MAE_CM = {"shoulder": 1.5, "chest": 2.5, "waist": 3.0, "hip": 2.5}
MIN_SUBJECTS_FOR_FIT = 8
MIN_SPREAD_CM = 1.0
SLOPE_BOUNDS = (0.7, 1.3)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class Observation:
    subject: str
    group: str
    metric: str
    tape_cm: float
    predicted_cm: float


def find_image(image_dir: Path, subject: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{subject}{suffix}"
        if candidate.exists():
            return candidate
    raise StudyError(f"No image found for subject {subject}")


def load_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"subject", "height_cm"} | {tape for _, tape in METRICS.values()}
    if not rows or not required <= set(rows[0]):
        raise StudyError(f"Manifest needs columns: {sorted(required)}")
    return rows


def collect_observations(
    manifest: list[dict], image_dir: str | Path, pipeline: MeasurementPipeline
) -> tuple[list[Observation], dict[str, str]]:
    observations: list[Observation] = []
    failures: dict[str, str] = {}
    for row in manifest:
        subject = row["subject"]
        try:
            image = cv2.imread(str(find_image(Path(image_dir), subject)))
            if image is None:
                raise StudyError("Image could not be read")
            request = MeasurementRequest(
                float(row["height_cm"]),
                row.get("sizing") or "women",
                row.get("clothing") or "fitted",
            )
            result = pipeline.measure_image(image, request)
        except (FitVisionError, ValueError) as error:
            failures[subject] = str(error)
            continue
        for metric, (field_name, tape_column) in METRICS.items():
            observations.append(
                Observation(subject, request.model_key, metric, float(row[tape_column]), getattr(result, field_name))
            )
    return observations, failures


def by_metric(observations: list[Observation]) -> dict[str, list[Observation]]:
    grouped: dict[str, list[Observation]] = {metric: [] for metric in METRICS}
    for observation in observations:
        grouped[observation.metric].append(observation)
    return grouped


def arrays(observations: list[Observation]) -> tuple[np.ndarray, np.ndarray]:
    tape = np.array([o.tape_cm for o in observations], dtype=float)
    predicted = np.array([o.predicted_cm for o in observations], dtype=float)
    return tape, predicted


def error_summary(tape: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - tape
    spread = float(error.std(ddof=1)) if len(error) > 1 else 0.0
    correlation = (
        float(np.corrcoef(tape, predicted)[0, 1])
        if len(tape) > 2 and tape.std() > 0 and predicted.std() > 0
        else float("nan")
    )
    return {
        "n": int(len(error)),
        "bias_cm": float(error.mean()),
        "mae_cm": float(np.abs(error).mean()),
        "rmse_cm": float(np.sqrt((error**2).mean())),
        "mape_percent": float((np.abs(error) / tape).mean() * 100.0),
        "within_2cm_percent": float((np.abs(error) <= 2.0).mean() * 100.0),
        "loa_low_cm": float(error.mean() - 1.96 * spread),
        "loa_high_cm": float(error.mean() + 1.96 * spread),
        "pearson_r": correlation,
    }


def summarize(observations: list[Observation]) -> dict[str, dict]:
    return {
        metric: error_summary(*arrays(items))
        for metric, items in by_metric(observations).items()
        if items
    }


def fit_correction(tape: np.ndarray, predicted: np.ndarray) -> Correction:
    if len(tape) < MIN_SUBJECTS_FOR_FIT:
        raise StudyError(f"Need at least {MIN_SUBJECTS_FOR_FIT} subjects to fit a correction")
    if predicted.std() < MIN_SPREAD_CM:
        slope = 1.0
    else:
        slope = float(np.clip(np.polyfit(predicted, tape, 1)[0], *SLOPE_BOUNDS))
    intercept = float(np.mean(tape) - slope * np.mean(predicted))
    return Correction(round(slope, 4), round(intercept, 3))


def leave_one_out_mae(tape: np.ndarray, predicted: np.ndarray) -> float:
    errors = []
    for index in range(len(tape)):
        keep = np.arange(len(tape)) != index
        correction = fit_correction(tape[keep], predicted[keep]) if keep.sum() >= MIN_SUBJECTS_FOR_FIT else Correction()
        errors.append(abs(correction.apply(predicted[index]) - tape[index]))
    return float(np.mean(errors))


def fit_all_corrections(observations: list[Observation]) -> tuple[dict, dict]:
    corrections: dict[str, Correction] = {}
    validation: dict[str, dict] = {}
    for metric, items in by_metric(observations).items():
        if len(items) < MIN_SUBJECTS_FOR_FIT + 1:
            continue
        tape, predicted = arrays(items)
        corrections[metric] = fit_correction(tape, predicted)
        validation[metric] = {
            "raw_mae_cm": float(np.abs(predicted - tape).mean()),
            "cross_validated_mae_cm": leave_one_out_mae(tape, predicted),
        }
    return corrections, validation


def by_group(observations: list[Observation]) -> dict[str, list[Observation]]:
    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.group, []).append(observation)
    return grouped


def render_group(
    group: str,
    summary: dict[str, dict],
    validation: dict[str, dict],
    corrections: dict[str, Correction],
) -> list[str]:
    lines = [
        f"## {group}",
        "",
        "| Metric | n | Bias (cm) | MAE (cm) | RMSE (cm) | MAPE (%) | Within 2 cm (%) | 95% limits (cm) | r | Target MAE | Result |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for metric, row in summary.items():
        target = ACCEPTABLE_MAE_CM[metric]
        verdict = "pass" if row["mae_cm"] <= target else "fail"
        lines.append(
            f"| {metric} | {row['n']} | {row['bias_cm']:+.2f} | {row['mae_cm']:.2f} | {row['rmse_cm']:.2f} "
            f"| {row['mape_percent']:.1f} | {row['within_2cm_percent']:.0f} "
            f"| {row['loa_low_cm']:+.1f} to {row['loa_high_cm']:+.1f} | {row['pearson_r']:.2f} | {target:.1f} | {verdict} |"
        )
    if corrections:
        lines += ["", "Fitted corrections", "", "| Metric | Slope | Intercept (cm) | MAE before | MAE after (leave-one-out) |", "|---|---|---|---|---|"]
        for metric, correction in corrections.items():
            check = validation[metric]
            lines.append(
                f"| {metric} | {correction.slope:.3f} | {correction.intercept:+.2f} "
                f"| {check['raw_mae_cm']:.2f} | {check['cross_validated_mae_cm']:.2f} |"
            )
    else:
        lines += ["", f"Not enough subjects to fit corrections (need {MIN_SUBJECTS_FOR_FIT + 1} or more)."]
    return lines + [""]


def render_report(sections: list[list[str]], failures: dict[str, str]) -> str:
    lines = ["# Accuracy study: pipeline vs tape measure", ""]
    for section in sections:
        lines += section
    if failures:
        lines += ["## Subjects that could not be measured", ""]
        lines += [f"- {subject}: {reason}" for subject, reason in failures.items()]
    return "\n".join(lines) + "\n"


def write_observations(observations: list[Observation], path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "group", "metric", "tape_cm", "predicted_cm", "error_cm"])
        for o in observations:
            writer.writerow([o.subject, o.group, o.metric, o.tape_cm, f"{o.predicted_cm:.2f}", f"{o.predicted_cm - o.tape_cm:.2f}"])


def run_study(
    manifest_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    pipeline: MeasurementPipeline,
    base_library: ModelLibrary | None = None,
) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observations, failures = collect_observations(load_manifest(manifest_path), image_dir, pipeline)
    if not observations:
        raise StudyError("No subject could be measured")

    library = base_library or ModelLibrary()
    sections = []
    for group, items in by_group(observations).items():
        try:
            corrections, validation = fit_all_corrections(items)
        except StudyError:
            corrections, validation = {}, {}
        sections.append(render_group(group, summarize(items), validation, corrections))
        if corrections:
            library = library.with_model(group, CircumferenceModel().with_corrections(corrections))

    report = render_report(sections, failures)
    write_observations(observations, output / "observations.csv")
    (output / "report.md").write_text(report)
    library.save(output / "model.json")
    return report
