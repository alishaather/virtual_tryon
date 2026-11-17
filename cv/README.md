# FitVision CV

Guided body capture, MediaPipe measurement, calibration, circumference model, accuracy study and HSV colour rules.

## Run (Python 3.10 to 3.12, Windows needs the Visual C++ Redistributable)

    pip install -r requirements.txt
    python -m pytest
    python -m fitvision

`python -m fitvision` opens the camera app: type your height (`165` or `5'4`), choose Women or Men and Fitted or
Loose / Kurta, then Enter. Each capture is saved in its own folder, so nothing is ever overwritten:

    captures/20261116_204743_ayesha/measurements.json
    captures/20261116_204743_ayesha/photo.jpg

Useful options: `--height 5'4 --sizing women --clothing loose --name ayesha --no-photo --out-dir captures --camera 1`.

Other commands:

    python -m fitvision measure photo.jpg --height 165 --sizing women --clothing fitted
    python -m fitvision study manifest.csv --images photos --out study_output

## Integration contract (for the rest of the team)

Core code has no UI and no global state. Import it into any backend or app.

    from fitvision import MeasurementRequest, BodyMeasurements, ModelLibrary
    from fitvision.service import FitVisionService

    library = ModelLibrary.load("study_output/model.json")
    with FitVisionService(library) as service:
        result = service.measure(image_bgr, MeasurementRequest(165, sizing="women", clothing="loose"))
    record = result.to_dict()

Create one `FitVisionService` and reuse it, because loading the pose model is the slow part.

Input: `MeasurementRequest(height_cm, sizing="women"|"men", clothing="fitted"|"loose", label=None)` plus a BGR numpy
image (full body, front view). `parse_height("5'4")` converts user text to cm.

Output, `BodyMeasurements` (schema_version 2, all lengths in cm, JSON safe via `to_dict()` / `from_dict()`):

| Field | Meaning |
|---|---|
| height_cm | height the person entered |
| shoulder_width_cm | seam to seam |
| chest_circumference_cm, waist_circumference_cm, hip_circumference_cm | full loop measurements |
| arm_length_cm, torso_length_cm | shoulder to wrist, shoulder to hip |
| cm_per_pixel | scale used |
| confidence | 0 to 1, results under 0.5 are rejected |
| sizing, clothing | what was requested |
| calibrated | true when a study-fitted model exists for this sizing and clothing |
| warnings | list of human readable cautions to show the user |
| captured_at | UTC ISO timestamp |

Failures raise subclasses of `FitVisionError` (`RequestError`, `MeasurementError`, `CaptureError`), and their messages
are written for end users ("Retake with your arms held further from your body").

Camera capture is optional: `fitvision.ui` and `fitvision.app` are only the desktop interface. A web or mobile app can
send its own photo to `FitVisionService.measure`. Colour rules live in `fitvision.color_rules` and need only OpenCV.

## Women and Eastern clothing

- Arms are removed from the silhouette using the arm landmarks, so sleeves do not inflate the chest.
- Loose mode asks for wider arms and removes a thicker band around each arm.
- Knees are no longer required, because long kurtas hide them.
- Silhouette measuring sees the garment, not the body. Loose kurtas overestimate circumferences, so loose results
  carry a warning and `calibrated=false` until a loose-clothing study exists. Fitted clothing is the most accurate.

## Accuracy study

Measure 20+ people per group with a tape (full loop for chest, waist, hips; seam to seam for shoulders).
Photograph each, name the photo after the subject, and fill `manifest.csv`:

    subject,height_cm,shoulder_cm,chest_cm,waist_cm,hip_cm,sizing,clothing
    ayesha,162,37,88,74,96,women,fitted

`sizing` and `clothing` are optional and default to women and fitted. The study reports bias, MAE, RMSE, MAPE, 2 cm hit
rate and limits of agreement per group, then fits a correction for every group with 9 or more subjects into
`study_output/model.json`. Pass it with `--model` or `ModelLibrary.load`.
