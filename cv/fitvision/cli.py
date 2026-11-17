import argparse
import sys
from pathlib import Path

import cv2

from .app import run_app
from .circumference import ModelLibrary
from .errors import FitVisionError
from .pipeline import MeasurementPipeline
from .request import CLOTHINGS, SIZINGS, MeasurementRequest, parse_height
from .landmarks import PoseExtractor
from .service import FitVisionService
from .store import CaptureStore
from .ui import SetupState
from .study import run_study


def load_library(path: str | None) -> ModelLibrary:
    return ModelLibrary.load(path) if path else ModelLibrary()


def ask_height() -> float:
    while True:
        try:
            return parse_height(input("Your height (cm, or feet and inches like 5'4): "))
        except ValueError as error:
            print(error)


def build_request(args: argparse.Namespace, height: float | None) -> MeasurementRequest:
    return MeasurementRequest(height, args.sizing, args.clothing, args.name)


def command_capture(args: argparse.Namespace) -> None:
    prefill = SetupState(args.height or "", args.sizing, args.clothing)
    print("Loading pose models, the first run downloads them once...")
    with FitVisionService(load_library(args.model)) as service:
        saved = run_app(
            service, CaptureStore(args.out_dir), args.camera, prefill, args.name, not args.no_photo
        )
    if saved is None:
        print("Capture cancelled")
        return
    print(saved.measurements.to_json())
    print(f"Saved to {saved.directory}")


def command_measure(args: argparse.Namespace) -> None:
    image = cv2.imread(args.image)
    if image is None:
        raise FitVisionError(f"Could not read {args.image}")
    height = parse_height(args.height) if args.height else ask_height()
    request = build_request(args, height)
    with FitVisionService(load_library(args.model)) as service:
        result = service.measure(image, request)
    print(result.to_json())
    if not args.no_save:
        saved = CaptureStore(args.out_dir).save(result, None if args.no_photo else image, args.name)
        print(f"Saved to {saved.directory}")


def command_study(args: argparse.Namespace) -> None:
    with PoseExtractor(static_image=True, model_complexity=2) as extractor:
        pipeline = MeasurementPipeline(extractor, ModelLibrary())
        print(run_study(args.manifest, args.images, args.out, pipeline))


def add_profile_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--height", help="centimetres (165) or feet and inches (5'4)")
    parser.add_argument("--sizing", choices=SIZINGS, default="women")
    parser.add_argument("--clothing", choices=CLOTHINGS, default="fitted")
    parser.add_argument("--name", help="label used in the saved folder name")
    parser.add_argument("--model", help="model library JSON produced by the study")
    parser.add_argument("--out-dir", default="captures")
    parser.add_argument("--no-photo", action="store_true", help="do not save the photo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fitvision")
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="open the camera (default)")
    add_profile_options(capture)
    capture.add_argument("--camera", type=int, default=0)
    capture.set_defaults(handler=command_capture)

    measure = commands.add_parser("measure", help="measure from an existing photo")
    measure.add_argument("image")
    add_profile_options(measure)
    measure.add_argument("--no-save", action="store_true")
    measure.set_defaults(handler=command_measure)

    study = commands.add_parser("study", help="compare against tape measurements")
    study.add_argument("manifest")
    study.add_argument("--images", required=True)
    study.add_argument("--out", default="study_output")
    study.set_defaults(handler=command_study)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(arguments or ["capture"])
    try:
        args.handler(args)
    except FitVisionError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
