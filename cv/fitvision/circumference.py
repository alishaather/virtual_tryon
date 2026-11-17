import json
import math
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DEPTH_RATIOS = {"chest": 0.72, "waist": 0.70, "hip": 0.68}


def ellipse_circumference(width_cm: float, depth_cm: float) -> float:
    a = width_cm / 2.0
    b = depth_cm / 2.0
    return math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))


@dataclass(frozen=True)
class Correction:
    slope: float = 1.0
    intercept: float = 0.0

    def apply(self, value: float) -> float:
        return self.slope * value + self.intercept


@dataclass(frozen=True)
class CircumferenceModel:
    depth_ratios: dict = field(default_factory=lambda: dict(DEFAULT_DEPTH_RATIOS))
    corrections: dict = field(default_factory=dict)

    def circumference(self, region: str, width_cm: float) -> float:
        return ellipse_circumference(width_cm, width_cm * self.depth_ratios[region])

    def correct(self, metric: str, value_cm: float) -> float:
        return self.corrections.get(metric, Correction()).apply(value_cm)

    def estimate(self, region: str, width_cm: float) -> float:
        return self.correct(region, self.circumference(region, width_cm))

    def with_corrections(self, corrections: dict) -> "CircumferenceModel":
        return CircumferenceModel(dict(self.depth_ratios), dict(corrections))

    def without_corrections(self) -> "CircumferenceModel":
        return CircumferenceModel(dict(self.depth_ratios), {})

    def to_dict(self) -> dict:
        return {
            "depth_ratios": self.depth_ratios,
            "corrections": {
                metric: {"slope": c.slope, "intercept": c.intercept}
                for metric, c in self.corrections.items()
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CircumferenceModel":
        data = json.loads(Path(path).read_text())
        corrections = {
            metric: Correction(**values)
            for metric, values in data.get("corrections", {}).items()
        }
        ratios = {**DEFAULT_DEPTH_RATIOS, **data.get("depth_ratios", {})}
        return cls(ratios, corrections)


@dataclass(frozen=True)
class ModelLibrary:
    models: dict = field(default_factory=dict)

    def resolve(self, sizing: str, clothing: str) -> tuple[CircumferenceModel, bool]:
        key = f"{sizing}:{clothing}"
        if key in self.models:
            return self.models[key], True
        return CircumferenceModel(), False

    def with_model(self, key: str, model: CircumferenceModel) -> "ModelLibrary":
        return ModelLibrary({**self.models, key: model})

    def save(self, path: str | Path) -> None:
        payload = {"models": {key: model.to_dict() for key, model in self.models.items()}}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelLibrary":
        data = json.loads(Path(path).read_text())
        models = {}
        for key, entry in data.get("models", {}).items():
            corrections = {m: Correction(**v) for m, v in entry.get("corrections", {}).items()}
            ratios = {**DEFAULT_DEPTH_RATIOS, **entry.get("depth_ratios", {})}
            models[key] = CircumferenceModel(ratios, corrections)
        return cls(models)
