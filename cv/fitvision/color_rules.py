import colorsys
from dataclasses import dataclass

import cv2
import numpy as np

NEUTRALS = frozenset({"black", "white", "grey", "beige", "brown", "navy"})
HUE_FAMILIES = (
    (15, "red"),
    (45, "orange"),
    (70, "yellow"),
    (165, "green"),
    (195, "teal"),
    (255, "blue"),
    (290, "purple"),
    (345, "pink"),
)
GREAT, GOOD, RISKY = 80, 65, 50
MONOCHROME_GAP = 12
ANALOGOUS_GAP = 40
CONTRAST_GAP = 75
SPLIT_GAP = 105
TRIAD_GAP = 150


@dataclass(frozen=True)
class HSV:
    h: float
    s: float
    v: float

    @classmethod
    def from_opencv(cls, h: float, s: float, v: float) -> "HSV":
        return cls(h * 2.0, s / 255.0, v / 255.0)

    @classmethod
    def from_rgb(cls, r: float, g: float, b: float) -> "HSV":
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        return cls(h * 360.0, s, v)

    def to_rgb(self) -> tuple[int, int, int]:
        r, g, b = colorsys.hsv_to_rgb(self.h / 360.0, self.s, self.v)
        return round(r * 255), round(g * 255), round(b * 255)

    def to_opencv(self) -> tuple[int, int, int]:
        return round(self.h / 2.0) % 180, round(self.s * 255), round(self.v * 255)


@dataclass(frozen=True)
class PairVerdict:
    score: int
    reason: str


@dataclass(frozen=True)
class OutfitVerdict:
    score: int
    rating: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Suggestion:
    color: HSV
    label: str
    score: int


def hue_gap(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def color_name(color: HSV) -> str:
    if color.v < 0.18:
        return "black"
    if color.s < 0.10:
        return "white" if color.v > 0.88 else "grey"
    if color.s < 0.18 and color.v > 0.90:
        return "white"
    if 20 <= color.h <= 55 and color.s < 0.32 and color.v > 0.70:
        return "beige"
    if 15 <= color.h <= 45 and color.v < 0.50 and color.s >= 0.35:
        return "brown"
    if 200 <= color.h <= 250 and color.v < 0.38 and color.s >= 0.30:
        return "navy"
    hue = color.h % 360.0
    for limit, name in HUE_FAMILIES:
        if hue < limit:
            return name
    return "red"


def is_neutral(color: HSV) -> bool:
    return color_name(color) in NEUTRALS


def is_vivid(color: HSV) -> bool:
    return not is_neutral(color) and color.s >= 0.60 and color.v >= 0.60


def is_soft(color: HSV) -> bool:
    return not is_neutral(color) and (color.s < 0.40 or color.v < 0.45)


def tone_label(color: HSV) -> str:
    name = color_name(color)
    if name in NEUTRALS:
        return name
    if color.s < 0.40 and color.v >= 0.70:
        return f"pastel {name}"
    if color.v < 0.45:
        return f"deep {name}"
    if is_vivid(color):
        return f"vivid {name}"
    return f"muted {name}"


def pair_score(first: HSV, second: HSV) -> PairVerdict:
    names = (color_name(first), color_name(second))
    neutral = [name in NEUTRALS for name in names]

    if all(neutral):
        if set(names) == {"navy", "black"}:
            return PairVerdict(60, "Navy and black are too close to separate")
        if names[0] == names[1]:
            return PairVerdict(75, "Same neutral in both pieces")
        return PairVerdict(88, "Neutrals pair cleanly")
    if any(neutral):
        return PairVerdict(92, "A neutral lets the colour stand out")

    gap = hue_gap(first.h, second.h)
    depth = max(abs(first.v - second.v), abs(first.s - second.s))
    both_vivid = is_vivid(first) and is_vivid(second)
    either_soft = is_soft(first) or is_soft(second)

    if gap <= MONOCHROME_GAP:
        if depth >= 0.20:
            return PairVerdict(90, "Tonal pairing with clear depth contrast")
        return PairVerdict(58, "Near-identical shades look flat together")
    if gap <= ANALOGOUS_GAP:
        return PairVerdict(84, "Neighbouring hues sit comfortably together")
    if gap > TRIAD_GAP:
        if both_vivid:
            return PairVerdict(74, "Strong complements, soften one for balance")
        return PairVerdict(86, "Complementary colours balance each other")

    score = 60 if gap <= CONTRAST_GAP else 72 if gap <= SPLIT_GAP else 66
    reason = "Mid-distance hues can look unplanned"
    if both_vivid:
        return PairVerdict(score - 20, "Two loud colours compete for attention")
    if either_soft:
        return PairVerdict(score + 12, "A softer shade calms the contrast")
    return PairVerdict(score, reason)


def rating_for(score: float) -> str:
    if score >= GREAT:
        return "great"
    if score >= GOOD:
        return "good"
    if score >= RISKY:
        return "risky"
    return "clash"


def score_outfit(top: HSV, bottom: HSV, dupatta: HSV) -> OutfitVerdict:
    pairs = {
        "top and bottom": pair_score(top, bottom),
        "dupatta and top": pair_score(dupatta, top),
        "dupatta and bottom": pair_score(dupatta, bottom),
    }
    score = (
        0.30 * pairs["top and bottom"].score
        + 0.35 * pairs["dupatta and top"].score
        + 0.35 * pairs["dupatta and bottom"].score
    )
    reasons = [f"{label}: {verdict.reason}" for label, verdict in pairs.items()]

    garments = (top, bottom, dupatta)
    if all(is_vivid(color) for color in garments):
        score -= 20
        reasons.append("Three loud colours: soften at least one piece")

    dupatta_neutral = is_neutral(dupatta)
    contrasting_set = (
        not is_neutral(top)
        and not is_neutral(bottom)
        and hue_gap(top.h, bottom.h) > ANALOGOUS_GAP
    )
    echoes_top = not dupatta_neutral and hue_gap(dupatta.h, top.h) <= 15
    echoes_bottom = not dupatta_neutral and hue_gap(dupatta.h, bottom.h) <= 15

    if contrasting_set and not dupatta_neutral and not (echoes_top or echoes_bottom):
        score -= 15
        reasons.append("Three different hues: echo one of them or choose a neutral dupatta")
    elif contrasting_set and (echoes_top or echoes_bottom):
        score += 8
        reasons.append("The dupatta ties the two main colours together")
    elif contrasting_set and dupatta_neutral and (is_vivid(top) or is_vivid(bottom)):
        score += 5
        reasons.append("A neutral dupatta calms two strong colours")

    identical_set = (
        not dupatta_neutral
        and hue_gap(dupatta.h, top.h) <= MONOCHROME_GAP
        and hue_gap(dupatta.h, bottom.h) <= MONOCHROME_GAP
        and abs(dupatta.v - top.v) < 0.12
        and abs(dupatta.v - bottom.v) < 0.12
    )
    if identical_set:
        score -= 10
        reasons.append("Dupatta matches both pieces exactly: add contrast in shade or print")

    final = int(round(max(0.0, min(100.0, score))))
    return OutfitVerdict(final, rating_for(final), tuple(reasons))


def candidate_palette() -> list[HSV]:
    neutrals = [
        HSV(0, 0, 0.08),
        HSV(0, 0, 0.96),
        HSV(0, 0, 0.55),
        HSV(40, 0.20, 0.88),
        HSV(25, 0.55, 0.35),
        HSV(225, 0.60, 0.28),
    ]
    chromatic = []
    for hue in range(0, 360, 30):
        chromatic.append(HSV(hue, 0.30, 0.92))
        chromatic.append(HSV(hue, 0.55, 0.75))
        chromatic.append(HSV(hue, 0.75, 0.42))
    return neutrals + chromatic


def recommend_pair(anchor: HSV, count: int = 5) -> list[Suggestion]:
    ranked = sorted(
        (
            Suggestion(color, tone_label(color), pair_score(anchor, color).score)
            for color in candidate_palette()
        ),
        key=lambda suggestion: suggestion.score,
        reverse=True,
    )
    return ranked[:count]


def recommend_dupatta(top: HSV, bottom: HSV, count: int = 5) -> list[Suggestion]:
    ranked = sorted(
        (
            Suggestion(color, tone_label(color), score_outfit(top, bottom, color).score)
            for color in candidate_palette()
        ),
        key=lambda suggestion: suggestion.score,
        reverse=True,
    )
    return ranked[:count]


def dominant_color(
    bgr: np.ndarray, mask: np.ndarray | None = None, clusters: int = 3, max_side: int = 160
) -> HSV:
    scale = max_side / max(bgr.shape[:2])
    if scale < 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(
                mask.astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
    pixels = bgr[mask] if mask is not None else bgr.reshape(-1, 3)
    pixels = pixels.astype(np.float32)
    if pixels.shape[0] == 0:
        raise ValueError("No garment pixels to analyse")
    clusters = min(clusters, pixels.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    dominant = centers[int(np.argmax(np.bincount(labels.flatten())))]
    b, g, r = dominant
    return HSV.from_rgb(r, g, b)
