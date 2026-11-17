import numpy as np
import pytest

from fitvision.color_rules import (
    HSV,
    color_name,
    dominant_color,
    hue_gap,
    pair_score,
    recommend_dupatta,
    recommend_pair,
    score_outfit,
)

MAROON = HSV(350, 0.80, 0.40)
MUSTARD = HSV(48, 0.85, 0.85)
EMERALD = HSV(150, 0.85, 0.55)
PASTEL_PINK = HSV(330, 0.25, 0.95)
CREAM = HSV(40, 0.12, 0.95)
BLACK = HSV(0, 0.0, 0.05)
NAVY = HSV(225, 0.60, 0.28)
VIVID_RED = HSV(0, 0.90, 0.90)
VIVID_GREEN = HSV(120, 0.90, 0.80)
VIVID_ORANGE = HSV(25, 0.9, 0.95)
VIVID_PINK = HSV(320, 0.9, 0.9)


def test_hue_gap_wraps():
    assert hue_gap(350, 10) == 20
    assert hue_gap(0, 180) == 180


@pytest.mark.parametrize(
    "color,name",
    [(BLACK, "black"), (CREAM, "white"), (NAVY, "navy"), (MUSTARD, "yellow"), (EMERALD, "green"), (MAROON, "red")],
)
def test_color_names(color, name):
    assert color_name(color) == name


def test_opencv_roundtrip():
    color = HSV.from_opencv(90, 128, 200)
    assert color.to_opencv() == (90, 128, 200)


def test_neutral_pairs_with_anything():
    assert pair_score(BLACK, VIVID_RED).score >= 90
    assert pair_score(CREAM, EMERALD).score >= 90


def test_navy_with_black_is_flagged():
    assert pair_score(NAVY, BLACK).score <= 60


def test_loud_mid_distance_hues_clash():
    verdict = pair_score(VIVID_RED, VIVID_GREEN)
    assert verdict.score < 50
    assert "compete" in verdict.reason


def test_softer_shade_rescues_contrast():
    assert pair_score(PASTEL_PINK, MUSTARD).score > pair_score(VIVID_PINK, MUSTARD).score


def test_tonal_pair_needs_depth():
    flat = pair_score(HSV(10, 0.6, 0.6), HSV(12, 0.6, 0.62)).score
    deep = pair_score(HSV(10, 0.6, 0.9), HSV(12, 0.6, 0.45)).score
    assert deep > flat


def test_complementary_is_strong():
    assert pair_score(MAROON, EMERALD).score >= 80


def test_dupatta_neutral_calms_loud_set():
    loud = score_outfit(VIVID_PINK, VIVID_ORANGE, CREAM)
    clash = score_outfit(VIVID_PINK, VIVID_ORANGE, VIVID_GREEN)
    assert loud.score > clash.score
    assert loud.rating in {"great", "good"}
    assert clash.rating in {"risky", "clash"}


def test_dupatta_echoing_a_colour_ties_the_outfit():
    verdict = score_outfit(MAROON, EMERALD, HSV(352, 0.35, 0.9))
    assert any("ties" in reason for reason in verdict.reasons)


def test_matching_everything_is_called_flat():
    verdict = score_outfit(MAROON, MAROON, MAROON)
    assert any("exactly" in reason for reason in verdict.reasons)


def test_recommendations_are_sorted_and_good():
    pairs = recommend_pair(MAROON, count=5)
    assert [s.score for s in pairs] == sorted((s.score for s in pairs), reverse=True)
    assert pairs[0].score >= 85
    dupattas = recommend_dupatta(MAROON, EMERALD, count=3)
    assert dupattas[0].score >= 80


def test_dominant_color_ignores_background():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = (30, 30, 30)
    image[20:80, 20:80] = (60, 40, 200)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    color = dominant_color(image, mask)
    assert color_name(color) == "red"
