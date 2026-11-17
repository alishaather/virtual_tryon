import cv2
import numpy as np


def clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def runs(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate(([0], row.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return edges[0::2], edges[1::2]


def central_run(row: np.ndarray, center: int) -> tuple[int, int] | None:
    starts, ends = runs(row)
    if starts.size == 0:
        return None
    inside = np.flatnonzero((starts <= center) & (center < ends))
    if inside.size:
        index = int(inside[0])
    else:
        distance = np.minimum(np.abs(starts - center), np.abs(ends - center))
        index = int(np.argmin(distance))
    return int(starts[index]), int(ends[index])


def row_width(mask: np.ndarray, row: int, center: int) -> int | None:
    if not 0 <= row < mask.shape[0]:
        return None
    run = central_run(mask[row], center)
    return None if run is None else run[1] - run[0]


def band_width(
    mask: np.ndarray, row: int, center: int, half_band: int
) -> tuple[float, float] | None:
    widths = []
    for offset in range(-half_band, half_band + 1):
        width = row_width(mask, row + offset, center)
        if width is not None:
            widths.append(width)
    if not widths:
        return None
    values = np.array(widths, dtype=float)
    median = float(np.median(values))
    spread = float(values.std() / median) if median > 0 else 1.0
    return median, spread


def find_armpit_row(
    mask: np.ndarray,
    top_row: int,
    bottom_row: int,
    center: int,
    window: int = 8,
    jump: float = 1.25,
) -> int | None:
    history: list[int] = []
    for row in range(bottom_row, top_row - 1, -1):
        width = row_width(mask, row, center)
        if width is None:
            continue
        if len(history) >= window and width > jump * np.median(history[-window:]):
            return row
        history.append(width)
    return None
