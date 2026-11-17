import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .capture import GuideState
from .landmarks import POSE_CONNECTIONS
from .models import BodyMeasurements, PoseResult
from .request import CLOTHINGS, SIZINGS, cm_to_feet_inches, parse_height

INK = (14, 20, 31)
CARD = (18, 25, 38, 238)
GLASS = (14, 20, 31, 190)
FIELD = (32, 41, 58, 255)
WHITE = (245, 248, 252, 255)
MUTED = (150, 163, 184, 255)
ACCENT = (45, 212, 170, 255)
AMBER = (251, 191, 36, 255)
RED = (248, 113, 113, 255)

REGULAR_FONTS = ("segoeui.ttf", "SegoeUI.ttf", "SFNS.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "arial.ttf")
BOLD_FONTS = ("segoeuib.ttf", "seguisb.ttf", "Helvetica.ttc", "DejaVuSans-Bold.ttf", "arialbd.ttf")
CHECK_TITLES = {
    "distance": "Distance",
    "centered": "Centered",
    "visibility": "Full body",
    "facing": "Facing camera",
    "upright": "Posture",
    "arms": "Arms",
    "image": "Lighting",
    "still": "Hold still",
}
SIZING_LABELS = {"women": "Women", "men": "Men"}
SIZING_KEYS = {"women": "W", "men": "M"}
CLOTHING_LABELS = {"fitted": "Fitted", "loose": "Loose / Kurta"}
CLOTHING_KEYS = {"fitted": "F", "loose": "L"}
TIPS = {
    "fitted": (
        "Stand 2 to 3 m from the camera, head to toe in view",
        "Hold your arms about 30 degrees away from your body",
        "Use even light and a plain wall behind you",
    ),
    "loose": (
        "Take off your dupatta, or drape it behind you",
        "A fitted layer underneath gives the best accuracy",
        "Hold your arms about 35 degrees away from your body",
    ),
}
KEY_ENTER, KEY_ESC, KEY_BACKSPACE = 13, 27, 8
HEIGHT_CHARACTERS = set(b"0123456789'.")
MAX_HEIGHT_TEXT = 6


@lru_cache(maxsize=64)
def load_font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    for name in BOLD_FONTS if bold else REGULAR_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Canvas:
    def __init__(self, bgr: np.ndarray):
        self._base = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        self._layer = Image.new("RGBA", self._base.size, (0, 0, 0, 0))
        self._draw = ImageDraw.Draw(self._layer)
        self.width, self.height = self._base.size
        self._unit = self.height / 720.0

    def u(self, value: float) -> int:
        return int(round(value * self._unit))

    def rect(self, box, fill, radius: int = 0, outline=None, width: int = 1) -> None:
        self._draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def circle(self, center, radius: int, fill=None, outline=None, width: int = 1) -> None:
        x, y = center
        self._draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)

    def arc(self, center, radius: int, start: float, end: float, color, width: int) -> None:
        x, y = center
        self._draw.arc((x - radius, y - radius, x + radius, y + radius), start, end, fill=color, width=width)

    def line(self, points, color, width: int) -> None:
        self._draw.line(points, fill=color, width=width, joint="curve")

    def text(self, xy, content: str, size: int, color=WHITE, bold: bool = False, anchor: str = "la") -> None:
        self._draw.text(xy, content, font=load_font(size, bold), fill=color, anchor=anchor)

    def text_width(self, content: str, size: int, bold: bool = False) -> float:
        return load_font(size, bold).getlength(content)

    def wrap(self, content: str, size: int, max_width: int, bold: bool = False) -> list[str]:
        lines, current = [], ""
        for word in content.split():
            trial = f"{current} {word}".strip()
            if self.text_width(trial, size, bold) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        return lines + ([current] if current else [])

    def to_bgr(self) -> np.ndarray:
        merged = Image.alpha_composite(self._base, self._layer).convert("RGB")
        return cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)


@dataclass
class SetupState:
    text: str = ""
    sizing: str = "women"
    clothing: str = "fitted"

    def parse(self) -> tuple[float | None, str | None]:
        if not self.text:
            return None, None
        try:
            return parse_height(self.text), None
        except ValueError as error:
            return None, str(error)

    @property
    def height_cm(self) -> float | None:
        return self.parse()[0]

    def handle_key(self, key: int) -> str | None:
        if key == -1:
            return None
        key &= 0xFF
        if key == KEY_ESC:
            return "quit"
        if key == KEY_ENTER:
            return "start" if self.height_cm is not None else None
        if key == KEY_BACKSPACE:
            self.text = self.text[:-1]
        elif key in HEIGHT_CHARACTERS and len(self.text) < MAX_HEIGHT_TEXT:
            self.text += chr(key)
        else:
            letter = chr(key).lower() if 32 <= key < 127 else ""
            for sizing, shortcut in SIZING_KEYS.items():
                if letter == shortcut.lower():
                    self.sizing = sizing
            for clothing, shortcut in CLOTHING_KEYS.items():
                if letter == shortcut.lower():
                    self.clothing = clothing
        return None


def dim_background(frame: np.ndarray, blur: bool = True) -> Canvas:
    base = cv2.GaussianBlur(frame, (0, 0), 14) if blur else frame
    canvas = Canvas(base)
    canvas.rect((0, 0, canvas.width, canvas.height), (6, 10, 18, 150))
    return canvas


def keycap(canvas: Canvas, x: int, y: int, key: str, label: str) -> int:
    size = canvas.u(14)
    key_w = int(canvas.text_width(key, size, True)) + canvas.u(18)
    canvas.rect((x, y, x + key_w, y + canvas.u(28)), FIELD, radius=canvas.u(8))
    canvas.text((x + key_w // 2, y + canvas.u(14)), key, size, WHITE, True, "mm")
    canvas.text((x + key_w + canvas.u(10), y + canvas.u(14)), label, size, MUTED, False, "lm")
    return x + key_w + canvas.u(20) + int(canvas.text_width(label, size))


def segmented(canvas: Canvas, x: int, y: int, width: int, options, labels, keys, selected: str) -> None:
    gap = canvas.u(10)
    segment = (width - gap * (len(options) - 1)) // len(options)
    height = canvas.u(46)
    for index, option in enumerate(options):
        left = x + index * (segment + gap)
        active = option == selected
        canvas.rect((left, y, left + segment, y + height), ACCENT if active else FIELD, radius=canvas.u(12))
        color = INK + (255,) if active else MUTED
        canvas.text((left + segment // 2, y + height // 2), labels[option], canvas.u(17), color, True, "mm")
        canvas.text((left + segment - canvas.u(14), y + height // 2), keys[option], canvas.u(12), color, False, "rm")


def render_setup(frame: np.ndarray, state: SetupState, now: float) -> np.ndarray:
    c = dim_background(frame)
    card_w, card_h = c.u(620), c.u(640)
    x0, y0 = (c.width - card_w) // 2, (c.height - card_h) // 2
    c.rect((x0, y0, x0 + card_w, y0 + card_h), CARD, radius=c.u(28))
    pad = c.u(40)
    x, inner = x0 + pad, card_w - 2 * pad
    y = y0 + c.u(34)

    c.text((x, y), "FitVision", c.u(32), WHITE, True)
    c.text((x, y + c.u(44)), "Body measurements for virtual try-on", c.u(15), MUTED)
    y += c.u(96)

    c.text((x, y), "HEIGHT", c.u(12), MUTED, True)
    y += c.u(24)
    height_cm, error = state.parse()
    border = RED if error else ACCENT if height_cm else FIELD
    c.rect((x, y, x + inner, y + c.u(60)), FIELD, radius=c.u(14), outline=border, width=c.u(2))
    shown = state.text or ""
    c.text((x + c.u(20), y + c.u(30)), shown, c.u(28), WHITE, True, "lm")
    if int(now * 2) % 2 == 0:
        cursor_x = x + c.u(22) + int(c.text_width(shown, c.u(28), True))
        c.line([(cursor_x, y + c.u(16)), (cursor_x, y + c.u(44))], ACCENT, c.u(2))
    c.text((x + inner - c.u(20), y + c.u(30)), "cm  or  5'4", c.u(14), MUTED, False, "rm")
    y += c.u(72)
    if error:
        c.text((x, y), error, c.u(14), RED)
    elif height_cm:
        feet, inches = cm_to_feet_inches(height_cm)
        c.text((x, y), f"{height_cm:.0f} cm   |   {feet}'{inches}\"", c.u(14), ACCENT)
    else:
        c.text((x, y), "Type your height, for example 165 or 5'4", c.u(14), MUTED)
    y += c.u(44)

    c.text((x, y), "SIZING", c.u(12), MUTED, True)
    y += c.u(24)
    segmented(c, x, y, inner, SIZINGS, SIZING_LABELS, SIZING_KEYS, state.sizing)
    y += c.u(70)

    c.text((x, y), "CLOTHING", c.u(12), MUTED, True)
    y += c.u(24)
    segmented(c, x, y, inner, CLOTHINGS, CLOTHING_LABELS, CLOTHING_KEYS, state.clothing)
    y += c.u(70)

    for tip in TIPS[state.clothing]:
        c.circle((x + c.u(4), y + c.u(10)), c.u(3), fill=AMBER)
        c.text((x + c.u(18), y), tip, c.u(14), MUTED)
        y += c.u(26)

    footer = y0 + card_h - c.u(52)
    next_x = keycap(c, x, footer, "Enter", "Start")
    keycap(c, next_x, footer, "Esc", "Quit")
    return c.to_bgr()


def draw_skeleton(display: np.ndarray, pose: PoseResult, color) -> None:
    height, width = display.shape[:2]
    points = [(int((1.0 - x) * width), int(y * height)) for x, y in pose.landmarks[:, :2]]
    for start, end in POSE_CONNECTIONS:
        cv2.line(display, points[start], points[end], color, 3, cv2.LINE_AA)


def brackets(c: Canvas, box, color) -> None:
    x1, y1, x2, y2 = box
    arm, width = c.u(44), c.u(4)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        c.line([(cx + dx * arm, cy), (cx, cy), (cx, cy + dy * arm)], color, width)


def render_guide(frame: np.ndarray, pose: PoseResult | None, state: GuideState) -> np.ndarray:
    display = cv2.flip(frame, 1)
    if pose is not None:
        draw_skeleton(display, pose, (170, 212, 45) if state.ready else (255, 255, 255))
    c = Canvas(display)

    row = c.u(34)
    panel_w = c.u(250)
    panel_h = c.u(70) + row * len(state.checks) + c.u(14)
    c.rect((c.u(24), c.u(24), c.u(24) + panel_w, c.u(24) + panel_h), GLASS, radius=c.u(20))
    c.text((c.u(48), c.u(40)), "FitVision", c.u(20), WHITE, True)
    for index, check in enumerate(state.checks):
        cy = c.u(24) + c.u(84) + index * row
        if check.passed:
            c.circle((c.u(56), cy), c.u(7), fill=ACCENT)
        else:
            c.circle((c.u(56), cy), c.u(7), outline=MUTED, width=c.u(2))
        title = CHECK_TITLES.get(check.name, check.name.title())
        c.text((c.u(76), cy), title, c.u(16), WHITE if check.passed else MUTED, False, "lm")

    box_h = int(c.height * 0.92)
    box_w = min(int(c.width * 0.46), int(box_h * 0.62))
    box = ((c.width - box_w) // 2, (c.height - box_h) // 2, (c.width + box_w) // 2, (c.height + box_h) // 2)
    brackets(c, box, ACCENT if state.ready else (255, 255, 255, 140))

    message = state.instruction
    size = c.u(24)
    pill_w = int(c.text_width(message, size, True)) + c.u(72)
    left = (c.width - pill_w) // 2
    top = c.height - c.u(96)
    c.rect((left, top, left + pill_w, top + c.u(56)), ACCENT if state.ready else GLASS, radius=c.u(28))
    c.circle((left + c.u(26), top + c.u(28)), c.u(6), fill=(14, 20, 31, 255) if state.ready else AMBER)
    c.text((left + c.u(44), top + c.u(28)), message, size, (14, 20, 31, 255) if state.ready else WHITE, True, "lm")

    if state.ready:
        center, radius = (c.width - c.u(130), c.height // 2), c.u(78)
        c.circle(center, radius, fill=GLASS)
        c.arc(center, radius, -90, -90 + 360 * state.hold_progress, ACCENT, c.u(9))
        c.text(center, str(max(1, math.ceil(state.seconds_left))), c.u(64), WHITE, True, "mm")
    return c.to_bgr()


def render_notice(frame: np.ndarray, title: str, body: str, hint: tuple[tuple[str, str], ...] = (), color=ACCENT) -> np.ndarray:
    c = dim_background(frame)
    card_w = c.u(560)
    lines = c.wrap(body, c.u(17), card_w - c.u(80))
    card_h = c.u(150) + c.u(26) * len(lines) + (c.u(70) if hint else 0)
    x0, y0 = (c.width - card_w) // 2, (c.height - card_h) // 2
    c.rect((x0, y0, x0 + card_w, y0 + card_h), CARD, radius=c.u(28))
    c.circle((x0 + c.u(48), y0 + c.u(58)), c.u(8), fill=color)
    c.text((x0 + c.u(72), y0 + c.u(58)), title, c.u(26), WHITE, True, "lm")
    for index, line in enumerate(lines):
        c.text((x0 + c.u(40), y0 + c.u(108) + index * c.u(26)), line, c.u(17), MUTED)
    next_x = x0 + c.u(40)
    for key, label in hint:
        next_x = keycap(c, next_x, y0 + card_h - c.u(58), key, label)
    return c.to_bgr()


def render_busy(frame: np.ndarray) -> np.ndarray:
    return render_notice(frame, "Measuring", "Reading your pose and body outline. This takes a few seconds.")


def render_error(frame: np.ndarray, message: str) -> np.ndarray:
    return render_notice(frame, "Could not measure", message, (("R", "Retake"), ("Esc", "Close")), RED)


def render_results(image: np.ndarray, result: BodyMeasurements, saved_to: str | None) -> np.ndarray:
    c = dim_background(image)
    rows = (
        ("Shoulder width", result.shoulder_width_cm),
        ("Chest", result.chest_circumference_cm),
        ("Waist", result.waist_circumference_cm),
        ("Hips", result.hip_circumference_cm),
        ("Arm length", result.arm_length_cm),
        ("Torso length", result.torso_length_cm),
    )
    card_w = c.u(600)
    warning_lines = [line for w in result.warnings for line in c.wrap(w, c.u(14), card_w - c.u(80))]
    card_h = c.u(228) + c.u(46) * len(rows) + c.u(22) * len(warning_lines) + c.u(40 if saved_to else 0)
    x0, y0 = (c.width - card_w) // 2, max(c.u(16), (c.height - card_h) // 2)
    c.rect((x0, y0, x0 + card_w, y0 + card_h), CARD, radius=c.u(28))
    pad = c.u(40)
    x, right = x0 + pad, x0 + card_w - pad
    y = y0 + c.u(34)

    c.text((x, y), "Your measurements", c.u(28), WHITE, True)
    chip = f"{int(round(result.confidence * 100))}% confidence"
    chip_w = int(c.text_width(chip, c.u(13), True)) + c.u(44)
    chip_color = ACCENT if result.confidence >= 0.8 else AMBER
    c.rect((right - chip_w, y + c.u(4), right, y + c.u(34)), FIELD, radius=c.u(15))
    c.circle((right - chip_w + c.u(18), y + c.u(19)), c.u(4), fill=chip_color)
    c.text((right - c.u(12), y + c.u(19)), chip, c.u(13), WHITE, True, "rm")
    y += c.u(48)
    c.text((x, y), f"{SIZING_LABELS[result.sizing]}  |  {CLOTHING_LABELS[result.clothing]}  |  {result.height_cm:.0f} cm tall", c.u(14), MUTED)
    y += c.u(40)

    for label, value in rows:
        c.line([(x, y), (right, y)], (36, 46, 64, 255), 1)
        c.text((x, y + c.u(23)), label, c.u(17), MUTED, False, "lm")
        c.text((right, y + c.u(23)), f"{value:.1f} cm", c.u(20), WHITE, True, "rm")
        y += c.u(46)
    y += c.u(14)

    for line in warning_lines:
        c.text((x, y), line, c.u(14), AMBER)
        y += c.u(22)
    if saved_to:
        c.text((x, y + c.u(6)), f"Saved to {saved_to}", c.u(13), MUTED)
    footer = y0 + card_h - c.u(58)
    next_x = keycap(c, x, footer, "R", "Retake")
    keycap(c, next_x, footer, "Esc", "Close")
    return c.to_bgr()
