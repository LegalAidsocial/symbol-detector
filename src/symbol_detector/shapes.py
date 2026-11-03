"""Programmatic line-art renderers for symbols and their hard negatives.

Drawn from scratch with PIL (no copied/downloaded source images), so there is
nothing to license for these ones. Covers the symbols simple enough to draw
accurately as geometry; the flag-based symbols (Imperial war flag, Islamic
State, Hamas, Hezbollah, PKK) need real licensed reference images instead -
see data/licenses.csv.
"""

import math
from typing import Callable

from PIL import Image, ImageDraw

SIZE = 512
LINE = 28


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 0))
    return img, ImageDraw.Draw(img)


def _arm(draw: ImageDraw.ImageDraw, cx: int, cy: int, dx: int, dy: int, length: int) -> None:
    """One straight swastika arm from the centre, plus its hook."""
    x2, y2 = cx + dx * length, cy + dy * length
    draw.line([(cx, cy), (x2, y2)], fill="black", width=LINE)
    hx, hy = -dy, dx  # perpendicular, for the hook
    hook_len = length // 2
    draw.line([(x2, y2), (x2 + hx * hook_len, y2 + hy * hook_len)], fill="black", width=LINE)


def swastika(tilted: bool = True) -> Image.Image:
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    arm_len = 150
    angle = 45 if tilted else 0
    rad = math.radians(angle)
    for base_dx, base_dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        dx = base_dx * math.cos(rad) - base_dy * math.sin(rad)
        dy = base_dx * math.sin(rad) + base_dy * math.cos(rad)
        _arm(draw, cx, cy, dx, dy, arm_len)
    return img


def swastika_religious() -> Image.Image:
    """Hard negative: upright, with corner dots (Hindu/Buddhist/Jain style)."""
    img = swastika(tilted=False)
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE // 2, SIZE // 2
    r = 10
    for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        x, y = cx + dx * 110, cy + dy * 110
        draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
    return img


def _sig_rune(draw: ImageDraw.ImageDraw, x: int, y: int, h: int) -> None:
    """One lightning-bolt-shaped Sig rune, top-left origin."""
    pts = [(x, y), (x + h * 0.4, y + h * 0.45), (x, y + h * 0.5), (x + h * 0.4, y + h)]
    draw.line(pts, fill="black", width=LINE, joint="curve")


def ss_runes() -> Image.Image:
    img, draw = _canvas()
    h = 260
    _sig_rune(draw, 150, 130, h)
    _sig_rune(draw, 260, 130, h)
    return img


def lightning_warning() -> Image.Image:
    """Hard negative: single electrical-hazard lightning bolt."""
    img, draw = _canvas()
    pts = [(260, 100), (190, 280), (250, 280), (210, 420), (330, 220), (260, 220)]
    draw.polygon(pts, fill="black")
    return img


def black_sun() -> Image.Image:
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    r_outer, r_inner = 220, 40
    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline="black", width=10)
    for i in range(12):
        angle = math.radians(i * 30)
        x1 = cx + r_inner * math.cos(angle)
        y1 = cy + r_inner * math.sin(angle)
        x2 = cx + r_outer * math.cos(angle)
        y2 = cy + r_outer * math.sin(angle)
        draw.line([(x1, y1), (x2, y2)], fill="black", width=14)
    return img


def plain_sun() -> Image.Image:
    """Hard negative: ordinary sun with rounded triangular rays, no zigzags."""
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    draw.ellipse([cx - 90, cy - 90, cx + 90, cy + 90], fill="black")
    for i in range(8):
        angle = math.radians(i * 45)
        x1 = cx + 100 * math.cos(angle)
        y1 = cy + 100 * math.sin(angle)
        x2 = cx + 200 * math.cos(angle)
        y2 = cy + 200 * math.sin(angle)
        draw.line([(x1, y1), (x2, y2)], fill="black", width=20)
    return img


def odal_rune(serifs: bool = True) -> Image.Image:
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2 - 30
    pts = [(cx, cy - 100), (cx + 90, cy), (cx, cy + 100), (cx - 90, cy)]
    draw.line(pts + [pts[0]], fill="black", width=LINE, joint="curve")
    draw.line([(cx, cy + 100), (cx, cy + 220)], fill="black", width=LINE)
    if serifs:
        for dy in (100, 220):
            draw.line([(cx - 35, cy + dy - 25), (cx, cy + dy)], fill="black", width=LINE)
            draw.line([(cx + 35, cy + dy - 25), (cx, cy + dy)], fill="black", width=LINE)
    return img


def wolfsangel() -> Image.Image:
    img, draw = _canvas()
    pts = [(170, 120), (170, 390), (340, 390), (340, 300), (230, 300), (230, 230), (340, 230), (340, 120)]
    draw.line(pts, fill="black", width=LINE, joint="curve")
    return img


def fishhook() -> Image.Image:
    """Hard negative: plain curved fishhook, no angular wolf-trap styling."""
    img, draw = _canvas()
    draw.arc([150, 150, 370, 370], start=30, end=330, fill="black", width=LINE)
    draw.line([(260, 150), (260, 90)], fill="black", width=LINE)
    return img


def celtic_cross_extremist() -> Image.Image:
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    draw.ellipse([cx - 180, cy - 180, cx + 180, cy + 180], outline="black", width=30)
    draw.line([(cx, cy - 220), (cx, cy + 220)], fill="black", width=40)
    draw.line([(cx - 220, cy), (cx + 220, cy)], fill="black", width=40)
    return img


def grave_cross() -> Image.Image:
    """Hard negative: thinner church/grave cross, no circle, no bold blocky style."""
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    draw.line([(cx, cy - 220), (cx, cy + 220)], fill="black", width=14)
    draw.line([(cx - 120, cy - 60), (cx + 120, cy - 60)], fill="black", width=14)
    return img


def plain_tricolor() -> Image.Image:
    """Hard negative for imperial_war_flag: same black/white/red bands, no iron cross."""
    img = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, SIZE, SIZE // 3], fill="black")
    draw.rectangle([0, 2 * SIZE // 3, SIZE, SIZE], fill=(221, 0, 0))
    return img


SYMBOLS: dict[str, dict[str, Callable[[], Image.Image]]] = {
    "swastika": {"positive": swastika, "hard_negative": swastika_religious},
    "ss_runes": {"positive": ss_runes, "hard_negative": lightning_warning},
    "black_sun": {"positive": black_sun, "hard_negative": plain_sun},
    "odal_rune": {"positive": odal_rune, "hard_negative": lambda: odal_rune(serifs=False)},
    "wolfsangel": {"positive": wolfsangel, "hard_negative": fishhook},
    "celtic_cross": {"positive": celtic_cross_extremist, "hard_negative": grave_cross},
}
