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


def swastika_ornament_tile() -> Image.Image:
    """Hard negative: a Greek-key/meander ornamental tile pattern - the kind
    of repeating angular geometric border design the hooked-cross shape is
    sometimes confused with in tile/ornament contexts (brief: 'tiles and
    ornaments')."""
    img, draw = _canvas()
    step = 85
    y0 = SIZE // 2 - step
    for i in range(5):
        x = 60 + i * step
        pts = [
            (x, y0),
            (x, y0 + step),
            (x + step * 0.6, y0 + step),
            (x + step * 0.6, y0 + step * 0.4),
            (x + step * 0.2, y0 + step * 0.4),
        ]
        draw.line(pts, fill="black", width=12, joint="curve")
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


def decorative_wheel() -> Image.Image:
    """Hard negative: a plain decorative spoked wheel (e.g. ship's-wheel
    style) - evenly spaced thin spokes, no sig-rune zigzag, no sun rays."""
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    draw.ellipse([cx - 200, cy - 200, cx + 200, cy + 200], outline="black", width=16)
    draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], outline="black", width=10)
    for i in range(10):
        angle = math.radians(i * 36)
        x1 = cx + 40 * math.cos(angle)
        y1 = cy + 40 * math.sin(angle)
        x2 = cx + 200 * math.cos(angle)
        y2 = cy + 200 * math.sin(angle)
        draw.line([(x1, y1), (x2, y2)], fill="black", width=8)
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


def odal_rune_jewellery() -> Image.Image:
    """Hard negative: the odal-rune shape rendered as a plain pendant charm -
    a common innocent context (Nordic-style jewellery), thin lines, no
    serifs, pendant loop and bezel (brief: 'Plain runes on jewellery')."""
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    draw.ellipse([cx - 20, cy - 220, cx + 20, cy - 180], outline="black", width=8)
    draw.ellipse([cx - 150, cy - 170, cx + 150, cy + 190], outline="black", width=8)
    pts = [(cx, cy - 70), (cx + 60, cy), (cx, cy + 70), (cx - 60, cy)]
    draw.line(pts + [pts[0]], fill="black", width=10, joint="curve")
    draw.line([(cx, cy + 70), (cx, cy + 130)], fill="black", width=10)
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


def wolfsangel_heraldry() -> Image.Image:
    """Hard negative: the wolf-hook shape inside a heraldic shield outline -
    this charge appears legitimately in several real German municipal coats
    of arms (brief: 'Heraldry, municipal coats of arms')."""
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    shield = [
        (cx - 140, cy - 180),
        (cx + 140, cy - 180),
        (cx + 140, cy + 60),
        (cx, cy + 220),
        (cx - 140, cy + 60),
    ]
    draw.line(shield + [shield[0]], fill="black", width=12, joint="curve")
    hx, hy = cx - 40, cy - 110
    hook = [
        (hx, hy),
        (hx, hy + 150),
        (hx + 90, hy + 150),
        (hx + 90, hy + 100),
        (hx + 30, hy + 100),
        (hx + 30, hy + 55),
        (hx + 90, hy + 55),
        (hx + 90, hy),
    ]
    draw.line(hook, fill="black", width=12, joint="curve")
    return img


def ss_deaths_head() -> Image.Image:
    """SS-style skull and crossbones (Totenkopf) insignia - bold outlined
    skull over crossed bones."""
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2 - 40
    draw.ellipse([cx - 110, cy - 130, cx + 110, cy + 90], outline="black", width=LINE)
    r = 28
    draw.ellipse([cx - 70, cy - 40, cx - 70 + 2 * r, cy - 40 + 2 * r], fill="black")
    draw.ellipse([cx + 70 - 2 * r, cy - 40, cx + 70, cy - 40 + 2 * r], fill="black")
    draw.line([(cx - 50, cy + 70), (cx + 50, cy + 70)], fill="black", width=10)
    draw.line([(cx - 140, cy + 180), (cx + 140, cy + 320)], fill="black", width=LINE)
    draw.line([(cx - 140, cy + 320), (cx + 140, cy + 180)], fill="black", width=LINE)
    return img


def pirate_flag_skull() -> Image.Image:
    """Hard negative: Jolly Roger-style skull and crossbones on a black flag
    field - pirate flag context, filled white silhouette rather than the
    bold outlined insignia style."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE // 2, SIZE // 2 - 40
    draw.ellipse([cx - 90, cy - 100, cx + 90, cy + 70], fill="white")
    r = 20
    draw.ellipse([cx - 55, cy - 30, cx - 55 + 2 * r, cy - 30 + 2 * r], fill="black")
    draw.ellipse([cx + 55 - 2 * r, cy - 30, cx + 55, cy - 30 + 2 * r], fill="black")
    draw.line([(cx - 120, cy + 160), (cx + 120, cy + 280)], fill="white", width=24)
    draw.line([(cx - 120, cy + 280), (cx + 120, cy + 160)], fill="white", width=24)
    return img


def poison_sign() -> Image.Image:
    """Hard negative: a minimal poison/toxic hazard placard icon - skull in
    a warning diamond, safety-sign style rather than an insignia."""
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    diamond = [(cx, cy - 200), (cx + 200, cy), (cx, cy + 200), (cx - 200, cy)]
    draw.polygon(diamond, outline="black", width=14)
    r = 70
    draw.ellipse([cx - r, cy - r - 20, cx + r, cy + r - 20], outline="black", width=10)
    draw.ellipse([cx - 35, cy - 10, cx - 5, cy + 20], fill="black")
    draw.ellipse([cx + 5, cy - 10, cx + 35, cy + 20], fill="black")
    draw.line([(cx - 90, cy + 110), (cx + 90, cy + 200)], fill="black", width=10)
    draw.line([(cx - 90, cy + 200), (cx + 90, cy + 110)], fill="black", width=10)
    return img


def halloween_skull() -> Image.Image:
    """Hard negative: a rounded cartoon Halloween-decoration skull - no
    crossbones, no insignia framing."""
    img, draw = _canvas()
    cx, cy = SIZE // 2, SIZE // 2
    draw.ellipse([cx - 130, cy - 150, cx + 130, cy + 100], fill="black")
    r = 36
    draw.ellipse([cx - 80, cy - 40, cx - 80 + 2 * r, cy - 40 + 2 * r], fill="white")
    draw.ellipse([cx + 80 - 2 * r, cy - 40, cx + 80, cy - 40 + 2 * r], fill="white")
    draw.polygon([(cx - 15, cy + 30), (cx + 15, cy + 30), (cx, cy + 65)], fill="white")
    for i in range(5):
        x = cx - 60 + i * 30
        draw.rectangle([x, cy + 90, x + 14, cy + 120], fill="white")
    return img


def plain_tricolor() -> Image.Image:
    """Hard negative for imperial_war_flag: same black/white/red bands, no iron cross."""
    img = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, SIZE, SIZE // 3], fill="black")
    draw.rectangle([0, 2 * SIZE // 3, SIZE, SIZE], fill=(221, 0, 0))
    return img


def plain_black_flag() -> Image.Image:
    """Hard negative for islamic_state_flag: other black flags, no script/seal -
    an abstract flowing mark stands in for 'ordinary calligraphy', since
    rendering specific invented script text isn't appropriate here."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE // 2, SIZE // 2 - 40
    pts = [(cx - 160, cy), (cx - 60, cy - 60), (cx + 40, cy + 30), (cx + 160, cy - 20)]
    draw.line(pts, fill="white", width=16, joint="curve")
    draw.ellipse([cx - 60, cy + 120, cx + 60, cy + 200], outline="white", width=10)
    return img


def palestinian_flag() -> Image.Image:
    """Hard negative for hamas_flag: the Palestinian national flag itself,
    not a militant-group flag."""
    img = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, SIZE, SIZE // 3], fill="black")
    draw.rectangle([0, 2 * SIZE // 3, SIZE, SIZE], fill=(0, 122, 61))
    draw.polygon([(0, 0), (0, SIZE), (SIZE // 3, SIZE // 2)], fill=(206, 17, 38))
    return img


def hezbollah_flag_approx() -> Image.Image:
    """Hezbollah's real flag/logo is not freely licensed anywhere we could find
    (Commons only hosts a restricted non-free placeholder, not reusable here),
    so this is an original, hand-drawn approximation of its most recognisable
    elements (yellow field, raised fist gripping a rifle, globe) - not a copy
    of the group's actual artwork. Flag as approximate wherever it's used."""
    img = Image.new("RGBA", (SIZE, SIZE), (252, 209, 22, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE // 2, SIZE // 2 + 20
    draw.ellipse([cx - 90, cy - 20, cx + 90, cy + 160], outline="black", width=14)
    draw.rectangle([cx - 35, cy - 120, cx + 35, cy + 10], fill="black")  # forearm
    draw.ellipse([cx - 55, cy - 150, cx + 55, cy - 60], fill="black")  # fist
    draw.line([(cx - 10, cy - 180), (cx + 120, cy - 260)], fill="black", width=20)  # rifle
    return img


def plain_yellow_flag() -> Image.Image:
    """Hard negative for hezbollah_flag: plain yellow field, no fist/rifle/globe."""
    return Image.new("RGBA", (SIZE, SIZE), (252, 209, 22, 255))


def lebanese_flag() -> Image.Image:
    """Hard negative for hezbollah_flag: the Lebanese national flag (brief:
    'Lebanese national flag') - red-white-red bands with a simplified cedar
    silhouette, the country's own flag rather than the militant group's."""
    img = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    band = SIZE // 4
    red = (237, 41, 57)
    draw.rectangle([0, 0, SIZE, band], fill=red)
    draw.rectangle([0, SIZE - band, SIZE, SIZE], fill=red)
    cx, cy = SIZE // 2, SIZE // 2
    green = (0, 122, 61)
    for i, w in enumerate([140, 110, 80]):
        top = cy - 90 + i * 55
        draw.polygon([(cx, top), (cx - w, top + 70), (cx + w, top + 70)], fill=green)
    draw.rectangle([cx - 15, cy + 100, cx + 15, cy + 140], fill=green)
    return img


def kurdistan_regional_flag() -> Image.Image:
    """Hard negative for pkk_flag: the Kurdistan Region's own flag (a
    legitimate regional government flag), not a militant party flag."""
    img = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, SIZE, SIZE // 3], fill=(206, 17, 38))
    draw.rectangle([0, 2 * SIZE // 3, SIZE, SIZE], fill=(0, 123, 76))
    cx, cy = SIZE // 2, SIZE // 2
    draw.ellipse([cx - 60, cy - 60, cx + 60, cy + 60], fill=(255, 205, 0))
    for i in range(21):
        angle = math.radians(i * 360 / 21)
        x1 = cx + 62 * math.cos(angle)
        y1 = cy + 62 * math.sin(angle)
        x2 = cx + 80 * math.cos(angle)
        y2 = cy + 80 * math.sin(angle)
        draw.line([(x1, y1), (x2, y2)], fill=(255, 205, 0), width=4)
    return img


Builder = Callable[[], Image.Image]

# "hard_negative" may be a single builder or a list of them - generate_dataset.py
# splits that symbol/label's variant budget evenly across however many there are,
# so every hard-negative category the brief calls out gets real coverage instead
# of only the first one found.
SYMBOLS: dict[str, dict[str, Builder | list[Builder]]] = {
    "swastika": {"positive": swastika, "hard_negative": [swastika_religious, swastika_ornament_tile]},
    "ss_runes": {"positive": ss_runes, "hard_negative": lightning_warning},
    "ss_deaths_head": {
        "positive": ss_deaths_head,
        "hard_negative": [pirate_flag_skull, poison_sign, halloween_skull],
    },
    "black_sun": {"positive": black_sun, "hard_negative": [plain_sun, decorative_wheel]},
    "odal_rune": {
        "positive": odal_rune,
        "hard_negative": [lambda: odal_rune(serifs=False), odal_rune_jewellery],
    },
    "wolfsangel": {"positive": wolfsangel, "hard_negative": [wolfsangel_heraldry, fishhook]},
    "celtic_cross": {"positive": celtic_cross_extremist, "hard_negative": grave_cross},
}
