"""Build the synthetic image dataset: own variants (rotated, small, partly
covered, on photos, different colours) for each symbol in shapes.SYMBOLS,
plus generic easy negatives. Everything here is generated from code - no
downloaded or copied images, so there is nothing to license for this part.
The flag-based symbols still need real licensed reference images; see
data/licenses.csv and README.md.

Usage: uv run python -m symbol_detector.generate_dataset
"""

import csv
import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from symbol_detector.backgrounds import random_background
from symbol_detector.shapes import (
    SIZE,
    SYMBOLS,
    hezbollah_flag_approx,
    kurdistan_regional_flag,
    palestinian_flag,
    plain_black_flag,
    plain_tricolor,
    plain_yellow_flag,
)

OUT_DIR = Path("data/images")
MANIFEST_PATH = Path("data/manifest.csv")
RAW_LICENSED_DIR = Path("data/raw_licensed")
VARIANTS_PER_CLASS = 25  # positive + hard_negative, per symbol
EASY_NEGATIVES = 200

# Real licensed reference images: symbol -> (positive image loader, hard negative builder).
# Not recolored like the drawn line-art symbols - a real flag's colors are part of what
# makes it that symbol, so changing them isn't a realistic augmentation.
REAL_SYMBOLS: dict[str, dict[str, Callable[[], Image.Image]]] = {
    "imperial_war_flag": {
        "positive": lambda: Image.open(RAW_LICENSED_DIR / "imperial_war_flag" / "original.png").convert(
            "RGBA"
        ),
        "hard_negative": plain_tricolor,
    },
    "islamic_state_flag": {
        "positive": lambda: Image.open(RAW_LICENSED_DIR / "islamic_state_flag" / "original.png").convert(
            "RGBA"
        ),
        "hard_negative": plain_black_flag,
    },
    "hamas_flag": {
        "positive": lambda: Image.open(RAW_LICENSED_DIR / "hamas_flag" / "original.png").convert("RGBA"),
        "hard_negative": palestinian_flag,
    },
    "pkk_flag": {
        "positive": lambda: Image.open(RAW_LICENSED_DIR / "pkk_flag" / "original.png").convert("RGBA"),
        "hard_negative": kurdistan_regional_flag,
    },
}

# Not sourced from anywhere - Hezbollah's actual flag/logo isn't freely licensed
# on Commons (only a restricted non-free placeholder exists), so this is an
# original hand-drawn approximation instead. See shapes.hezbollah_flag_approx.
APPROX_SYMBOLS: dict[str, dict[str, Callable[[], Image.Image]]] = {
    "hezbollah_flag": {"positive": hezbollah_flag_approx, "hard_negative": plain_yellow_flag},
}

COLORS = [(0, 0, 0), (255, 255, 255), (200, 0, 0), (0, 0, 150), (40, 40, 40)]


def recolor(img: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    arr = np.array(img)
    alpha = arr[:, :, 3]
    out = np.zeros_like(arr)
    out[:, :, 0] = color[0]
    out[:, :, 1] = color[1]
    out[:, :, 2] = color[2]
    out[:, :, 3] = alpha
    return Image.fromarray(out)


def occlude(img: Image.Image) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w = random.randint(SIZE // 4, SIZE // 2)
    h = random.randint(SIZE // 4, SIZE // 2)
    x = random.randint(0, SIZE - w)
    y = random.randint(0, SIZE - h)
    draw.rectangle([x, y, x + w, y + h], fill=(128, 128, 128, 255))
    return img


def composite_on_background(img: Image.Image, scale: float = 1.0) -> Image.Image:
    bg = random_background().convert("RGBA")
    fg = img.resize((int(SIZE * scale), int(SIZE * scale)))
    max_x, max_y = SIZE - fg.width, SIZE - fg.height
    x = random.randint(0, max(max_x, 0))
    y = random.randint(0, max(max_y, 0))
    bg.alpha_composite(fg, (x, y))
    return bg.convert("RGB")


def make_variant(base: Image.Image, kind: str, allow_recolor: bool = True) -> Image.Image:
    img = recolor(base, random.choice(COLORS)) if allow_recolor else base.resize((SIZE, SIZE))
    if kind == "rotated":
        img = img.rotate(random.choice([15, 30, 60, 90, 120, 180, -20, -45]), expand=False)
        return composite_on_background(img)
    if kind == "small":
        return composite_on_background(img, scale=random.uniform(0.15, 0.35))
    if kind == "occluded":
        img = occlude(img)
        return composite_on_background(img)
    if kind == "on_photo":
        return composite_on_background(img)
    if kind == "color":
        return composite_on_background(img)
    raise ValueError(kind)


def easy_negative_shape() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = random.choice(COLORS)
    shape = random.choice(["circle", "square", "triangle", "star", "check", "line"])
    cx, cy = SIZE // 2, SIZE // 2
    if shape == "circle":
        draw.ellipse([cx - 150, cy - 150, cx + 150, cy + 150], outline=color, width=20)
    elif shape == "square":
        draw.rectangle([cx - 140, cy - 140, cx + 140, cy + 140], outline=color, width=20)
    elif shape == "triangle":
        draw.polygon([(cx, cy - 160), (cx - 160, cy + 120), (cx + 160, cy + 120)], outline=color, width=20)
    elif shape == "star":
        pts = []
        for i in range(10):
            r = 170 if i % 2 == 0 else 70
            angle = math.pi / 5 * i - math.pi / 2
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(pts, outline=color, width=14)
    elif shape == "check":
        draw.line([(cx - 120, cy), (cx - 20, cy + 100), (cx + 150, cy - 130)], fill=color, width=28)
    else:
        draw.line([(cx - 180, cy), (cx + 180, cy)], fill=color, width=20)
    return composite_on_background(img, scale=random.uniform(0.4, 1.0))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    sources = [
        (SYMBOLS, True, "synthetic (generated, no license needed)"),
        (REAL_SYMBOLS, False, None),
        (APPROX_SYMBOLS, False, "hand-drawn approximation, not the group's actual artwork - see shapes.py"),
    ]
    for symbol_dict, allow_recolor, fixed_source in sources:
        for symbol, kinds in symbol_dict.items():
            for label, builder in kinds.items():
                folder = OUT_DIR / symbol / label
                folder.mkdir(parents=True, exist_ok=True)
                base = builder()
                variant_kinds = (
                    ["rotated", "small", "occluded", "on_photo", "color"]
                    if allow_recolor
                    else ["rotated", "small", "occluded", "on_photo", "on_photo"]
                )
                source = fixed_source or "see data/licenses.csv"
                for i in range(VARIANTS_PER_CLASS):
                    kind = variant_kinds[i % len(variant_kinds)]
                    out = make_variant(base, kind, allow_recolor=allow_recolor)
                    fname = f"{symbol}_{label}_{kind}_{i:03d}.jpg"
                    out.convert("RGB").save(folder / fname, quality=90)
                    rows.append(
                        {
                            "path": str((folder / fname).as_posix()),
                            "symbol": symbol,
                            "label": label,
                            "variant": kind,
                            "source": source,
                        }
                    )

    neg_folder = OUT_DIR / "easy_negative"
    neg_folder.mkdir(parents=True, exist_ok=True)
    for i in range(EASY_NEGATIVES):
        out = easy_negative_shape()
        fname = f"easy_negative_{i:03d}.jpg"
        out.save(neg_folder / fname, quality=90)
        rows.append(
            {
                "path": str((neg_folder / fname).as_posix()),
                "symbol": "none",
                "label": "easy_negative",
                "variant": "random_shape",
                "source": "synthetic (generated, no license needed)",
            }
        )

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "symbol", "label", "variant", "source"])
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(1 for r in rows if r["label"] == "positive")
    negatives = len(rows) - positives
    print(f"Wrote {len(rows)} images ({positives} positive, {negatives} negative) to {OUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
