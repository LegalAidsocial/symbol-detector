"""Turn data/manifest.csv + data/images/ into a YOLO classification-mode
folder tree (one subfolder per class, per split) for yolov8n-cls training.

Classes: each of the 12 banned symbols (from a "positive" row), plus
"swastika_religious" (the swastika hard negative drawn upright with corner
dots - shapes.SYMBOLS["swastika"]["hard_negative"][0], group suffix "_0" -
SCRUM-53 requires this reported as needs_context, never banned, so it gets
its own class rather than being folded into "none"), plus "none" for every
other hard negative and all easy negatives.

Split: SCRUM-52's locked manifest.csv "split" column groups each symbol's
25 variants as a single group (only one drawn/sourced artwork per symbol),
so group-based splitting puts a whole class's images in only one of
train/val/test - unusable for training a classifier (most classes end up
completely absent from train). This script does its own stratified
per-image split instead, per class, so every class appears in all three.
That means variants of the same base artwork can land in both train and
test for single-source classes - a real accuracy-inflation trade-off,
accepted deliberately (see SCRUM-53 discussion) because there is currently
only one real/drawn example per symbol. manifest.csv and its locked split
are left untouched for whatever else relies on them.

Usage: uv run python -m symbol_detector.prepare_yolo_dataset
"""

import csv
import random
import shutil
from pathlib import Path

MANIFEST = Path("data/manifest.csv")
OUT_DIR = Path("yolo_dataset")
SEED = 20261109
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# base_idx 0 in shapes.SYMBOLS["swastika"]["hard_negative"] is
# swastika_religious (first in the list) - see generate_dataset.py's
# group = f"{symbol}_{label}_{base_idx}".
SWASTIKA_RELIGIOUS_GROUP_SUFFIX = "_0"


def target_class(row: dict[str, str]) -> str:
    if row["label"] == "positive":
        return row["symbol"]
    if row["symbol"] == "swastika" and row["label"] == "hard_negative":
        if row["group"].endswith(SWASTIKA_RELIGIOUS_GROUP_SUFFIX):
            return "swastika_religious"
    return "none"


def split_class_images(paths: list[Path]) -> dict[str, list[Path]]:
    rng = random.Random(SEED)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * TEST_FRACTION))
    n_val = max(1, round(n * VAL_FRACTION))
    n_test = min(n_test, n - 2)  # leave at least 2 for train+val
    n_val = min(n_val, n - n_test - 1)  # leave at least 1 for train
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return {"train": train, "val": val, "test": test}


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_class: dict[str, list[Path]] = {}
    for row in rows:
        by_class.setdefault(target_class(row), []).append(Path(row["path"]))

    counts: dict[str, int] = {}
    for cls, paths in sorted(by_class.items()):
        splits = split_class_images(paths)
        for split, split_paths in splits.items():
            dest_dir = OUT_DIR / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_paths:
                shutil.copy2(src, dest_dir / src.name)
            counts[f"{split}/{cls}"] = len(split_paths)

    print(f"Wrote {sum(counts.values())} images to {OUT_DIR}/")
    for split in ("train", "val", "test"):
        classes = sorted(c for c in by_class if (OUT_DIR / split / c).exists())
        print(f"{split}: {len(classes)} classes")
        for cls in classes:
            print(f"  {cls}: {counts[f'{split}/{cls}']}")


if __name__ == "__main__":
    main()
