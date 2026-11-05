"""Split data/manifest.csv into train/val/test and write the split back as a
new "split" column - no images are moved, the manifest just gains a column.

Splits by GROUP, never by individual image: every image in a group is a
rotated/scaled/recolored/occluded copy of the same one drawn or sourced
artwork (see generate_dataset.py), so letting siblings land on both sides of
train/test would leak - the model could "recognise" the exact base image
instead of generalising to an unseen rendering. Each group goes to exactly
one split.

Locked test set requirement (SCRUM-52): 500+ images, at least half negative,
never used for training. Within that floor, test is built negative-heavy
(targets 30% positive / 70% negative) on purpose - this dataset only has 300
positive images total, so giving test a full 50/50 share would leave very
few positives for training. The remainder (whatever positive/negative groups
aren't in test) is split 80/20 between train and validation.

Usage: uv run python -m symbol_detector.split_dataset
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

MANIFEST_PATH = Path("data/manifest.csv")
SEED = 20261105  # fixed for a reproducible split

TEST_MIN_TOTAL = 500
TEST_POSITIVE_SHARE = 0.3  # of the test set itself, not of all positives
VAL_SHARE_OF_REMAINDER = 0.2


def _is_positive(label: str) -> bool:
    return label == "positive"


def _group_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    return groups


def assign_splits(rows: list[dict], seed: int = SEED) -> dict[str, str]:
    """Return {group: split} for every group in rows."""
    groups = _group_groups(rows)
    pos_groups = [g for g, r in groups.items() if _is_positive(r[0]["label"])]
    neg_groups = [g for g, r in groups.items() if not _is_positive(r[0]["label"])]

    rng = random.Random(seed)
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    total_positive = sum(len(groups[g]) for g in pos_groups)

    # Greedily add whole groups to test until both the size floor and the
    # negative-share target are met. Positive groups are capped so test
    # doesn't end up starving train of positives; negative supply is large
    # enough (500 total) that the floor is reached from negatives alone if
    # needed, but we mix in some positives on purpose so test actually
    # measures true-positive detection, not just true-negative rejection.
    test_groups: set[str] = set()
    test_pos = test_neg = 0
    target_pos = int(TEST_MIN_TOTAL * TEST_POSITIVE_SHARE)

    for g in pos_groups:
        if test_pos >= target_pos:
            break
        test_groups.add(g)
        test_pos += len(groups[g])

    for g in neg_groups:
        if test_pos + test_neg >= TEST_MIN_TOTAL and test_neg >= test_pos:
            break
        test_groups.add(g)
        test_neg += len(groups[g])

    assert test_pos + test_neg >= TEST_MIN_TOTAL, "test set came up short of the 500-image floor"
    assert test_neg >= (test_pos + test_neg) / 2, "test set has under 50% negatives"
    assert test_pos < total_positive, "test ate every positive image - nothing left to train on"

    remaining = [g for g in (*pos_groups, *neg_groups) if g not in test_groups]
    rng.shuffle(remaining)
    n_val = int(len(remaining) * VAL_SHARE_OF_REMAINDER)
    val_groups = set(remaining[:n_val])

    assignment: dict[str, str] = {}
    for g in groups:
        if g in test_groups:
            assignment[g] = "test"
        elif g in val_groups:
            assignment[g] = "val"
        else:
            assignment[g] = "train"
    return assignment


def main() -> None:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assignment = assign_splits(rows)
    for row in rows:
        row["split"] = assignment[row["group"]]

    fieldnames = [*rows[0].keys()]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for split in ("train", "val", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        pos = sum(1 for r in split_rows if _is_positive(r["label"]))
        neg = len(split_rows) - pos
        print(f"{split}: {len(split_rows)} images ({pos} positive, {neg} negative)")

    # Group leakage check: no group should ever appear in more than one split.
    group_splits = defaultdict(set)
    for row in rows:
        group_splits[row["group"]].add(row["split"])
    leaky = {g: s for g, s in group_splits.items() if len(s) > 1}
    assert not leaky, f"group(s) split across train/val/test: {leaky}"
    print(f"Updated {MANIFEST_PATH} with a 'split' column - no group crosses a split boundary.")


if __name__ == "__main__":
    main()
