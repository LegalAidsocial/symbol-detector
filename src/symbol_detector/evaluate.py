"""SCRUM-57: run the trained classifier against SCRUM-52's locked test split
(manifest.csv, split=="test" - 507 images) and report precision/recall for
"banned" and images/s on CPU.

**This is NOT a true held-out evaluation - found by Hilda's cross-review,
SCRUM-68.** manifest.csv's own "split" column is ignored by the training
pipeline: prepare_yolo_dataset.py does its own fresh 70/15/15 per-image
split across all 800 images (a deliberate, documented trade-off there, since
only one base artwork exists per symbol), so roughly 70% of these same 507
"test" rows are actually in the training set the model was trained on. The
precision/recall this script reports is therefore inflated and should NOT
be read as a real-world accuracy number - it's useful only as a smoke test
(did training produce a model that fits its own training distribution at
all) and for relative before/after comparisons where both runs share the
same inflation. For an honest accuracy number on content the model has
never seen, use evaluate_real_photos.py's output instead - that's the real
headline metric.

Ground truth verdict per row follows prepare_yolo_dataset.target_class's
logic, restated in verdict terms:
    label == "positive"                                   -> banned
    label == "hard_negative" and symbol == swastika        -> needs_context
        and group ends with "_0" (the religious swastika)
    everything else (other hard_negatives, easy_negatives) -> clean

Writes every misclassification to failures.csv (path, true verdict,
predicted verdict/symbol/confidence) - the actual input list for the
"add examples, retrain" half of the failure loop.

Usage: uv run python -m symbol_detector.evaluate
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from symbol_detector.classify_image import (
    MODEL_PATH,
    NAMES_PATH,
    load_names,
    verdict_for,
)

MANIFEST = Path("data/manifest.csv")
FAILURES_OUT = Path("failures.csv")
IMG_SIZE = 224
VERDICTS = ("banned", "needs_context", "clean")


def ground_truth_verdict(row: dict[str, str]) -> str:
    if row["label"] == "positive":
        return "banned"
    if (
        row["label"] == "hard_negative"
        and row["symbol"] == "swastika"
        and row["group"].endswith("_0")
    ):
        return "needs_context"
    return "clean"


def preprocess_batch(paths: list[Path], img_size: int = IMG_SIZE) -> np.ndarray:
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((img_size, img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arrs.append(arr.transpose(2, 0, 1))
    return np.stack(arrs)


def main() -> None:
    with MANIFEST.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

    names = load_names(NAMES_PATH)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    confusion = {t: dict.fromkeys(VERDICTS, 0) for t in VERDICTS}
    failures = []

    start = time.perf_counter()
    for row in rows:
        x = preprocess_batch([Path(row["path"])])
        (probs,) = session.run(None, {input_name: x})
        probs = probs[0]
        top_idx = int(np.argmax(probs))
        symbol_class = names[top_idx]
        predicted = verdict_for(symbol_class)
        truth = ground_truth_verdict(row)
        confusion[truth][predicted] += 1
        if predicted != truth:
            failures.append(
                {
                    "path": row["path"],
                    "true_symbol": row["symbol"],
                    "true_label": row["label"],
                    "true_verdict": truth,
                    "predicted_symbol": symbol_class,
                    "predicted_verdict": predicted,
                    "confidence": round(float(probs[top_idx]), 4),
                }
            )
    elapsed = time.perf_counter() - start

    tp = confusion["banned"]["banned"]
    fp = sum(confusion[t]["banned"] for t in VERDICTS if t != "banned")
    fn = sum(confusion["banned"][p] for p in VERDICTS if p != "banned")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    images_per_sec = len(rows) / elapsed

    never_banned_violations = [
        f
        for f in failures
        if f["true_verdict"] != "banned" and f["predicted_verdict"] == "banned"
    ]

    with FAILURES_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "true_symbol",
                "true_label",
                "true_verdict",
                "predicted_symbol",
                "predicted_verdict",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(failures)

    report = {
        "CAVEAT": (
            "precision/recall here are inflated - ~70% of these rows are also "
            "in the training set (SCRUM-68). Use evaluate_real_photos.py for a "
            "real accuracy number."
        ),
        "n_test_images": len(rows),
        "precision_banned": round(precision, 4),
        "recall_banned": round(recall, 4),
        "images_per_sec": round(images_per_sec, 1),
        "meets_precision_95": precision >= 0.95,
        "meets_recall_90": recall >= 0.90,
        "meets_speed_20": images_per_sec >= 20,
        "never_banned_violations": len(never_banned_violations),
        "confusion_matrix(true_row x predicted_col)": confusion,
        "n_failures_logged": len(failures),
        "failures_file": str(FAILURES_OUT),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
