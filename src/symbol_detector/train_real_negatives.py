"""SCRUM-142: retrain yolov8n-cls with real-photo hard negatives added to the
"none" class (data/images/real_negative/, 147 manually-reviewed real photos
from Fahim's cleanup dataset - see data/manifest.csv rows with
label=="real_negative"). Same hyperparameters as SCRUM-53's original training
run for a fair before/after comparison.

Usage (from repo root):
    uv run python -m symbol_detector.prepare_yolo_dataset
    uv run --group training python -m symbol_detector.train_real_negatives

Ultralytics ignores a relative `project=` value's exact placement and always
nests it under runs/<task>/ - actual output lands at
runs/classify/runs_scrum142/yolov8n_cls_symbols_realneg/weights/{best.pt,best.onnx}.
`runs/` is gitignored; the exported ONNX + names.json used for evaluation are
copied to CANDIDATE_DIR, a staging directory separate from models/ (the
currently deployed model) - promoting a candidate into models/ is a
deliberate, separate step, never done automatically by this script.

This round's candidate (models_candidate_scrum142/) was reviewed and
promoted into models/ on 26 Nov (see dsa-moderation-monitor PR #14,
merged 23 Nov, which already deployed it to the live service) - update
CANDIDATE_DIR's name for the next retraining round rather than reusing
this one.
"""

import json
import shutil
from pathlib import Path

from ultralytics import YOLO

RUN_DIR = "runs/classify/runs_scrum142/yolov8n_cls_symbols_realneg"
CANDIDATE_DIR = Path("models_candidate_scrum142")


def main() -> None:
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data="yolo_dataset",
        epochs=60,
        imgsz=224,
        patience=15,
        project="runs_scrum142",
        name="yolov8n_cls_symbols_realneg",
        seed=20261109,
    )

    best = YOLO(f"{RUN_DIR}/weights/best.pt")
    test_metrics = best.val(data="yolo_dataset", split="test")
    print("top1 accuracy on fresh per-image split test:", test_metrics.top1)
    print("top5 accuracy on fresh per-image split test:", test_metrics.top5)

    onnx_path = best.export(format="onnx", imgsz=224, opset=12, simplify=True)
    CANDIDATE_DIR.mkdir(exist_ok=True)
    shutil.copy2(onnx_path, CANDIDATE_DIR / "symbol_classifier.onnx")
    names = best.names
    with open(CANDIDATE_DIR / "symbol_classifier.names.json", "w") as f:
        json.dump(names, f, indent=2)
    print(names)


if __name__ == "__main__":
    main()
