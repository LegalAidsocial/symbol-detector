"""SCRUM-142: retrain yolov8n-cls with real-photo hard negatives added to the
"none" class (data/images/real_negative/, 147 manually-reviewed real photos
from Fahim's cleanup dataset - see data/manifest.csv rows with
label=="real_negative"). Same hyperparameters as SCRUM-53's original training
run for a fair before/after comparison.

Usage (from repo root):
    uv run python -m symbol_detector.prepare_yolo_dataset
    uv run --with ultralytics --with onnx --with onnxruntime python -m symbol_detector.train_real_negatives

Ultralytics ignores a relative `project=` value's exact placement and always
nests it under runs/<task>/ - actual output lands at
runs/classify/runs_scrum142/yolov8n_cls_symbols_realneg/weights/{best.pt,best.onnx}.
`runs/` is gitignored; the exported ONNX + names.json used for evaluation are
copied to models_candidate_scrum142/ (NOT models/ - that dir is the currently
deployed model and is deliberately left untouched pending a review decision).
"""

import json

from ultralytics import YOLO

RUN_DIR = "runs/classify/runs_scrum142/yolov8n_cls_symbols_realneg"


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

    best.export(format="onnx", imgsz=224, opset=12, simplify=True)
    names = best.names
    with open("models_candidate_scrum142/symbol_classifier.names.json", "w") as f:
        json.dump(names, f, indent=2)
    print(names)


if __name__ == "__main__":
    main()
