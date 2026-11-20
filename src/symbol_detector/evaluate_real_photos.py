"""SCRUM-142: evaluate a model (default: the models_candidate_scrum142/
retrained one) against a directory of real photos, reporting the verdict
breakdown. Used to measure the real-photo false-positive rate the way
evaluate.py measures the locked synthetic test set - but this one has no
ground-truth labels (real photos pulled from Fahim's cleanup dataset, not a
curated positive/negative set), so it only reports the resulting verdict
counts/rate, not precision/recall.

Usage: uv run python -m symbol_detector.evaluate_real_photos <image_dir> [--model path] [--names path]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

IMG_SIZE = 224
DEFAULT_MODEL = Path("models_candidate_scrum142/symbol_classifier.onnx")
DEFAULT_NAMES = Path("models_candidate_scrum142/symbol_classifier.names.json")


def verdict_for(symbol_class: str) -> str:
    if symbol_class == "none":
        return "clean"
    if symbol_class == "swastika_religious":
        return "needs_context"
    return "banned"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image_dir")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--names", type=Path, default=DEFAULT_NAMES)
    args = p.parse_args()

    names = {int(k): v for k, v in json.loads(args.names.read_text()).items()}
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    files = sorted(Path(args.image_dir).iterdir())
    counts: dict[str, int] = {}
    flagged = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = np.expand_dims(arr.transpose(2, 0, 1), 0)
        (probs,) = session.run(None, {input_name: x})
        probs = probs[0]
        top = int(np.argmax(probs))
        symbol = names[top]
        verdict = verdict_for(symbol)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == "banned":
            flagged.append((f.name, symbol, float(probs[top])))

    total = len(files)
    print(f"n={total}  counts={counts}")
    banned = counts.get("banned", 0)
    print(f"banned rate: {banned}/{total} = {banned / total * 100:.1f}%")
    for name, symbol, conf in flagged:
        print(f"  flagged: {name}  {symbol}  {conf * 100:.1f}%")


if __name__ == "__main__":
    main()
