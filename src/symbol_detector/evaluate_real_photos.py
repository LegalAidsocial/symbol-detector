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
from pathlib import Path

import onnxruntime as ort
import pillow_heif

from symbol_detector.classify_image import load_names, preprocess, verdict_for

pillow_heif.register_heif_opener()

DEFAULT_MODEL = Path("models_candidate_scrum142/symbol_classifier.onnx")
DEFAULT_NAMES = Path("models_candidate_scrum142/symbol_classifier.names.json")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image_dir")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--names", type=Path, default=DEFAULT_NAMES)
    args = p.parse_args()

    names = load_names(args.names)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    files = sorted(
        f for f in Path(args.image_dir).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        print(f"no images found in {args.image_dir}")
        return

    counts: dict[str, int] = {}
    flagged = []
    errors = []
    for f in files:
        try:
            x = preprocess(f)
        except Exception as e:
            errors.append((f.name, f"{type(e).__name__}: {e}"))
            continue
        (probs,) = session.run(None, {input_name: x})
        probs = probs[0]
        top = int(probs.argmax())
        symbol = names[top]
        verdict = verdict_for(symbol)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict == "banned":
            flagged.append((f.name, symbol, float(probs[top])))

    total = len(files) - len(errors)
    print(f"n={total}  counts={counts}")
    if total:
        banned = counts.get("banned", 0)
        print(f"banned rate: {banned}/{total} = {banned / total * 100:.1f}%")
    for name, symbol, conf in flagged:
        print(f"  flagged: {name}  {symbol}  {conf * 100:.1f}%")
    for name, error in errors:
        print(f"  unreadable, skipped: {name}  {error}")


if __name__ == "__main__":
    main()
