"""Run the trained symbol classifier (ONNX, CPU) on an image and map its
prediction to a verdict: banned / needs_context / clean.

SCRUM-53: a religious swastika (class "swastika_religious") is reported as
needs_context, never banned. Every other symbol class is banned; "none" is
clean.

Usage: uv run python -m symbol_detector.classify_image path/to/image.jpg
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = Path("models/symbol_classifier.onnx")
NAMES_PATH = Path("models/symbol_classifier.names.json")
IMG_SIZE = 224

NEEDS_CONTEXT_CLASSES = {"swastika_religious"}
CLEAN_CLASSES = {"none"}


def load_names(names_path: Path = NAMES_PATH) -> dict[int, str]:
    raw = json.loads(names_path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def preprocess(image_path: Path, img_size: int = IMG_SIZE) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0)  # add batch dim


def verdict_for(symbol_class: str) -> str:
    if symbol_class in CLEAN_CLASSES:
        return "clean"
    if symbol_class in NEEDS_CONTEXT_CLASSES:
        return "needs_context"
    return "banned"


def classify(
    image_path: Path,
    model_path: Path = MODEL_PATH,
    names_path: Path = NAMES_PATH,
) -> dict[str, object]:
    names = load_names(names_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = preprocess(image_path)
    (probs,) = session.run(None, {input_name: x})
    probs = probs[0]
    top_idx = int(np.argmax(probs))
    symbol_class = names[top_idx]
    return {
        "symbol": symbol_class,
        "confidence": float(probs[top_idx]),
        "verdict": verdict_for(symbol_class),
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python -m symbol_detector.classify_image <image_path>")
        sys.exit(1)
    result = classify(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
