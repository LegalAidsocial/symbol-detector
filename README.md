# symbol-detector

Task 4 (banned symbols in text and images). This part covers the image
dataset (SCRUM-51): synthetic variants of each symbol, plus hard negatives
and easy negatives, for a detector's locked test set.

## What's done

6 of 12 symbols from `symbols-for-images.csv` are simple enough to draw
accurately as line art in code (`src/symbol_detector/shapes.py`), so they
need no licensed source image: swastika, SS runes, Black Sun, Odal rune,
Wolfsangel, Celtic cross (extremist style). Each has a matched hard negative
(the look-alike from the brief's `hard_negatives_to_include` column) drawn
the same way.

`generate_dataset.py` turns each into 25 variants per symbol per label
(rotated, scaled small, partly occluded, composited onto a procedural
background, recoloured) plus 200 generic easy negatives (unrelated shapes).
Current output: **500 images, 150 positive / 350 negative** in
`data/images/`, logged in `data/manifest.csv`.

```bash
uv sync
uv run python -m symbol_detector.generate_dataset
```

## What's still needed

The 5 flag/banner symbols (Imperial war flag, Islamic State flag, Hamas,
Hezbollah, PKK) are too detailed to recreate accurately from code - they
need real reference images with the licence recorded, e.g. from Wikimedia
Commons. Tracked as "NOT YET SOURCED" in `data/licenses.csv`.

No personal data or faces are used anywhere in this dataset - everything is
generated shapes on procedural backgrounds.
