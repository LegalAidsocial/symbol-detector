# symbol-detector

Task 4 (banned symbols in text and images). This part covers the image
dataset (SCRUM-51): synthetic variants of each symbol, plus hard negatives
and easy negatives, for a detector's locked test set.

## What's done

All 12 symbols from `symbols-for-images.csv` now have a positive source and
a matched hard negative (the look-alike from the brief's
`hard_negatives_to_include` column):

- **6 drawn as line art in code** (`src/symbol_detector/shapes.py`), no
  licence needed: swastika, SS runes, Black Sun, Odal rune, Wolfsangel,
  Celtic cross (extremist style).
- **4 real flags sourced from Wikimedia Commons**, public domain, logged in
  `data/licenses.csv`: Imperial war flag, Islamic State flag, Hamas flag,
  PKK flag (1978-1995 version).
- **1 hand-drawn approximation**: Hezbollah's actual flag/logo isn't freely
  licensed anywhere we could find (Commons only hosts a restricted non-free
  placeholder), so `hezbollah_flag_approx()` draws an original
  approximation of its recognisable elements instead - not a copy of the
  group's real artwork. Flagged as approximate in `data/licenses.csv`.

`generate_dataset.py` turns each into 25 variants per symbol per label
(rotated, scaled small, partly occluded, composited onto a procedural
background; the 6 drawn symbols are also recoloured - the real/approximate
flags are not, since their actual colours are part of the symbol) plus 200
generic easy negatives (unrelated shapes).

Current output: **750 images, 275 positive / 475 negative** in
`data/images/`, logged in `data/manifest.csv`.

```bash
uv sync
uv run python -m symbol_detector.generate_dataset
```

## What's still worth doing

- Add more real licensed background photos (currently procedural
  gradients/textures/noise, not real photos) for a more realistic test set.
- Double-check the PKK flag's 1978-1995 design is still the right reference
  if a later redesign exists.
- Revisit the Hezbollah approximation if a real freely-licensed source
  turns up later.

No personal data or faces are used anywhere in this dataset - everything is
either generated shapes, real flags (which depict no people), or procedural
backgrounds.
