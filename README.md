# symbol-detector

Task 4 (banned symbols in text and images). This part covers the image
dataset (SCRUM-51) and its hard negatives + split (SCRUM-52): synthetic
variants of each symbol, plus hard negatives and easy negatives, for a
detector's train/val/locked-test split.

## What's done

All 12 symbols from `symbols-for-images.csv` now have a positive source and
hard negative(s) matching the brief's `hard_negatives_to_include` column
(as of 2026-11-05 - SS death's head was missing entirely until today, and
the Wolfsangel/Hezbollah hard negatives didn't match the brief; see git
history for the fix):

- **7 drawn as line art in code** (`src/symbol_detector/shapes.py`), no
  licence needed: swastika, SS runes, SS death's head, Black Sun, Odal
  rune, Wolfsangel, Celtic cross (extremist style).
- **4 real flags sourced from Wikimedia Commons**, public domain, logged in
  `data/licenses.csv`: Imperial war flag, Islamic State flag, Hamas flag,
  PKK flag (1978-1995 version).
- **1 hand-drawn approximation**: Hezbollah's actual flag/logo isn't freely
  licensed anywhere we could find (Commons only hosts a restricted non-free
  placeholder), so `hezbollah_flag_approx()` draws an original
  approximation of its recognisable elements instead - not a copy of the
  group's real artwork. Flagged as approximate in `data/licenses.csv`.

Several symbols now have **more than one hard negative** where the brief
lists more than one look-alike category (e.g. swastika: religious swastika
*and* an ornamental tile pattern; Odal rune: plain rune *and* a jewellery
pendant rendering; SS death's head: pirate flag, poison sign, *and*
Halloween skull). `generate_dataset.py` splits that symbol's 25-image
budget evenly across however many hard-negative builders it has, so every
listed category gets real coverage instead of only the first one found.

`generate_dataset.py` turns each into variants (rotated, scaled small,
partly occluded, composited onto a procedural background; the drawn
symbols are also recoloured - the real/approximate flags are not, since
their actual colours are part of the symbol) plus 200 generic easy
negatives (unrelated shapes).

Current output: **800 images, 300 positive / 500 negative** in
`data/images/`, logged in `data/manifest.csv`.

```bash
uv sync
uv run python -m symbol_detector.generate_dataset
```

### Train / validation / locked-test split

```bash
uv run python -m symbol_detector.split_dataset
```

Adds a `split` column to `data/manifest.csv` (train/val/test) - no images
are moved. Splits by **group**, not by individual image: every image in a
group is a rotated/scaled/recoloured/occluded copy of the *same* one drawn
or sourced artwork, so letting siblings land on both sides of train/test
would leak (the model could recognise the exact base image instead of
generalising). The script asserts no group ever crosses a split boundary,
and fails loudly if it would.

Locked test set requirement (SCRUM-52): 500+ images, at least half
negative, never used for training. Current split:

- **train**: 246 images (125 positive, 121 negative)
- **val**: 47 images (25 positive, 22 negative)
- **test**: 507 images (150 positive, 357 negative) - meets the 500+/50%+
  negative floor

Test is deliberately built negative-heavy (~70%) rather than an even 50/50,
because this dataset only has 300 positive images total - splitting them
evenly would leave very few for training. **Known limitation**: with the
brief's test-set floor this large relative to the whole corpus, train only
gets 125 positive images - more positive source material (beyond the 12
symbols drawn/sourced so far) would materially help before training a real
model on this.

## What's still worth doing

- Add more positive source variety - right now each symbol has exactly one
  drawn/sourced base image (just transformed 25+ ways), which caps how much
  the model can learn to generalise versus memorise.
- Add more real licensed background photos (currently procedural
  gradients/textures/noise, not real photos) for a more realistic test set.
- Double-check the PKK flag's 1978-1995 design is still the right reference
  if a later redesign exists.
- Revisit the Hezbollah approximation if a real freely-licensed source
  turns up later.

No personal data or faces are used anywhere in this dataset - everything is
either generated shapes, real flags (which depict no people), or procedural
backgrounds.
