"""Procedural stand-in 'photo' backgrounds (no personal data, no faces, no
downloaded images - generated purely from code, so nothing to license).
Used for the 'symbol on a photo' variant. Should be supplemented with real
licensed photo backgrounds later for a more realistic test set.
"""

import random

import numpy as np
from PIL import Image

SIZE = 512


def _gradient(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for y in range(SIZE):
        t = y / SIZE
        arr[y, :] = [int(c1[i] * (1 - t) + c2[i] * t) for i in range(3)]
    return Image.fromarray(arr)


def sky() -> Image.Image:
    return _gradient((135, 206, 235), (255, 255, 255))


def brick() -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), (178, 94, 68))
    arr = np.array(img)
    for y in range(0, SIZE, 40):
        arr[y : y + 3, :] = (90, 60, 50)
    for y in range(0, SIZE, 40):
        offset = 0 if (y // 40) % 2 == 0 else 60
        for x in range(offset, SIZE, 120):
            arr[y : y + 40, x : x + 3] = (90, 60, 50)
    return Image.fromarray(arr)


def fabric_noise() -> Image.Image:
    rng = np.random.default_rng()
    base = rng.integers(140, 180, size=(SIZE, SIZE, 1), dtype=np.uint8)
    arr = np.repeat(base, 3, axis=2).astype(np.int16)
    arr += rng.integers(-15, 15, size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def wall() -> Image.Image:
    return _gradient((200, 200, 195), (150, 150, 145))


GENERATORS = [sky, brick, fabric_noise, wall]


def random_background() -> Image.Image:
    return random.choice(GENERATORS)()
