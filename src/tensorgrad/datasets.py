"""Datasets for the examples and tests, and a character tokenizer.

The spirals and shapes are generated procedurally. :func:`tiny_shakespeare` downloads Andrej
Karpathy's ~1.1 MB char-rnn corpus once, verifies its SHA-256 and caches it (see
:func:`fetch`); nothing else touches the network.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from tensorgrad._types import Array

__all__ = [
    "SHAPE_CLASSES",
    "TINY_SHAKESPEARE_SHA256",
    "TINY_SHAKESPEARE_URL",
    "CharTokenizer",
    "cache_dir",
    "fetch",
    "make_shapes",
    "make_spiral",
    "tiny_shakespeare",
]

#: Karpathy's tiny-shakespeare (40,000 lines of Shakespeare, 1,115,394 characters).
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
#: SHA-256 of that file, pinned so a changed or corrupted download is rejected.
TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def cache_dir() -> Path:
    """Where downloads are cached: ``$TENSORGRAD_CACHE``, else ``$XDG_CACHE_HOME/tensorgrad``,
    else ``~/.cache/tensorgrad``."""
    if explicit := os.environ.get("TENSORGRAD_CACHE"):
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "tensorgrad"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    url: str, sha256: str, filename: str, *, cache: Path | None = None, timeout: float = 60.0
) -> Path:
    """Return a verified local copy of ``url``, downloading it on first use.

    The file lives at ``cache / filename`` (default :func:`cache_dir`). A cached copy whose
    SHA-256 does not match is discarded and downloaded again; a download that does not match
    raises ``ValueError`` and leaves nothing behind. Writes go to a temporary file that is
    renamed into place, so an interrupted download never leaves a truncated file.
    """
    path = (cache if cache is not None else cache_dir()) / filename
    if path.exists():
        if _sha256(path) == sha256:
            return path
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(partial, "wb") as fh:
            shutil.copyfileobj(response, fh)
        actual = _sha256(partial)
        if actual != sha256:
            raise ValueError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def tiny_shakespeare(cache: Path | None = None) -> str:
    """Text of the tiny-shakespeare corpus (downloaded and verified once, then cached)."""
    path = fetch(TINY_SHAKESPEARE_URL, TINY_SHAKESPEARE_SHA256, "tinyshakespeare.txt", cache=cache)
    return path.read_text(encoding="utf-8")


def make_spiral(
    n_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.2,
    seed: int | None = None,
) -> tuple[Array, Array]:
    """Interleaved 2-D spiral arms, the classic "not linearly separable" toy problem.

    Returns:
        ``(x, y)``: float32 points of shape ``(n_per_class * n_classes, 2)`` in roughly
        ``[-1, 1]^2`` and int64 labels.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        jitter = rng.standard_normal(n_per_class) * noise
        theta = np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + jitter
        xs.append(np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1))
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    points: Array = np.concatenate(xs).astype(np.float32)
    labels: Array = np.concatenate(ys)
    return points, labels


#: Class names of :func:`make_shapes`, indexed by label.
SHAPE_CLASSES = ("circle", "square", "triangle", "cross")


def _segment_distance(px: Array, py: Array, a: Array, b: Array) -> Array:
    """Distance from every grid point to the segment ``a-b``."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    t = ((px - a[0]) * abx + (py - a[1]) * aby) / (abx * abx + aby * aby)
    t = np.clip(t, 0.0, 1.0)
    dist: Array = np.hypot(px - (a[0] + t * abx), py - (a[1] + t * aby))
    return dist


def _polygon_edges(vertices: Array, closed: bool = True) -> Iterable[tuple[Array, Array]]:
    n = len(vertices)
    for i in range(n if closed else n - 1):
        yield vertices[i], vertices[(i + 1) % n]


def _draw(label: int, size: int, rng: np.random.Generator) -> Array:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx, cy = (size - 1) / 2 + rng.uniform(-2.0, 2.0, size=2)
    radius = rng.uniform(0.22, 0.36) * size
    angle = rng.uniform(-math.pi / 8, math.pi / 8)
    thickness = rng.uniform(0.8, 1.4)

    def rotate(points: Array) -> Array:
        c, s = math.cos(angle), math.sin(angle)
        return np.stack(
            [cx + c * points[:, 0] - s * points[:, 1], cy + s * points[:, 0] + c * points[:, 1]],
            axis=1,
        )

    if label == 0:  # circle outline
        dist = np.abs(np.hypot(xx - cx, yy - cy) - radius)
    else:
        if label == 1:  # square outline
            r = radius * 0.85
            verts = rotate(np.array([[-r, -r], [r, -r], [r, r], [-r, r]]))
            edges = list(_polygon_edges(verts))
        elif label == 2:  # equilateral triangle outline
            angles = np.array([-math.pi / 2, math.pi / 6, 5 * math.pi / 6])
            verts = rotate(np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1))
            edges = list(_polygon_edges(verts))
        elif label == 3:  # plus-shaped cross
            arms = rotate(np.array([[-radius, 0.0], [radius, 0.0], [0.0, -radius], [0.0, radius]]))
            edges = [(arms[0], arms[1]), (arms[2], arms[3])]
        else:
            raise ValueError(f"unknown shape label {label}")
        dist = np.min([_segment_distance(xx, yy, a, b) for a, b in edges], axis=0)
    image: Array = np.clip(1.0 - (dist - thickness / 2), 0.0, 1.0)  # anti-aliased stroke
    return image


def make_shapes(
    n: int = 1000, size: int = 16, noise: float = 0.1, seed: int | None = None
) -> tuple[Array, Array]:
    """Procedurally drawn grayscale images of circles, squares, triangles and crosses.

    Each image gets a random position jitter, scale, rotation and stroke width plus Gaussian
    pixel noise, so the classes have to be recognised by shape rather than by fixed pixels.

    Returns:
        ``(images, labels)``: float32 ``(n, 1, size, size)`` images in ``[0, 1]`` and int64
        labels in ``range(len(SHAPE_CLASSES))`` (balanced, shuffled).
    """
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.arange(n) % len(SHAPE_CLASSES)).astype(np.int64)
    images = np.stack([_draw(int(label), size, rng) for label in labels])
    images = np.clip(images + rng.normal(0.0, noise, size=images.shape), 0.0, 1.0)
    return images[:, None].astype(np.float32), labels


class CharTokenizer:
    """Maps each distinct character of a corpus to an integer id."""

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("cannot build a vocabulary from empty text")
        self.chars = sorted(set(text))
        self._index = {ch: i for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters."""
        return len(self.chars)

    def encode(self, text: str) -> Array:
        """Text -> int64 ids (raises ``KeyError`` on characters outside the vocabulary)."""
        return np.array([self._index[ch] for ch in text], dtype=np.int64)

    def decode(self, ids: Iterable[int] | Array) -> str:
        """Ids -> text."""
        return "".join(self.chars[int(i)] for i in np.asarray(ids).reshape(-1))
