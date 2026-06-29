"""Helpers for the real-data SVM-GMU vs SVM-GSU experiments (Plan 1: MNIST).

Pure functions plus per-seed and full-experiment drivers. All numerical logic
lives here so the notebooks stay thin, mirroring experiments/_common.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


def augment_image(
    img28: NDArray[np.floating],
    rng: np.random.Generator,
    rot_deg: float,
    max_shift: float,
) -> NDArray[np.float64]:
    """Rotate a 28x28 image about its center, then translate by a small shift.

    Rotation keeps the output shape fixed; the shift is drawn uniformly in
    [-max_shift, max_shift] pixels per axis. Bilinear interpolation, zero fill.
    """
    out = ndimage.rotate(img28, rot_deg, reshape=False, order=1, mode="constant", cval=0.0)
    sx = rng.uniform(-max_shift, max_shift)
    sy = rng.uniform(-max_shift, max_shift)
    out = ndimage.shift(out, (sy, sx), order=1, mode="constant", cval=0.0)
    return out.astype(np.float64)


def augmentation_cloud(
    img_flat: NDArray[np.floating],
    n_aug: int,
    rng: np.random.Generator,
    rot_range: float,
    max_shift: float,
) -> NDArray[np.float64]:
    """Return an (n_aug, 784) cloud of flattened augmentations of one image.

    Rotation angle of each draw is uniform in [-rot_range, rot_range] degrees.
    """
    img28 = np.asarray(img_flat, dtype=np.float64).reshape(28, 28)
    rows = np.empty((n_aug, 784), dtype=np.float64)
    for j in range(n_aug):
        ang = rng.uniform(-rot_range, rot_range)
        rows[j] = augment_image(img28, rng, ang, max_shift).ravel()
    return rows
