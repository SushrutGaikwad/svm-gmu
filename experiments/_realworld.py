"""Helpers for the real-data SVM-GMU vs SVM-GSU experiments (Plan 1: MNIST).

Pure functions plus per-seed and full-experiment drivers. All numerical logic
lives here so the notebooks stay thin, mirroring experiments/_common.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


_VAR_FLOOR = 1e-6


def moment_gaussian(cloud: NDArray[np.floating], cov_type: str = "diag") -> dict:
    """Single-Gaussian uncertainty (SVM-GSU) by moment matching the cloud."""
    cloud = np.asarray(cloud, dtype=np.float64)
    mu = cloud.mean(axis=0)
    if cov_type == "diag":
        var = np.maximum(cloud.var(axis=0), _VAR_FLOOR)
        covs = var[None, :]
    elif cov_type == "full":
        d = cloud.shape[1]
        cov = np.cov(cloud, rowvar=False, bias=True) + _VAR_FLOOR * np.eye(d)
        covs = cov[None, :, :]
    else:
        raise ValueError(f"cov_type must be 'diag' or 'full', got {cov_type!r}.")
    return {"weights": np.array([1.0]), "means": mu[None, :], "covariances": covs}


def iso_gaussian(cloud: NDArray[np.floating]) -> dict:
    """Single isotropic-diagonal Gaussian (LSVM-iso baseline)."""
    cloud = np.asarray(cloud, dtype=np.float64)
    mu = cloud.mean(axis=0)
    var = max(float(cloud.var(axis=0).mean()), _VAR_FLOOR)
    d = cloud.shape[1]
    return {
        "weights": np.array([1.0]),
        "means": mu[None, :],
        "covariances": np.full((1, d), var),
    }


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


def structural_components(
    img_flat: NDArray[np.floating],
    k: int,
    rot_range: float,
    n_per_anchor: int,
    rng: np.random.Generator,
    max_shift: float,
    pca,
    cov_type: str = "diag",
) -> dict:
    """EM-free K-component mixture along the known rotation arc, in PCA space."""
    img28 = np.asarray(img_flat, dtype=np.float64).reshape(28, 28)
    angles = np.linspace(-rot_range, rot_range, k)
    means, covs = [], []
    for ang in angles:
        batch = np.array(
            [augment_image(img28, rng, float(ang), max_shift).ravel() for _ in range(n_per_anchor)]
        )
        comp = moment_gaussian(pca.transform(batch), cov_type)
        means.append(comp["means"][0])
        covs.append(comp["covariances"][0])
    return {
        "weights": np.full(k, 1.0 / k),
        "means": np.array(means),
        "covariances": np.array(covs),
    }


def evaluate_metrics(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_score: NDArray[np.floating],
) -> dict:
    """Accuracy, macro-F1, ROC-AUC, and average precision for a binary task."""
    y_bin = (np.asarray(y_true) == 1.0).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "auc": float(roc_auc_score(y_bin, y_score)),
        "ap": float(average_precision_score(y_bin, y_score)),
    }


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
