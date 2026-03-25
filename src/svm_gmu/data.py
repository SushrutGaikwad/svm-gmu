"""
Data validation and generation utilities for SVM-GMU.
"""

import numpy as np


def validate_gmm_dataset(X_means, y, gmm_params, verbose=False):
    """
    Validate that a GMM uncertainty dataset is well-formed.

    Parameters
    ----------
    X_means : ndarray of shape (n_samples, d)
        Observed locations (overall means) of each data point.
    y : ndarray of shape (n_samples,)
        Labels in {-1, +1}.
    gmm_params : list of dict
        One dict per sample with keys:
            'weights' : ndarray of shape (K_i,) summing to 1.
            'means'   : ndarray of shape (K_i, d).
            'covs'    : ndarray of shape (K_i, d, d) of PSD matrices.
    verbose : bool, default=False
        If True, print summary information.

    Returns
    -------
    bool
        True if all checks pass.

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    n_samples, d = X_means.shape

    assert y.shape == (n_samples,), f"y has shape {y.shape}, expected ({n_samples},)"
    assert set(np.unique(y)).issubset({-1, +1}), (
        f"Labels must be in {{-1, +1}}, got {np.unique(y)}"
    )
    assert len(gmm_params) == n_samples, (
        f"gmm_params has {len(gmm_params)} entries, expected {n_samples}"
    )

    for i, gp in enumerate(gmm_params):
        w = gp["weights"]
        m = gp["means"]
        c = gp["covs"]
        K_i = len(w)

        assert np.all(w >= 0), f"Point {i}: negative weights found"
        assert np.isclose(np.sum(w), 1.0), (
            f"Point {i}: weights sum to {np.sum(w):.4f}, not 1.0"
        )
        assert m.shape == (K_i, d), (
            f"Point {i}: means shape {m.shape}, expected ({K_i}, {d})"
        )
        assert c.shape == (K_i, d, d), (
            f"Point {i}: covs shape {c.shape}, expected ({K_i}, {d}, {d})"
        )

        for j in range(K_i):
            cov_j = c[j]
            assert np.allclose(cov_j, cov_j.T), (
                f"Point {i}, component {j}: covariance not symmetric"
            )
            eigvals = np.linalg.eigvalsh(cov_j)
            assert np.all(eigvals >= -1e-10), (
                f"Point {i}, component {j}: covariance not PSD, eigvals={eigvals}"
            )

    if verbose:
        print("All validations passed!")
        print(f"  Samples: {n_samples}, Dimensionality: {d}")
        print(f"  Class +1: {np.sum(y == +1)}, Class -1: {np.sum(y == -1)}")
        for i, gp in enumerate(gmm_params):
            print(f"  Point {i:2d} (y={y[i]:+d}): {len(gp['weights'])} component(s)")

    return True


def generate_gmm_dataset(
    d=2,
    n_per_class=15,
    class_sep=4.0,
    uncertainty_scale=0.4,
    n_components_range=(1, 4),
    seed=42,
):
    """
    Generate a d-dimensional binary classification dataset with
    per-point GMM uncertainties.

    Parameters
    ----------
    d : int, default=2
        Dimensionality of the feature space.
    n_per_class : int, default=15
        Number of points per class.
    class_sep : float, default=4.0
        Separation between class centers along the first axis.
    uncertainty_scale : float, default=0.4
        Controls the spread of uncertainty clouds.
    n_components_range : tuple of (int, int), default=(1, 4)
        Range for number of GMM components per point (inclusive).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_means : ndarray of shape (n_samples, d)
        Observed locations.
    y : ndarray of shape (n_samples,)
        Labels in {-1, +1}.
    gmm_params : list of dict
        GMM parameters for each point.
    """
    rng = np.random.RandomState(seed)

    X_means = []
    y_labels = []
    gmm_params = []

    for cls in [+1, -1]:
        center = np.zeros(d)
        center[0] = cls * class_sep / 2.0

        for _ in range(n_per_class):
            mean_i = center + rng.randn(d) * 0.8
            X_means.append(mean_i)
            y_labels.append(cls)

            K_i = rng.randint(n_components_range[0], n_components_range[1] + 1)

            weights = rng.dirichlet(np.ones(K_i))
            means = np.zeros((K_i, d))
            covs = np.zeros((K_i, d, d))

            for j in range(K_i):
                offset = rng.randn(d) * uncertainty_scale * 0.5
                if rng.rand() < 0.3:
                    offset[0] = -cls * abs(offset[0]) * 3.0
                means[j] = mean_i + offset

                A = rng.randn(d, d) * uncertainty_scale * 0.3
                cov_j = A @ A.T + np.eye(d) * 0.01

                if rng.rand() < 0.4:
                    stretch_dir = rng.randn(d)
                    stretch_dir /= np.linalg.norm(stretch_dir)
                    cov_j += np.outer(stretch_dir, stretch_dir) * uncertainty_scale**2

                covs[j] = cov_j

            gmm_params.append({"weights": weights, "means": means, "covs": covs})

    X_means = np.array(X_means)
    y_labels = np.array(y_labels, dtype=int)

    return X_means, y_labels, gmm_params
