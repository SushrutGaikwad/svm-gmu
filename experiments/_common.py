"""Shared helpers for the SVM-GMU experiments.

Single source of truth for the close_separable dataset, sampling, boundary
metrics, moment matching, EM+BIC fitting, the SVM-GMU fit helper, and the
high-dimensional lift. Imported by every experiment figure script and notebook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from svm_gmu import SvmGmu

# --- close_separable dataset (verbatim, the only definition in the repo) ---
X = np.array([
    [-1.8,  0.0],
    [-2.0,  1.5],
    [-1.0,  2.5],
    [ 1.5, -1.0],
    [ 1.0,  1.0],
    [ 2.0,  2.0],
])
y = np.array([+1, +1, +1, -1, -1, -1])

sample_1 = {
    "weights": np.array([0.10, 0.20, 0.40, 0.20, 0.10]),
    "means": np.array([
        [-4.2, -0.3],
        [-3.8,  0.3],
        [-3.0,  0.5],
        [-2.2,  0.3],
        [-1.8, -0.3],
    ]),
    "covariances": np.array([
        [[ 0.08, -0.05], [-0.05,  0.15]],
        [[ 0.10, -0.03], [-0.03,  0.12]],
        [[ 0.18,  0.00], [ 0.00,  0.06]],
        [[ 0.10,  0.03], [ 0.03,  0.12]],
        [[ 0.08,  0.05], [ 0.05,  0.15]],
    ]),
}

sample_2 = {
    "weights": np.array([0.08, 0.15, 0.22, 0.30, 0.15, 0.10]),
    "means": np.array([
        [-2.2,  0.6],
        [-1.8,  1.1],
        [-1.4,  1.5],
        [-0.8,  1.7],
        [-0.4,  1.5],
        [-0.1,  1.1],
    ]),
    "covariances": np.array([
        [[ 0.07,  0.04], [ 0.04,  0.13]],
        [[ 0.09,  0.03], [ 0.03,  0.10]],
        [[ 0.14,  0.02], [ 0.02,  0.07]],
        [[ 0.18,  0.00], [ 0.00,  0.05]],
        [[ 0.12, -0.02], [-0.02,  0.07]],
        [[ 0.08, -0.04], [-0.04,  0.12]],
    ]),
}

sample_3 = {
    "weights": np.array([0.10, 0.20, 0.40, 0.20, 0.10]),
    "means": np.array([
        [-0.6,  2.2],
        [-0.2,  2.7],
        [ 0.3,  3.0],
        [ 0.8,  3.4],
        [ 1.1,  3.8],
    ]),
    "covariances": np.array([
        [[ 0.14, -0.06], [-0.06,  0.08]],
        [[ 0.12, -0.07], [-0.07,  0.10]],
        [[ 0.10, -0.05], [-0.05,  0.14]],
        [[ 0.08,  0.03], [ 0.03,  0.16]],
        [[ 0.10,  0.06], [ 0.06,  0.15]],
    ]),
}

sample_4 = {
    "weights": np.array([0.08, 0.17, 0.25, 0.25, 0.17, 0.08]),
    "means": np.array([
        [-0.9, -0.3],
        [-0.5, -0.7],
        [-0.1, -1.0],
        [ 0.4, -1.2],
        [ 0.8, -1.1],
        [ 1.1, -0.8],
    ]),
    "covariances": np.array([
        [[ 0.07,  0.04], [ 0.04,  0.12]],
        [[ 0.09,  0.03], [ 0.03,  0.10]],
        [[ 0.15,  0.00], [ 0.00,  0.07]],
        [[ 0.15,  0.00], [ 0.00,  0.07]],
        [[ 0.09, -0.03], [-0.03,  0.10]],
        [[ 0.12, -0.05], [-0.05,  0.09]],
    ]),
}

sample_5 = {
    "weights": np.array([0.10, 0.20, 0.40, 0.20, 0.10]),
    "means": np.array([
        [1.3,  0.7],
        [1.7,  0.3],
        [2.3,  0.1],
        [2.9,  0.3],
        [3.3,  0.7],
    ]),
    "covariances": np.array([
        [[ 0.09,  0.05], [ 0.05,  0.14]],
        [[ 0.12,  0.03], [ 0.03,  0.10]],
        [[ 0.16,  0.00], [ 0.00,  0.07]],
        [[ 0.12, -0.03], [-0.03,  0.10]],
        [[ 0.09, -0.05], [-0.05,  0.14]],
    ]),
}

sample_6 = {
    "weights": np.array([0.08, 0.17, 0.25, 0.25, 0.17, 0.08]),
    "means": np.array([
        [2.4,  2.3],
        [2.8,  2.0],
        [3.3,  2.0],
        [3.8,  2.1],
        [4.2,  2.5],
        [4.4,  3.0],
    ]),
    "covariances": np.array([
        [[ 0.08,  0.05], [ 0.05,  0.11]],
        [[ 0.12,  0.03], [ 0.03,  0.08]],
        [[ 0.15,  0.00], [ 0.00,  0.06]],
        [[ 0.13, -0.02], [-0.02,  0.07]],
        [[ 0.09, -0.04], [-0.04,  0.10]],
        [[ 0.07, -0.05], [-0.05,  0.12]],
    ]),
}

SAMPLE_UNCERTAINTY = [sample_1, sample_2, sample_3, sample_4, sample_5, sample_6]

# --- shared experiment constants ---
LAM = 0.01
SVM_ITER = 1000
SVM_SEED = 42
MASTER_SEED = 2026
M_CANDIDATES = list(range(1, 9))
N_INIT = 5

GRAPHICS_DIR = Path(__file__).resolve().parents[1] / "docs" / "reports" / "report_3" / "graphics"

# --- fixed 2D grid for the in-plane boundary-disagreement metric ---
_grid_x = np.linspace(-5, 5, 60)
_grid_y = np.linspace(-4, 5, 60)
_GX, _GY = np.meshgrid(_grid_x, _grid_y)
GRID_PTS_2D = np.column_stack([_GX.ravel(), _GY.ravel()])


def sample_from_gmm(gmm: dict, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_samples IID points from a single GMM."""
    weights = gmm["weights"]
    means = gmm["means"]
    covs = gmm["covariances"]
    n_components, d = means.shape
    comp_idx = rng.choice(n_components, size=n_samples, p=weights)
    out = np.empty((n_samples, d), dtype=np.float64)
    for m in range(n_components):
        mask = comp_idx == m
        k = int(mask.sum())
        if k == 0:
            continue
        out[mask] = rng.multivariate_normal(means[m], covs[m], size=k)
    return out


def boundary_metrics(w_a, b_a, w_b, b_b, eval_points):
    """Return (angle_deg, offset_diff, rms) between two linear boundaries.

    rms is the root-mean-square difference of the two unit-normalized decision
    functions over eval_points. Pass GRID_PTS_2D in 2D; pass a Monte Carlo
    cloud in higher dimensions.
    """
    wa_hat = w_a / np.linalg.norm(w_a)
    wb_hat = w_b / np.linalg.norm(w_b)
    cos_t = np.clip(np.dot(wa_hat, wb_hat), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_t)))
    offset_a = b_a / np.linalg.norm(w_a)
    offset_b = b_b / np.linalg.norm(w_b)
    offset_diff = float(abs(offset_a - offset_b))
    f_a = eval_points @ wa_hat + offset_a
    f_b = eval_points @ wb_hat + offset_b
    rms = float(np.sqrt(np.mean((f_a - f_b) ** 2)))
    return angle_deg, offset_diff, rms


def boundary_metrics_2d(w_a, b_a, w_b, b_b):
    """Boundary metrics over the fixed 2D grid (matches the report's numbers)."""
    return boundary_metrics(w_a, b_a, w_b, b_b, GRID_PTS_2D)


def make_seeds(master, count):
    """Spawn count independent integer seeds from a master seed."""
    ss = np.random.SeedSequence(master)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(count)]


def band(values):
    """Return (median, q25, q75) of a 1D array. Shared by every error-band script."""
    values = np.asarray(values, dtype=float)
    return float(np.median(values)), float(np.percentile(values, 25)), float(np.percentile(values, 75))


def configure_pgf():
    """Set matplotlib pgf rcParams (10pt serif, amsmath+bm). Lazy matplotlib import."""
    import matplotlib
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 10,
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join([r"\usepackage{amsmath}", r"\usepackage{bm}"]),
    })


def moment_match_gmm(gmm: dict) -> dict:
    """Reduce a GMM to one Gaussian with the same overall mean and covariance.

    Uses the law of total covariance: the overall covariance is the weighted
    average of the component covariances (within) plus the weighted covariance
    of the component means (between).
    """
    weights = gmm["weights"]
    means = gmm["means"]
    covs = gmm["covariances"]
    mu = (weights[:, None] * means).sum(axis=0)
    within = (weights[:, None, None] * covs).sum(axis=0)
    diffs = means - mu
    outers = diffs[:, :, None] * diffs[:, None, :]
    between = (weights[:, None, None] * outers).sum(axis=0)
    cov = within + between
    return {
        "weights": np.array([1.0]),
        "means": mu[None, :],
        "covariances": cov[None, :, :],
    }


def svm_iter_for(n_total):
    """Scale Pegasos SGD iterations with the training-cloud size."""
    return max(2000, 500 * int(np.log10(max(n_total, 10)) + 1))


def fit_gmu(X, y, sample_uncertainty, lam=LAM, max_iter=SVM_ITER, seed=SVM_SEED, batch_size=None):
    """Fit SVM-GMU (or a standard SVM if sample_uncertainty is None); return (w, b)."""
    bs = len(X) if batch_size is None else min(batch_size, len(X))
    model = SvmGmu(lam=lam, max_iter=max_iter, batch_size=bs, random_state=seed)
    model.fit(X, y, sample_uncertainty=sample_uncertainty)
    return model.coef_.copy(), float(model.intercept_)


def fit_gmm_bic(points, m_candidates, n_init, seed) -> dict:
    """Fit a GMM to points, choosing the component count by minimum BIC."""
    best_bic = np.inf
    best_gmm = None
    for m in m_candidates:
        gmm = GaussianMixture(
            n_components=m,
            covariance_type="full",
            n_init=n_init,
            random_state=seed,
        ).fit(points)
        bic = gmm.bic(points)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    return {
        "weights": best_gmm.weights_.copy(),
        "means": best_gmm.means_.copy(),
        "covariances": best_gmm.covariances_.copy(),
    }


def build_sample_cloud(sample_uncertainty, y, n_per, rng):
    """Stack n_per draws from every GMM, each labeled by its parent y."""
    parts_x, parts_y = [], []
    for gmm, yi in zip(sample_uncertainty, y):
        parts_x.append(sample_from_gmm(gmm, n_per, rng))
        parts_y.append(np.full(n_per, yi, dtype=np.float64))
    return np.vstack(parts_x), np.concatenate(parts_y)


def random_rotation(d, rng):
    """Return a random orthogonal (d, d) matrix via QR of a Gaussian matrix."""
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    # Fix column signs so the result is deterministic given `a`.
    q = q * np.sign(np.diag(r))
    return q


def lift_dataset(X, sample_uncertainty, d, sigma2_noise, rng):
    """Lift the 2D dataset and per-sample GMMs into R^d with a shared rotation.

    The non-Gaussian structure and all class signal stay in the original 2D
    subspace; the extra d-2 dimensions carry shared isotropic noise of variance
    sigma2_noise. A single random rotation Q makes the signal subspace
    non-axis-aligned. Returns (X_lifted, su_lifted, Q).
    """
    n, d0 = X.shape
    if d < d0:
        raise ValueError(f"target dimension d={d} must be >= base dimension {d0}.")
    q = random_rotation(d, rng)

    x_pad = np.zeros((n, d))
    x_pad[:, :d0] = X
    x_lifted = x_pad @ q.T

    noise_block = sigma2_noise * np.eye(d - d0) if d > d0 else None
    su_lifted = []
    for gmm in sample_uncertainty:
        means = gmm["means"]
        covs = gmm["covariances"]
        m_count = means.shape[0]
        means_pad = np.zeros((m_count, d))
        means_pad[:, :d0] = means
        means_lifted = means_pad @ q.T
        covs_lifted = np.empty((m_count, d, d))
        for m in range(m_count):
            full = np.zeros((d, d))
            full[:d0, :d0] = covs[m]
            if noise_block is not None:
                full[d0:, d0:] = noise_block
            covs_lifted[m] = q @ full @ q.T
        su_lifted.append({
            "weights": gmm["weights"].copy(),
            "means": means_lifted,
            "covariances": covs_lifted,
        })
    return x_lifted, su_lifted, q


def mc_eval_cloud(sample_uncertainty, n_per, rng):
    """Monte Carlo evaluation cloud: n_per draws from every GMM, stacked."""
    return np.vstack([sample_from_gmm(g, n_per, rng) for g in sample_uncertainty])


def make_convergence_metrics_figure(agg):
    """1x3 median + IQR band panel (angle, offset, grid RMS) vs n. Shared by Exp 2 and 4.

    Each agg entry has 'n' and 'angle'/'offset'/'rms' as (median, q25, q75) tuples.
    """
    import matplotlib.pyplot as plt

    Ns = np.array([a["n"] for a in agg], dtype=float)
    specs = [
        ("angle", "Angle between normals (degrees)", "Normal-vector disagreement", "#2563eb"),
        ("offset", r"$\left|\frac{b_N}{\|\mathbf{w}_N\|} - \frac{b_{\mathrm{ref}}}{\|\mathbf{w}_{\mathrm{ref}}\|}\right|$", "Offset disagreement", "#059669"),
        ("rms", "Grid RMS of decision-function difference", "Overall boundary disagreement", "#dc2626"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, ylabel, title, color) in zip(axes, specs):
        med = np.array([a[key][0] for a in agg])
        lo = np.array([a[key][1] for a in agg])
        hi = np.array([a[key][2] for a in agg])
        ax.plot(Ns, med, marker="o", color=color)
        ax.fill_between(Ns, lo, hi, color=color, alpha=0.2)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xlabel(r"$N$ (samples per GMM)")
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig
