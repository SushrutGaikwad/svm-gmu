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
    """Render figures with the report's LaTeX serif font (pdflatex + lmodern).

    Delegates the font setup to ``svm_gmu.plotting.use_latex_serif`` (the single
    definition) and adds the report's 10pt size. Lazy matplotlib import.
    """
    import matplotlib
    from svm_gmu.plotting import use_latex_serif

    use_latex_serif()
    matplotlib.rcParams["font.size"] = 10


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


def fit_standard_svm_on_cloud(X_cloud, y_cloud):
    """Fit a standard SVM (no uncertainty) on a sampled cloud; return (w, b).

    The single definition of the protocol used whenever an experiment spends its
    sample budget on raw points rather than on a mixture description: no
    uncertainty, iterations scaled to the cloud size, mini-batches of 256.
    """
    return fit_gmu(
        X_cloud, y_cloud, None,
        max_iter=svm_iter_for(len(X_cloud)), seed=SVM_SEED, batch_size=256,
    )


def add_standard_svm_boundaries(records):
    """Attach a standard-SVM boundary (w_svm, b_svm) to each EM record, in place.

    Every record already carries the sampled cloud its EM fits were trained on,
    so the standard SVM fitted here consumes exactly the same points at exactly
    the same budget: it is the uncertainty-blind counterpart of that record's
    SVM-GMU fit. Deriving it from the records instead of computing it inside the
    sweep keeps the (slow) EM cache valid. Shared by the EM notebook and the
    paper's figure script so the two draw the identical boundary.
    """
    for r in records:
        r["w_svm"], r["b_svm"] = fit_standard_svm_on_cloud(r["X_exp"], r["y_exp"])
    return records


def check_ascending_dimensions(res):
    """Raise unless a sweep's rows are in strictly ascending dimension order.

    Anything plotting against d joins consecutive rows into a curve, so rows that
    are out of order do not fail: they quietly draw a line that doubles back on
    itself, which is easy to miss and easy to publish. The cache stores rows in
    the order the dimensions were computed, which need not be ascending, so this
    guards the boundary where a wrong order turns into a wrong figure. Requiring
    *strict* ascent also catches a duplicated dimension, which would mean a merge
    went wrong. Returns res so it can wrap a value in place.
    """
    d = np.asarray(res["d_values"])
    if d.size > 1 and not np.all(np.diff(d) > 0):
        raise ValueError(
            f"sweep rows must be in strictly ascending dimension order, got "
            f"{[int(v) for v in d]}. Read the cache through load_highdim_cache, "
            f"which orders them, and keep the requested dimension list sorted."
        )
    return res


def load_highdim_cache(path):
    """Load a high-dimensional sweep cache with its rows ordered by dimension.

    The cache accumulates rows in the order dimensions were first computed, which
    need not be ascending: extending an existing sweep appends the new dimension
    after the ones already there. That order is an implementation detail of the
    cache, not a display order, and plotting straight from it joins the points in
    the wrong sequence and draws a curve that doubles back on itself. Every
    reader should come through here rather than indexing the raw file.
    """
    data = np.load(path, allow_pickle=False)
    res = {k: data[k] for k in data.files}
    order = np.argsort(res["d_values"])
    res["d_values"] = res["d_values"][order]
    for key in ("angle", "offset", "rms"):
        res[key] = res[key][order]
    return check_ascending_dimensions(res)


def nstar_from_angles(res, tau):
    """N*(d) per seed: smallest ladder N reaching angle <= tau, else inf.

    A seed that never reaches the tolerance within the ladder is *censored*, not
    missing: its true N* is unknown but certainly larger than every rung, so inf
    is the honest placeholder and it ranks above every seed that did reach the
    tolerance. Recording nan instead and summarizing with nan-skipping functions
    would silently drop exactly the worst seeds and bias the result downward,
    reporting the median of the seeds that happened to succeed as though it were
    the median over all of them. With inf in place an ordinary median runs over
    every seed and evaluates to inf precisely when at least half never got there.

    Callers should summarize with plain median/percentile (percentile needs
    method="lower"/"higher": interpolating across inf computes inf - inf and
    silently yields nan), then clip to the ladder ceiling for drawing.
    """
    angle, ladder = res["angle"], res["n_ladder"]
    nd, ns, _ = angle.shape
    nstar = np.full((nd, ns), np.inf)
    for di in range(nd):
        for si in range(ns):
            below = np.where(angle[di, si, :] <= tau)[0]
            if below.size:
                nstar[di, si] = ladder[below[0]]
    return nstar


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


def make_highdim_gmm_dataset(
    d, rng, n_per_class=5, n_components=3, sep=2.0, jitter=0.5, spread=0.9, sigma=1.1
):
    """Build a genuinely d-dimensional GMM-per-example dataset (two close classes).

    Each example is a real ``n_components``-component isotropic Gaussian mixture
    in R^d: the component means are offset from the example center in random
    d-dimensional directions, and every component has covariance sigma^2 I_d. The
    two classes are separated along the first axis (with off-axis jitter so the
    optimal boundary is genuinely d-dimensional) and placed close enough that
    their uncertainty overlaps. Returns ``(X, y, sample_uncertainty)``, where each
    row of X is the example's overall mean. Fitting the closed form on this is
    literally SVM-GMU (every example has more than one component).
    """
    X, y, su = [], [], []
    for cls in (1.0, -1.0):
        for _ in range(n_per_class):
            center = jitter * rng.standard_normal(d)
            center[0] += cls * sep
            dirs = rng.standard_normal((n_components, d))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            means = center + spread * dirs
            covs = np.stack([sigma**2 * np.eye(d) for _ in range(n_components)])
            su.append({
                "weights": np.full(n_components, 1.0 / n_components),
                "means": means,
                "covariances": covs,
            })
            X.append(means.mean(axis=0))
            y.append(cls)
    return np.array(X), np.array(y), su


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
