"""Experiment 4: SVM-GMU on EM-fitted, BIC-selected per-sample GMMs.

Canonical figure generator. Computes the n-sweep and writes two PGF
figures into docs/reports/report_3/graphics/. Run:

    uv run python experiments/em_fitted_gmu_figs.py            # full run, writes PGFs
    uv run python experiments/em_fitted_gmu_figs.py --smoke    # fast PNG self-check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from svm_gmu import SvmGmu

# --------------------------------------------------------------------------
# Dataset (copy verbatim from experiments/svm_gmu_convergence.ipynb)
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Experiment constants
# --------------------------------------------------------------------------
LAM = 0.01
SVM_ITER = 1000
SVM_SEED = 42
MASTER_SEED = 2026
M_CANDIDATES = list(range(1, 9))   # BIC chooses M in 1..8
N_INIT = 5
N_VALUES_FULL = [50, 100, 250, 500, 1000, 2000, 5000, 20000]
N_VALUES_SMOKE = [50, 200, 1000]

GRAPHICS_DIR = Path(__file__).resolve().parents[1] / "docs" / "reports" / "report_3" / "graphics"

# Fixed grid for the boundary-disagreement metric (same as the other notebooks).
_grid_x = np.linspace(-5, 5, 60)
_grid_y = np.linspace(-4, 5, 60)
_GX, _GY = np.meshgrid(_grid_x, _grid_y)
_GRID_PTS = np.column_stack([_GX.ravel(), _GY.ravel()])


def sample_from_gmm(gmm: dict, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_samples IID points from a single GMM (verbatim from convergence nb)."""
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


def fit_gmm_bic(
    points: np.ndarray,
    m_candidates: list[int],
    n_init: int,
    seed: int,
) -> dict:
    """Fit a GMM to points, choosing the component count by BIC.

    Parameters
    ----------
    points : ndarray of shape (n_samples, d)
        Samples to fit.
    m_candidates : iterable of int
        Candidate component counts to try; BIC selects the best.
    n_init : int
        Number of EM restarts per candidate (passed to GaussianMixture).
    seed : int
        Random seed for GaussianMixture.

    Returns
    -------
    dict
        A sample_uncertainty dict with keys 'weights' (M,),
        'means' (M, d), 'covariances' (M, d, d).
    """
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


def boundary_metrics(w_a, b_a, w_b, b_b):
    """Return (angle_deg, offset_diff, grid_rms) between two linear boundaries."""
    wa_hat = w_a / np.linalg.norm(w_a)
    wb_hat = w_b / np.linalg.norm(w_b)
    cos_t = np.clip(np.dot(wa_hat, wb_hat), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_t)))
    offset_a = b_a / np.linalg.norm(w_a)
    offset_b = b_b / np.linalg.norm(w_b)
    offset_diff = float(abs(offset_a - offset_b))
    f_a = _GRID_PTS @ wa_hat + offset_a
    f_b = _GRID_PTS @ wb_hat + offset_b
    grid_rms = float(np.sqrt(np.mean((f_a - f_b) ** 2)))
    return angle_deg, offset_diff, grid_rms


def fit_reference(X, y, true_su):
    """Fit SVM-GMU on the true GMMs to get the target boundary."""
    model = SvmGmu(lam=LAM, max_iter=SVM_ITER, batch_size=len(X), random_state=SVM_SEED)
    model.fit(X, y, sample_uncertainty=true_su)
    return model.coef_.copy(), float(model.intercept_)


def run_sweep(X, y, true_su, n_values, w_ref, b_ref):
    """Run the sample-size sweep, recording boundary metrics at each n.

    For each n in ``n_values`` we draw n samples from every ground-truth GMM
    in ``true_su``, fit a fresh GMM to each via BIC-selected EM, fit SVM-GMU
    on the learned mixtures, and record its disagreement with the reference
    boundary ``(w_ref, b_ref)``.

    Seeding contract: a single ``master_rng`` (from ``MASTER_SEED``) derives
    one ``draw_seed`` per n, which seeds both the sampling and every EM fit at
    that n, so the whole sweep is reproducible.

    Returns a list of per-n record dicts with keys ``n``, ``n_total``,
    ``su_hat``, ``w``, ``b``, ``angle``, ``offset``, ``rms``, ``m_chosen``.
    """
    master_rng = np.random.default_rng(MASTER_SEED)
    records = []
    for n in n_values:
        draw_seed = int(master_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(draw_seed)

        su_hat = []
        # The shared per-n draw_seed seeds every EM fit at this n. The six fits
        # still differ because each sees different sampled data; reusing the
        # seed keeps the sweep reproducible.
        for gmm in true_su:
            pts = sample_from_gmm(gmm, n, rng)
            su_hat.append(fit_gmm_bic(pts, M_CANDIDATES, N_INIT, seed=draw_seed))

        model = SvmGmu(lam=LAM, max_iter=SVM_ITER, batch_size=len(X), random_state=SVM_SEED)
        model.fit(X, y, sample_uncertainty=su_hat)
        w_n, b_n = model.coef_.copy(), float(model.intercept_)
        angle, offset, rms = boundary_metrics(w_n, b_n, w_ref, b_ref)

        m_chosen = [len(s["weights"]) for s in su_hat]
        records.append({
            "n": n, "n_total": len(true_su) * n, "su_hat": su_hat,
            "w": w_n, "b": b_n,
            "angle": angle, "offset": offset, "rms": rms,
            "m_chosen": m_chosen,
        })
        print(
            f"N={n:>6d}  n_total={len(true_su) * n:>7d}  M={m_chosen}  "
            f"angle={angle:7.3f}  offset={offset:.4f}  rms={rms:.4f}"
        )
    return records


def _draw_line(ax, w, b, xlim, **kwargs):
    """Draw the line w^T x + b = 0 within the given x-limits."""
    xs = np.linspace(xlim[0], xlim[1], 400)
    if abs(w[1]) > 1e-8:
        ys = -(w[0] * xs + b) / w[1]
        return ax.plot(xs, ys, **kwargs)[0]
    return ax.axvline(-b / w[0], **kwargs)


def make_metrics_figure(records):
    """1x3 panel: angle, offset, grid RMS vs n on a log-x axis."""
    import matplotlib.pyplot as plt

    Ns = np.array([r["n"] for r in records], dtype=float)
    angles = np.array([r["angle"] for r in records])
    offsets = np.array([r["offset"] for r in records])
    rms = np.array([r["rms"] for r in records])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(Ns, angles, marker="o", color="#2563eb")
    axes[0].set_ylabel("Angle between normals (degrees)")
    axes[0].set_title("Normal-vector disagreement")
    axes[1].plot(Ns, offsets, marker="o", color="#059669")
    axes[1].set_ylabel(r"$\left|\frac{b_N}{\|\mathbf{w}_N\|} - \frac{b_{\mathrm{ref}}}{\|\mathbf{w}_{\mathrm{ref}}\|}\right|$")
    axes[1].set_title("Offset disagreement")
    axes[2].plot(Ns, rms, marker="o", color="#dc2626")
    axes[2].set_ylabel("Grid RMS of decision-function difference")
    axes[2].set_title("Overall boundary disagreement")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$N$ (samples per GMM)")
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def make_panel_figure(X, y, records, w_ref, b_ref):
    """4x2 grid: per-n EM-fitted contours + reference and learned boundaries."""
    import matplotlib.pyplot as plt
    from svm_gmu.plotting import plot_uncertainty

    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.0 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    xlim, ylim = (-5.5, 5.5), (-4.0, 5.5)
    ref_color, learned_color = "#059669", "black"

    for k, r in enumerate(records):
        ax = axes[k // n_cols, k % n_cols]
        plot_uncertainty(
            X, y, r["su_hat"],
            sigmas=(3,),
            title=rf"$N = {r['n']}$  $\left(n = {r['n_total']}\right)$",
            ax=ax,
        )
        _draw_line(ax, w_ref, b_ref, xlim, color=ref_color, linewidth=2.2,
                   linestyle="--", zorder=4, label="SVM-GMU (true GMMs, reference)")
        _draw_line(ax, r["w"], r["b"], xlim, color=learned_color, linewidth=2.0,
                   linestyle="-", zorder=5, label="SVM-GMU (EM-fitted GMMs)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        handles, labels = ax.get_legend_handles_labels()
        keep = {"Class +1", "Class −1",
                "SVM-GMU (true GMMs, reference)", "SVM-GMU (EM-fitted GMMs)"}
        seen, uniq = set(), []
        for h, l in zip(handles, labels):
            if l in keep and l not in seen:
                seen.add(l)
                uniq.append((h, l))
        ax.legend([h for h, _ in uniq], [l for _, l in uniq],
                  loc="upper left", fontsize=8, framealpha=0.9)

    for k in range(len(records), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")
    fig.tight_layout()
    return fig


def _configure_pgf():
    """rcParams matching the existing committed PGFs (10pt, amsmath+bm)."""
    import matplotlib
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 10,
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join([r"\usepackage{amsmath}", r"\usepackage{bm}"]),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="fast PNG self-check on a tiny n-grid")
    args = parser.parse_args()

    n_values = N_VALUES_SMOKE if args.smoke else N_VALUES_FULL
    w_ref, b_ref = fit_reference(X, y, SAMPLE_UNCERTAINTY)
    print(f"reference  w = {w_ref}  b = {b_ref:.4f}")
    records = run_sweep(X, y, SAMPLE_UNCERTAINTY, n_values, w_ref, b_ref)

    if args.smoke:
        assert all(r["rms"] < 0.3 for r in records), "smoke rms outside sane range"
        make_metrics_figure(records).savefig("em_metrics_smoke.png", dpi=120, bbox_inches="tight")
        make_panel_figure(X, y, records, w_ref, b_ref).savefig("em_panels_smoke.png", dpi=120, bbox_inches="tight")
        print("smoke figures written")
        return

    _configure_pgf()
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = GRAPHICS_DIR / "em_convergence_metrics.pgf"
    panel_path = GRAPHICS_DIR / "em_fitted_gmu_convergence.pgf"
    make_metrics_figure(records).savefig(metrics_path, backend="pgf", bbox_inches="tight")
    make_panel_figure(X, y, records, w_ref, b_ref).savefig(panel_path, backend="pgf", bbox_inches="tight")
    print(f"wrote {metrics_path}")
    print(f"wrote {panel_path}")


if __name__ == "__main__":
    main()
