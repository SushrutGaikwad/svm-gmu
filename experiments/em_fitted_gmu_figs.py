"""Experiment 4: SVM-GMU on EM-fitted, BIC-selected per-sample GMMs.

Canonical figure generator. Computes the n-sweep and writes two PGF
figures into docs/reports/report_3/graphics/. Run:

    uv run python experiments/em_fitted_gmu_figs.py            # full run, writes PGFs
    uv run python experiments/em_fitted_gmu_figs.py --smoke    # fast PNG self-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

X, y, SAMPLE_UNCERTAINTY = C.X, C.y, C.SAMPLE_UNCERTAINTY
LAM, MASTER_SEED = C.LAM, C.MASTER_SEED
M_CANDIDATES, N_INIT = C.M_CANDIDATES, C.N_INIT
GRAPHICS_DIR = C.GRAPHICS_DIR

N_VALUES_FULL = [50, 100, 250, 500, 1000, 2000, 5000, 20000]
N_VALUES_SMOKE = [50, 200, 1000]
R_SEEDS_FULL = 30
R_SEEDS_SMOKE = 3


def run_sweep_seed(n_values, w_ref, b_ref, seed):
    """One full n-sweep for a single seed. Returns a list of per-n records."""
    rng = np.random.default_rng(seed)
    recs = []
    for n in n_values:
        su_hat = [
            C.fit_gmm_bic(C.sample_from_gmm(gmm, n, rng), M_CANDIDATES, N_INIT, seed=seed)
            for gmm in SAMPLE_UNCERTAINTY
        ]
        w_n, b_n = C.fit_gmu(X, y, su_hat)
        angle, offset, rms = C.boundary_metrics_2d(w_n, b_n, w_ref, b_ref)
        recs.append({
            "n": n, "n_total": len(SAMPLE_UNCERTAINTY) * n,
            "angle": angle, "offset": offset, "rms": rms,
            "m_chosen": [len(s["weights"]) for s in su_hat],
            "su_hat": su_hat, "w": w_n, "b": b_n,
        })
    return recs


def run_sweep_multiseed(n_values, w_ref, b_ref, seeds):
    """Run run_sweep_seed for every seed; return (agg, per_seed).

    agg[j] holds median/IQR bands for angle, offset, rms at n_values[j], plus
    the per-point modal BIC component count across seeds.
    """
    per_seed = []
    for s in seeds:
        recs = run_sweep_seed(n_values, w_ref, b_ref, s)
        per_seed.append(recs)
        print(f"seed {s}: rms@largest_n = {recs[-1]['rms']:.4f}")
    agg = []
    for j, n in enumerate(n_values):
        angles = np.array([per_seed[i][j]["angle"] for i in range(len(seeds))])
        offsets = np.array([per_seed[i][j]["offset"] for i in range(len(seeds))])
        rmss = np.array([per_seed[i][j]["rms"] for i in range(len(seeds))])
        m_mat = np.array([per_seed[i][j]["m_chosen"] for i in range(len(seeds))])
        am, alo, ahi = C.band(angles)
        om, olo, ohi = C.band(offsets)
        rm, rlo, rhi = C.band(rmss)
        m_modal = [int(np.bincount(m_mat[:, i]).argmax()) for i in range(m_mat.shape[1])]
        agg.append({
            "n": n, "n_total": len(SAMPLE_UNCERTAINTY) * n,
            "angle": (am, alo, ahi), "offset": (om, olo, ohi), "rms": (rm, rlo, rhi),
            "m_modal": m_modal,
        })
    return agg, per_seed


def _draw_line(ax, w, b, xlim, **kwargs):
    """Draw the line w^T x + b = 0 within the given x-limits."""
    xs = np.linspace(xlim[0], xlim[1], 400)
    if abs(w[1]) > 1e-8:
        ys = -(w[0] * xs + b) / w[1]
        return ax.plot(xs, ys, **kwargs)[0]
    return ax.axvline(-b / w[0], **kwargs)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_values = N_VALUES_SMOKE if args.smoke else N_VALUES_FULL
    r_seeds = R_SEEDS_SMOKE if args.smoke else R_SEEDS_FULL
    seeds = C.make_seeds(MASTER_SEED, r_seeds)

    w_ref, b_ref = C.fit_gmu(X, y, SAMPLE_UNCERTAINTY)
    print(f"reference  w = {w_ref}  b = {b_ref:.4f}")
    agg, per_seed = run_sweep_multiseed(n_values, w_ref, b_ref, seeds)

    if args.smoke:
        assert all(a["rms"][0] < 0.3 for a in agg), "smoke median rms outside sane range"
        C.make_convergence_metrics_figure(agg).savefig("em_metrics_smoke.png", dpi=120, bbox_inches="tight")
        make_panel_figure(X, y, per_seed[0], w_ref, b_ref).savefig("em_panels_smoke.png", dpi=120, bbox_inches="tight")
        print("smoke figures written")
        return

    C.configure_pgf()
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    C.make_convergence_metrics_figure(agg).savefig(GRAPHICS_DIR / "em_convergence_metrics.pgf", backend="pgf", bbox_inches="tight")
    make_panel_figure(X, y, per_seed[0], w_ref, b_ref).savefig(GRAPHICS_DIR / "em_fitted_gmu_convergence.pgf", backend="pgf", bbox_inches="tight")
    print("wrote em_convergence_metrics.pgf and em_fitted_gmu_convergence.pgf")


if __name__ == "__main__":
    main()
