"""Experiment 5: high-dimensional scaling of sampling vs closed-form SVM-GMU.

For each dimension d we build a genuinely d-dimensional dataset (two close
classes; each example a real isotropic Gaussian mixture in R^d, see
C.make_highdim_gmm_dataset), fit the closed-form SVM-GMU reference on the
mixtures, and measure how many Monte Carlo samples per example a standard SVM
needs to match that boundary (angle <= tau). Headline: N*(d). Secondary:
fixed-budget angle vs d.

    uv run python experiments/high_dim_scaling_figs.py
    uv run python experiments/high_dim_scaling_figs.py --smoke
    uv run python experiments/high_dim_scaling_figs.py --force   # ignore cache
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

MASTER_SEED, GRAPHICS_DIR = C.MASTER_SEED, C.GRAPHICS_DIR

D_VALUES_FULL = [2, 3, 5, 10, 20, 50]
D_VALUES_SMOKE = [2, 3, 5]
N_LADDER_FULL = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
N_LADDER_SMOKE = [10, 100, 1000]
R_SEEDS_FULL = 30
R_SEEDS_SMOKE = 3

# Per-example d-dimensional GMM dataset parameters (see make_highdim_gmm_dataset).
N_PER_CLASS = 5      # examples per class -> n = 10
M_COMPONENTS = 3     # components per example (a genuine SVM-GMU mixture)
SEP = 2.0            # class-center separation along the first axis
JITTER = 0.5         # off-axis within-class jitter (boundary genuinely d-dim)
SPREAD = 0.9         # component-mean offset from the example center
SIGMA = 1.1          # per-component isotropic std

TAUS = [1.0, 2.0]            # degrees; primary curve uses tau = 2.0
FIXED_BUDGETS = [1000, 10000]
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def run_highdim_sweep(d_values, n_ladder, seeds):
    """Return dict with angle/offset/rms arrays of shape (len(d), len(seeds), len(ladder))."""
    nd, ns, nl = len(d_values), len(seeds), len(n_ladder)
    angle = np.full((nd, ns, nl), np.nan)
    offset = np.full((nd, ns, nl), np.nan)
    rms = np.full((nd, ns, nl), np.nan)
    for di, d in enumerate(d_values):
        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            X, y, su = C.make_highdim_gmm_dataset(
                d, rng, n_per_class=N_PER_CLASS, n_components=M_COMPONENTS,
                sep=SEP, jitter=JITTER, spread=SPREAD, sigma=SIGMA,
            )
            w_ref, b_ref = C.fit_gmu(X, y, su)
            eval_cloud = C.mc_eval_cloud(su, 2000, rng)
            for ni, n in enumerate(n_ladder):
                Xc, yc = C.build_sample_cloud(su, y, n, rng)
                w_n, b_n = C.fit_gmu(
                    Xc, yc, None, max_iter=C.svm_iter_for(len(Xc)),
                    seed=C.SVM_SEED, batch_size=256,
                )
                a, o, r = C.boundary_metrics(w_n, b_n, w_ref, b_ref, eval_cloud)
                angle[di, si, ni], offset[di, si, ni], rms[di, si, ni] = a, o, r
            print(f"d={d:>3d} seed={seed}: angle@maxN={angle[di, si, -1]:.3f}")
    return {
        "d_values": np.array(d_values), "n_ladder": np.array(n_ladder, dtype=float),
        "seeds": np.array(seeds),
        "angle": angle, "offset": offset, "rms": rms,
    }


def nstar_from_angles(res, tau):
    """N*(d) per seed: smallest ladder N with angle <= tau, else nan (cap exceeded)."""
    angle, ladder = res["angle"], res["n_ladder"]
    nd, ns, _ = angle.shape
    nstar = np.full((nd, ns), np.nan)
    for di in range(nd):
        for si in range(ns):
            below = np.where(angle[di, si, :] <= tau)[0]
            if below.size:
                nstar[di, si] = ladder[below[0]]
    return nstar


def make_nstar_figure(res):
    import matplotlib.pyplot as plt

    d_values = res["d_values"]
    cap = res["n_ladder"][-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {1.0: "#dc2626", 2.0: "#2563eb"}
    for tau in TAUS:
        nstar = nstar_from_angles(res, tau)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(nstar, axis=1)
            lo = np.nanpercentile(nstar, 25, axis=1)
            hi = np.nanpercentile(nstar, 75, axis=1)
        ax.plot(d_values, med, marker="o", color=colors[tau], label=rf"$\tau = {tau:g}^\circ$")
        ax.fill_between(d_values, lo, hi, color=colors[tau], alpha=0.2)
        # Mark dimensions where some seeds never reached tau (cap exceeded).
        exceeded = np.isnan(nstar).any(axis=1)
        if exceeded.any():
            ax.scatter(d_values[exceeded], np.full(exceeded.sum(), cap),
                       marker="^", color=colors[tau], zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel(r"$N^\ast(d)$: samples per example to match SVM-GMU")
    ax.set_title("Samples needed for standard SVM to match closed-form SVM-GMU")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_fixedbudget_figure(res):
    import matplotlib.pyplot as plt

    d_values = res["d_values"]
    ladder = [float(v) for v in res["n_ladder"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    for budget in FIXED_BUDGETS:
        if float(budget) not in ladder:
            continue  # budget absent from this ladder (e.g. smoke mode); skip
        ni = ladder.index(float(budget))
        med = np.nanmedian(res["angle"][:, :, ni], axis=1)
        lo = np.nanpercentile(res["angle"][:, :, ni], 25, axis=1)
        hi = np.nanpercentile(res["angle"][:, :, ni], 75, axis=1)
        line, = ax.plot(d_values, med, marker="o", label=rf"$N = {budget}$")
        ax.fill_between(d_values, lo, hi, color=line.get_color(), alpha=0.2)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0,
               label="SVM-GMU (closed form)")
    ax.set_xlabel(r"Dimension $d$")
    ax.set_ylabel("Angle to SVM-GMU boundary (degrees)")
    ax.set_title("Fixed sampling budget: boundary error grows with dimension")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _cache_path(mode):
    return CACHE_DIR / f"high_dim_gmm_{mode}.npz"


def _load_or_run(mode, force, d_values, n_ladder, seeds):
    path = _cache_path(mode)
    if path.exists() and not force:
        data = np.load(path, allow_pickle=False)
        print(f"loaded cache {path}")
        return {k: data[k] for k in data.files}
    res = run_highdim_sweep(d_values, n_ladder, seeds)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **res)
    print(f"wrote cache {path}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore cache")
    args = parser.parse_args()

    mode = "smoke" if args.smoke else "full"
    d_values = D_VALUES_SMOKE if args.smoke else D_VALUES_FULL
    n_ladder = N_LADDER_SMOKE if args.smoke else N_LADDER_FULL
    seeds = C.make_seeds(MASTER_SEED, R_SEEDS_SMOKE if args.smoke else R_SEEDS_FULL)

    res = _load_or_run(mode, args.force, d_values, n_ladder, seeds)

    if args.smoke:
        # Sanity: by the top of the smoke ladder, d=2 should be well matched.
        assert np.nanmin(res["angle"][0, :, -1]) < 5.0, "smoke: d=2 did not converge"
        make_nstar_figure(res).savefig("highdim_nstar_smoke.png", dpi=120, bbox_inches="tight")
        make_fixedbudget_figure(res).savefig("highdim_budget_smoke.png", dpi=120, bbox_inches="tight")
        print("smoke figures written")
        return

    C.configure_pgf()
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    make_nstar_figure(res).savefig(GRAPHICS_DIR / "high_dim_nstar.pgf", backend="pgf", bbox_inches="tight")
    make_fixedbudget_figure(res).savefig(GRAPHICS_DIR / "high_dim_fixed_budget.pgf", backend="pgf", bbox_inches="tight")
    print("wrote high_dim_nstar.pgf and high_dim_fixed_budget.pgf")


if __name__ == "__main__":
    main()
