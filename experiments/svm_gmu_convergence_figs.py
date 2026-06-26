"""Experiment 2: Monte Carlo convergence of a standard SVM to SVM-GMU.

For each N, fit a standard SVM on 6N samples drawn from the per-sample GMMs and
measure its boundary disagreement with the closed-form SVM-GMU reference, over
several seeds. Run:

    uv run python experiments/svm_gmu_convergence_figs.py
    uv run python experiments/svm_gmu_convergence_figs.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

X, y, SAMPLE_UNCERTAINTY = C.X, C.y, C.SAMPLE_UNCERTAINTY
MASTER_SEED, GRAPHICS_DIR = C.MASTER_SEED, C.GRAPHICS_DIR

N_VALUES_FULL = [1, 5, 25, 100, 500, 2000, 5000, 20000]
N_VALUES_SMOKE = [1, 25, 500]
R_SEEDS_FULL = 30
R_SEEDS_SMOKE = 4


def svm_iter_for(n_total):
    """Scale SGD iterations with cloud size (matches the original notebook)."""
    return max(2000, 500 * int(np.log10(max(n_total, 10)) + 1))


def run_conv_seed(n_values, w_ref, b_ref, seed):
    rng = np.random.default_rng(seed)
    recs = []
    for n in n_values:
        Xc, yc = C.build_sample_cloud(SAMPLE_UNCERTAINTY, y, n, rng)
        w_n, b_n = C.fit_gmu(
            Xc, yc, None, max_iter=svm_iter_for(len(Xc)),
            seed=C.SVM_SEED, batch_size=256,
        )
        angle, offset, rms = C.boundary_metrics_2d(w_n, b_n, w_ref, b_ref)
        recs.append({"n": n, "angle": angle, "offset": offset, "rms": rms})
    return recs


def run_conv_multiseed(n_values, w_ref, b_ref, seeds):
    per_seed = []
    for s in seeds:
        per_seed.append(run_conv_seed(n_values, w_ref, b_ref, s))
        print(f"seed {s}: rms@largest_n = {per_seed[-1][-1]['rms']:.4f}")
    agg = []
    for j, n in enumerate(n_values):
        angles = np.array([per_seed[i][j]["angle"] for i in range(len(seeds))])
        offsets = np.array([per_seed[i][j]["offset"] for i in range(len(seeds))])
        rmss = np.array([per_seed[i][j]["rms"] for i in range(len(seeds))])
        agg.append({"n": n, "angle": C.band(angles), "offset": C.band(offsets), "rms": C.band(rmss)})
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_values = N_VALUES_SMOKE if args.smoke else N_VALUES_FULL
    seeds = C.make_seeds(MASTER_SEED, R_SEEDS_SMOKE if args.smoke else R_SEEDS_FULL)
    w_ref, b_ref = C.fit_gmu(X, y, SAMPLE_UNCERTAINTY)
    print(f"reference  w = {w_ref}  b = {b_ref:.4f}")
    agg = run_conv_multiseed(n_values, w_ref, b_ref, seeds)

    if args.smoke:
        assert agg[-1]["rms"][0] < agg[0]["rms"][0], "smoke: rms did not decrease with N"
        C.make_convergence_metrics_figure(agg).savefig("conv_metrics_smoke.png", dpi=120, bbox_inches="tight")
        print("smoke figure written")
        return

    C.configure_pgf()
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    C.make_convergence_metrics_figure(agg).savefig(GRAPHICS_DIR / "convergence_metrics.pgf", backend="pgf", bbox_inches="tight")
    print("wrote convergence_metrics.pgf")


if __name__ == "__main__":
    main()
