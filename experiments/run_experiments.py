"""
Comprehensive experiments for SVM-GMU evaluation.

Runs all experiments and saves results to a CSV file.

Usage:
    uv run python experiments/run_experiments.py
    uv run python experiments/run_experiments.py --output results.csv
"""

import argparse
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from svm_gmu import SVMGMU, validate_gmm_dataset

warnings.filterwarnings("ignore")


# ============================================================
# Dataset generators for each experiment category
# ============================================================


def make_dataset_basic(
    d, n_per_class, class_sep, n_components, uncertainty_scale, seed
):
    """Generate a basic dataset with controlled parameters."""
    rng = np.random.RandomState(seed)
    X = []
    y = []
    gmm_params = []

    for cls in [+1, -1]:
        center = np.zeros(d)
        center[0] = cls * class_sep / 2.0

        for _ in range(n_per_class):
            mean_i = center + rng.randn(d) * 0.8
            X.append(mean_i)
            y.append(cls)

            K_i = n_components
            weights = rng.dirichlet(np.ones(K_i))
            means = np.zeros((K_i, d))
            covs = np.zeros((K_i, d, d))

            for j in range(K_i):
                offset = rng.randn(d) * uncertainty_scale * 0.5
                if rng.rand() < 0.3:
                    offset[0] = -cls * abs(offset[0]) * 2.0
                means[j] = mean_i + offset

                A = rng.randn(d, d) * uncertainty_scale * 0.2
                covs[j] = A @ A.T + np.eye(d) * 0.01

            gmm_params.append({"weights": weights, "means": means, "covs": covs})

    return np.array(X), np.array(y, dtype=int), gmm_params


def make_dataset_identical_uncertainty(d, n_per_class, class_sep, seed):
    """All points share the same GMM uncertainty shape."""
    rng = np.random.RandomState(seed)

    # Define a shared GMM shape (relative offsets and covariances)
    shared_offsets = np.array([[0.0] * d, [0.3] + [0.0] * (d - 1)])
    A1 = rng.randn(d, d) * 0.15
    A2 = rng.randn(d, d) * 0.15
    shared_covs = np.array([A1 @ A1.T + np.eye(d) * 0.01, A2 @ A2.T + np.eye(d) * 0.01])
    shared_weights = np.array([0.6, 0.4])

    X = []
    y = []
    gmm_params = []

    for cls in [+1, -1]:
        center = np.zeros(d)
        center[0] = cls * class_sep / 2.0
        for _ in range(n_per_class):
            mean_i = center + rng.randn(d) * 0.8
            X.append(mean_i)
            y.append(cls)
            gmm_params.append(
                {
                    "weights": shared_weights.copy(),
                    "means": mean_i + shared_offsets,
                    "covs": shared_covs.copy(),
                }
            )

    return np.array(X), np.array(y, dtype=int), gmm_params


def make_dataset_asymmetric_classes(d, n_per_class, class_sep, seed):
    """Class +1 has large uncertainty, class -1 has compact uncertainty."""
    rng = np.random.RandomState(seed)
    X = []
    y = []
    gmm_params = []

    for cls in [+1, -1]:
        center = np.zeros(d)
        center[0] = cls * class_sep / 2.0

        for _ in range(n_per_class):
            mean_i = center + rng.randn(d) * 0.8
            X.append(mean_i)
            y.append(cls)

            if cls == +1:
                # Large, multi-component uncertainty
                K_i = 3
                weights = rng.dirichlet(np.ones(K_i))
                means = np.array([mean_i + rng.randn(d) * 0.5 for _ in range(K_i)])
                covs = np.array(
                    [
                        rng.randn(d, d) * 0.3 @ (rng.randn(d, d) * 0.3).T
                        + np.eye(d) * 0.01
                        for _ in range(K_i)
                    ]
                )
            else:
                # Compact, single Gaussian
                K_i = 1
                weights = np.array([1.0])
                means = mean_i.reshape(1, -1)
                covs = (np.eye(d) * 0.02).reshape(1, d, d)

            gmm_params.append({"weights": weights, "means": means, "covs": covs})

    return np.array(X), np.array(y, dtype=int), gmm_params


def make_dataset_overlapping(d, n_per_class, seed):
    """Classes with heavily overlapping uncertainty clouds."""
    rng = np.random.RandomState(seed)
    X = []
    y = []
    gmm_params = []

    for cls in [+1, -1]:
        center = np.zeros(d)
        center[0] = cls * 0.5  # very close classes

        for _ in range(n_per_class):
            mean_i = center + rng.randn(d) * 0.3
            X.append(mean_i)
            y.append(cls)

            K_i = 2
            weights = np.array([0.5, 0.5])
            means = np.array(
                [
                    mean_i + rng.randn(d) * 0.3,
                    mean_i + rng.randn(d) * 0.3,
                ]
            )
            covs = np.array(
                [
                    rng.randn(d, d) * 0.2 @ (rng.randn(d, d) * 0.2).T + np.eye(d) * 0.01
                    for _ in range(K_i)
                ]
            )

            gmm_params.append({"weights": weights, "means": means, "covs": covs})

    return np.array(X), np.array(y, dtype=int), gmm_params


def make_dataset_zero_covariance(d, n_per_class, class_sep, seed):
    """All covariances are near-zero (should recover standard SVM)."""
    rng = np.random.RandomState(seed)
    X = []
    y = []
    gmm_params = []

    for cls in [+1, -1]:
        center = np.zeros(d)
        center[0] = cls * class_sep / 2.0
        for _ in range(n_per_class):
            mean_i = center + rng.randn(d) * 0.8
            X.append(mean_i)
            y.append(cls)
            gmm_params.append(
                {
                    "weights": np.array([1.0]),
                    "means": mean_i.reshape(1, -1),
                    "covs": (np.eye(d) * 1e-10).reshape(1, d, d),
                }
            )

    return np.array(X), np.array(y, dtype=int), gmm_params


def make_dataset_concentrated_weights(d, n_per_class, class_sep, seed):
    """GMM weights heavily concentrated on one component."""
    rng = np.random.RandomState(seed)
    X = []
    y = []
    gmm_params = []

    for cls in [+1, -1]:
        center = np.zeros(d)
        center[0] = cls * class_sep / 2.0
        for _ in range(n_per_class):
            mean_i = center + rng.randn(d) * 0.8
            X.append(mean_i)
            y.append(cls)

            K_i = 3
            weights = np.array([0.98, 0.01, 0.01])
            means = np.array(
                [
                    mean_i,
                    mean_i + rng.randn(d) * 0.5,
                    mean_i + rng.randn(d) * 0.5,
                ]
            )
            covs = np.array(
                [
                    np.eye(d) * 0.03,
                    rng.randn(d, d) * 0.2 @ (rng.randn(d, d) * 0.2).T
                    + np.eye(d) * 0.01,
                    rng.randn(d, d) * 0.2 @ (rng.randn(d, d) * 0.2).T
                    + np.eye(d) * 0.01,
                ]
            )

            gmm_params.append({"weights": weights, "means": means, "covs": covs})

    return np.array(X), np.array(y, dtype=int), gmm_params


def make_dataset_single_point_per_class(d, seed):
    """Extreme edge case: one point per class."""
    rng = np.random.RandomState(seed)
    X = np.array([rng.randn(d) + 2.0, rng.randn(d) - 2.0])
    y = np.array([+1, -1])
    gmm_params = []
    for i in range(2):
        K_i = 2
        gmm_params.append(
            {
                "weights": np.array([0.5, 0.5]),
                "means": np.array(
                    [X[i] + rng.randn(d) * 0.3, X[i] + rng.randn(d) * 0.3]
                ),
                "covs": np.array(
                    [
                        rng.randn(d, d) * 0.15 @ (rng.randn(d, d) * 0.15).T
                        + np.eye(d) * 0.01
                        for _ in range(K_i)
                    ]
                ),
            }
        )
    return X, y, gmm_params


def make_dataset_many_components(d, n_per_class, class_sep, n_components, seed):
    """Many GMM components per point."""
    rng = np.random.RandomState(seed)
    X = []
    y = []
    gmm_params = []

    for cls in [+1, -1]:
        center = np.zeros(d)
        center[0] = cls * class_sep / 2.0
        for _ in range(n_per_class):
            mean_i = center + rng.randn(d) * 0.8
            X.append(mean_i)
            y.append(cls)

            weights = rng.dirichlet(np.ones(n_components))
            means = np.array([mean_i + rng.randn(d) * 0.3 for _ in range(n_components)])
            covs = np.array(
                [
                    rng.randn(d, d) * 0.1 @ (rng.randn(d, d) * 0.1).T + np.eye(d) * 0.01
                    for _ in range(n_components)
                ]
            )

            gmm_params.append({"weights": weights, "means": means, "covs": covs})

    return np.array(X), np.array(y, dtype=int), gmm_params


# ============================================================
# Evaluation
# ============================================================


def evaluate(
    X, y, gmm_params, lam=0.01, n_epochs=300, lr_init=0.5, n_mc=20000, seed=42
):
    """
    Train SVM-GMU and standard SVM, return evaluation metrics.

    Returns a dict with all metrics.
    """
    n_samples, d = X.shape

    # Train SVM-GMU
    t0 = time.time()
    clf_gmu = SVMGMU(lam=lam, n_epochs=n_epochs, lr_init=lr_init, seed=seed)
    clf_gmu.fit(X, y, gmm_params)
    time_gmu = time.time() - t0

    # Train standard SVM
    t0 = time.time()
    C = 1.0 / max(lam, 1e-10)
    clf_std = SVC(kernel="linear", C=C)
    clf_std.fit(X, y)
    time_std = time.time() - t0

    w_std = clf_std.coef_[0]
    b_std = clf_std.intercept_[0]

    # Evaluate expected misclassification
    p_gmu = clf_gmu.expected_misclassification(n_mc=n_mc, seed=seed)
    avg_p_gmu = np.mean(p_gmu)

    # Evaluate standard SVM misclassification via MC
    rng = np.random.RandomState(seed)
    p_std_list = []
    for i in range(n_samples):
        gp = gmm_params[i]
        K_i = len(gp["weights"])
        comp = rng.choice(K_i, size=n_mc, p=gp["weights"])
        samples = np.zeros((n_mc, d))
        for j in range(K_i):
            mask = comp == j
            n_j = np.sum(mask)
            if n_j > 0:
                samples[mask] = rng.multivariate_normal(
                    gp["means"][j],
                    gp["covs"][j],
                    size=n_j,
                )
        scores = y[i] * (samples @ w_std + b_std)
        p_std_list.append(np.mean(scores < 0))
    avg_p_std = np.mean(p_std_list)

    # Accuracy on means
    acc_gmu = clf_gmu.score(X, y)
    acc_std = clf_std.score(X, y)

    # Relative improvement
    if avg_p_std > 1e-10:
        rel_improvement = (avg_p_std - avg_p_gmu) / avg_p_std * 100
    else:
        rel_improvement = 0.0

    # Convergence check
    objs = clf_gmu.history_["objective"]
    final_objs = objs[-min(50, len(objs)) :]
    converged = (max(final_objs) - min(final_objs)) < 0.001

    return {
        "avg_p_gmu": avg_p_gmu,
        "avg_p_std": avg_p_std,
        "rel_improvement": rel_improvement,
        "acc_gmu": acc_gmu,
        "acc_std": acc_std,
        "time_gmu": time_gmu,
        "time_std": time_std,
        "converged": converged,
        "final_objective": objs[-1],
        "w_norm_gmu": np.linalg.norm(clf_gmu.w_),
        "b_gmu": clf_gmu.b_,
    }


# ============================================================
# Experiment definitions
# ============================================================


def run_all_experiments(seed=42):
    """Run all experiments and return a DataFrame of results."""
    results = []

    def record(category, name, params, X, y, gmm_params, **eval_kwargs):
        print(f"  Running: {category} / {name} ...", end=" ", flush=True)
        try:
            metrics = evaluate(X, y, gmm_params, seed=seed, **eval_kwargs)
            metrics["category"] = category
            metrics["experiment"] = name
            metrics.update(params)
            results.append(metrics)
            print(
                f"P_gmu={metrics['avg_p_gmu']:.4f}, "
                f"P_std={metrics['avg_p_std']:.4f}, "
                f"improvement={metrics['rel_improvement']:.1f}%"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(
                {
                    "category": category,
                    "experiment": name,
                    "error": str(e),
                    **params,
                }
            )

    # ── 1. Varying uncertainty structure ──────────────────────

    print("\n=== 1. Uncertainty Structure ===")

    X, y, gmm = make_dataset_basic(2, 15, 4.0, 3, 0.4, seed)
    record("uncertainty", "heterogeneous_3comp", {"d": 2, "n": 30}, X, y, gmm)

    X, y, gmm = make_dataset_identical_uncertainty(2, 15, 4.0, seed)
    record("uncertainty", "identical_uncertainty", {"d": 2, "n": 30}, X, y, gmm)

    X, y, gmm = make_dataset_asymmetric_classes(2, 15, 4.0, seed)
    record("uncertainty", "asymmetric_classes", {"d": 2, "n": 30}, X, y, gmm)

    X, y, gmm = make_dataset_overlapping(2, 15, seed)
    record("uncertainty", "heavily_overlapping", {"d": 2, "n": 30}, X, y, gmm)

    X, y, gmm = make_dataset_zero_covariance(2, 15, 4.0, seed)
    record("uncertainty", "zero_covariance", {"d": 2, "n": 30}, X, y, gmm)

    X, y, gmm = make_dataset_concentrated_weights(2, 15, 4.0, seed)
    record("uncertainty", "concentrated_weights", {"d": 2, "n": 30}, X, y, gmm)

    # ── 2. Varying geometry ───────────────────────────────────

    print("\n=== 2. Geometry ===")

    for sep in [1.0, 2.0, 4.0, 8.0]:
        X, y, gmm = make_dataset_basic(2, 15, sep, 2, 0.4, seed)
        record("geometry", f"class_sep={sep}", {"d": 2, "n": 30, "sep": sep}, X, y, gmm)

    # Imbalanced
    rng_imb = np.random.RandomState(seed)
    X1, y1, g1 = make_dataset_basic(2, 5, 4.0, 2, 0.4, seed)
    X2, y2, g2 = make_dataset_basic(2, 30, 4.0, 2, 0.4, seed + 1)
    # Take 5 from class +1, 30 from class -1
    mask_pos = y1 == +1
    mask_neg = y2 == -1
    X_imb = np.vstack([X1[mask_pos], X2[mask_neg]])
    y_imb = np.concatenate([y1[mask_pos], y2[mask_neg]])
    g_imb = [g1[i] for i in range(len(g1)) if mask_pos[i]] + [
        g2[i] for i in range(len(g2)) if mask_neg[i]
    ]
    record(
        "geometry", "imbalanced_5vs30", {"d": 2, "n": len(y_imb)}, X_imb, y_imb, g_imb
    )

    # ── 3. Varying dimensionality ─────────────────────────────

    print("\n=== 3. Dimensionality ===")

    for d in [2, 5, 10, 20, 50, 100]:
        n_pc = min(15, max(d + 2, 10))
        X, y, gmm = make_dataset_basic(d, n_pc, 4.0, 2, 0.4, seed)
        record("dimensionality", f"d={d}", {"d": d, "n": 2 * n_pc}, X, y, gmm)

    # ── 4. Varying sample size ────────────────────────────────

    print("\n=== 4. Sample Size ===")

    for n_pc in [3, 5, 10, 25, 50, 100]:
        X, y, gmm = make_dataset_basic(2, n_pc, 4.0, 2, 0.4, seed)
        record("sample_size", f"n_per_class={n_pc}", {"d": 2, "n": 2 * n_pc}, X, y, gmm)

    # ── 5. Varying number of components ───────────────────────

    print("\n=== 5. Number of Components ===")

    for K in [1, 2, 3, 5, 10, 20]:
        X, y, gmm = make_dataset_many_components(2, 15, 4.0, K, seed)
        record("n_components", f"K={K}", {"d": 2, "n": 30, "K": K}, X, y, gmm)

    # ── 6. Edge cases ─────────────────────────────────────────

    print("\n=== 6. Edge Cases ===")

    X, y, gmm = make_dataset_single_point_per_class(2, seed)
    record("edge_case", "single_point_per_class", {"d": 2, "n": 2}, X, y, gmm)

    X, y, gmm = make_dataset_single_point_per_class(10, seed)
    record("edge_case", "single_point_per_class_10d", {"d": 10, "n": 2}, X, y, gmm)

    # High d, low n
    X, y, gmm = make_dataset_basic(50, 3, 4.0, 2, 0.4, seed)
    record("edge_case", "high_d_low_n_50d_6pts", {"d": 50, "n": 6}, X, y, gmm)

    # ── 7. Hyperparameter sensitivity ─────────────────────────

    print("\n=== 7. Hyperparameter Sensitivity ===")

    X_base, y_base, gmm_base = make_dataset_basic(2, 15, 4.0, 2, 0.4, seed)

    for lam in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        record(
            "hyperparam",
            f"lambda={lam}",
            {"d": 2, "n": 30, "lambda": lam},
            X_base,
            y_base,
            gmm_base,
            lam=lam,
        )

    for lr in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        record(
            "hyperparam",
            f"lr={lr}",
            {"d": 2, "n": 30, "lr": lr},
            X_base,
            y_base,
            gmm_base,
            lr_init=lr,
        )

    for n_ep in [50, 100, 200, 500, 1000]:
        record(
            "hyperparam",
            f"epochs={n_ep}",
            {"d": 2, "n": 30, "epochs": n_ep},
            X_base,
            y_base,
            gmm_base,
            n_epochs=n_ep,
        )

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Run SVM-GMU experiments")
    parser.add_argument(
        "--output",
        "-o",
        default="experiments/results.csv",
        help="Output CSV file path (default: experiments/results.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SVM-GMU Comprehensive Experiments")
    print("=" * 60)

    df = run_all_experiments(seed=args.seed)

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY BY CATEGORY")
    print("=" * 60)

    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        if "rel_improvement" in sub.columns:
            valid = sub.dropna(subset=["rel_improvement"])
            if len(valid) > 0:
                avg_imp = valid["rel_improvement"].mean()
                min_imp = valid["rel_improvement"].min()
                max_imp = valid["rel_improvement"].max()
                print(
                    f"  {cat:25s}: "
                    f"avg improvement = {avg_imp:+.1f}%, "
                    f"range = [{min_imp:+.1f}%, {max_imp:+.1f}%], "
                    f"n_experiments = {len(valid)}"
                )


if __name__ == "__main__":
    main()
