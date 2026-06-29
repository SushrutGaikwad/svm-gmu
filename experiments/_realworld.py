"""Helpers for the real-data SVM-GMU vs SVM-GSU experiments (Plan 1: MNIST).

Pure functions plus per-seed and full-experiment drivers. All numerical logic
lives here so the notebooks stay thin, mirroring experiments/_common.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.stats import binomtest, ttest_rel, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    accuracy_score as _acc,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

from svm_gmu import SvmGmu


_M_CANDIDATES = list(range(1, 9))
_N_INIT = 3


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


def mcnemar_pvalue(
    y_true: NDArray[np.floating],
    pred_a: NDArray[np.floating],
    pred_b: NDArray[np.floating],
) -> float:
    """Exact two-sided McNemar p-value comparing two classifiers' predictions."""
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)


def paired_seed_tests(acc_a, acc_b) -> dict:
    """Paired Wilcoxon signed-rank and paired t-test over per-seed accuracies."""
    acc_a = np.asarray(acc_a, dtype=float)
    acc_b = np.asarray(acc_b, dtype=float)
    try:
        w_p = float(wilcoxon(acc_a, acc_b).pvalue)
    except ValueError:
        w_p = 1.0  # zero differences across all seeds
    t_p = float(ttest_rel(acc_a, acc_b).pvalue)
    return {"wilcoxon_p": w_p, "ttest_p": t_p}


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


def run_ladder(X_tr, y_tr, X_te, y_te, su_by_model, lam_grid, n_folds, svm_kwargs, seed) -> dict:
    """Fit each model in the ladder and return its boundary, predictions, metrics."""
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    y_te = np.asarray(y_te, dtype=np.float64)
    out = {}
    for key, su in su_by_model.items():
        lam = select_lambda_cv(X_tr, y_tr, su, lam_grid, n_folds, seed, svm_kwargs)
        model = SvmGmu(lam=lam, random_state=seed, **svm_kwargs)
        model.fit(X_tr, y_tr, sample_uncertainty=su)
        y_pred = model.predict(X_te)
        y_score = model.decision_function(X_te)
        out[key] = {
            "lam": lam,
            "w": model.coef_.copy(),
            "b": float(model.intercept_),
            "y_pred": y_pred,
            "y_score": y_score,
            "metrics": evaluate_metrics(y_te, y_pred, y_score),
        }
    return out


def select_lambda_cv(X, y, su, lam_grid, n_folds, seed, svm_kwargs) -> float:
    """Pick lambda by best mean stratified k-fold validation accuracy."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_lam, best_acc = lam_grid[0], -1.0
    for lam in lam_grid:
        fold_acc = []
        for tr, va in skf.split(X, y):
            su_tr = None if su is None else [su[i] for i in tr]
            model = SvmGmu(lam=lam, random_state=seed, **svm_kwargs)
            model.fit(X[tr], y[tr], sample_uncertainty=su_tr)
            fold_acc.append(_acc(y[va], model.predict(X[va])))
        mean_acc = float(np.mean(fold_acc))
        if mean_acc > best_acc:
            best_acc, best_lam = mean_acc, lam
    return best_lam


def fit_gmm_bic_cov(cloud, m_candidates, n_init, seed, cov_type="diag") -> dict:
    """Fit a GMM choosing the component count by BIC, with the given cov type.

    Returns weights (M,), means (M, d), and covariances shaped (M, d) for
    'diag' or (M, d, d) for 'full', matching the SvmGmu uncertainty format.
    """
    best_bic, best = np.inf, None
    for m in m_candidates:
        gm = GaussianMixture(
            n_components=m, covariance_type=cov_type, n_init=n_init, random_state=seed
        ).fit(cloud)
        bic = gm.bic(cloud)
        if bic < best_bic:
            best_bic, best = bic, gm
    return {
        "weights": best.weights_.copy(),
        "means": best.means_.copy(),
        "covariances": best.covariances_.copy(),
    }


def _select_examples(images, labels, digit, n, rng):
    idx = np.where(labels == digit)[0]
    chosen = rng.choice(idx, size=n, replace=False)
    return images[chosen]


def run_mnist_seed(
    images, labels, digit_pos, digit_neg, seed, *,
    n_train, n_test, n_aug, rot_range, max_shift, pca_dim,
    k_anchors, n_per_anchor, lam_grid, n_folds, svm_kwargs, cov_type="diag",
) -> dict:
    """One seed of the MNIST GMU-vs-GSU comparison; builds the ladder and metrics."""
    rng = np.random.default_rng(seed)

    # Disjoint train and test images for each digit.
    pos = _select_examples(images, labels, digit_pos, n_train + n_test, rng)
    neg = _select_examples(images, labels, digit_neg, n_train + n_test, rng)
    pos_tr, pos_te = pos[:n_train], pos[n_train:]
    neg_tr, neg_te = neg[:n_train], neg[n_train:]

    train_imgs = np.vstack([pos_tr, neg_tr])
    test_imgs = np.vstack([pos_te, neg_te])
    y_tr = np.array([1.0] * n_train + [-1.0] * n_train)
    y_te = np.array([1.0] * n_test + [-1.0] * n_test)

    # Augmentation clouds for the training images, then PCA on the pooled clouds.
    clouds = [augmentation_cloud(img, n_aug, rng, rot_range, max_shift) for img in train_imgs]
    pca = PCA(n_components=pca_dim, random_state=seed).fit(np.vstack(clouds))

    clouds_pca = [pca.transform(c) for c in clouds]
    X_tr = pca.transform(train_imgs)
    X_te = pca.transform(test_imgs)

    # Uncertainty ladder per training example.
    su_iso = [iso_gaussian(c) for c in clouds_pca]
    su_m0 = [moment_gaussian(c, cov_type) for c in clouds_pca]
    su_m1 = [
        structural_components(img, k_anchors, rot_range, n_per_anchor, rng, max_shift, pca, cov_type)
        for img in train_imgs
    ]
    su_m2 = [fit_gmm_bic_cov(c, _M_CANDIDATES, _N_INIT, seed, cov_type) for c in clouds_pca]
    bic_counts = [len(g["weights"]) for g in su_m2]

    su_by_model = {"B0": None, "B1": su_iso, "M0": su_m0, "M1": su_m1, "M2": su_m2}
    models = run_ladder(X_tr, y_tr, X_te, y_te, su_by_model, lam_grid, n_folds, svm_kwargs, seed)

    mcnemar_p = mcnemar_pvalue(y_te, models["M2"]["y_pred"], models["M0"]["y_pred"])
    return {"models": models, "mcnemar_p": mcnemar_p, "bic_counts": bic_counts}
