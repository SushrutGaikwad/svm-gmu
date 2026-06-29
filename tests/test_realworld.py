# tests/test_realworld.py
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

_EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXP_DIR))
_spec = importlib.util.spec_from_file_location("_realworld", _EXP_DIR / "_realworld.py")
RW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RW)


def test_augment_image_zero_is_identity():
    img = np.zeros((28, 28))
    img[8:14, 6:10] = 1.0  # asymmetric patch
    rng = np.random.default_rng(0)
    out = RW.augment_image(img, rng, rot_deg=0.0, max_shift=0.0)
    assert out.shape == (28, 28)
    assert np.allclose(out, img, atol=1e-6)


def test_augment_image_rotation_changes_asymmetric_patch():
    img = np.zeros((28, 28))
    img[8:14, 6:10] = 1.0
    rng = np.random.default_rng(0)
    out = RW.augment_image(img, rng, rot_deg=90.0, max_shift=0.0)
    assert not np.allclose(out, img)


def test_augmentation_cloud_shape_and_zero_range():
    img = np.zeros(784)
    img[300:320] = 1.0
    rng = np.random.default_rng(0)
    cloud = RW.augmentation_cloud(img, n_aug=16, rng=rng, rot_range=0.0, max_shift=0.0)
    assert cloud.shape == (16, 784)
    assert np.allclose(cloud, img[None, :], atol=1e-6)


def test_moment_gaussian_diag():
    cloud = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    g = RW.moment_gaussian(cloud, "diag")
    assert g["weights"].shape == (1,) and np.isclose(g["weights"][0], 1.0)
    assert np.allclose(g["means"][0], [1.0, 1.0])
    assert g["covariances"].shape == (1, 2)
    assert np.allclose(g["covariances"][0], [1.0, 1.0])


def test_moment_gaussian_full_shape():
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(50, 3))
    g = RW.moment_gaussian(cloud, "full")
    assert g["covariances"].shape == (1, 3, 3)


def test_iso_gaussian_is_isotropic():
    cloud = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    g = RW.iso_gaussian(cloud)
    assert g["covariances"].shape == (1, 2)
    assert np.allclose(g["covariances"][0], [2.0, 2.0])  # mean of per-dim var (4, 0)


def test_structural_components_shapes():
    rng = np.random.default_rng(0)
    img = np.zeros(784)
    img[300:316] = 1.0
    pool = RW.augmentation_cloud(img, 60, rng, rot_range=20.0, max_shift=0.0)
    pca = PCA(n_components=3, random_state=0).fit(pool)
    su = RW.structural_components(
        img, k=4, rot_range=20.0, n_per_anchor=20, rng=rng, max_shift=0.0, pca=pca, cov_type="diag"
    )
    assert su["weights"].shape == (4,) and np.isclose(su["weights"].sum(), 1.0)
    assert su["means"].shape == (4, 3)
    assert su["covariances"].shape == (4, 3)


def test_evaluate_metrics_perfect():
    y = np.array([1.0, 1.0, -1.0, -1.0])
    pred = y.copy()
    score = np.array([2.0, 1.0, -1.0, -2.0])
    m = RW.evaluate_metrics(y, pred, score)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["auc"] == 1.0
    assert m["ap"] == 1.0


def test_evaluate_metrics_half_accuracy():
    y = np.array([1.0, 1.0, -1.0, -1.0])
    pred = np.array([1.0, -1.0, 1.0, -1.0])
    score = np.array([0.1, -0.1, 0.1, -0.1])
    m = RW.evaluate_metrics(y, pred, score)
    assert np.isclose(m["accuracy"], 0.5)


def test_mcnemar_identical_is_one():
    y = np.array([1.0, -1.0, 1.0, -1.0])
    assert RW.mcnemar_pvalue(y, y.copy(), y.copy()) == 1.0


def test_mcnemar_one_sided_is_small():
    y = np.ones(12)
    a = np.ones(12)        # a always right
    b = -np.ones(12)       # b always wrong
    assert RW.mcnemar_pvalue(y, a, b) < 0.05


def test_paired_seed_tests_detects_difference():
    a = np.array([0.90, 0.92, 0.88, 0.91, 0.93])
    b = np.array([0.80, 0.82, 0.79, 0.81, 0.83])
    r = RW.paired_seed_tests(a, b)
    assert r["wilcoxon_p"] < 0.1
    assert r["ttest_p"] < 0.05


def test_select_lambda_returns_grid_value():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(2, 0.5, (20, 2)), rng.normal(-2, 0.5, (20, 2))])
    y = np.array([1.0] * 20 + [-1.0] * 20)
    grid = [1e-3, 1e-2, 1e-1]
    lam = RW.select_lambda_cv(X, y, None, grid, n_folds=3, seed=0,
                              svm_kwargs=dict(max_iter=500, batch_size=16))
    assert lam in grid
