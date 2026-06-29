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


def test_paired_seed_tests_all_equal_is_not_significant():
    a = np.array([0.9, 0.9, 0.9])
    r = RW.paired_seed_tests(a, a.copy())
    assert r["wilcoxon_p"] == 1.0
    assert r["ttest_p"] == 1.0


def test_select_lambda_returns_grid_value():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(2, 0.5, (20, 2)), rng.normal(-2, 0.5, (20, 2))])
    y = np.array([1.0] * 20 + [-1.0] * 20)
    grid = [1e-3, 1e-2, 1e-1]
    lam = RW.select_lambda_cv(X, y, None, grid, n_folds=3, seed=0,
                              svm_kwargs=dict(max_iter=500, batch_size=16))
    assert lam in grid


def test_run_ladder_keys_and_separable_accuracy():
    rng = np.random.default_rng(0)
    X_tr = np.vstack([rng.normal(2, 0.4, (20, 2)), rng.normal(-2, 0.4, (20, 2))])
    y_tr = np.array([1.0] * 20 + [-1.0] * 20)
    X_te = np.vstack([rng.normal(2, 0.4, (10, 2)), rng.normal(-2, 0.4, (10, 2))])
    y_te = np.array([1.0] * 10 + [-1.0] * 10)
    clouds = [X_tr[i] + 0.05 * rng.normal(size=(30, 2)) for i in range(40)]
    su_iso = [RW.iso_gaussian(c) for c in clouds]
    su_m0 = [RW.moment_gaussian(c, "diag") for c in clouds]
    su_by_model = {"B0": None, "B1": su_iso, "M0": su_m0, "M2": su_m0}
    res = RW.run_ladder(X_tr, y_tr, X_te, y_te, su_by_model,
                        lam_grid=[1e-2, 1e-1], n_folds=3,
                        svm_kwargs=dict(max_iter=500, batch_size=16), seed=0)
    assert set(res) == {"B0", "B1", "M0", "M2"}
    for key in res:
        assert 0.0 <= res[key]["metrics"]["accuracy"] <= 1.0
    assert res["M0"]["metrics"]["accuracy"] > 0.8


def test_run_mnist_seed_smoke():
    rng = np.random.default_rng(0)
    imgs, labs = [], []
    for _ in range(30):
        im = np.zeros((28, 28)); im[5:10, :] = rng.uniform(0.5, 1.0, (5, 28))
        imgs.append(im.ravel()); labs.append(4)
    for _ in range(30):
        im = np.zeros((28, 28)); im[18:23, :] = rng.uniform(0.5, 1.0, (5, 28))
        imgs.append(im.ravel()); labs.append(9)
    images = np.array(imgs); labels = np.array(labs)
    out = RW.run_mnist_seed(
        images, labels, digit_pos=4, digit_neg=9, seed=0,
        n_train=10, n_test=10, n_aug=40, rot_range=15.0, max_shift=1.0,
        pca_dim=5, k_anchors=4, n_per_anchor=15, lam_grid=[1e-2, 1e-1],
        n_folds=3, svm_kwargs=dict(max_iter=300, batch_size=8), cov_type="diag",
    )
    assert {"B0", "B1", "M0", "M1", "M2"}.issubset(out["models"].keys())
    assert "mcnemar_p" in out and 0.0 <= out["mcnemar_p"] <= 1.0
    assert len(out["bic_counts"]) == 20  # n_train per class x 2 = 20 training mixtures


def _fake_mnist(rng):
    imgs, labs = [], []
    for _ in range(40):
        im = np.zeros((28, 28)); im[5:11, :] = rng.uniform(0.5, 1.0, (6, 28))
        imgs.append(im.ravel()); labs.append(4)
    for _ in range(40):
        im = np.zeros((28, 28)); im[17:23, :] = rng.uniform(0.5, 1.0, (6, 28))
        imgs.append(im.ravel()); labs.append(9)
    return np.array(imgs), np.array(labs)


def test_run_mnist_experiment_smoke():
    images, labels = _fake_mnist(np.random.default_rng(1))
    config = dict(
        digit_pos=4, digit_neg=9, master_seed=2026, n_seeds=2,
        rot_ladder=[0.0, 20.0], train_ladder=[8], fixed_R=20.0, fixed_n_train=8,
        n_test=8, n_aug=40, max_shift=1.0, pca_dim=5, k_anchors=4, n_per_anchor=15,
        lam_grid=[1e-2, 1e-1], n_folds=3, svm_kwargs=dict(max_iter=300, batch_size=8),
        cov_type="diag",
    )
    res = RW.run_mnist_experiment(images, labels, config)
    assert "rot_sweep" in res and "train_sweep" in res and "significance" in res
    assert set(res["rot_sweep"].keys()) == {0.0, 20.0}
    # Each cell holds per-model metric bands (median, q25, q75).
    cell = res["rot_sweep"][20.0]["M2"]["accuracy"]
    assert len(cell) == 3


def test_plot_mnist_2d_panel_runs():
    import matplotlib
    matplotlib.use("Agg")
    images, labels = _fake_mnist(np.random.default_rng(5))
    config = dict(
        digit_pos=4, digit_neg=9, master_seed=2026, n_seeds=2,
        rot_ladder=[20.0], train_ladder=[8], fixed_R=20.0, fixed_n_train=8,
        n_test=8, n_aug=40, max_shift=1.0, pca_dim=20, k_anchors=4, n_per_anchor=15,
        lam_grid=[1e-2, 1e-1], n_folds=3, svm_kwargs=dict(max_iter=300, batch_size=8),
        cov_type="diag",
    )
    fig = RW.plot_mnist_2d_panel(images, labels, config, seed=0)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_make_sweep_figure_runs():
    import matplotlib
    matplotlib.use("Agg")
    images, labels = _fake_mnist(np.random.default_rng(2))
    config = dict(
        digit_pos=4, digit_neg=9, master_seed=2026, n_seeds=2,
        rot_ladder=[0.0, 20.0], train_ladder=[8], fixed_R=20.0, fixed_n_train=8,
        n_test=8, n_aug=40, max_shift=1.0, pca_dim=5, k_anchors=4, n_per_anchor=15,
        lam_grid=[1e-2, 1e-1], n_folds=3, svm_kwargs=dict(max_iter=300, batch_size=8),
        cov_type="diag",
    )
    res = RW.run_mnist_experiment(images, labels, config)
    fig = RW.make_sweep_figure(res["rot_sweep"], xlabel="Rotation range (deg)", title="Accuracy vs rotation")
    assert fig is not None
    assert len(fig.axes) >= 1
