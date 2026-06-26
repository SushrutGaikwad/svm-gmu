# tests/test_common.py
import importlib.util
import sys
from pathlib import Path

import numpy as np

from svm_gmu._loss import compute_d_mu, compute_d_sigma
from svm_gmu._validation import validate_sample_uncertainty

# Load experiments/_common.py by path (experiments/ is not a package).
_EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXP_DIR))
_spec = importlib.util.spec_from_file_location("_common", _EXP_DIR / "_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_dataset_shapes():
    assert C.X.shape == (6, 2)
    assert C.y.shape == (6,)
    assert len(C.SAMPLE_UNCERTAINTY) == 6
    for su in C.SAMPLE_UNCERTAINTY:
        assert np.isclose(su["weights"].sum(), 1.0)
        assert su["means"].shape[1] == 2
        assert su["covariances"].shape[1:] == (2, 2)


def test_sample_from_gmm_shape_and_reproducible():
    gmm = C.SAMPLE_UNCERTAINTY[0]
    a = C.sample_from_gmm(gmm, 500, np.random.default_rng(0))
    b = C.sample_from_gmm(gmm, 500, np.random.default_rng(0))
    assert a.shape == (500, 2)
    assert np.allclose(a, b)  # same seed -> identical draws


def test_boundary_metrics_identity_and_orthogonal():
    w = np.array([1.0, 0.0])
    # Identical boundaries -> all metrics zero.
    angle, offset, rms = C.boundary_metrics_2d(w, 0.0, w, 0.0)
    assert angle == 0.0 and offset == 0.0 and rms == 0.0
    # Orthogonal normals -> 90 degrees.
    angle2, _, _ = C.boundary_metrics_2d(w, 0.0, np.array([0.0, 1.0]), 0.0)
    assert np.isclose(angle2, 90.0)


def test_make_seeds_and_band():
    seeds = C.make_seeds(2026, 5)
    assert len(seeds) == 5
    assert len(set(seeds)) == 5  # independent, distinct
    assert C.make_seeds(2026, 5) == seeds  # reproducible from the master
    med, lo, hi = C.band([1.0, 2.0, 3.0, 4.0])
    assert med == 2.5 and lo <= med <= hi


def test_moment_match_single_component_is_identity():
    gmm = {
        "weights": np.array([1.0]),
        "means": np.array([[1.5, -2.0]]),
        "covariances": np.array([[[0.3, 0.1], [0.1, 0.4]]]),
    }
    out = C.moment_match_gmm(gmm)
    assert out["weights"].shape == (1,)
    assert np.allclose(out["means"][0], [1.5, -2.0])
    assert np.allclose(out["covariances"][0], [[0.3, 0.1], [0.1, 0.4]])


def test_moment_match_two_symmetric_components():
    # Two equal-weight points at +/-(2,0) with tiny isotropic cov.
    gmm = {
        "weights": np.array([0.5, 0.5]),
        "means": np.array([[-2.0, 0.0], [2.0, 0.0]]),
        "covariances": np.array([np.eye(2) * 0.01, np.eye(2) * 0.01]),
    }
    out = C.moment_match_gmm(gmm)
    assert np.allclose(out["means"][0], [0.0, 0.0])
    # Between-component variance along x is 0.5*4 + 0.5*4 = 4, plus 0.01 within.
    assert np.isclose(out["covariances"][0][0, 0], 4.01)
    assert np.isclose(out["covariances"][0][1, 1], 0.01)
    # Output is a valid single-sample uncertainty.
    validate_sample_uncertainty([out], n_samples=1, n_features=2)


def test_fit_gmu_returns_boundary_shapes():
    w, b = C.fit_gmu(C.X, C.y, C.SAMPLE_UNCERTAINTY)
    assert w.shape == (2,)
    assert isinstance(b, float)


def test_build_sample_cloud_shapes_and_labels():
    Xc, yc = C.build_sample_cloud(C.SAMPLE_UNCERTAINTY, C.y, 10, np.random.default_rng(0))
    assert Xc.shape == (60, 2)
    assert yc.shape == (60,)
    # First 10 rows are sample 0 (label +1), rows 30-40 are sample 3 (label -1).
    assert np.all(yc[:10] == 1)
    assert np.all(yc[30:40] == -1)


def test_fit_gmm_bic_prefers_two_components():
    gmm = {
        "weights": np.array([0.5, 0.5]),
        "means": np.array([[-3.0, 0.0], [3.0, 0.0]]),
        "covariances": np.array([np.eye(2) * 0.2, np.eye(2) * 0.2]),
    }
    pts = C.sample_from_gmm(gmm, 4000, np.random.default_rng(0))
    out = C.fit_gmm_bic(pts, [1, 2, 3, 4], n_init=3, seed=0)
    validate_sample_uncertainty([out], n_samples=1, n_features=2)
    assert len(out["weights"]) >= 2


def test_random_rotation_is_orthogonal():
    Q = C.random_rotation(7, np.random.default_rng(1))
    assert Q.shape == (7, 7)
    assert np.allclose(Q @ Q.T, np.eye(7), atol=1e-10)


def test_lift_dataset_shapes_and_psd():
    rng = np.random.default_rng(2)
    Xl, sul, Q = C.lift_dataset(C.X, C.SAMPLE_UNCERTAINTY, d=6, sigma2_noise=0.05, rng=rng)
    assert Xl.shape == (6, 6)
    assert len(sul) == 6
    for su in sul:
        M = su["means"].shape[0]
        assert su["means"].shape == (M, 6)
        assert su["covariances"].shape == (M, 6, 6)
        for cov in su["covariances"]:
            assert np.allclose(cov, cov.T, atol=1e-10)
            assert np.linalg.eigvalsh(cov).min() > -1e-8


def test_lift_preserves_loss_geometry_in_signal_subspace():
    # The key invariant: with w_d = Q @ [w2; 0], every component's d_mu and
    # d_sigma equal their 2D values, because the noise dims carry zero weight.
    rng = np.random.default_rng(3)
    d, sigma2 = 9, 0.07
    Xl, sul, Q = C.lift_dataset(C.X, C.SAMPLE_UNCERTAINTY, d, sigma2, rng)
    w2 = np.array([-1.2527, 1.2589])
    b = -0.9470
    w_d = Q @ np.concatenate([w2, np.zeros(d - 2)])
    for gmm2, gmmd in zip(C.SAMPLE_UNCERTAINTY, sul):
        for m in range(len(gmm2["weights"])):
            dmu2 = compute_d_mu(w2, b, gmm2["means"][m], 1.0)
            dmud = compute_d_mu(w_d, b, gmmd["means"][m], 1.0)
            assert np.isclose(dmu2, dmud, atol=1e-9)
            dsig2 = compute_d_sigma(w2, gmm2["covariances"][m])
            dsigd = compute_d_sigma(w_d, gmmd["covariances"][m])
            assert np.isclose(dsig2, dsigd, atol=1e-9)


def test_mc_eval_cloud_shape():
    cloud = C.mc_eval_cloud(C.SAMPLE_UNCERTAINTY, 100, np.random.default_rng(4))
    assert cloud.shape == (600, 2)
