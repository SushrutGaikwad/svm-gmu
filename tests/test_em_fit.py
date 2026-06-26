"""Unit test for the BIC-selected EM fit used in Experiment 4."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from svm_gmu._validation import validate_sample_uncertainty

_EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXP_DIR))
_spec = importlib.util.spec_from_file_location("_common", _EXP_DIR / "_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_fit_gmm_bic_recovers_moments_and_valid_shape():
    rng = np.random.default_rng(0)
    gmm = {
        "weights": np.array([0.5, 0.5]),
        "means": np.array([[-3.0, 0.0], [3.0, 0.0]]),
        "covariances": np.array([np.eye(2) * 0.2, np.eye(2) * 0.2]),
    }
    pts = C.sample_from_gmm(gmm, 4000, rng)

    out = C.fit_gmm_bic(pts, m_candidates=[1, 2, 3, 4], n_init=3, seed=0)

    validated = validate_sample_uncertainty([out], n_samples=1, n_features=2)
    assert len(validated) == 1

    overall_mean = (out["weights"][:, None] * out["means"]).sum(axis=0)
    assert np.allclose(overall_mean, [0.0, 0.0], atol=0.3)
    assert len(out["weights"]) >= 2
