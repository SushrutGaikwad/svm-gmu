"""Unit test for the BIC-selected EM fit used in Experiment 4."""

import importlib.util
from pathlib import Path

import numpy as np

from svm_gmu._validation import validate_sample_uncertainty

# Load the generator module by path (experiments/ is not a package).
_MOD_PATH = Path(__file__).resolve().parents[1] / "experiments" / "em_fitted_gmu_figs.py"
_spec = importlib.util.spec_from_file_location("em_fitted_gmu_figs", _MOD_PATH)
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)


def test_fit_gmm_bic_recovers_moments_and_valid_shape():
    rng = np.random.default_rng(0)
    # A clear 2-component GMM well separated so EM/BIC behave.
    gmm = {
        "weights": np.array([0.5, 0.5]),
        "means": np.array([[-3.0, 0.0], [3.0, 0.0]]),
        "covariances": np.array([np.eye(2) * 0.2, np.eye(2) * 0.2]),
    }
    pts = em.sample_from_gmm(gmm, 4000, rng)

    out = em.fit_gmm_bic(pts, m_candidates=[1, 2, 3, 4], n_init=3, seed=0)

    # Output must be a valid single-sample uncertainty dict.
    validated = validate_sample_uncertainty([out], n_samples=1, n_features=2)
    assert len(validated) == 1

    # Overall mean of the fitted mixture matches the true overall mean (~origin).
    overall_mean = (out["weights"][:, None] * out["means"]).sum(axis=0)
    assert np.allclose(overall_mean, [0.0, 0.0], atol=0.3)

    # BIC should prefer >= 2 components for this clearly bimodal data.
    assert len(out["weights"]) >= 2
