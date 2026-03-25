"""
SVM-GMU: Support Vector Machines with Gaussian Mixture Sample Uncertainty.

A classifier that accounts for per-sample non-Gaussian uncertainty
modelled as Gaussian mixtures, extending the SVM-GSU framework.
"""

from .classifier import SVMGMU
from .data import generate_gmm_dataset, validate_gmm_dataset
from .loss import expected_hinge_loss_gaussian, expected_hinge_loss_gmm
from .visualization import plot_uncertainties, plot_decision_boundary, plot_comparison

__all__ = [
    "SVMGMU",
    "validate_gmm_dataset",
    "generate_gmm_dataset",
    "expected_hinge_loss_gaussian",
    "expected_hinge_loss_gmm",
    "plot_uncertainties",
    "plot_decision_boundary",
    "plot_comparison",
]
