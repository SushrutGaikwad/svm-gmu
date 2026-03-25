"""
Expected hinge loss functions for SVM-GSU and SVM-GMU.

SVM-GSU: closed-form expected hinge loss under a single Gaussian.
SVM-GMU: weighted sum of SVM-GSU losses over Gaussian mixture components.
"""

import numpy as np
from scipy.special import erf


def expected_hinge_loss_gaussian(w, b, mu_ij, cov_ij, y_i):
    """
    Compute the expected hinge loss for a single Gaussian component.

    This is L_GSU(w, b; mu_ij, Sigma_ij, y_i) from the SVM-GSU paper [1].

    Parameters
    ----------
    w : ndarray of shape (d,)
        Weight vector.
    b : float
        Bias term.
    mu_ij : ndarray of shape (d,)
        Mean of the j-th Gaussian component of point i.
    cov_ij : ndarray of shape (d, d)
        Covariance matrix of the j-th component of point i.
    y_i : int
        Label in {-1, +1}.

    Returns
    -------
    loss : float
        Scalar expected hinge loss.
    grad_w : ndarray of shape (d,)
        Gradient with respect to w.
    grad_b : float
        Gradient with respect to b.

    References
    ----------
    [1] C. Tzelepis, V. Mezaris, I. Patras, "Linear Maximum Margin
        Classifier for Learning from Uncertain Data", IEEE TPAMI, 2018.
    """
    d_x_ij = 1.0 - y_i * (np.dot(w, mu_ij) + b)

    wSw = np.dot(w, cov_ij @ w)
    d_sigma_ij = np.sqrt(2.0 * max(wSw, 1e-15))

    ratio = d_x_ij / d_sigma_ij

    erf_term = erf(ratio)
    exp_term = np.exp(-(ratio**2))

    loss = (
        0.5 * d_x_ij * (erf_term + 1.0)
        + (d_sigma_ij / (2.0 * np.sqrt(np.pi))) * exp_term
    )

    prob_active = 0.5 * (erf_term + 1.0)

    dd_x_dw = -y_i * mu_ij
    dd_x_db = -y_i

    Sw = cov_ij @ w
    dd_sigma_dw = (2.0 * Sw) / max(d_sigma_ij, 1e-15)

    grad_w = (
        prob_active * dd_x_dw + (1.0 / (2.0 * np.sqrt(np.pi))) * exp_term * dd_sigma_dw
    )
    grad_b = prob_active * dd_x_db

    return loss, grad_w, grad_b


def expected_hinge_loss_gmm(w, b, gmm_param_i, y_i):
    """
    Compute the expected hinge loss for a data point with GMM uncertainty.

    L_{GMU,i}(w, b) = sum_{j=1}^{K_i} pi_ij * L_GSU(w, b; mu_ij, Sigma_ij, y_i)

    Parameters
    ----------
    w : ndarray of shape (d,)
        Weight vector.
    b : float
        Bias term.
    gmm_param_i : dict
        GMM parameters for point i with keys:
            'weights' : ndarray of shape (K_i,) — mixture weights pi_ij.
            'means'   : ndarray of shape (K_i, d) — component means mu_ij.
            'covs'    : ndarray of shape (K_i, d, d) — component covariances Sigma_ij.
    y_i : int
        Label in {-1, +1}.

    Returns
    -------
    loss : float
        Scalar expected hinge loss.
    grad_w : ndarray of shape (d,)
        Gradient with respect to w.
    grad_b : float
        Gradient with respect to b.
    """
    pi_ij = gmm_param_i["weights"]
    mu_ij = gmm_param_i["means"]
    sigma_ij = gmm_param_i["covs"]
    K_i = len(pi_ij)
    d = len(w)

    total_loss = 0.0
    total_grad_w = np.zeros(d)
    total_grad_b = 0.0

    for j in range(K_i):
        loss_j, grad_w_j, grad_b_j = expected_hinge_loss_gaussian(
            w, b, mu_ij[j], sigma_ij[j], y_i
        )
        total_loss += pi_ij[j] * loss_j
        total_grad_w += pi_ij[j] * grad_w_j
        total_grad_b += pi_ij[j] * grad_b_j

    return total_loss, total_grad_w, total_grad_b
