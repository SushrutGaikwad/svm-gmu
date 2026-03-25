"""
Tests for the expected hinge loss functions.
"""

import numpy as np
import pytest

from svm_gmu import expected_hinge_loss_gaussian, expected_hinge_loss_gmm


class TestExpectedHingeLossGaussian:
    """Tests for the single-Gaussian expected hinge loss."""

    def test_returns_three_values(self):
        w = np.array([1.0, 0.0])
        loss, grad_w, grad_b = expected_hinge_loss_gaussian(
            w,
            b=0.0,
            mu_ij=np.array([1.0, 0.0]),
            cov_ij=np.eye(2) * 0.1,
            y_i=+1,
        )
        assert isinstance(loss, float)
        assert grad_w.shape == (2,)
        assert isinstance(grad_b, float)

    def test_loss_nonnegative(self):
        rng = np.random.RandomState(0)
        for _ in range(20):
            d = rng.randint(2, 10)
            w = rng.randn(d)
            b = rng.randn()
            mu = rng.randn(d)
            A = rng.randn(d, d) * 0.3
            cov = A @ A.T + np.eye(d) * 0.01
            y_i = rng.choice([-1, +1])
            loss, _, _ = expected_hinge_loss_gaussian(w, b, mu, cov, y_i)
            assert loss >= -1e-10, f"Loss should be non-negative, got {loss}"

    def test_gradient_w_numerical(self):
        w = np.array([1.0, -0.5])
        b = 0.3
        mu = np.array([0.5, 1.0])
        cov = np.array([[0.2, 0.05], [0.05, 0.15]])
        y_i = +1
        eps = 1e-5

        _, grad_w, _ = expected_hinge_loss_gaussian(w, b, mu, cov, y_i)

        num_grad_w = np.zeros_like(w)
        for k in range(len(w)):
            w_plus = w.copy()
            w_plus[k] += eps
            w_minus = w.copy()
            w_minus[k] -= eps
            l_plus, _, _ = expected_hinge_loss_gaussian(w_plus, b, mu, cov, y_i)
            l_minus, _, _ = expected_hinge_loss_gaussian(w_minus, b, mu, cov, y_i)
            num_grad_w[k] = (l_plus - l_minus) / (2 * eps)

        np.testing.assert_allclose(grad_w, num_grad_w, atol=1e-4)

    def test_gradient_b_numerical(self):
        w = np.array([0.8, 0.3])
        b = -0.2
        mu = np.array([-1.0, 0.5])
        cov = np.array([[0.1, 0.0], [0.0, 0.2]])
        y_i = -1
        eps = 1e-5

        _, _, grad_b = expected_hinge_loss_gaussian(w, b, mu, cov, y_i)

        l_plus, _, _ = expected_hinge_loss_gaussian(w, b + eps, mu, cov, y_i)
        l_minus, _, _ = expected_hinge_loss_gaussian(w, b - eps, mu, cov, y_i)
        num_grad_b = (l_plus - l_minus) / (2 * eps)

        np.testing.assert_allclose(grad_b, num_grad_b, atol=1e-4)

    def test_reduces_to_hinge_loss_zero_covariance(self):
        w = np.array([1.0, 0.5])
        b = 0.1
        mu = np.array([0.3, -0.2])
        cov = np.eye(2) * 1e-12
        y_i = +1

        loss, _, _ = expected_hinge_loss_gaussian(w, b, mu, cov, y_i)
        hinge = max(0.0, 1.0 - y_i * (np.dot(w, mu) + b))

        np.testing.assert_allclose(loss, hinge, atol=1e-4)


class TestExpectedHingeLossGMM:
    """Tests for the GMM expected hinge loss."""

    def test_single_component_matches_gaussian(self):
        w = np.array([1.0, 0.0])
        b = 0.0
        mu = np.array([1.0, 0.0])
        cov = np.array([[0.1, 0.0], [0.0, 0.1]])
        y_i = +1

        loss_gsu, gw_gsu, gb_gsu = expected_hinge_loss_gaussian(w, b, mu, cov, y_i)

        gmm = {
            "weights": np.array([1.0]),
            "means": np.array([[1.0, 0.0]]),
            "covs": np.array([[[0.1, 0.0], [0.0, 0.1]]]),
        }
        loss_gmu, gw_gmu, gb_gmu = expected_hinge_loss_gmm(w, b, gmm, y_i)

        np.testing.assert_allclose(loss_gsu, loss_gmu)
        np.testing.assert_allclose(gw_gsu, gw_gmu)
        np.testing.assert_allclose(gb_gsu, gb_gmu)

    def test_gradient_w_numerical_multicomponent(self):
        w = np.array([1.0, 0.0])
        b = 0.0
        y_i = +1
        eps = 1e-5

        gmm = {
            "weights": np.array([0.6, 0.4]),
            "means": np.array([[1.0, 0.5], [0.5, -0.5]]),
            "covs": np.array(
                [
                    [[0.1, 0.02], [0.02, 0.1]],
                    [[0.2, -0.05], [-0.05, 0.15]],
                ]
            ),
        }

        _, grad_w, _ = expected_hinge_loss_gmm(w, b, gmm, y_i)

        num_grad_w = np.zeros_like(w)
        for k in range(len(w)):
            w_plus = w.copy()
            w_plus[k] += eps
            w_minus = w.copy()
            w_minus[k] -= eps
            l_plus, _, _ = expected_hinge_loss_gmm(w_plus, b, gmm, y_i)
            l_minus, _, _ = expected_hinge_loss_gmm(w_minus, b, gmm, y_i)
            num_grad_w[k] = (l_plus - l_minus) / (2 * eps)

        np.testing.assert_allclose(grad_w, num_grad_w, atol=1e-4)

    def test_gradient_b_numerical_multicomponent(self):
        w = np.array([1.0, 0.0])
        b = 0.0
        y_i = +1
        eps = 1e-5

        gmm = {
            "weights": np.array([0.6, 0.4]),
            "means": np.array([[1.0, 0.5], [0.5, -0.5]]),
            "covs": np.array(
                [
                    [[0.1, 0.02], [0.02, 0.1]],
                    [[0.2, -0.05], [-0.05, 0.15]],
                ]
            ),
        }

        _, _, grad_b = expected_hinge_loss_gmm(w, b, gmm, y_i)

        l_plus, _, _ = expected_hinge_loss_gmm(w, b + eps, gmm, y_i)
        l_minus, _, _ = expected_hinge_loss_gmm(w, b - eps, gmm, y_i)
        num_grad_b = (l_plus - l_minus) / (2 * eps)

        np.testing.assert_allclose(grad_b, num_grad_b, atol=1e-4)

    def test_higher_dimensional(self):
        d = 10
        rng = np.random.RandomState(42)
        w = rng.randn(d) * 0.5
        b = rng.randn() * 0.5
        y_i = +1
        eps = 1e-5

        K = 3
        gmm = {
            "weights": np.array([0.5, 0.3, 0.2]),
            "means": rng.randn(K, d) * 0.5,
            "covs": np.array(
                [
                    A @ A.T + np.eye(d) * 0.01
                    for A in [rng.randn(d, d) * 0.1 for _ in range(K)]
                ]
            ),
        }

        _, grad_w, grad_b = expected_hinge_loss_gmm(w, b, gmm, y_i)

        num_grad_w = np.zeros(d)
        for k in range(d):
            w_plus = w.copy()
            w_plus[k] += eps
            w_minus = w.copy()
            w_minus[k] -= eps
            l_p, _, _ = expected_hinge_loss_gmm(w_plus, b, gmm, y_i)
            l_m, _, _ = expected_hinge_loss_gmm(w_minus, b, gmm, y_i)
            num_grad_w[k] = (l_p - l_m) / (2 * eps)

        np.testing.assert_allclose(grad_w, num_grad_w, atol=1e-4)
