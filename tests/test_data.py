"""
Tests for data validation and generation utilities.
"""

import numpy as np
import pytest

from svm_gmu import validate_gmm_dataset, generate_gmm_dataset


class TestValidateGMMDataset:
    """Tests for dataset validation."""

    def test_valid_dataset_passes(self):
        X = np.array([[1.0, 0.0], [-1.0, 0.0]])
        y = np.array([+1, -1])
        gmm = [
            {
                "weights": np.array([1.0]),
                "means": np.array([[1.0, 0.0]]),
                "covs": np.array([[[0.1, 0.0], [0.0, 0.1]]]),
            },
            {
                "weights": np.array([0.5, 0.5]),
                "means": np.array([[-1.0, 0.0], [-0.5, 0.5]]),
                "covs": np.array(
                    [
                        [[0.1, 0.0], [0.0, 0.1]],
                        [[0.2, 0.0], [0.0, 0.2]],
                    ]
                ),
            },
        ]
        assert validate_gmm_dataset(X, y, gmm) is True

    def test_wrong_label_raises(self):
        X = np.array([[1.0, 0.0]])
        y = np.array([0])  # invalid label
        gmm = [
            {
                "weights": np.array([1.0]),
                "means": np.array([[1.0, 0.0]]),
                "covs": np.array([[[0.1, 0.0], [0.0, 0.1]]]),
            },
        ]
        with pytest.raises(AssertionError):
            validate_gmm_dataset(X, y, gmm)

    def test_weights_not_summing_to_one_raises(self):
        X = np.array([[1.0, 0.0]])
        y = np.array([+1])
        gmm = [
            {
                "weights": np.array([0.3, 0.3]),  # sums to 0.6
                "means": np.array([[1.0, 0.0], [0.5, 0.5]]),
                "covs": np.array(
                    [
                        [[0.1, 0.0], [0.0, 0.1]],
                        [[0.1, 0.0], [0.0, 0.1]],
                    ]
                ),
            },
        ]
        with pytest.raises(AssertionError):
            validate_gmm_dataset(X, y, gmm)

    def test_nonsymmetric_covariance_raises(self):
        X = np.array([[1.0, 0.0]])
        y = np.array([+1])
        gmm = [
            {
                "weights": np.array([1.0]),
                "means": np.array([[1.0, 0.0]]),
                "covs": np.array([[[0.1, 0.5], [0.0, 0.1]]]),  # not symmetric
            },
        ]
        with pytest.raises(AssertionError):
            validate_gmm_dataset(X, y, gmm)

    def test_mismatched_lengths_raises(self):
        X = np.array([[1.0, 0.0], [-1.0, 0.0]])
        y = np.array([+1, -1])
        gmm = [
            {
                "weights": np.array([1.0]),
                "means": np.array([[1.0, 0.0]]),
                "covs": np.array([[[0.1, 0.0], [0.0, 0.1]]]),
            },
        ]  # only 1 entry, but 2 samples
        with pytest.raises(AssertionError):
            validate_gmm_dataset(X, y, gmm)


class TestGenerateGMMDataset:
    """Tests for random dataset generation."""

    def test_output_shapes(self):
        X, y, gmm = generate_gmm_dataset(d=5, n_per_class=10, seed=0)
        assert X.shape == (20, 5)
        assert y.shape == (20,)
        assert len(gmm) == 20

    def test_labels_balanced(self):
        X, y, gmm = generate_gmm_dataset(n_per_class=15, seed=0)
        assert np.sum(y == +1) == 15
        assert np.sum(y == -1) == 15

    def test_generated_dataset_validates(self):
        X, y, gmm = generate_gmm_dataset(d=3, n_per_class=8, seed=123)
        assert validate_gmm_dataset(X, y, gmm) is True

    def test_reproducible_with_seed(self):
        X1, y1, gmm1 = generate_gmm_dataset(seed=42)
        X2, y2, gmm2 = generate_gmm_dataset(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        for g1, g2 in zip(gmm1, gmm2):
            np.testing.assert_array_equal(g1["weights"], g2["weights"])
            np.testing.assert_array_equal(g1["means"], g2["means"])
            np.testing.assert_array_equal(g1["covs"], g2["covs"])

    def test_different_seeds_differ(self):
        X1, _, _ = generate_gmm_dataset(seed=0)
        X2, _, _ = generate_gmm_dataset(seed=999)
        assert not np.allclose(X1, X2)

    def test_components_within_range(self):
        _, _, gmm = generate_gmm_dataset(
            n_components_range=(2, 3), n_per_class=20, seed=0
        )
        for gp in gmm:
            K = len(gp["weights"])
            assert 2 <= K <= 3, f"Expected 2-3 components, got {K}"
