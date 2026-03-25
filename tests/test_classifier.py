"""
Tests for the SVMGMU classifier.
"""

import numpy as np
import pytest

from svm_gmu import SVMGMU, generate_gmm_dataset


@pytest.fixture
def simple_dataset():
    """A well-separated 2D dataset for testing."""
    return generate_gmm_dataset(d=2, n_per_class=10, class_sep=6.0, seed=42)


class TestSVMGMU:
    """Tests for the SVMGMU classifier."""

    def test_fit_returns_self(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=10)
        result = clf.fit(X, y, gmm)
        assert result is clf

    def test_fit_creates_attributes(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=10)
        clf.fit(X, y, gmm)
        assert hasattr(clf, "w_")
        assert hasattr(clf, "b_")
        assert hasattr(clf, "history_")
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "y_")
        assert hasattr(clf, "gmm_params_")

    def test_w_shape(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=10)
        clf.fit(X, y, gmm)
        assert clf.w_.shape == (2,)

    def test_history_length(self, simple_dataset):
        X, y, gmm = simple_dataset
        n_epochs = 50
        clf = SVMGMU(n_epochs=n_epochs)
        clf.fit(X, y, gmm)
        assert len(clf.history_["objective"]) == n_epochs

    def test_objective_decreases(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=200, lr_init=0.5)
        clf.fit(X, y, gmm)
        objs = clf.history_["objective"]
        assert objs[-1] < objs[0], "Objective should decrease during training"

    def test_predict_shape(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=50)
        clf.fit(X, y, gmm)
        preds = clf.predict(X)
        assert preds.shape == y.shape

    def test_predict_labels_valid(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=50)
        clf.fit(X, y, gmm)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({-1, +1})

    def test_high_accuracy_well_separated(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=300, lr_init=0.5, lam=0.01)
        clf.fit(X, y, gmm)
        accuracy = clf.score(X, y)
        assert accuracy >= 0.9, f"Expected >= 90% accuracy, got {accuracy:.2%}"

    def test_decision_function_shape(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=50)
        clf.fit(X, y, gmm)
        scores = clf.decision_function(X)
        assert scores.shape == (len(X),)

    def test_decision_function_consistent_with_predict(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=100)
        clf.fit(X, y, gmm)
        scores = clf.decision_function(X)
        preds = clf.predict(X)
        expected_preds = np.where(scores >= 0, 1, -1)
        np.testing.assert_array_equal(preds, expected_preds)

    def test_expected_misclassification_shape(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=100)
        clf.fit(X, y, gmm)
        probs = clf.expected_misclassification(n_mc=1000)
        assert probs.shape == (len(X),)

    def test_expected_misclassification_bounded(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf = SVMGMU(n_epochs=100)
        clf.fit(X, y, gmm)
        probs = clf.expected_misclassification(n_mc=1000)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_reproducible_with_seed(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf1 = SVMGMU(n_epochs=50, seed=123)
        clf1.fit(X, y, gmm)
        clf2 = SVMGMU(n_epochs=50, seed=123)
        clf2.fit(X, y, gmm)
        np.testing.assert_array_equal(clf1.w_, clf2.w_)
        np.testing.assert_equal(clf1.b_, clf2.b_)

    def test_different_seeds_differ(self, simple_dataset):
        X, y, gmm = simple_dataset
        clf1 = SVMGMU(n_epochs=50, seed=0)
        clf1.fit(X, y, gmm)
        clf2 = SVMGMU(n_epochs=50, seed=999)
        clf2.fit(X, y, gmm)
        assert not np.allclose(clf1.w_, clf2.w_)

    def test_higher_dimensional(self):
        X, y, gmm = generate_gmm_dataset(d=20, n_per_class=10, seed=42)
        clf = SVMGMU(n_epochs=100, lr_init=0.5)
        clf.fit(X, y, gmm)
        assert clf.w_.shape == (20,)
        assert clf.score(X, y) >= 0.8
