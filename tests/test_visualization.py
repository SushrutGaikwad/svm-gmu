"""
Tests for the visualization module.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt
import pytest

from svm_gmu import (
    SVMGMU,
    generate_gmm_dataset,
    plot_uncertainties,
    plot_decision_boundary,
    plot_comparison,
)


@pytest.fixture
def dataset_2d():
    """A 2D dataset for visualization tests."""
    return generate_gmm_dataset(d=2, n_per_class=5, seed=42)


@pytest.fixture
def dataset_3d():
    """A 3D dataset for visualization tests."""
    return generate_gmm_dataset(d=3, n_per_class=5, seed=42)


@pytest.fixture
def fitted_clf_2d(dataset_2d):
    """A fitted SVMGMU classifier on 2D data."""
    X, y, gmm = dataset_2d
    clf = SVMGMU(n_epochs=50, lr_init=0.5)
    clf.fit(X, y, gmm)
    return clf


class TestPlotUncertainties:
    """Tests for the plot_uncertainties function."""

    def test_returns_axes_2d(self, dataset_2d):
        X, y, gmm = dataset_2d
        ax = plot_uncertainties(X, y, gmm, show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_returns_axes_3d(self, dataset_3d):
        X, y, gmm = dataset_3d
        ax = plot_uncertainties(X, y, gmm, show=False)
        assert ax is not None
        plt.close("all")

    def test_rejects_higher_dimensions(self):
        X, y, gmm = generate_gmm_dataset(d=5, n_per_class=3, seed=0)
        with pytest.raises(ValueError, match="2D and 3D"):
            plot_uncertainties(X, y, gmm, show=False)

    def test_custom_n_sigma_list(self, dataset_2d):
        X, y, gmm = dataset_2d
        ax = plot_uncertainties(X, y, gmm, n_sigma_list=[1, 2, 3], show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_custom_figsize(self, dataset_2d):
        X, y, gmm = dataset_2d
        ax = plot_uncertainties(X, y, gmm, figsize=(12, 10), show=False)
        fig = ax.get_figure()
        width, height = fig.get_size_inches()
        assert width == 12
        assert height == 10
        plt.close("all")

    def test_custom_grid_resolution(self, dataset_2d):
        X, y, gmm = dataset_2d
        ax = plot_uncertainties(X, y, gmm, grid_resolution=50, show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_existing_axes_2d(self, dataset_2d):
        X, y, gmm = dataset_2d
        fig, ax_input = plt.subplots()
        ax_output = plot_uncertainties(X, y, gmm, ax=ax_input, show=False)
        assert ax_output is ax_input
        plt.close("all")

    def test_default_n_sigma(self, dataset_2d):
        X, y, gmm = dataset_2d
        ax = plot_uncertainties(X, y, gmm, n_sigma_list=None, show=False)
        assert "$3\\sigma$" in ax.get_title()
        plt.close("all")


class TestPlotDecisionBoundary:
    """Tests for the plot_decision_boundary function."""

    def test_returns_axes(self, dataset_2d, fitted_clf_2d):
        X, y, gmm = dataset_2d
        ax = plot_decision_boundary(
            fitted_clf_2d,
            X,
            y,
            gmm,
            show=False,
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_rejects_non_2d(self, fitted_clf_2d):
        X, y, gmm = generate_gmm_dataset(d=3, n_per_class=3, seed=0)
        with pytest.raises(ValueError, match="2D"):
            plot_decision_boundary(fitted_clf_2d, X, y, gmm, show=False)

    def test_has_boundary_line(self, dataset_2d, fitted_clf_2d):
        X, y, gmm = dataset_2d
        ax = plot_decision_boundary(
            fitted_clf_2d,
            X,
            y,
            gmm,
            show=False,
        )
        # Check that there is at least one line (the decision boundary)
        lines = ax.get_lines()
        assert len(lines) >= 1
        plt.close("all")

    def test_has_legend(self, dataset_2d, fitted_clf_2d):
        X, y, gmm = dataset_2d
        ax = plot_decision_boundary(
            fitted_clf_2d,
            X,
            y,
            gmm,
            show=False,
        )
        legend = ax.get_legend()
        assert legend is not None
        plt.close("all")

    def test_custom_n_sigma(self, dataset_2d, fitted_clf_2d):
        X, y, gmm = dataset_2d
        ax = plot_decision_boundary(
            fitted_clf_2d,
            X,
            y,
            gmm,
            n_sigma_list=[1, 2],
            show=False,
        )
        assert "$1\\sigma$" in ax.get_title()
        assert "$2\\sigma$" in ax.get_title()
        plt.close("all")

    def test_vertical_boundary(self, dataset_2d):
        """Test with a classifier whose w[1] ≈ 0 (vertical boundary)."""
        X, y, gmm = dataset_2d
        clf = SVMGMU(n_epochs=50)
        clf.fit(X, y, gmm)
        # Force a near-vertical boundary
        clf.w_ = np.array([1.0, 1e-15])
        clf.b_ = 0.0
        ax = plot_decision_boundary(clf, X, y, gmm, show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")


class TestPlotComparison:
    """Tests for the plot_comparison function."""

    def test_returns_axes(self, dataset_2d, fitted_clf_2d):
        X, y, gmm = dataset_2d
        w_std = np.array([1.0, 0.0])
        b_std = 0.0
        ax = plot_comparison(
            fitted_clf_2d,
            w_std,
            b_std,
            X,
            y,
            gmm,
            show=False,
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_rejects_non_2d(self, fitted_clf_2d):
        X, y, gmm = generate_gmm_dataset(d=3, n_per_class=3, seed=0)
        with pytest.raises(ValueError, match="2D"):
            plot_comparison(
                fitted_clf_2d,
                np.zeros(3),
                0.0,
                X,
                y,
                gmm,
                show=False,
            )

    def test_has_two_boundary_lines(self, dataset_2d, fitted_clf_2d):
        X, y, gmm = dataset_2d
        w_std = np.array([1.0, 0.1])
        b_std = -0.5
        ax = plot_comparison(
            fitted_clf_2d,
            w_std,
            b_std,
            X,
            y,
            gmm,
            show=False,
        )
        lines = ax.get_lines()
        assert len(lines) >= 2
        plt.close("all")

    def test_legend_has_both_classifiers(self, dataset_2d, fitted_clf_2d):
        X, y, gmm = dataset_2d
        w_std = np.array([1.0, 0.1])
        b_std = -0.5
        ax = plot_comparison(
            fitted_clf_2d,
            w_std,
            b_std,
            X,
            y,
            gmm,
            show=False,
        )
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert any("GMU" in t for t in legend_texts)
        assert any("Standard" in t for t in legend_texts)
        plt.close("all")

    def test_custom_figsize(self, dataset_2d, fitted_clf_2d):
        X, y, gmm = dataset_2d
        w_std = np.array([1.0, 0.0])
        b_std = 0.0
        ax = plot_comparison(
            fitted_clf_2d,
            w_std,
            b_std,
            X,
            y,
            gmm,
            figsize=(14, 10),
            show=False,
        )
        fig = ax.get_figure()
        width, height = fig.get_size_inches()
        assert width == 14
        assert height == 10
        plt.close("all")
