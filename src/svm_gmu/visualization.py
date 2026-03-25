"""
Visualization utilities for SVM-GMU (2D and 3D).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def _eval_gmm_density(grid_points, gmm_param):
    """Evaluate GMM density at grid points."""
    w = gmm_param["weights"]
    m = gmm_param["means"]
    c = gmm_param["covs"]
    density = np.zeros(len(grid_points))
    for j in range(len(w)):
        rv = multivariate_normal(mean=m[j], cov=c[j])
        density += w[j] * rv.pdf(grid_points)
    return density


def _get_n_sigma_level(gmm_param, n_sigma):
    """Get density threshold corresponding to n-sigma for a GMM."""
    w = gmm_param["weights"]
    m = gmm_param["means"]
    c = gmm_param["covs"]
    decay = np.exp(-0.5 * n_sigma**2)
    level = 0.0
    for j in range(len(w)):
        peak_j = multivariate_normal(mean=m[j], cov=c[j]).pdf(m[j])
        level += w[j] * peak_j * decay
    return level


def plot_uncertainties(
    X,
    y,
    gmm_params,
    n_sigma_list=None,
    grid_resolution=200,
    figsize=None,
    ax=None,
    show=True,
):
    """
    Plot data points with GMM uncertainty contours.

    Parameters
    ----------
    X : ndarray of shape (n_samples, d)
        Observed locations. Must have d=2 or d=3.
    y : ndarray of shape (n_samples,)
        Labels in {-1, +1}.
    gmm_params : list of dict
        GMM parameters for each point.
    n_sigma_list : list of float or None
        Sigma levels to plot. Defaults to [3].
    grid_resolution : int, default=200
        Grid density for contour evaluation (2D only).
    figsize : tuple or None
        Figure size. Defaults to (8, 6) for 2D, (10, 8) for 3D.
    ax : matplotlib Axes or None
        If provided, plot on this axes. Otherwise create a new figure.
    show : bool, default=True
        If True, call plt.show() at the end.

    Returns
    -------
    ax : matplotlib Axes
        The axes with the plot.
    """
    d = X.shape[1]
    if d not in (2, 3):
        raise ValueError(f"Plotting only supported for 2D and 3D data, got d={d}")

    if n_sigma_list is None:
        n_sigma_list = [3]

    if d == 2:
        ax = _plot_uncertainties_2d(
            X,
            y,
            gmm_params,
            n_sigma_list,
            grid_resolution,
            figsize,
            ax,
        )
    else:
        ax = _plot_uncertainties_3d(
            X,
            y,
            gmm_params,
            n_sigma_list,
            figsize,
            ax,
        )

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def _plot_uncertainties_2d(
    X, y, gmm_params, n_sigma_list, grid_resolution, figsize, ax
):
    """Plot 2D data with GMM contours."""
    if figsize is None:
        figsize = (8, 6)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    all_means = np.vstack([gp["means"] for gp in gmm_params])
    x_min, x_max = all_means[:, 0].min() - 1.5, all_means[:, 0].max() + 1.5
    y_min, y_max = all_means[:, 1].min() - 1.5, all_means[:, 1].max() + 1.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    linestyles = ["-", "--", ":", "-."]

    for i, gp in enumerate(gmm_params):
        density = _eval_gmm_density(grid_points, gp).reshape(xx.shape)
        color = "blue" if y[i] == +1 else "red"

        for s_idx, n_sigma in enumerate(n_sigma_list):
            level = _get_n_sigma_level(gp, n_sigma)
            ls = linestyles[s_idx % len(linestyles)]
            ax.contour(
                xx,
                yy,
                density,
                levels=[level],
                colors=color,
                alpha=0.5,
                linewidths=0.8,
                linestyles=ls,
            )

    pos_mask = y == +1
    neg_mask = y == -1
    ax.scatter(
        X[pos_mask, 0],
        X[pos_mask, 1],
        c="blue",
        s=80,
        marker="o",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="Class $+1$",
    )
    ax.scatter(
        X[neg_mask, 0],
        X[neg_mask, 1],
        c="red",
        s=80,
        marker="s",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="Class $-1$",
    )

    for i in range(len(X)):
        ax.annotate(
            f"{i}",
            (X[i, 0], X[i, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    sigma_str = ", ".join([f"${s}\\sigma$" for s in n_sigma_list])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"GMM Uncertainty Contours ({sigma_str})")
    ax.legend()
    ax.set_aspect("equal")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    return ax


def _plot_uncertainties_3d(X, y, gmm_params, n_sigma_list, figsize, ax):
    """Plot 3D data with GMM uncertainty clouds via scatter samples."""
    if figsize is None:
        figsize = (10, 8)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    rng = np.random.RandomState(42)
    n_samples_per_point = 500

    for i, gp in enumerate(gmm_params):
        pi_ij = gp["weights"]
        mu_ij = gp["means"]
        sigma_ij = gp["covs"]
        K_i = len(pi_ij)

        comp_assign = rng.choice(K_i, size=n_samples_per_point, p=pi_ij)
        samples = np.zeros((n_samples_per_point, 3))
        for j in range(K_i):
            mask = comp_assign == j
            n_j = np.sum(mask)
            if n_j > 0:
                samples[mask] = rng.multivariate_normal(
                    mu_ij[j],
                    sigma_ij[j],
                    size=n_j,
                )

        color = "blue" if y[i] == +1 else "red"
        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            samples[:, 2],
            c=color,
            alpha=0.03,
            s=5,
            depthshade=True,
        )

    pos_mask = y == +1
    neg_mask = y == -1
    ax.scatter(
        X[pos_mask, 0],
        X[pos_mask, 1],
        X[pos_mask, 2],
        c="blue",
        s=100,
        marker="o",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="Class $+1$",
    )
    ax.scatter(
        X[neg_mask, 0],
        X[neg_mask, 1],
        X[neg_mask, 2],
        c="red",
        s=100,
        marker="s",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="Class $-1$",
    )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("GMM Uncertainty Clouds (3D)")
    ax.legend()

    return ax


def plot_decision_boundary(
    clf,
    X,
    y,
    gmm_params,
    n_sigma_list=None,
    grid_resolution=200,
    figsize=None,
    show=True,
):
    """
    Plot the SVM-GMU decision boundary with uncertainty contours (2D only).

    Parameters
    ----------
    clf : SVMGMU
        A fitted SVMGMU classifier.
    X : ndarray of shape (n_samples, 2)
        Observed locations.
    y : ndarray of shape (n_samples,)
        Labels in {-1, +1}.
    gmm_params : list of dict
        GMM parameters for each point.
    n_sigma_list : list of float or None
        Sigma levels for uncertainty contours. Defaults to [3].
    grid_resolution : int, default=200
        Grid density.
    figsize : tuple or None
        Figure size. Defaults to (8, 6).
    show : bool, default=True
        If True, call plt.show() at the end.

    Returns
    -------
    ax : matplotlib Axes
        The axes with the plot.
    """
    d = X.shape[1]
    if d != 2:
        raise ValueError(
            f"Decision boundary plotting only supported for 2D data, got d={d}"
        )

    if n_sigma_list is None:
        n_sigma_list = [3]
    if figsize is None:
        figsize = (8, 6)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot uncertainty contours (without showing yet)
    _plot_uncertainties_2d(
        X,
        y,
        gmm_params,
        n_sigma_list,
        grid_resolution,
        None,
        ax,
    )
    ax.set_title("")

    # Compute plot bounds
    all_means = np.vstack([gp["means"] for gp in gmm_params])
    x_min, x_max = all_means[:, 0].min() - 1.5, all_means[:, 0].max() + 1.5
    y_min, y_max = all_means[:, 1].min() - 1.5, all_means[:, 1].max() + 1.5

    # Decision regions
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_2d = np.column_stack([xx.ravel(), yy.ravel()])
    Z = clf.decision_function(grid_2d).reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        Z,
        levels=[-1e10, 0, 1e10],
        colors=["#FFCCCC", "#CCCCFF"],
        alpha=0.2,
        zorder=0,
    )

    # Decision boundary
    w = clf.w_
    b = clf.b_
    x_line = np.linspace(x_min, x_max, 500)

    if abs(w[1]) > 1e-10:
        y_line = -(w[0] * x_line + b) / w[1]
        ax.plot(x_line, y_line, "k-", linewidth=2.5, label="SVM-GMU boundary")
    else:
        x_boundary = -b / w[0]
        ax.axvline(
            x=x_boundary,
            color="k",
            linewidth=2.5,
            label="SVM-GMU boundary",
        )

    sigma_str = ", ".join([f"${s}\\sigma$" for s in n_sigma_list])
    ax.set_title(f"SVM-GMU Decision Boundary (contours: {sigma_str})")
    ax.legend()
    ax.set_ylim(y_min, y_max)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_comparison(
    clf_gmu,
    clf_std_w,
    clf_std_b,
    X,
    y,
    gmm_params,
    n_sigma_list=None,
    grid_resolution=200,
    figsize=None,
    show=True,
):
    """
    Plot SVM-GMU vs standard SVM decision boundaries (2D only).

    Parameters
    ----------
    clf_gmu : SVMGMU
        A fitted SVMGMU classifier.
    clf_std_w : ndarray of shape (2,)
        Weight vector from standard SVM.
    clf_std_b : float
        Bias from standard SVM.
    X : ndarray of shape (n_samples, 2)
        Observed locations.
    y : ndarray of shape (n_samples,)
        Labels in {-1, +1}.
    gmm_params : list of dict
        GMM parameters for each point.
    n_sigma_list : list of float or None
        Sigma levels for uncertainty contours. Defaults to [3].
    grid_resolution : int, default=200
        Grid density.
    figsize : tuple or None
        Figure size. Defaults to (10, 7).
    show : bool, default=True
        If True, call plt.show() at the end.

    Returns
    -------
    ax : matplotlib Axes
        The axes with the plot.
    """
    d = X.shape[1]
    if d != 2:
        raise ValueError(f"Comparison plotting only supported for 2D data, got d={d}")

    if n_sigma_list is None:
        n_sigma_list = [3]
    if figsize is None:
        figsize = (10, 7)

    fig, ax = plt.subplots(figsize=figsize)

    # Uncertainty contours (fainter)
    all_means = np.vstack([gp["means"] for gp in gmm_params])
    x_min, x_max = all_means[:, 0].min() - 1.5, all_means[:, 0].max() + 1.5
    y_min, y_max = all_means[:, 1].min() - 1.5, all_means[:, 1].max() + 1.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    linestyles = ["-", "--", ":", "-."]
    for i, gp in enumerate(gmm_params):
        density = _eval_gmm_density(grid_points, gp).reshape(xx.shape)
        color = "blue" if y[i] == +1 else "red"
        for s_idx, n_sigma in enumerate(n_sigma_list):
            level = _get_n_sigma_level(gp, n_sigma)
            ls = linestyles[s_idx % len(linestyles)]
            ax.contour(
                xx,
                yy,
                density,
                levels=[level],
                colors=color,
                alpha=0.3,
                linewidths=0.6,
                linestyles=ls,
            )

    # Data points
    pos_mask = y == +1
    neg_mask = y == -1
    ax.scatter(
        X[pos_mask, 0],
        X[pos_mask, 1],
        c="blue",
        s=80,
        marker="o",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="Class $+1$",
    )
    ax.scatter(
        X[neg_mask, 0],
        X[neg_mask, 1],
        c="red",
        s=80,
        marker="s",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="Class $-1$",
    )

    for i in range(len(X)):
        ax.annotate(
            f"{i}",
            (X[i, 0], X[i, 1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    # SVM-GMU boundary
    x_line = np.linspace(x_min, x_max, 500)
    w_gmu = clf_gmu.w_
    b_gmu = clf_gmu.b_

    if abs(w_gmu[1]) > 1e-10:
        y_line = -(w_gmu[0] * x_line + b_gmu) / w_gmu[1]
        ax.plot(x_line, y_line, "k-", linewidth=2.5, label="SVM-GMU")
    else:
        ax.axvline(
            x=-b_gmu / w_gmu[0],
            color="k",
            linewidth=2.5,
            label="SVM-GMU",
        )

    # Standard SVM boundary
    if abs(clf_std_w[1]) > 1e-10:
        y_line_std = -(clf_std_w[0] * x_line + clf_std_b) / clf_std_w[1]
        ax.plot(
            x_line,
            y_line_std,
            "k--",
            linewidth=2.5,
            label="Standard SVM",
        )
    else:
        ax.axvline(
            x=-clf_std_b / clf_std_w[0],
            color="k",
            linewidth=2.5,
            linestyle="--",
            label="Standard SVM",
        )

    sigma_str = ", ".join([f"${s}\\sigma$" for s in n_sigma_list])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"SVM-GMU vs Standard SVM (contours: {sigma_str})")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return ax
