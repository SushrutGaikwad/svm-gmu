"""
SVM-GMU classifier with scikit-learn compatible API.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .loss import expected_hinge_loss_gmm


class SVMGMU(BaseEstimator, ClassifierMixin):
    """
    Support Vector Machine with Gaussian Mixture Uncertainty (SVM-GMU).

    Minimises the objective:

        (lambda/2) ||w||^2 + (1/ell) sum_i L_{GMU,i}(w, b)

    where L_{GMU,i} is the expected hinge loss under the i-th sample's
    Gaussian mixture uncertainty distribution.

    Parameters
    ----------
    lam : float, default=0.01
        Regularisation parameter (lambda).
    n_epochs : int, default=500
        Number of SGD passes over the dataset.
    lr_init : float, default=0.5
        Initial learning rate.
    lr_decay : bool, default=True
        If True, use schedule lr = lr_init / (1 + epoch).
    batch_size : int or None, default=None
        Mini-batch size. If None, use full-batch gradient descent.
    seed : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, print progress every 50 epochs.

    Attributes
    ----------
    w_ : ndarray of shape (d,)
        Learned weight vector.
    b_ : float
        Learned bias.
    history_ : dict
        Training history with key 'objective' containing per-epoch values.
    classes_ : ndarray of shape (2,)
        Unique class labels.
    gmm_params_ : list of dict
        GMM parameters stored from fit.

    Examples
    --------
    >>> from svm_gmu import SVMGMU
    >>> clf = SVMGMU(lam=0.01, n_epochs=500)
    >>> clf.fit(X_means, y, gmm_params)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        lam=0.01,
        n_epochs=500,
        lr_init=0.5,
        lr_decay=True,
        batch_size=None,
        seed=42,
        verbose=False,
    ):
        self.lam = lam
        self.n_epochs = n_epochs
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y, gmm_params):
        """
        Fit the SVM-GMU classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            Observed means of each data point. Used for dimensionality
            and for predict/score when gmm_params are not available.
        y : ndarray of shape (n_samples,)
            Labels in {-1, +1}.
        gmm_params : list of dict
            One dict per sample with keys 'weights', 'means', 'covs'.

        Returns
        -------
        self
            Fitted classifier.
        """
        rng = np.random.RandomState(self.seed)
        n_samples, d = X.shape
        ell = n_samples

        self.classes_ = np.unique(y)
        self.gmm_params_ = gmm_params
        self.y_ = y.copy()

        batch_size = self.batch_size if self.batch_size is not None else n_samples

        w = rng.randn(d) * 0.01
        b = 0.0

        history = {"objective": []}

        for epoch in range(self.n_epochs):
            if self.lr_decay:
                lr = self.lr_init / (1.0 + epoch)
            else:
                lr = self.lr_init

            indices = rng.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start : start + batch_size]

                grad_w = self.lam * w
                grad_b = 0.0

                for i in batch_idx:
                    _, gw_i, gb_i = expected_hinge_loss_gmm(w, b, gmm_params[i], y[i])
                    grad_w += gw_i / ell
                    grad_b += gb_i / ell

                w -= lr * grad_w
                b -= lr * grad_b

            reg = 0.5 * self.lam * np.dot(w, w)
            total_loss = 0.0
            for i in range(n_samples):
                loss_i, _, _ = expected_hinge_loss_gmm(w, b, gmm_params[i], y[i])
                total_loss += loss_i / ell
            obj = reg + total_loss
            history["objective"].append(obj)

            if self.verbose and (epoch % 50 == 0 or epoch == self.n_epochs - 1):
                print(
                    f"  Epoch {epoch:4d}/{self.n_epochs}: "
                    f"objective = {obj:.6f}, "
                    f"||w|| = {np.linalg.norm(w):.4f}, "
                    f"b = {b:.4f}, "
                    f"lr = {lr:.6f}"
                )

        self.w_ = w
        self.b_ = b
        self.history_ = history

        return self

    def decision_function(self, X):
        """
        Compute the signed distance to the decision boundary.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            Input data.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Signed distances: w^T x + b.
        """
        return X @ self.w_ + self.b_

    def predict(self, X):
        """
        Predict class labels for input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels in {-1, +1}.
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def score(self, X, y):
        """
        Compute classification accuracy on the means.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            Input data (typically the observed means).
        y : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
            Fraction of correctly classified points.
        """
        return np.mean(self.predict(X) == y)

    def expected_misclassification(
        self, gmm_params=None, y=None, n_mc=50000, seed=None
    ):
        """
        Estimate per-point expected misclassification probability
        via Monte Carlo sampling from each point's GMM.

        Parameters
        ----------
        gmm_params : list of dict or None
            GMM parameters. If None, uses those from fit.
        y : ndarray or None
            True labels. If None, uses those from fit.
        n_mc : int, default=50000
            Number of Monte Carlo samples per point.
        seed : int or None
            Random seed. If None, uses self.seed.

        Returns
        -------
        probs : ndarray of shape (n_samples,)
            Estimated misclassification probability for each point.
        """
        if gmm_params is None:
            gmm_params = self.gmm_params_
        if y is None:
            y = self.y_
        if seed is None:
            seed = self.seed

        rng = np.random.RandomState(seed)
        n_samples = len(gmm_params)
        probs = np.zeros(n_samples)

        for i in range(n_samples):
            gp = gmm_params[i]
            pi_ij = gp["weights"]
            mu_ij = gp["means"]
            sigma_ij = gp["covs"]
            K_i = len(pi_ij)
            d = mu_ij.shape[1]

            comp_assign = rng.choice(K_i, size=n_mc, p=pi_ij)
            samples = np.zeros((n_mc, d))
            for j in range(K_i):
                mask = comp_assign == j
                n_j = np.sum(mask)
                if n_j > 0:
                    samples[mask] = rng.multivariate_normal(
                        mu_ij[j], sigma_ij[j], size=n_j
                    )

            scores = y[i] * (samples @ self.w_ + self.b_)
            probs[i] = np.mean(scores < 0)

        return probs
