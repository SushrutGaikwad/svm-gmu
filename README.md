# SVM-GMU

## Support Vector Machines with Gaussian Mixture Sample Uncertainty

SVM-GMU extends the SVM-GSU framework to handle per-sample non-Gaussian
uncertainty modelled as Gaussian mixtures. By exploiting the linearity of
expectation, the expected hinge loss under a Gaussian mixture decomposes
into a weighted sum of closed-form SVM-GSU losses, yielding a convex
objective with analytic gradients.

## Installation

Install SVM-GMU directly from GitHub:

```bash
pip install git+https://github.com/SushrutGaikwad/svm-gmu.git
```

Or with uv:

```bash
uv add git+https://github.com/SushrutGaikwad/svm-gmu.git
```

Or clone and install locally:

```bash
git clone https://github.com/SushrutGaikwad/svm-gmu.git
cd svm-gmu
uv sync
```

## Quick Start

```python
import numpy as np
from svm_gmu import SVMGMU, validate_gmm_dataset

# Define data: means, labels, and per-point GMM uncertainties
X_means = np.array([[2.0, 1.0], [-2.0, -1.0]])
y = np.array([+1, -1])
gmm_params = [
    {
        "weights": np.array([0.5, 0.5]),
        "means": np.array([[2.0, 1.0], [1.5, 1.5]]),
        "covs": np.array([
            [[0.1, 0.0], [0.0, 0.1]],
            [[0.1, 0.0], [0.0, 0.1]],
        ]),
    },
    {
        "weights": np.array([1.0]),
        "means": np.array([[-2.0, -1.0]]),
        "covs": np.array([[[0.1, 0.0], [0.0, 0.1]]]),
    },
]

# Validate
validate_gmm_dataset(X_means, y, gmm_params)

# Train
clf = SVMGMU(lam=0.01, n_epochs=500, lr_init=0.5)
clf.fit(X_means, y, gmm_params)

# Predict
predictions = clf.predict(X_means)
print(f"w = {clf.w_}, b = {clf.b_:.4f}")

# Evaluate expected misclassification under uncertainty
probs = clf.expected_misclassification()
print(f"Per-point P(misclass): {probs}")
```

## How It Works

Each data point has uncertainty described by a Gaussian mixture:

$$p_i(\mathbf{x}) = \sum_{j=1}^{K_i} \pi_{ij} \, \mathcal{N}(\boldsymbol{\mu}_{ij}, \boldsymbol{\Sigma}_{ij})$$

The SVM-GMU loss for point $i$ is:

$$\mathcal{L}_{\mathrm{GMU},i}(\mathbf{w}, b) = \sum_{j=1}^{K_i} \pi_{ij} \cdot \mathcal{L}_{\mathrm{GSU}}(\mathbf{w}, b; \boldsymbol{\mu}_{ij}, \boldsymbol{\Sigma}_{ij}, y_i)$$

where each $\mathcal{L}_{\mathrm{GSU}}$ is the closed-form expected hinge
loss from the SVM-GSU paper.

## References

- C. Tzelepis, V. Mezaris, I. Patras, "Linear Maximum Margin Classifier for Learning from Uncertain Data", IEEE TPAMI, 2018.
- S. Shalev-Shwartz, Y. Singer, N. Srebro, "Pegasos: Primal Estimated Sub-Gradient Solver for SVM", ICML, 2007.
