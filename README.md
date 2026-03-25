# SVM-GMU

## Support Vector Machines with Gaussian Mixture Sample Uncertainty

[![Open Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SushrutGaikwad/svm-gmu/blob/main/examples/demo_2d.ipynb)

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

# Two data points in 2D, one per class
X = np.array([
    [ 2.0,  1.0],   # Point 0: class +1
    [-2.0, -1.0],   # Point 1: class -1
])
y = np.array([+1, -1])

gmm_params = [
    # Point 0: Banana-shaped uncertainty (3 Gaussians along a curve)
    #
    #   Component 0 (weight 0.4): centered at [2.0, 1.0]
    #   Component 1 (weight 0.3): shifted to [1.3, 1.4], elongated
    #   Component 2 (weight 0.3): shifted to [0.6, 1.0], elongated
    #
    # Together these three form a banana curving to the left.
    {
        "weights": np.array([0.4, 0.3, 0.3]),
        "means": np.array([
            [2.0, 1.0],
            [1.3, 1.4],
            [0.6, 1.0],
        ]),
        "covs": np.array([
            [
                [0.05, 0.00],
                [0.00, 0.05]
            ],
            [
                [0.10, 0.03],
                [0.03, 0.05]
            ],
            [
                [0.12, 0.00],
                [0.00, 0.05]
            ],
        ]),
    },
    # Point 1: Crescent-shaped uncertainty (3 Gaussians along an arc)
    #
    #   Component 0 (weight 0.3): top of the arc at [-1.7, -0.7]
    #   Component 1 (weight 0.4): bottom of the arc at [-2.0, -1.3]
    #   Component 2 (weight 0.3): top of the arc at [-2.3, -0.7]
    #
    # Together these three form a crescent opening upward.
    {
        "weights": np.array([0.3, 0.4, 0.3]),
        "means": np.array([
            [-1.7, -0.7],
            [-2.0, -1.3],
            [-2.3, -0.7],
        ]),
        "covs": np.array([
            [
                [0.03, -0.02],
                [-0.02, 0.04]
            ],
            [
                [0.05,  0.00],
                [ 0.00, 0.02]
            ],
            [
                [0.03,  0.02],
                [ 0.02, 0.04]
            ],
        ]),
    },
]

validate_gmm_dataset(X, y, gmm_params)

clf = SVMGMU(lam=0.01, n_epochs=500, lr_init=0.5)
clf.fit(X, y, gmm_params)
print(f"w = {clf.w_}, b = {clf.b_:.4f}")

probs = clf.expected_misclassification()
print(f"Per-point P(misclass): {probs}")
```

Here are more examples of uncertainty shapes you can construct:

```python
# Bimodal: two well-separated blobs (dumbbell shape)
bimodal = {
    "weights": np.array([0.5, 0.5]),
    "means": np.array([[3.0, 0.0], [3.0, 1.5]]),
    "covs": np.array([
        [[0.04, 0.00], [0.00, 0.04]],
        [[0.04, 0.00], [0.00, 0.04]],
    ]),
}

# Trimodal triangle: three blobs at vertices of a triangle
triangle = {
    "weights": np.array([0.35, 0.35, 0.30]),
    "means": np.array([[0.0, 0.5], [0.4, -0.3], [-0.4, -0.3]]),
    "covs": np.array([
        [[0.03, 0.00], [0.00, 0.03]],
        [[0.03, 0.01], [0.01, 0.03]],
        [[0.03, -0.01], [-0.01, 0.03]],
    ]),
}

# Elongated diagonal: single Gaussian stretched along a 45-degree line
diagonal = {
    "weights": np.array([0.5, 0.5]),
    "means": np.array([[1.8, 0.8], [2.2, 1.2]]),
    "covs": np.array([
        [[0.06, 0.04], [0.04, 0.06]],
        [[0.06, 0.04], [0.04, 0.06]],
    ]),
}

# L-shape: two perpendicular elongated components
l_shape = {
    "weights": np.array([0.5, 0.5]),
    "means": np.array([[-1.0, 0.0], [-1.0, 0.6]]),
    "covs": np.array([
        [[0.15, 0.00], [0.00, 0.02]],   # horizontal elongation
        [[0.02, 0.00], [0.00, 0.12]],   # vertical elongation
    ]),
}

# Ring approximation: 4 components placed in a circle
ring = {
    "weights": np.array([0.25, 0.25, 0.25, 0.25]),
    "means": np.array([
        [ 0.3,  0.0],
        [ 0.0,  0.3],
        [-0.3,  0.0],
        [ 0.0, -0.3],
    ]),
    "covs": np.array([
        [[0.01, 0.00], [0.00, 0.03]],   # vertical at right
        [[0.03, 0.00], [0.00, 0.01]],   # horizontal at top
        [[0.01, 0.00], [0.00, 0.03]],   # vertical at left
        [[0.03, 0.00], [0.00, 0.01]],   # horizontal at bottom
    ]),
}
```

Each shape is just a collection of weighted Gaussians — by controlling how many components you use, where you place their means, and how you orient their covariances, you can approximate any shape you want.

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
