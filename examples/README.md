# Examples

Worked, self-contained examples of using the `svm_gmu` library on small
two-dimensional datasets. Each notebook builds a dataset whose per-sample
uncertainty is a Gaussian mixture, trains the uncertainty-aware **SVM-GMU**
classifier, trains a **standard linear SVM** on the same points for comparison,
and visualizes both. These are library usage demos; the experiments that
reproduce the figures in the report live in [`../experiments/`](../experiments/).

## Running

From the repository root:

```bash
uv sync              # installs svm_gmu, matplotlib, and JupyterLab
uv run jupyter lab   # open and run a notebook interactively
```

The notebooks are **clone-and-run**: the first code cell searches upward from the
working directory for the repo's `src/` folder and puts it on the import path, so
no separate `pip install` is needed and the notebook works from any local clone.

Figures are rendered with the **LaTeX serif font** (Latin Modern) through
`svm_gmu.plotting.use_latex_serif()`, called once in the imports cell. This
requires a local LaTeX installation (`pdflatex`). If you do not have LaTeX,
comment out the `use_latex_serif()` line in the imports cell and matplotlib's
default fonts are used instead.

## The two notebooks

Both follow the same seven-step structure (define data, visualize uncertainty,
train SVM-GMU, train a standard SVM, plot the boundary, compare the two, check
predictions). They differ only in the dataset.

### `close_separable.ipynb`

The two classes sit close together but remain linearly separable, so the
orientation of the uncertainty has a clearly visible effect on the learned
boundary. This is the same `close_separable` dataset used by the report's first
four experiments. With `lam = 0.01`, `max_iter = 1000`, `batch_size = 6`,
`random_state = 42`, SVM-GMU converges to `w = [-1.2527, 1.2589]`,
`b = -0.9470`, while the standard SVM settles on a visibly different boundary.
A final "Saving the plots" section shows how to export a figure with the
`save_path` argument; it writes `comparison.{png,pdf,pgf}` into `figures/`.

### `banana_crescent.ipynb`

A second six-point dataset whose mixtures trace more pronounced banana and
crescent shapes (five to seven components each), drawn at the 1, 2, 3, and 4
sigma contour levels. It illustrates the same workflow on uncertainty that is
even more strongly non-Gaussian. Figures are shown inline only.

## API surface demonstrated

```python
from svm_gmu import SvmGmu
from svm_gmu.plotting import (
    plot_uncertainty, plot_boundary, plot_boundary_comparison, use_latex_serif,
)

use_latex_serif()                       # LaTeX serif fonts (needs pdflatex)

model = SvmGmu(lam=0.01, max_iter=1000, batch_size=6, random_state=42)
model.fit(X, y, sample_uncertainty=su)  # uncertainty-aware SVM-GMU
model.fit(X, y)                          # omit su -> standard linear SVM
model.predict(X)                         # labels in {+1, -1}
model.decision_function(X)               # signed distance w^T x + b
```

- `X` has shape `(n, 2)` (the observed feature vectors), `y` is in `{+1, -1}`.
- `sample_uncertainty` is a list of `n` dicts, one per sample, each with
  `"weights"` `(M_i,)`, `"means"` `(M_i, 2)`, and `"covariances"`
  `(M_i, 2, 2)` full or `(M_i, 2)` diagonal. Weights must be non-negative and
  sum to one; full covariances must be symmetric positive semi-definite.

The three plotting helpers take the data plus a fitted model (except
`plot_uncertainty`, which needs no model) and accept `sigmas`, `ax`,
`save_path`, and `savefig_kwargs`. They support two-dimensional data only.

## Output

`close_separable.ipynb` writes its exported figures into `figures/` (created on
demand and not tracked by git). The example notebooks do **not** write into the
report's `graphics/` directory; that is the job of the experiment notebooks.
