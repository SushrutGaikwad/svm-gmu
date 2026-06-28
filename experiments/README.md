# Experiments

The five experiments from Part V of the SVM-GMU report
([`../docs/reports/report_3/report.tex`](../docs/reports/report_3/report.tex)).
Each notebook is the **generator** for the corresponding report figures: running
it recomputes the results and saves the figures as `.pgf` into
[`../docs/reports/report_3/graphics/`](../docs/reports/report_3/graphics/), under
the exact filenames the report `\input`s, rendered with the LaTeX serif font
(Latin Modern) so they match the document. Rebuilding the report then picks up
the regenerated figures directly.

## Running

From the repository root:

```bash
uv sync                                   # installs svm_gmu + JupyterLab
uv run jupyter lab                        # run a notebook interactively
# or, headless, regenerate one notebook's figures in place:
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 experiments/<notebook>.ipynb
```

Like the examples, the notebooks are **clone-and-run**: a bootstrap cell searches
upward for the repo and puts both `src/` and `experiments/` on the import path.
Rendering requires a local LaTeX installation (`pdflatex` with the `lmodern`
package), configured once per notebook by `_common.configure_pgf()`.

## Shared module: `_common.py`

The single source of truth shared by every notebook (and by the repository's
tests under `../tests/`). It holds:

- **The `close_separable` dataset**: `X` (6x2), `y` in `{+1, -1}`, and
  `SAMPLE_UNCERTAINTY` (six per-sample GMMs with banana and crescent shapes).
  Used by experiments 1 to 4.
- **Constants**: `LAM = 0.01`, `SVM_SEED = 42`, `SVM_ITER = 1000`,
  `MASTER_SEED = 2026` (all seeds spawn from this), `M_CANDIDATES = 1..8` and
  `N_INIT = 5` (for EM model selection), and `GRAPHICS_DIR` (the report's
  graphics path).
- **Helpers**: `sample_from_gmm`, `build_sample_cloud`, `mc_eval_cloud`
  (sampling); `boundary_metrics` / `boundary_metrics_2d` (angle, offset, and
  grid/Monte-Carlo RMS between two boundaries); `moment_match_gmm` (law of total
  covariance); `fit_gmu` (fit SVM-GMU, or a standard SVM when uncertainty is
  `None`); `fit_gmm_bic` (EM with the component count chosen by BIC);
  `make_highdim_gmm_dataset` (the genuinely d-dimensional dataset for Exp 5);
  `make_seeds`, `band`, `svm_iter_for`, `configure_pgf`, and
  `make_convergence_metrics_figure`.

## Notebooks and the figures they generate

| Notebook | Report experiment | Figures written to `graphics/` | Cache | First-run cost |
| --- | --- | --- | --- | --- |
| `svm_gmu_fit.ipynb` | Exp 1: fit SVM-GMU | `dataset_gmm_uncertainty_contours.pgf`, `svm-gmu_decision_boundary.pgf`, `comparison.pgf` | none | seconds |
| `svm_gmu_convergence.ipynb` | Exp 2: Monte-Carlo convergence | `convergence_metrics.pgf`, `svm_gmu_convergence.pgf` | `.cache/conv_metrics_full.npz` | ~10 min |
| `gmu_vs_gsu_approximation.ipynb` | Exp 3: does the mixture matter | `gmu_vs_gsu_approximation.pgf` | none | seconds |
| `em_fitted_gmu.ipynb` | Exp 4: learn uncertainty by EM | `em_convergence_metrics.pgf`, `em_fitted_gmu_convergence.pgf` | `.cache/em_full.pkl` | ~28 min |
| `high_dim_scaling.ipynb` | Exp 5: scaling with dimension | `high_dim_nstar.pgf`, `high_dim_fixed_budget.pgf` | `.cache/high_dim_gmm_full.npz` | minutes |

### Exp 1 - `svm_gmu_fit.ipynb`

Fits SVM-GMU on the six `close_separable` points with their full GMM uncertainty,
and a standard linear SVM on the same points, then draws the three showcase
figures: the dataset's GMM contours, the SVM-GMU decision boundary with margins,
and the SVM-GMU-vs-standard-SVM comparison. Closed-form and seed-independent.

### Exp 2 - `svm_gmu_convergence.ipynb`

Draws `N` Monte-Carlo samples per point from the true GMMs, trains a standard SVM
on the `6N`-point cloud, and measures how close its boundary gets to the
closed-form SVM-GMU reference, sweeping `N` from 1 to 20000. The metric figure is
the median and inter-quartile band over 30 seeds; the panel figure shows the
boundaries at each `N` for one representative seed.

### Exp 3 - `gmu_vs_gsu_approximation.ipynb`

Replaces each GMM with its moment-matched single Gaussian (SVM-GSU) and compares
the two boundaries, isolating the value of the mixture structure. Both objectives
are strongly convex, so the reported metrics are seed-independent (verified over
30 seeds). Produces the three-panel contour-and-boundary comparison.

### Exp 4 - `em_fitted_gmu.ipynb`

The realistic pipeline: sample each true GMM, re-estimate a GMM per point by EM
with the component count chosen by BIC, then fit SVM-GMU on the estimated
mixtures, sweeping `N`. Reports the 30-seed median metrics, the per-point modal
BIC component count, and the per-`N` panels. This is the slowest notebook (EM
over many seeds); the first run is about 28 minutes, then instant from cache.

### Exp 5 - `high_dim_scaling.ipynb`

Builds a genuinely d-dimensional Gaussian-mixture dataset for each
`d` in {2, 3, 5, 10, 20, 50} and measures `N*(d)`, the samples-per-example a
standard SVM needs to match the closed-form SVM-GMU boundary to a fixed angular
tolerance, plus a fixed-budget angle-versus-`d` view.

## Caching and reproducibility

The multi-seed experiments (2, 4, 5) run the full 30-seed sweeps and cache the
aggregated result to `.cache/` (git-ignored). The first run is slow; subsequent
runs load the cache and are instant. Each cached notebook has a
`FORCE_RECOMPUTE` flag near the top: set it to `True` to recompute and overwrite
the cache. All seeds are spawned from `MASTER_SEED = 2026`, so the regenerated
numbers reproduce the report's tables exactly.
