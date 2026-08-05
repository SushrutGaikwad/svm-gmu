# Experiments

The five experiments from Part V of the SVM-GMU report
([`../docs/reports/report_3/report.tex`](../docs/reports/report_3/report.tex)).
Each notebook is the **generator** for the corresponding report figures: running
it recomputes the results and saves the figures as `.pgf` into
[`../docs/reports/report_3/graphics/`](../docs/reports/report_3/graphics/), under
the exact filenames the report `\input`s, rendered with the LaTeX serif font
(Latin Modern) so they match the document. Rebuilding the report then picks up
the regenerated figures directly.

## Self-contained by design

Every notebook stands on its own. Each one defines its own dataset, its own
constants and its own helpers, and imports nothing from this repository except
the `svm_gmu` library that is the subject of the experiment. There is no shared
module: you can copy a single notebook out of the repository and it will still
tell the whole story of its experiment.

That does mean the six-point `close_separable` dataset is repeated in the four
notebooks that use it. This is deliberate. The duplication costs one literal
cell; sharing it would cost every reader a trip to another file to find out what
the data actually is.

The notebooks are written as worked experiments rather than as scripts. Each
step is explained before it runs, most cells are plain step-by-step code, and
where a helper does appear it is small and comes *after* the same thing has been
done longhand at least once.

## Running

From the repository root:

```bash
uv sync                                   # installs svm_gmu + JupyterLab
uv run jupyter lab                        # run a notebook interactively
# or, headless, regenerate one notebook's figures in place:
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 experiments/<notebook>.ipynb
```

The notebooks are **clone-and-run**: the setup cell searches upward for the
repository and puts `src/` on the import path, so no separate install is needed
and they work from any local clone regardless of where Jupyter was started.

Rendering requires a local LaTeX installation (`pdflatex` with the `lmodern`
package), configured once per notebook by `use_latex_serif()`. Without LaTeX,
comment out that call in the setup cell and matplotlib's own fonts are used
instead.

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
and the SVM-GMU-vs-standard-SVM comparison. Also reports the decision scores of
both models and the angle between their normals, and confirms over 30 seeds that
the boundary is seed-independent, as strong convexity predicts.

### Exp 2 - `svm_gmu_convergence.ipynb`

Draws `N` Monte-Carlo samples per point from the true GMMs, trains a standard SVM
on the `6N`-point cloud, and measures how close its boundary gets to the
closed-form SVM-GMU reference, sweeping `N` from 1 to 20000. The three
disagreement metrics are built up longhand at a single `N` before being swept.
The metric figure is the median and inter-quartile band over 30 seeds; the panel
figure shows the boundaries and the sampled clouds at each `N` for one
representative seed.

### Exp 3 - `gmu_vs_gsu_approximation.ipynb`

Replaces each GMM with its moment-matched single Gaussian (SVM-GSU) and compares
the two boundaries, isolating the value of the mixture structure. The law of
total covariance is applied by hand to one example, showing the within- and
between-component terms separately, before being applied to all six. Both
objectives are strongly convex, so the reported metrics are seed-independent
(verified over 30 seeds). Produces the three-panel contour-and-boundary
comparison.

### Exp 4 - `em_fitted_gmu.ipynb`

The realistic pipeline: sample each true GMM, re-estimate a GMM per point by EM
with the component count chosen by BIC, then fit SVM-GMU on the estimated
mixtures, sweeping `N`. BIC is demonstrated on a single example first, printing
the score for every candidate component count so the fit-versus-complexity
trade-off is visible. Reports the 30-seed median metrics, the per-point modal BIC
component count, and the per-`N` panels. This is the slowest notebook (EM over
many seeds); the first run is about 28 minutes, then instant from cache.

### Exp 5 - `high_dim_scaling.ipynb`

Builds a genuinely d-dimensional Gaussian-mixture dataset for each
`d` in {2, 3, 5, 10, 20, 35, 50} and measures `N*(d)`, the samples-per-example a
standard SVM needs to match the closed-form SVM-GMU boundary to a fixed angular
tolerance, plus a fixed-budget angle-versus-`d` view. The construction is built
and plotted at `d = 2` first, where it can be seen, before being generalized.

## Caching and reproducibility

The multi-seed experiments (2, 4, 5) run the full 30-seed sweeps and cache the
aggregated result to `.cache/` (git-ignored). The first run is slow; subsequent
runs load the cache and are instant. Each cached notebook has a
`FORCE_RECOMPUTE` flag next to its cache path: set it to `True` to recompute and
overwrite the cache. All seeds are spawned from a master seed of 2026, so the
regenerated numbers reproduce the report's tables exactly.

Experiment 5 picks its dimensions out of the cache by name rather than plotting
whatever the file happens to contain, and errors out if any are missing. That
keeps the figures determined by the notebook, so a cache left over from a run
with a different ladder of dimensions cannot quietly change what gets plotted.
