# Experiments

The six experiments from Part V of the SVM-GMU report
([`../docs/reports/report_3/report.tex`](../docs/reports/report_3/report.tex)).
Each notebook is the **generator** for the corresponding report figures: running
it recomputes the results and saves the figures as `.pgf` into
[`../docs/reports/report_3/graphics/`](../docs/reports/report_3/graphics/), under
the exact filenames the report `\input`s, rendered with the LaTeX serif font
(Latin Modern) so they match the document. Rebuilding the report then picks up
the regenerated figures directly.

The two large panel figures (`svm_gmu_convergence.pgf` and
`em_fitted_gmu_convergence.pgf`) rasterize their sample-cloud scatter layers, so
each `.pgf` comes with sidecar `<figure>-img*.png` files that must sit next to
it in `graphics/` (the report resolves them via `\graphicspath`). Commit the
PNGs together with their `.pgf` whenever these figures are regenerated.

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

## Shared module: `_realworld.py`

Helpers for Experiment 6 (MNIST), mirroring `_common.py` so the two MNIST
notebooks stay thin. It holds the augmentation-cloud builders (`augment_image`,
`augmentation_cloud`, `bimodal_augmentation_cloud`), the ladder's uncertainty
constructors (`iso_gaussian`, `moment_gaussian`, `structural_components`,
`bimodal_structural_components`, `fit_gmm_bic_cov`), metric and significance
helpers (`evaluate_metrics`, `mcnemar_pvalue`, `paired_seed_tests`), the
per-seed and full-experiment drivers (`run_mnist_seed`,
`run_mnist_experiment`), and the figure makers (`make_sweep_figure`,
`plot_mnist_2d_panel`).

## Notebooks and the figures they generate

| Notebook | Report experiment | Figures written to `graphics/` | Cache | First-run cost |
| --- | --- | --- | --- | --- |
| `svm_gmu_fit.ipynb` | Exp 1: fit SVM-GMU | `dataset_gmm_uncertainty_contours.pgf`, `svm-gmu_decision_boundary.pgf`, `comparison.pgf` | none | seconds |
| `svm_gmu_convergence.ipynb` | Exp 2: Monte-Carlo convergence | `convergence_metrics.pgf`, `svm_gmu_convergence.pgf` | `.cache/conv_metrics_full.npz` | ~10 min |
| `gmu_vs_gsu_approximation.ipynb` | Exp 3: does the mixture matter | `gmu_vs_gsu_approximation.pgf` | none | seconds |
| `em_fitted_gmu.ipynb` | Exp 4: learn uncertainty by EM | `em_convergence_metrics.pgf`, `em_fitted_gmu_convergence.pgf` | `.cache/em_full.pkl` | ~28 min |
| `high_dim_scaling.ipynb` | Exp 5: scaling with dimension | `high_dim_nstar.pgf`, `high_dim_fixed_budget.pgf`, `high_dim_rms.pgf`, `high_dim_offset.pgf` | `.cache/high_dim_gmm_full.npz` | ~6 h (incremental) |
| `realworld_mnist_30seed.ipynb` | Exp 6: real-data SVM-GMU vs SVM-GSU on MNIST | `realworld_mnist_30seed_rotation.pgf`, `realworld_mnist_30seed_trainsize.pgf`, `realworld_mnist_30seed_panel.pgf` | `.cache/realworld_mnist_30seed.pkl` | ~2-3 h |
| `realworld_mnist.ipynb` | Exp 6 pilot (10 seeds, same config) | `realworld_mnist_rotation.pgf`, `realworld_mnist_trainsize.pgf`, `realworld_mnist_panel.pgf` (not referenced by the report) | `.cache/realworld_mnist.pkl` | ~40-60 min |

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
`d` in {2, 3, 5, 10, 20, 35, 50} and measures `N*(d)`, the samples-per-example a
standard SVM needs to match the closed-form SVM-GMU boundary to a fixed angular
tolerance, plus three fixed-budget views versus `d`: the angle, the Monte-Carlo
decision-function RMS, and the offset difference.

**Incremental cache.** Each `(d, seed)` cell reseeds its own generator, so its
result depends only on `(d, seed)` and the `N` ladder. `_load_or_run` exploits
this: it computes only the dimensions the cache is missing and merges them in,
so extending the sweep by one `d` costs one `d` (about 50 min) rather than a
full recompute (about 6 h at 30 seeds x 7 dimensions x 8 rungs). A cache built
for a different ladder or seed set is discarded rather than trusted, since those
would change every cell. `FORCE_RECOMPUTE = True` still rebuilds everything.

### Exp 6 - `realworld_mnist_30seed.ipynb`

Experiment 6 benchmarks the model ladder B0/B1/M0/M1/M2 on real MNIST data
(digits 4 vs 9), replacing the synthetic close-separable toy problem with a
genuine 28x28 image classification task. Per-image uncertainty is built by a
deliberately **bimodal** augmentation: each noisy copy is rotated by `+R` *or*
`-R` degrees (equiprobable) plus Gaussian jitter of 6 degrees and a small pixel
shift, so every cloud has two lobes that a single Gaussian cannot represent.
Clouds are projected to a 20-dimensional PCA subspace (fit on the pooled
training clouds) before fitting the SVM variants, with diagonal covariances.

**Model ladder** (all fit by the same `SvmGmu` estimator):

- **B0** (LSVM point): standard linear SVM; no uncertainty.
- **B1** (LSVM-iso): isotropic diagonal GSU from the augmentation cloud.
- **M0** (SVM-GSU): diagonal moment-matched single Gaussian per sample.
- **M1** (SVM-GMU structural): two-component mixture placed at the known
  `+R`/`-R` rotation modes, no EM.
- **M2** (SVM-GMU EM): free diagonal GMM fitted by EM with BIC component
  selection (`M` in 1..8).

**Evaluation on augmented test points.** Each test image is passed through the
same bimodal augmentation (10 copies) and the models are scored on those
points, the noisy-test-set regime of the SVM-GSU paper. This is one of the two
conditions for the mixture to matter (the other is genuinely multimodal
uncertainty); on clean test images the plain SVM wins and GMU ties GSU.

**Fixed lambda.** All five models share `lambda = 0.01` (a single-value
`lam_grid`), so accuracy differences are attributable to the uncertainty model
alone. A per-model cross-validated lambda grid is infeasible at this scale
(~130 h).

**Metrics:** accuracy, macro-F1, ROC-AUC, and average precision, reported
as 30-seed median with IQR bands.

**Significance:** GMU (M2) vs GSU (M0) at the fixed operating point
(R=30 deg, 80 training images/class) is tested by paired Wilcoxon
signed-rank, paired t-test over per-seed accuracies, and median McNemar
p-value.

**Sweeps:**

- *Rotation sweep*: `rot_ladder = [15, 30, 45]` degrees at fixed
  80 training images/class; saved as `realworld_mnist_30seed_rotation.pgf`.
- *Training-size sweep*: `train_ladder = [25, 80, 200]` images
  per class at fixed R=30 deg; saved as `realworld_mnist_30seed_trainsize.pgf`.
- *2D panel*: PCA-2 projection of augmentation clouds with fitted SVM
  boundaries overlaid; saved as `realworld_mnist_30seed_panel.pgf`.

**Cache:** `.cache/realworld_mnist_30seed.pkl` (git-ignored). The first run is
roughly 2-3 hours (30 seeds x 7 operating points x 5 models); subsequent runs
load from cache and are instant. Set `FORCE_RECOMPUTE = True` to recompute.
When running headless, raise the nbconvert cell timeout well above 3600 s (the
compute cell exceeds one hour).

`realworld_mnist.ipynb` is the same experiment at 10 seeds: the pilot run kept
for a quick (~40-60 min) reproduction. It caches to
`.cache/realworld_mnist.pkl` and writes the `realworld_mnist_*.pgf` figures,
which the report does not reference (the report uses the 30-seed figures).

## Caching and reproducibility

The multi-seed experiments (2, 4, 5, 6) run the full 30-seed sweeps and cache
the aggregated result to `.cache/` (git-ignored). The first run is slow;
subsequent runs load the cache and are instant. Each cached notebook has a
`FORCE_RECOMPUTE` flag near the top: set it to `True` to recompute and overwrite
the cache. All seeds are spawned from `MASTER_SEED = 2026`, so the regenerated
numbers reproduce the report's tables exactly.
