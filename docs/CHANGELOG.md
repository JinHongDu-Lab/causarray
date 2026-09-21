# Changelog

## [0.0.10] - Unreleased

Inference fix for small perturbation arms. Motivated by the SCARF mouse-brain
Perturb-seq pilot (58 perturbations, 68-227 cells each), where 83% of the
discoveries were genes with zero counts in the perturbed arm and real effects
were estimated but not called. See `plan/20260920_lfc_inference_fix_plan.md`
and the "Investigation" section of `docs/source/tutorial/SCARF/SCARF-py.ipynb`.

### Changed

- `LFC` uses the influence-function variance `var(eta)/n` of the estimator
  (`usevar='pooled'`) as its only variance estimator. `'unequal'` applied a
  two-sample Welch formula by arm to an estimator that averages over all
  cells; for equal arms it is exactly twice the correct standard error and
  for rare treatments far more. After validation on the Perturb-seq, SEA-AD
  and Adamson tutorials (point estimates unchanged or moved toward the raw
  log-ratio; permutation and fake-perturbation nulls calibrated) the Welch
  path was removed; `usevar='unequal'` is accepted as an alias of `'pooled'`
  with a `FutureWarning` for one release.
- `LFC`, `compute_causal_estimand`, `cross_fitting`,
  `estimate_propensity_scores` and `refit_propensity_scores` default to
  calibrated propensity scores (`class_weight=None`). The former
  `'balanced'` default centred scores near 0.5 regardless of prevalence,
  shrinking the AIPW correction by roughly twice the prevalence and turning
  the estimator into an outcome-model plug-in. `'balanced'` remains available.
- `ps_clip` defaults to `'auto'`: per treatment,
  `lower = min(0.01, prevalence/10)` and symmetrically for the upper bound.
  The fixed `(0.01, 0.99)` clipped every calibrated score of a treatment with
  prevalence below 1%. Resolved bounds are returned as
  `estimation['ps_clip_bounds']`.
- `LFC` default `thres_min` is `'auto'`: a gene is tested only if its larger
  arm mean implies about `min_counts` (new argument, default 5) expected
  counts in the smaller arm, i.e. `mean >= 5 / min(n0, n1)`. This equals the
  former 0.01 at 500 cells, 0.05 at 100 cells, and 0.007 at 700 cells, so
  small arms are protected without discarding testable genes in large arms
  (a fixed 0.05 removed 43% of Adamson pairs). The test is applied to the
  observed arm means as well as the counterfactual means, because the
  outcome model's prediction for an all-zero arm can be inflated.

### Added

- Model-based variance floor in `LFC`: the log-scale variance is bounded
  below by `1/(n1*tau1) + 1/(n0*tau0)`, the Poisson lower bound for means
  estimated from `n_k` cells. An arm whose cells all have zero counts has an
  empirical influence-function variance of zero; the floor gives it a
  standard error of at least ~1 for 100 cells, so chance all-zero arms of
  sparse genes are no longer called while genuine complete knockouts remain
  significant. The new `var_floored` column marks affected pairs. An arm
  with no observed counts stays estimable at the floor (its AIPW mean is
  exactly zero), so complete knockouts of expressed genes are reported
  instead of being dropped as non-estimable.
- Small-sample correction for in-sample nuisance fits (`K=1`): the pooled
  variance is rescaled by `n/(n-d)` (`d` = outcome-model parameters) and
  p-values use a t reference with `n-d` degrees of freedom. No-op for large
  `n`; brings null t-statistics on 85-donor pseudo-bulk (SEA-AD) and
  ~100-cell arms (Perturb-seq) from SD 1.08-1.10 to 1.0.
- Per-pair support columns `n_treated`, `n_control`, `count_treated`,
  `count_control`, plus `var_floored` and the pre-floor `std_raw`, in every
  `LFC` result frame.
- A `RuntimeWarning` when a treatment has fewer than 200 cells in one arm and
  the variance floor bound for some genes, and a `RuntimeWarning` when
  `backend='fast'` is requested but `crispyx` is not importable (previously a
  silent fall-back to gene-by-gene statsmodels).
- `tests/test_small_arm_inference.py`: oracle-simulation SE calibration for
  prevalence 0.5%-50%, chance all-zero arm not called, complete knockout
  still called, type-I error for 100 vs 5,000 cells across 0.02-5 counts per
  cell, class-weight invariance for balanced designs, prevalence-aware clip.

### Deprecated

- `LFC(usevar='unequal')` is an alias of `'pooled'` and warns (see above).
- `LFC(eps_var=...)` is ignored; the variance floor supersedes it.

## [0.0.9] - 2026-07-23

### Added

- `align_test_mask()` aligns named treatment-by-gene diagnostic masks to
  existing causarray result tables without refitting effects or changing
  inference.
- `summarize_treatment_associations()` and
  `plot_treatment_associations()` diagnose treatment associations with both
  observed covariates and estimated latent factors.
- `refit_propensity_scores()` supports treatment-specific covariate removal and
  feature-specific logistic L2 penalties while preserving untreated score
  columns and reusable outcome predictions.
- The Replogle tutorial now compares Wilcoxon and causarray effect estimates in
  the extreme negative LFC tail, reports expression support, and demonstrates
  diagnostic support rules.

### Changed

- Perturb-seq propensity diagnostics now include observed library size and a
  treatment-specific library-size regularization sensitivity analysis.
- Public diagnostics are documented in the README and LFC API guide, including
  the distinction between post-hoc result annotation and formal refitting or
  multiple-testing changes.
- `summarize_treatment_associations()` accepts `bh_scope='per_treatment'` to
  adjust within each treatment instead of across the whole treatment-by-covariate
  grid, and reports `n_tests_in_family` so the correction is self-describing.
- `plot_treatment_associations()` accepts an explicit symmetric `vmax`.
- `plot_propensity_scores()` accepts and forwards `clip_bounds`, which was
  previously fixed at its default in the summary it returns.
- `refit_propensity_scores()` reports `degenerate_design` and `score_std` per
  refitted treatment.

### Fixed

- `refit_propensity_scores()` documented that carried-over score columns are
  preserved *exactly*, while `clip` was in fact applied to the whole returned
  matrix. The uniform behaviour is kept, since `LFC` consumes a single bound,
  but it is now stated and covered by a test. Pass `clip=None` to leave
  carried-over scores untouched.
- `refit_propensity_scores()` now warns when covariate filtering leaves a
  constant design, which previously collapsed to a covariate-free `0.5` score
  in silence and quietly turned AIPW into an unweighted contrast.
- A scalar `clip` raised `TypeError: object of type 'float' has no len()`
  instead of the documented `ValueError` in `estimate_propensity_scores()` and
  `refit_propensity_scores()`.
- `plot_treatment_associations()` scaled the colour bar to the largest observed
  magnitude even for `spearman_rho`, which is bounded in [-1, 1]. Tiny
  correlations rendered as fully saturated and panels were not comparable
  across subsets; bounded statistics now use a fixed `(-1, 1)` range.
- `summarize_propensity_scores()` reported `clipped_fraction = 0.0` for raw
  scores by inferring clipping from a default `clip_bounds` the caller may
  never have applied. `clip_bounds=None` now yields `NaN`.
- The Replogle propensity cache script derived log-library size from raw counts
  while `prep_causarray_data` derives it from capped counts, so its design
  matrix did not match the tutorial's. The script now caps first and asserts
  the preprocessing contract before fitting.

## [0.0.8] - 2026-07-20

### Added

- `LFC()` and `gcate_lfc_batch()` results now include `log2fc` and
  `log2fc_se`, the base-2 equivalents of the existing natural-log `tau` and
  `std` columns. Existing statistics, p-values, and discoveries are unchanged.

### Changed

- Older compatible `gcate_lfc_batch()` caches are upgraded in memory with the
  new log2-scale columns when resumed.
- The Python environment now requires NumPy >2. A separate `causarray-r`
  environment supplies R 4.4 and a NumPy-2-compatible `reticulate`, avoiding
  changes to the Python numerical stack for R interoperability.

## [0.0.7] - 2026-07-18

### Added

- `estimate_propensity_scores()` estimates in-sample or out-of-fold treatment
  probabilities without rerunning the outcome model.
- `summarize_propensity_scores()` and `plot_propensity_scores()` report and
  visualize common support, tail mass, and inverse-weight effective sample size.
- `LFC()` now returns both clipped `pi_hat` and diagnostic `pi_hat_raw` scores.
- `LFC()` results include raw AIPW `mean_control`, `mean_treated`, and
  `estimable` diagnostics.
- Propensity and latent-factor association diagnostics in the Perturb-seq
  tutorial.

### Changed

- `estimate_propensity_scores()` now defaults to class-balanced logistic
  weighting, matching `LFC()` and historical causarray behavior. Pass
  `class_weight=None` explicitly for calibrated treatment probabilities.

### Fixed

- `LFC()` now always uses standard, unclipped AIPW pseudo-outcomes while
  retaining its established class-balanced propensity nuisance fit to limit
  unrelated result drift. This removes directional bias from clipping negative
  cell-level influence-function values under treatment-group imbalance.
  Calibrated scores remain available with `ps_class_weight=None`; the legacy
  cell-level clipping option has been removed.
- Valid aggregate counterfactual means are floored only for the final log-ratio
  and delta-method calculation, after averaging the unchanged pseudo-outcomes.
- Nonfinite or nonpositive aggregate counterfactual means are reported as
  non-estimable rather than silently floored before taking a log ratio.
- `cross_est=True` now enables two-fold nuisance cross-fitting, with an explicit
  `K` taking precedence.
- Intercept-only propensity models now respect the requested class weighting:
  balanced fits return 0.5 and unweighted fits return the case prevalence.
  Masks are indexed correctly within cross-fitting folds.

## [0.0.6] - 2026-05-31

### Added

- **Batch-wise fitting API** (`causarray/gcate.py`, `causarray/DR_learner.py`, `causarray/utils.py`)
  - `fit_gcate_batch(Y, X, A, r, batch_size=10, max_cells=2000, n_ctrl=2000, ...)`:
    Fits GCATE independently on batches of perturbations. A shared control
    subsample of `n_ctrl` cells is reused across batches; dispersion is
    pre-estimated once on the control pool. Supports `skip_batches` to resume
    interrupted runs; reports per-batch wall time and ETA when `verbose=True`.
  - `gcate_lfc_batch(Y, X, A, r, batch_size=10, max_cells=2000, n_ctrl=2000, cache_path=None, ...)`:
    End-to-end batch pipeline — runs GCATE and LFC per batch, freeing large
    intermediate arrays after each batch. `cache_path` enables HDF5 disk
    caching via `pandas.HDFStore` so interrupted runs resume from the last
    completed batch. Returns a concatenated DataFrame with a `'batch'` column.
  - `LFC_batch(...)`: deprecated alias for `gcate_lfc_batch`; emits
    `DeprecationWarning` and will be removed in a future release.
  - `n_batches` parameter for `fit_gcate_batch` and `gcate_lfc_batch`:
    specifies total number of batches instead of per-batch count; overrides
    `batch_size` when set.
  - `estimate_r(max_cells=N, random_state=0)`: new parameter that
    automatically subsamples to at most `N` cells before running JIC
    selection, prioritising control cells.

- **Fast GLM backend via crispyx** (`nb_glm_fast.py`, `gcate_glm.py`)
  - `fit_glm_fast()`: Batch NB-GLM fitting using crispyx's `NBGLMBatchFitter`,
    replacing per-gene statsmodels IRLS with vectorized batch IRLS.
  - `estimate_disp_fast()`: Vectorized method-of-moments dispersion estimation.
  - `fit_glm_ondisk()`: On-disk streaming GLM fitting for large h5ad files.
  - Per-perturbation fitting (`_fit_glm_fast_per_perturbation`): for
    multi-treatment data, fits binary (ctrl vs. treatment_k) models
    independently, then assembles the full coefficient matrix.
  - `fit_glm_auto()`: Routes to `fit_glm_fast()` when crispyx is available and
    the effective design dimension is small; falls back to statsmodels otherwise.
  - `estimate_disp_auto()`: Routes to `estimate_disp_fast()` for large gene
    counts; falls back to statsmodels otherwise.

### Fixed

- **Numba TBB fork warning**: Set `NUMBA_THREADING_LAYER_PRIORITY` to prefer
  OpenMP over TBB in `__init__.py`, eliminating fork warnings when Joblib forks
  after Numba parallel execution. Added `llvm-openmp` to conda dependencies.
- **Fast-path threshold** (`gcate_glm.py`): Raised the effective design-dimension
  ceiling so larger `r` values and wide batch designs correctly use the crispyx path.
- **Backend toggle** (`gcate_glm.py`): Re-added `_USE_FAST_BACKEND` module flag
  and `_backend_override()` context manager for reliable statsmodels fallback.
- **Weighted dispersion** (`nb_glm_fast.py`): Dispersion averaging is now
  cell-count-weighted; low-coverage perturbations contribute proportionally less.
- **Control-cell residuals** (`nb_glm_fast.py`): Fixed last-perturbation overwrite
  bug; control-cell deviance residuals and fitted values are now initialised from
  the global covariate model.
- **Module-qualified imports** (`gcate_opt.py`, `gcate.py`, `DR_estimation.py`):
  Backend toggles now propagate correctly at call time.
- **`estimate_r` bare name** (`gcate.py`): Fixed `NameError` caused by a bare
  `fit_glm_auto` reference.
- **crispyx availability check** (`gcate_glm.py`): Users without crispyx now get
  a transparent fallback to statsmodels instead of a traceback.

### Changed

- **⚠️ `alter_min()` early-stopping defaults** (`gcate_opt.py`):
  - Default `kwargs_es['max_iters']` reduced from 500 → 50.
  - Default `tolerance` reduced from `1e-3` → `0.0`; new scale-invariant
    `rel_tol=2e-4` introduced. To reproduce pre-v0.0.6 behavior, pass
    `kwargs_es_1=dict(max_iters=500)` and `kwargs_es_2=dict(max_iters=500)`.

- **⚠️ BREAKING — `LFC()` variance and default `usevar`** (`DR_learner.py`):
  - Default `usevar` changed from `'pooled'` to `'unequal'` (Welch). Revert
    with `LFC(..., usevar='pooled')` if reproducing pre-v0.0.6 results.
  - `'unequal'` formula corrected: variance is now `s₀²/n₀ + s₁²/n₁`
    (standard Welch); the prior version used `(s₀²/n₀ + s₁²/n₁)/2`
    ("half-Welch"), under-estimating the standard error by √2.
  - p-values now use the t-distribution with Welch-Satterthwaite degrees of
    freedom per gene; the prior version used a Normal approximation.

- `alter_min()` initialisation, `_check_input()`, `estimate_r()`, and
  `cross_fitting()` now use the auto-dispatch GLM/dispersion paths.
- `LFC()` now accepts `backend: str = "auto"` (`"fast"` forces crispyx,
  `"original"` forces statsmodels).
- `comp_size_factor()` vectorized with `np.nanmean`/`np.nanmedian`.

### Performance

Benchmarked on Perturb-seq data (n = 2,926 cells, p = 3,221 genes, 29 perturbations):

| Component | Original | Fast | Speedup |
|-----------|----------|------|---------|
| GCATE     | 331.6 s  | 298.5 s | 1.1×  |
| LFC       | 87.8 s   | 65.7 s  | 1.3×  |
| **Total** | **419.3 s** | **364.2 s** | **1.2×** |

On synthetic data (n = 500, p = 200): 61.5× GLM fit speedup, 7.1× imputation speedup.
Latent factor recovery: mean canonical correlation 0.998. LFC correlation: 0.856.

**Additional LFC throughput improvements** on the Replogle tutorial dataset
(79,865 cells × 8,563 genes, 200 perturbations, 14 batches):

| Change | Speedup contribution | Accuracy impact |
|--------|----------------------|-----------------|
| Stage 1 `max_iter` 50 → 5 (NB) / 10 (Poisson) | −10 min | identical (r=1.000) |
| Stage 1 ≤3,000-cell mixed subsample | −55 min | tau r=0.992, Jaccard=0.80 |
| Stage 2 joint fit | −5 min | tau r=0.9994, Jaccard=0.975 |
| **Combined** | **−70.6 min / 1.48×** | **tau r=0.9994, Jaccard=0.975** |

Full-run: 217.5 min → 146.9 min (**1.48×** faster); sig pairs −0.2%, perts with ≥1 hit −0.6%.

## [0.0.5] - 2025-01-30

- GCATE model for gene-level causal effect estimation from CRISPR screens.
- Doubly-robust learner (LFC, VIM) with AIPW pseudo-outcomes.
- Alternating minimization with Numba+OpenMP acceleration.
- Statsmodels-based per-gene NB-GLM fitting.
- Multiple testing correction (BH, step-down, FDX).
