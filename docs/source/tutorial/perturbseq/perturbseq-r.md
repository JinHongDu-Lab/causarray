# Perturb-seq Tutorial (R)

This tutorial uses an excitatory-neuron subset from [Jin et al. (2020),
*Science*](https://doi.org/10.1126/science.aaz6063), which studied gene
perturbations in the developing mouse brain. Data are available from the
[Broad Single Cell
Portal](https://singlecell.broadinstitute.org/single_cell/study/SCP1184).

The workflow is **prepare counts and treatment indicators → fit latent
factors → check propensity support and refit where it fails → estimate
log-fold changes**. Support is checked before the effects are read,
because an arm whose weights concentrate can report a discovery count
that reflects the weights rather than the biology. Saved outputs
illustrate one run; counts and rankings may change after refitting.

``` r
library(Seurat)
```

    ## Loading required package: SeuratObject

    ## Loading required package: sp

    ## 'SeuratObject' was built under R 4.4.1 but the current version is
    ## 4.4.2; it is recomended that you reinstall 'SeuratObject' as the ABI
    ## for R may have changed

    ## 'SeuratObject' was built with package 'Matrix' 1.6.5 but the current
    ## version is 1.7.2; it is recomended that you reinstall 'SeuratObject' as
    ## the ABI for 'Matrix' may have changed

    ## 
    ## Attaching package: 'SeuratObject'

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, t

``` r
sc.seurat <- readRDS("data/perturbseq-exneu.rds")

# Access the counts through Seurat when its installed version recognizes the
# serialized Assay5 object; otherwise recover the same layer and dimnames
# directly. The fallback keeps this tutorial runnable with older Seurat builds.
if ("RNA" %in% Assays(sc.seurat)) {
  counts <- GetAssayData(sc.seurat, assay = "RNA", layer = "counts")
} else {
  rna.assay <- sc.seurat@assays[["RNA"]]
  counts <- rna.assay@layers[["counts"]]
  rownames(counts) <- rownames(rna.assay@features)
  colnames(counts) <- rownames(rna.assay@cells)
}
Y <- data.frame(t(as.matrix(counts)), check.names = FALSE) # cell-by-gene matrix
metadata <- sc.seurat@meta.data

perturb <- metadata
colnames(perturb) <- gsub("Perturbation", "trt_", colnames(perturb))
perturb$trt_ <- relevel(as.factor(perturb$trt_), ref = "GFP")
A <- data.frame(
  model.matrix(~ trt_ - 1, data = perturb)[, -1, drop = FALSE],
  check.names = FALSE
) # cell-by-trt matrix; remove the first (GFP control) column
colnames(A) <- sub("^trt_", "", colnames(A))
```

## Prepare inputs

Use `Y` for cell-by-gene counts, `A` for perturbation indicators, and
`X`/`X_A` for outcome/propensity covariates. GFP control cells have
all-zero treatment indicators. `prep_causarray_data` prepares the
designs, including an intercept and log-library size in the propensity
model.

Use `reticulate` to call the Python package from R. The R environment is
defined in `environment-r.yaml`; `use_condaenv` selects the Python
environment.

``` r
require(reticulate)
```

    ## Loading required package: reticulate

``` r
Sys.setenv(PYTHONUNBUFFERED = TRUE)
use_condaenv('causarray')
causarray <- import("causarray")
cat(causarray$`__version__`)
```

    ## 0.0.10

``` r
# (Y, A) should be either data.frame or matrix
# optional covariates can be provided as matrices
dat <- causarray$prep_causarray_data(Y, A)
names(dat) <- c("Y", "A", "X", "X_A")
list2env(dat, .GlobalEnv)
```

    ## <environment: R_GlobalEnv>

## Estimate latent factors

This example uses a fixed illustrative rank, matching the Python fitting
cell. Inspect the JIC curve and nearby ranks in the Python example
before changing it. GCATE factors represent shared expression variation;
their interpretation as confounders depends on the model assumptions.

``` r
r <- 10
# Use the same early-stopping settings as the Python example.
# Equivalent results also require matching inputs and package versions.
res_gate <- causarray$fit_gcate(
  Y, X, A, r, verbose = TRUE,
  kwargs_es_1 = list(rel_tol = 2e-4, max_iters = 30L),
  kwargs_es_2 = list(rel_tol = 2e-4, max_iters = 30L)
) # a list of results from 2 stages optimization
```

    ## {'d': 30, 'n': 2926, 'p': 3221, 'r': 10}
    ## 'Estimating initial latent variables with GLMs...'
    ## 'Fitting nb GLM (structured, 1 covariates + 29 treatments)...'
    ## 'Estimating initial coefficients with GLMs...'
    ## 'Fitting nb GLM (structured, 11 covariates + 29 treatments)...'
    ## {'kwargs_es': {'max_iters': 30,
    ##                'patience': 5,
    ##                'rel_tol': 0.0002,
    ##                'tolerance': 0.0,
    ##                'warmup': 0},
    ##  'kwargs_glm': {'disp_glm': array([ 1.31485844,  2.49333886,  0.66275436, ..., 12.58894812,
    ##        17.78187864, 10.8189429 ]),
    ##                 'family': 'nb',
    ##                 'size_factor': array([0.53193358, 0.87362742, 1.2235467 , ..., 0.5593801 , 0.73025856,
    ##        0.77857223])},
    ##  'kwargs_ls': {'C': 1000.0,
    ##                'alpha': 0.1,
    ##                'beta': 0.5,
    ##                'max_iters': 20,
    ##                'recheck_interval': 10,
    ##                'sparsity_boost': 2.0,
    ##                'sparsity_threshold': 0.5,
    ##                'tol': 0.0001,
    ##                'tol_cell': 0.0001,
    ##                'tol_gene': 0.0001,
    ##                'warmup_iters': 0}}
    ## 'Fitting GCATE (step 1)...'
    ## {'d': 30, 'n': 2926, 'p': 3221, 'r': 10}
    ## {'kwargs_es': {'max_iters': 30,
    ##                'patience': 5,
    ##                'rel_tol': 0.0002,
    ##                'tolerance': 0.0,
    ##                'warmup': 0},
    ##  'kwargs_glm': {'disp_glm': array([ 1.31485844,  2.49333886,  0.66275436, ..., 12.58894812,
    ##        17.78187864, 10.8189429 ]),
    ##                 'family': 'nb',
    ##                 'size_factor': array([0.53193358, 0.87362742, 1.2235467 , ..., 0.5593801 , 0.73025856,
    ##        0.77857223])},
    ##  'kwargs_ls': {'C': 1000.0,
    ##                'alpha': 0.1,
    ##                'beta': 0.5,
    ##                'max_iters': 20,
    ##                'recheck_interval': 10,
    ##                'sparsity_boost': 2.0,
    ##                'sparsity_threshold': 0.5,
    ##                'tol': 0.0001,
    ##                'tol_cell': 0.0001,
    ##                'tol_gene': 0.0001,
    ##                'warmup_iters': 0}}
    ## 'Fitting GCATE (step 2)...'

``` r
U <- res_gate[[2]]$U
# reticulate returns `hist` as an R list, so unlist() before min().
cat(sprintf("Step 1 -- epochs: %d, best NLL: %.6f\n",
            as.integer(res_gate[[1]]$n_iter), min(unlist(res_gate[[1]]$hist))))
```

    ## Step 1 -- epochs: 25, best NLL: 1.683635

``` r
cat(sprintf("Step 2 -- epochs: %d, best NLL: %.6f\n",
            as.integer(res_gate[[2]]$n_iter), min(unlist(res_gate[[2]]$hist))))
```

    ## Step 2 -- epochs: 29, best NLL: 1.705020

### Treatment-association diagnostics

The summary compares each perturbation with shared controls using
Spearman correlations and standardized mean differences. P-values are
BH-adjusted across treatment–covariate pairs by default. Association
alone does not identify a confounder or justify removing a covariate.

Library size and latent factors are derived from the measured
transcriptome and may reflect perturbation responses as well as
technical variation. Their causal role requires scientific
justification. Factor labels refer to this fit; they need not retain the
same meaning after refitting.

``` r
W_A <- cbind(X_A, U)
factor_names <- paste0("U", seq_len(ncol(U)))
propensity_names <- c("intercept", "log_library_size", factor_names)
propensity_types <- c("observed", "observed", rep("latent", ncol(U)))

association_summary <- causarray$summarize_treatment_associations(
  A, W_A,
  covariate_names = propensity_names,
  covariate_types = propensity_types
)
observed_associations <- subset(
  association_summary, covariate_type == "observed" & !constant
)
observed_associations <- observed_associations[
  order(observed_associations$padj),
]
head(observed_associations, 10)
```

    ##     treatment        covariate covariate_type n_control n_treated spearman_rho
    ## 218     Satb2 log_library_size       observed       106        51   -0.6206012
    ## 134      Mbd5 log_library_size       observed       106       119   -0.5251680
    ## 26      Asxl3 log_library_size       observed       106       130   -0.5011708
    ## 326     Upf3b log_library_size       observed       106       100   -0.5244644
    ## 230    Scn2a1 log_library_size       observed       106        93   -0.4791687
    ## 146    Med13l log_library_size       observed       106        75   -0.4911242
    ## 242     Setd2 log_library_size       observed       106        76   -0.4504232
    ## 14      Ash1l log_library_size       observed       106       122   -0.3905315
    ## 254     Setd5 log_library_size       observed       106        71   -0.4383442
    ## 110    Fbxo11 log_library_size       observed       106       111   -0.3943441
    ##           pvalue         padj standardized_mean_difference constant
    ## 218 4.352985e-18 1.388602e-15                   -1.6704318    FALSE
    ## 134 2.377952e-17 3.792833e-15                   -1.1965476    FALSE
    ## 26  2.061826e-16 2.192409e-14                   -1.1478666    FALSE
    ## 326 5.917208e-16 4.718973e-14                   -1.1938053    FALSE
    ## 230 8.091610e-13 5.162447e-11                   -1.0708596    FALSE
    ## 146 2.226940e-12 1.183990e-10                   -1.0729179    FALSE
    ## 242 1.771080e-10 8.071064e-09                   -0.9716459    FALSE
    ## 14  1.003845e-09 3.710293e-08                   -0.8068996    FALSE
    ## 254 1.046791e-09 3.710293e-08                   -0.9526579    FALSE
    ## 110 1.730357e-09 5.519838e-08                   -0.8442098    FALSE
    ##     n_tests_in_family
    ## 218               319
    ## 134               319
    ## 26                319
    ## 326               319
    ## 230               319
    ## 146               319
    ## 242               319
    ## 14                319
    ## 254               319
    ## 110               319

``` r
latent_associations <- subset(
  association_summary, covariate_type == "latent"
)
latent_associations$abs_smd <- abs(
  latent_associations$standardized_mean_difference
)
latent_associations <- latent_associations[
  order(-latent_associations$abs_smd),
]
head(subset(latent_associations, select = -abs_smd), 10)
```

    ##     treatment covariate covariate_type n_control n_treated spearman_rho
    ## 227     Satb2        U9         latent       106        51    0.2847921
    ## 224     Satb2        U6         latent       106        51    0.2232722
    ## 154    Med13l        U8         latent       106        75    0.2034902
    ## 221     Satb2        U3         latent       106        51   -0.1881609
    ## 225     Satb2        U7         latent       106        51    0.2331754
    ## 223     Satb2        U5         latent       106        51   -0.2001647
    ## 346       Wac        U8         latent       106        85    0.1530500
    ## 335     Upf3b        U9         latent       106       100    0.1667636
    ## 142      Mbd5        U8         latent       106       119    0.1461313
    ## 222     Satb2        U4         latent       106        51   -0.1527494
    ##           pvalue       padj standardized_mean_difference constant
    ## 227 0.0003002609 0.00383133                    0.6475238    FALSE
    ## 224 0.0049417025 0.05630011                    0.4205666    FALSE
    ## 154 0.0060047742 0.06522452                    0.4086299    FALSE
    ## 221 0.0182784891 0.16659537                   -0.4020029    FALSE
    ## 225 0.0032932091 0.03890866                    0.3867679    FALSE
    ## 223 0.0119537822 0.11916427                   -0.3719454    FALSE
    ## 346 0.0345342442 0.28247241                    0.3021644    FALSE
    ## 335 0.0165866545 0.16033766                    0.3017549    FALSE
    ## 142 0.0284128718 0.23851858                    0.2944328    FALSE
    ## 222 0.0561495252 0.41655113                   -0.2855483    FALSE
    ##     n_tests_in_family
    ## 227               319
    ## 224               319
    ## 154               319
    ## 221               319
    ## 225               319
    ## 223               319
    ## 346               319
    ## 335               319
    ## 142               319
    ## 222               319

``` r
dir.create(
  "perturbseq-r_files/figure-markdown_github",
  recursive = TRUE, showWarnings = FALSE
)
association_plot <- causarray$plot_treatment_associations(
  association_summary
)
association_plot[[1]]$savefig(
  "perturbseq-r_files/figure-markdown_github/treatment-associations-1.png",
  dpi = 120L, bbox_inches = "tight"
)
knitr::asis_output(
  "![](perturbseq-r_files/figure-markdown_github/treatment-associations-1.png)"
)
```

![](perturbseq-r_files/figure-markdown_github/treatment-associations-1.png)

### Propensity-score diagnostics and refit

Read this before the effects: an arm whose weights concentrate on a few
cells reports a discovery count that reflects the weights rather than
the biology. Estimate five-fold out-of-fold logistic scores, where each
cell’s treatment label is held out of its propensity fit. The latent
factors were estimated using the full dataset, so this is conditional
validation of the propensity model, not cross-fitting of the entire
workflow. Unweighted logistic fits match the model class used by default
in `LFC`; they do not guarantee calibrated probabilities.

The table reports histogram overlap, scores outside the display interval
`[0.05, 0.95]`, Brier prediction error, and inverse-weight effective
sample size (ESS). There is no universal overlap cutoff. For weights
$w_i$, Kish’s ESS is

$$\mathrm{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2},$$

using $1/\hat\pi_i$ for treated cells and $1/(1-\hat\pi_i)$ for
controls. A low ESS fraction indicates concentrated weights. Inspect
both arms; neither a high ESS nor greater score overlap establishes
adequate confounding adjustment. See [Cole and
Hernán](https://doi.org/10.1093/aje/kwn164) for discussion of weighting
assumptions and diagnostics.

Pass `clip_bounds = NULL` for these raw scores; clipping cannot be
inferred from their values alone.

``` r
pi_oof <- causarray$estimate_propensity_scores(
  A, W_A, K = 5L, random_state = 0L
)
ps_summary <- causarray$summarize_propensity_scores(
  A, pi_oof, clip_bounds = NULL
)
ps_summary <- ps_summary[order(ps_summary$overlap_ratio),]
head(ps_summary[, c(
  "treatment", "n_treated", "overlap_ratio", "outside_overlap_fraction",
  "ess_control_fraction", "ess_treated_fraction", "brier_score"
)], 8)
```

    ##    treatment n_treated overlap_ratio outside_overlap_fraction
    ## 19     Satb2        51     0.2497225               0.22292994
    ## 12      Mbd5       119     0.2739020               0.04000000
    ## 20    Scn2a1        93     0.3082775               0.04020101
    ## 28     Upf3b       100     0.3103774               0.02912621
    ## 13    Med13l        75     0.3109434               0.05524862
    ## 3      Asxl3       130     0.3306241               0.02542373
    ## 18    Qrich1        86     0.3835015               0.00000000
    ## 2      Ash1l       122     0.3923600               0.00000000
    ##    ess_control_fraction ess_treated_fraction brier_score
    ## 19            0.4828017            0.5370468  0.09886707
    ## 12            0.3860983            0.4860468  0.13130423
    ## 20            0.7228786            0.5439490  0.14496125
    ## 28            0.6608906            0.7117094  0.13642822
    ## 13            0.8021303            0.3649246  0.14214569
    ## 3             0.5313239            0.9064971  0.12883687
    ## 18            0.8176082            0.5352798  0.16356239
    ## 2             0.5934557            0.7280965  0.16966052

``` r
weakest <- head(ps_summary$treatment, 4)
propensity_plot <- causarray$plot_propensity_scores(
  A, pi_oof, treatments = as.list(weakest), clip_bounds = NULL
)
propensity_plot[[1]]$savefig(
  "perturbseq-r_files/figure-markdown_github/propensity-overlap-1.png",
  dpi = 120L, bbox_inches = "tight"
)
knitr::asis_output(
  "![](perturbseq-r_files/figure-markdown_github/propensity-overlap-1.png)"
)
```

![](perturbseq-r_files/figure-markdown_github/propensity-overlap-1.png)

### Tuning the library-size penalty

`prep_causarray_data` appends standardized log library size to `X_A`.
For a few perturbations it differs enough between the perturbed and
control cells that the propensity model can tell them apart almost
perfectly. Measured library size mixes capture depth with total RNA
content, and these data do not identify which of the two drives a given
arm’s shift, so how much to adjust for it is a modelling choice rather
than a settled one.

`tune_penalty_factor` penalizes **that coefficient alone**, and only for
arms failing a pre-specified support check. Dropping the covariate is
the infinite-penalty limit, so it bounds what any finite factor can
achieve: the search evaluates that endpoint first, then returns the
smallest factor meeting the target. The report shows both, so the
penalty can be read against the limit rather than on its own.

Choose the target to match the failure. These arms lose histogram
overlap while keeping most of their treated sample, so overlap is the
target; a screen whose arms instead lose effective sample would target
that.

``` r
pi_before <- causarray$estimate_propensity_scores(
  A, W_A, K = 1L, C = 1.0, class_weight = NULL, clip = NULL, random_state = 0L
)
tuned <- causarray$tune_penalty_factor(
  A, W_A, "log_library_size",
  covariate_names = propensity_names,
  trigger = list(auc_gt = 0.9, ess_treated_fraction_lt = 0.5),
  target = list(overlap_ratio_gt = 0.3),
  K = 1L, C = 1.0, class_weight = NULL, random_state = 0L
)
penalty_factors <- tuned[[1]]
tuning_report <- tuned[[2]]
tuning_report[order(-tuning_report$penalty_factor), c(
  "treatment", "penalty_factor", "feasible", "n_fits",
  "overlap_ratio_unpenalized", "overlap_ratio_chosen",
  "overlap_ratio_dropped", "auc_unpenalized", "auc_chosen"
)]
```

    ##   treatment penalty_factor feasible n_fits overlap_ratio_unpenalized
    ## 4     Satb2      27.384196     TRUE      8                 0.1335553
    ## 2      Mbd5      17.782794     TRUE      8                 0.2308546
    ## 1     Asxl3      10.000000     TRUE      8                 0.2397678
    ## 3    Med13l       8.659643     TRUE      8                 0.2205031
    ## 5    Scn2a1       7.498942     TRUE      8                 0.2692230
    ## 6     Upf3b       5.623413     TRUE      8                 0.2503774
    ##   overlap_ratio_chosen overlap_ratio_dropped auc_unpenalized auc_chosen
    ## 4            0.3383278             0.3187199       0.9698483  0.8751387
    ## 2            0.3337561             0.7371175       0.9269859  0.8959886
    ## 1            0.3142235             0.7217707       0.9330189  0.9105951
    ## 3            0.3076730             0.5737107       0.9086792  0.8881761
    ## 5            0.3122337             0.7710489       0.9183404  0.9042402
    ## 6            0.3009434             0.6767925       0.9229245  0.9093396

``` r
refit <- causarray$refit_propensity_scores(
  A, W_A, pi_hat = pi_before, covariate_names = propensity_names,
  penalty_factors_by_treatment = penalty_factors,
  K = 1L, C = 1.0, class_weight = NULL, clip = NULL, random_state = 0L
)
pi_after <- refit[[1]]

support_after <- causarray$summarize_propensity_scores(
  A, pi_after, clip_bounds = NULL
)
support_after <- support_after[support_after$treatment %in% names(penalty_factors), c(
  "treatment", "overlap_ratio", "auc", "ess_treated_fraction"
)]
support_after
```

    ##    treatment overlap_ratio       auc ess_treated_fraction
    ## 3      Asxl3     0.3142235 0.9105951            0.9577911
    ## 12      Mbd5     0.3337561 0.8959886            0.9286911
    ## 13    Med13l     0.3076730 0.8881761            0.6634694
    ## 19     Satb2     0.3383278 0.8751387            0.6811826
    ## 20    Scn2a1     0.3122337 0.9042402            0.8689060
    ## 28     Upf3b     0.3009434 0.9093396            0.8823206

``` r
tuned_plot <- causarray$plot_propensity_scores(
  A, pi_after, treatments = as.list(weakest), clip_bounds = NULL
)
invisible(tuned_plot[[1]]$suptitle(
  "After: library size penalized on flagged arms only", y = 1.02
))
tuned_plot[[1]]$savefig(
  "perturbseq-r_files/figure-markdown_github/propensity-overlap-tuned-1.png",
  dpi = 120L, bbox_inches = "tight"
)
knitr::asis_output(
  "![](perturbseq-r_files/figure-markdown_github/propensity-overlap-tuned-1.png)"
)
```

![](perturbseq-r_files/figure-markdown_github/propensity-overlap-tuned-1.png)

### Estimate log-fold changes

`LFC` combines outcome predictions and propensity scores to estimate
adjusted log-fold changes. Reuse the GCATE size factors so both stages
use the same normalization. Inference uses the AIPW influence-function
variance with small-sample adjustments and a size-factor-aware Poisson
variance floor; see the `LFC` documentation for details.

``` r
offsets <- log(res_gate[[2]][['kwargs_glm']][['size_factor']]) # use the precomputed size factors
# pi_hat carries the tuned propensity from the section above, so the estimate
# and the diagnostics describe the same scores.
res <- causarray$LFC(Y, cbind(X, U), A, W_A, offset = offsets,
                     pi_hat = pi_after, verbose = TRUE)
```

    ## 'Estimating LFC...'
    ## {'a': 29, 'd': 11, 'd_A': 12, 'estimands': 'LFC', 'n': 2926, 'p': 3221}
    ## {'offset': array([-0.63123664, -0.13510128,  0.20175377, ..., -0.58092607,
    ##        -0.31435661, -0.25029351]),
    ##  'random_state': 0,
    ##  'verbose': True}
    ## 'Fit outcome models...'
    ## 'Fitting nb GLM (structured, 11 covariates + 29 treatments)...'
    ## 'Estimating AIPW mean...'

``` r
names(res) <- c("df_res", "estimation")
list2env(res, .GlobalEnv)
```

    ## <environment: R_GlobalEnv>

### Covariate filtering for one treatment

The tuner above adjusts how strongly one coefficient is penalized.
`refit_propensity_scores` can also remove a covariate from a single
treatment’s propensity design, which is a different operation: a penalty
shrinks a coefficient the data cannot support, whereas dropping asserts
the covariate does not belong in that model.

Satb2 is used here to show the mechanics: remove U9 or U8, raise the
library-size penalty, or shrink every coefficient with a smaller `C`.
These are illustrative settings for reading the audit output, not
candidates to choose among by discovery count; recheck factor
associations after refitting.

Only Satb2’s scores change. Out-of-fold fits supply the diagnostics,
while separate in-sample fits supply the variant LFC estimates using the
primary fit’s cached outcome predictions. Removing a factor here does
not remove it from the outcome model.

``` r
satb2_variants <- list(
  `drop U9` = list(drop_by_treatment = list(Satb2 = "U9")),
  `drop U8` = list(drop_by_treatment = list(Satb2 = "U8")),
  `10x library penalty` = list(
    penalty_factors_by_treatment = list(Satb2 = list(log_library_size = 10))
  ),
  `Satb2 C=0.1` = list(drop_by_treatment = list(Satb2 = list()), C = 0.1)
)

refit_satb2 <- function(pi_hat, K, options) {
  do.call(causarray$refit_propensity_scores, c(
    list(A, W_A, pi_hat = pi_hat, covariate_names = propensity_names,
         K = K, random_state = 0L),
    options
  ))
}

# Out-of-fold scores drive the overlap diagnostics.
oof_variants <- lapply(satb2_variants, function(options) {
  refit_satb2(pi_oof, 5L, options)
})

do.call(rbind, lapply(names(oof_variants), function(name) {
  audit <- oof_variants[[name]][[2]]
  data.frame(
    model = name,
    n_retained = audit$n_retained,
    degenerate_design = audit$degenerate_design,
    score_std = round(audit$score_std, 3)
  )
}))
```

    ##                 model n_retained degenerate_design score_std
    ## 1             drop U9         11             FALSE     0.304
    ## 2             drop U8         11             FALSE     0.300
    ## 3 10x library penalty         12             FALSE     0.221
    ## 4         Satb2 C=0.1         12             FALSE     0.221

``` r
satb2_row <- function(scores, name) {
  ps <- causarray$summarize_propensity_scores(A, scores, clip_bounds = NULL)
  transform(subset(ps, treatment == "Satb2"), model = name)
}
satb2_overlap <- rbind(
  satb2_row(pi_oof, "all factors"),
  do.call(rbind, lapply(names(oof_variants), function(name) {
    satb2_row(oof_variants[[name]][[1]], name)
  }))
)
satb2_overlap[, c(
  "model", "overlap_ratio", "outside_overlap_fraction",
  "ess_control_fraction", "ess_treated_fraction", "brier_score"
)]
```

    ##                   model overlap_ratio outside_overlap_fraction
    ## 19          all factors     0.2497225               0.22292994
    ## 194             drop U9     0.2606363               0.22929936
    ## 191             drop U8     0.2388087               0.25477707
    ## 192 10x library penalty     0.3492416               0.05095541
    ## 193         Satb2 C=0.1     0.2787643               0.01910828
    ##     ess_control_fraction ess_treated_fraction brier_score
    ## 19             0.4828017            0.5370468  0.09886707
    ## 194            0.2785753            0.4102262  0.10732713
    ## 191            0.3407912            0.5248322  0.10333950
    ## 192            0.8138645            0.5851291  0.13516335
    ## 193            0.7513981            0.7528600  0.13615172

``` r
regularized_plot <- causarray$plot_propensity_scores(
  A, oof_variants[["Satb2 C=0.1"]][[1]],
  treatments = list("Satb2"), clip_bounds = NULL
)
invisible(regularized_plot[[1]]$suptitle(
  "Satb2 after C=0.1 regularization", y = 1.02
))
regularized_plot[[1]]$savefig(
  "perturbseq-r_files/figure-markdown_github/satb2-c01-regularized-1.png",
  dpi = 120L, bbox_inches = "tight"
)
knitr::asis_output(
  "![](perturbseq-r_files/figure-markdown_github/satb2-c01-regularized-1.png)"
)
```

![](perturbseq-r_files/figure-markdown_github/satb2-c01-regularized-1.png)

``` r
# Analysis scores reuse the cached outcome model, so no outcome model is refitted.
satb2_all <- subset(
  df_res, trt == "Satb2", select = c(gene_names, tau, padj)
)
names(satb2_all)[-1] <- c("tau_all", "padj_all")

variant_summary <- do.call(rbind, lapply(names(satb2_variants), function(name) {
  analysis <- refit_satb2(estimation[["pi_hat_raw"]], 1L, satb2_variants[[name]])
  fit <- causarray$LFC(
    Y, cbind(X, U), A, W_A,
    offset = offsets,
    Y_hat = estimation[["Y_hat"]], pi_hat = analysis[[1]]
  )
  alternative <- subset(
    fit[[1]], trt == "Satb2", select = c(gene_names, tau, padj)
  )
  merged <- merge(satb2_all, alternative, by = "gene_names")
  data.frame(
    model = name,
    effect_correlation = cor(merged$tau_all, merged$tau),
    median_absolute_change = median(abs(merged$tau_all - merged$tau)),
    discoveries = sum(merged$padj < 0.1, na.rm = TRUE)
  )
}))
variant_summary$discoveries_all <- sum(satb2_all$padj_all < 0.1, na.rm = TRUE)
variant_summary
```

    ##                 model effect_correlation median_absolute_change discoveries
    ## 1             drop U9          0.9667417             0.03177003        1063
    ## 2             drop U8          0.9444852             0.04340389        1149
    ## 3 10x library penalty          0.9506038             0.01513041        1106
    ## 4         Satb2 C=0.1          0.9456426             0.02983502        1119
    ##   discoveries_all
    ## 1             979
    ## 2             979
    ## 3             979
    ## 4             979

### Reading the filtering comparison

The saved examples show that removing a factor can reduce control ESS
even when histogram overlap improves. Stronger penalties give less
concentrated weights, but their out-of-fold Brier scores are worse. The
effect estimates remain broadly correlated with the primary fit, while
some discovery decisions change.

That is a tradeoff, not evidence that one variant is more trustworthy.
Read prediction, weight concentration, and effect stability together,
and do not pick a design to raise overlap or discovery counts. Keep
factor removal for cases with a scientific reason to change the
adjustment set, and re-examine the pattern in a new run rather than
carrying a fixed `C` across datasets.

## Discovery summary

The plot reports genes passing the chosen BH-adjusted threshold for each
perturbation. Counts describe this fit and can change across runs.

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(ggplot2)

# Filter the results for significant discoveries
significant_discoveries <- df_res[df_res$padj < 0.1, ]

# Count the number of discoveries for each perturbation condition
discovery_counts <- as.data.frame(table(significant_discoveries$trt))
colnames(discovery_counts) <- c('Perturbation', 'Count')

# Order the discovery_counts by Count in descending order
discovery_counts <- discovery_counts %>% arrange(desc(Count))

# Set the factor levels of Perturbation to ensure ggplot respects the order
discovery_counts$Perturbation <- factor(discovery_counts$Perturbation, levels = discovery_counts$Perturbation)

# Plot the number of discoveries for each perturbation condition
ggplot(discovery_counts, aes(x = Perturbation, y = Count)) +
  geom_bar(stat = "identity", fill = "royalblue") +  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle('Number of Discoveries (padj < 0.1) for Each Perturbation Condition') +
  xlab('Perturbation Condition') +
  ylab('Number of Discoveries')
```

![](perturbseq-r_files/figure-markdown_github/unnamed-chunk-6-1.png)
