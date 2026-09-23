Doubly-robust semiparametric inference
======================================

:func:`LFC` estimates per-gene log-fold changes using a doubly-robust AIPW
estimator.  It requires the augmented covariate matrix ``W = [X | U]`` where
``U`` are the latent factors from :func:`fit_gcate`, and produces a DataFrame
with effect estimates, standard errors, and BH-adjusted p-values.

For screens with hundreds of perturbations use :func:`gcate_lfc_batch`, which
runs GCATE and LFC in batches to keep peak memory bounded.

Effect-size columns
-------------------

The returned DataFrame reports the treatment-versus-control effect on both
natural-log and base-2 scales:

=================  ===========================================================
Column             Definition
=================  ===========================================================
``tau``            Natural logarithm of the treated/control mean ratio.
``std``            Standard error of ``tau`` on the natural-log scale.
``log2fc``         Base-2 log fold change, exactly ``tau / log(2)``.
``log2fc_se``      Standard error of ``log2fc``, exactly ``std / log(2)``.
=================  ===========================================================

``log2fc_se`` is the standard error of the estimated log2 fold change, not the
sample standard deviation of expression. The rescaling leaves ``stat``,
p-values, adjusted p-values, and discoveries unchanged. ``tau`` and ``std``
remain available for backward compatibility. The same columns are returned by
``gcate_lfc_batch``; compatible caches made by older versions are upgraded in
memory when loaded.

Choosing the variance estimator
-------------------------------

Since 0.1.0 ``LFC`` uses ``usevar='pooled'``, the influence-function
(sandwich) variance ``var(eta)/n`` of the AIPW estimator, where ``eta`` are the
per-cell influence values of the log-ratio and ``n`` counts every cell that
enters the estimand. With calibrated propensity scores this equals the
efficient two-sample form ``Var(Y|A=1)/n1 + Var(Y|A=0)/n0`` up to the
outcome-model correction, for balanced and unbalanced arms alike, and it
matches the estimator's actual sampling variability in oracle simulations.

For in-sample nuisance fits (``K=1``, the default) the variance is rescaled by
``n/(n-d)``, with ``d`` the number of outcome-model parameters, and p-values
use a t reference with ``n-d`` degrees of freedom. Both are no-ops for large
``n``; on donor-level pseudo-bulk data (tens of donors) and ~100-cell
perturbation arms they remove the small-sample anti-conservativeness of the
raw sandwich variance.

Two further safeguards apply to every gene:

* **Model-based variance floor.** The log-scale variance is bounded below by
  ``1/(n1*tau1) + 1/(n0*tau0)``, the Poisson lower bound for arm means
  estimated from ``n1`` and ``n0`` cells. An arm whose cells all have zero
  counts has an empirical influence-function variance of zero; without the
  floor the floored log mean is reported with a spuriously tiny standard error
  and the pair is called. With ~100 perturbed cells the floor gives a standard
  error of at least about 1, so a chance all-zero arm of a sparse gene is not
  significant while a genuine complete knockout (``tau`` of -5 or more) still
  is. The ``var_floored`` column marks affected pairs and ``std_raw`` reports
  the pre-floor standard error.
* **Expression threshold.** ``thres_min='auto'`` (default since 0.1.0)
  requires about ``min_counts`` (5) expected counts in the smaller arm, i.e. a
  larger-arm mean of at least ``5 / min(n0, n1)`` counts per cell: 0.05 for a
  100-cell arm, 0.007 for a 700-cell arm. A fixed float can be passed
  instead.

``usevar='unequal'`` (the 0.0.6-0.0.9 default) applied a two-sample Welch
formula ``s0²/n0 + s1²/n1`` by arm. That is not the variance of an estimator
that averages pseudo-outcomes over all cells: for equal arm sizes it is exactly
twice the correct standard error, and for a rare treatment fitted with
class-balanced propensity scores it is an order of magnitude too large, so
real effects were estimated but not called. It was removed in 0.1.0 after
re-validation on the Perturb-seq, SEA-AD and Adamson tutorials; the argument
is accepted as an alias of ``'pooled'`` with a ``FutureWarning`` for one
release. Neither estimator models within-donor correlation; repeated cells
from one biological unit should still be pseudo-bulked or analysed with a
cluster-aware method.

Propensity diagnostics
----------------------

:func:`estimate_propensity_scores` estimates propensity scores without fitting
the outcome model.  Use ``K=5`` to obtain out-of-fold scores for positivity and
overfitting diagnostics.  :func:`summarize_propensity_scores` reports overlap,
tail mass, and inverse-weight effective sample size, while
:func:`plot_propensity_scores` compares treatment and control distributions.

Since 0.1.0 both the standalone estimator and ``LFC`` fit calibrated
logistic propensity scores by default (``class_weight=None``), which is what
the AIPW weights ``A/pi`` require. The former ``'balanced'`` default centred
the scores near 0.5 whatever the prevalence; for a treatment with 0.6%
prevalence that shrank the AIPW correction term by roughly twice the
prevalence and turned the estimator into an outcome-model plug-in whose
uncertainty the influence function no longer reflected. ``'balanced'`` remains
available to reproduce earlier analyses. Because in-sample logistic fits with
~100 cases against thousands of controls overstate separation, use out-of-fold
scores (``K=5``) when judging overlap.

Propensity scores used by AIPW are clipped with a prevalence-aware bound by
default (``ps_clip='auto'``: ``lower = min(0.01, prevalence/10)`` per
treatment, and symmetrically above). The fixed ``(0.01, 0.99)`` used before
0.1.0 clipped every calibrated score of a treatment with prevalence below 1%.
The resolved bounds are returned as ``estimation['ps_clip_bounds']`` and the
raw scores as ``estimation['pi_hat_raw']``.

``LFC`` uses the standard AIPW pseudo-outcome, which may be negative for
individual cells even though its counterfactual mean is positive. Individual
pseudo-outcomes are never clipped because doing so can bias the arm means,
particularly when a large shared control group is compared with much smaller
treatment groups.

Small perturbation arms
-----------------------

Screens with fewer than ~200 cells per perturbation and thousands of shared
controls are the regime in which the pre-0.1.0 defaults failed (SCARF
tutorial, "Investigation" section): 83% of discoveries were genes with zero
counts in the perturbed arm, and real effects had t-statistics halved by the
Welch formula. In this regime inspect the ``count_treated`` and
``var_floored`` columns, keep the default expression threshold, and expect a
``RuntimeWarning`` listing how many pairs the variance floor
bound.

Treatment-specific covariate diagnostics
----------------------------------------

``summarize_treatment_associations`` compares every observed covariate or
latent factor with each treatment using shared all-zero controls. It reports
pairwise Spearman correlations, standardized mean differences, and
Benjamini--Hochberg adjusted p-values. ``plot_treatment_associations`` displays
either effect-size measure as a heatmap. These are descriptive diagnostics:
there is deliberately no automatic threshold or drop decision.

By default the adjustment pools every treatment-by-covariate test into one
family, which with many perturbations is large and correspondingly
conservative. Pass ``bh_scope='per_treatment'`` to adjust within each
treatment's own block; the ``n_tests_in_family`` column records how many tests
entered each row's correction either way. In the heatmap, ``spearman_rho`` uses
a fixed ``(-1, 1)`` colour range so that panels stay comparable across subsets,
while the unbounded standardized mean difference scales to the data; pass
``vmax`` to set a symmetric limit explicitly.

When a scientifically justified sensitivity analysis uses a different
propensity design for each treatment, ``refit_propensity_scores`` accepts a
treatment-specific mapping such as ``{'Satb2': ['U9']}``. Supplying existing
raw scores refits only the named treatment columns and carries the others over
unchanged, up to the shared ``clip``, which is applied to the whole returned
matrix so a single consistent bound reaches ``LFC``; pass ``clip=None`` to leave
carried-over scores exactly as supplied.
For logistic L2 propensity models, ``penalty_factors_by_treatment`` can instead
retain a covariate while shrinking its coefficient more strongly. For example,
``{'Satb2': {'log_library_size': 10}}`` applies ten times the ordinary ridge
penalty to standardized log-library size for Satb2 only. This weighted penalty
is implemented by rescaling that feature during both fitting and prediction;
it is intentionally unavailable for tree and ensemble propensity models.
The updated scores can then be passed to ``LFC`` together with cached outcome
predictions::

   associations = summarize_treatment_associations(
       A, W_A, covariate_names=covariate_names,
       covariate_types=covariate_types,
   )
   pi_filtered, audit = refit_propensity_scores(
       A, W_A,
       pi_hat=estimation['pi_hat_raw'],
       covariate_names=covariate_names,
       penalty_factors_by_treatment={
           'Satb2': {'log_library_size': 10},
       },
   )
   filtered_results, _ = LFC(
       Y, W, A, W_A,
       Y_hat=estimation['Y_hat'], pi_hat=pi_filtered,
   )

Removing a treatment predictor does not create overlap in the underlying
population and can omit a genuine measured confounder. Treat filtered fits as
sensitivity analyses, distinguish pre-treatment covariates from possible
post-treatment variables, and compare propensity overlap, effective sample
sizes, and effect estimates before and after filtering.

Choosing the penalty factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``tune_penalty_factor`` selects the factor for one covariate per treatment
instead of fixing it by hand. It triggers on treatments failing a support
check, then returns the **smallest** factor meeting a target, so the covariate
keeps as much of its adjustment role as the data support::

   factors, report = tune_penalty_factor(
       A, W_A, 'log_library_size',
       treatment_names=treatment_names, covariate_names=covariate_names,
       trigger={'auc_gt': 0.9, 'ess_treated_fraction_lt': 0.5},
       target={'auc_lt': 0.9},
   )
   pi_tuned, audit = refit_propensity_scores(
       A, W_A, pi_hat=estimation['pi_hat_raw'],
       treatment_names=treatment_names, covariate_names=covariate_names,
       penalty_factors_by_treatment=factors,
   )

Dropping the covariate is the infinite-penalty limit, so it bounds what any
finite factor can achieve. The search evaluates that endpoint first: when the
dropped fit already misses the target, the treatment is reported with
``feasible=False`` after a single extra fit rather than an exhausted search,
and no penalty is applied. Otherwise the factor is found by bisection on a log
scale, and ``tol`` trades fits against how tightly the smallest qualifying
factor is resolved.

Target a monotone metric. ``auc`` and ``overlap_ratio`` move monotonically with
the penalty, so the endpoints bracket the search. ``ess_treated_fraction`` does
not: a completely separated arm has near-uniform weights and a deceptively high
ESS that *falls* as the penalty restores genuine overlap. Use ESS to trigger
and to report, and a separation metric as the target.

The returned ``report`` records, for every triggered treatment, the chosen
factor, whether the target was feasible, the number of fits used, and the
metrics unpenalized, at the chosen factor, and with the covariate dropped.
Report the trigger, target and grid alongside the results: tuning a nuisance
model against an overlap diagnostic is a specification choice, and choosing it
to maximise overlap or discoveries would be tuning toward the answer.

Penalizing and dropping are two points on one continuum. When a covariate is
affected by treatment, no factor makes that contrast identified; the penalty
only decides how much of a known bias to retain in exchange for precision. That
belongs in the analysis plan rather than in the search.

Filtering can go too far. If it leaves a constant design, the propensity model
degenerates to a covariate-free constant and AIPW reduces to an unweighted
contrast; ``refit_propensity_scores`` raises a ``RuntimeWarning`` and sets
``degenerate_design`` in the returned audit table. A very large penalty factor
shrinks a coefficient without making the design constant, so check
``score_std`` in the same table: a value near zero means the scores carry
almost no covariate information even though nothing was formally dropped.

Post-hoc diagnostic masks
-------------------------

``align_test_mask`` aligns a Boolean treatment-by-gene diagnostic mask to an
existing causarray result table by ``(trt, gene_names)`` labels. It accepts a
named DataFrame, a two-level indexed Series, or an array with explicit
treatment and gene names. The returned one-dimensional mask follows the
original row order of the result table, so it remains correct for reordered or
batched results::

   keep = align_test_mask(
       df_res, support_mask,
       treatment_names=perturbation_names,
       gene_names=gene_names,
   )
   df_res_flagged = df_res.assign(support_keep=keep)

Mask alignment does not refit effects, mutate ``df_res``, or change standard
errors and p-values. A mask chosen after inspecting outcomes should be reported
as a sensitivity diagnostic; retain the original BH-adjusted p-values rather
than redefining the testing family post hoc. The Replogle tutorial demonstrates
this workflow for several expression-support rules.

The result includes ``mean_control``, ``mean_treated``, and ``estimable``. These
are computed from the unclipped pseudo-outcomes. For numerical stability, valid
aggregate arm means are floored at ``thres_diff`` only when constructing the
log ratio and delta-method denominator. A pair with a nonfinite or nonpositive
raw aggregate mean remains non-estimable, so the aggregate floor cannot create
an extreme discovery from an invalid estimate.

.. automodule:: causarray.DR_learner
   :members:

.. automodule:: causarray.DR_estimation
   :members: estimate_propensity_scores, refit_propensity_scores,
             tune_penalty_factor

.. automodule:: causarray.diagnostics
   :members:
