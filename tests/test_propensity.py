import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss
from scipy import stats
from statsmodels.stats.multitest import multipletests
from unittest.mock import patch

from causarray import (
    LFC,
    estimate_propensity_scores,
    tune_penalty_factor,
    plot_propensity_scores,
    plot_treatment_associations,
    refit_propensity_scores,
    summarize_propensity_scores,
    summarize_treatment_associations,
)
from causarray.DR_estimation import cross_fitting


def test_intercept_only_scores_respect_class_weight():
    A = np.zeros((100, 2))
    A[60:80, 0] = 1
    A[80:100, 1] = 1

    # Since 0.1.0 the default is calibrated; 'balanced' is the legacy option.
    calibrated = estimate_propensity_scores(A, np.ones((100, 1)))
    balanced = estimate_propensity_scores(
        A, np.ones((100, 1)), class_weight='balanced')

    np.testing.assert_allclose(balanced, 0.5)
    np.testing.assert_allclose(calibrated[:, 0], 0.25)
    np.testing.assert_allclose(calibrated[:, 1], 0.25)


def test_calibrated_option_improves_probability_calibration():
    rng = np.random.default_rng(12)
    n = 5000
    z = rng.standard_normal(n)
    true_pi = 1 / (1 + np.exp(-(-2.0 + 0.8 * z)))
    A = rng.binomial(1, true_pi)
    X = np.c_[np.ones(n), z]

    calibrated = estimate_propensity_scores(A, X)
    balanced = estimate_propensity_scores(A, X, class_weight='balanced')

    assert abs(calibrated.mean() - A.mean()) < 0.02
    assert abs(balanced.mean() - A.mean()) > 0.15
    assert brier_score_loss(A, calibrated[:, 0]) < brier_score_loss(A, balanced[:, 0])


def test_cross_fitted_scores_are_reproducible_and_unclipped():
    rng = np.random.default_rng(9)
    X = np.c_[np.ones(300), rng.standard_normal((300, 2))]
    A = rng.binomial(1, 1 / (1 + np.exp(-X[:, 1])))

    pi_1 = estimate_propensity_scores(A, X, K=5, random_state=4)
    pi_2 = estimate_propensity_scores(A, X, K=5, random_state=4)

    np.testing.assert_allclose(pi_1, pi_2)
    assert np.all((pi_1 > 0) & (pi_1 < 1))


def test_fold_specific_mask_uses_training_rows():
    rng = np.random.default_rng(5)
    n = 240
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    A = np.zeros((n, 2))
    A[80:120, 0] = 1
    A[120:160, 1] = 1
    mask = np.zeros_like(A, dtype=bool)
    controls = np.arange(n) < 80
    mask[:, 0] = controls | (A[:, 0] == 1)
    mask[:, 1] = controls | (A[:, 1] == 1)

    pi = estimate_propensity_scores(A, X, K=3, mask=mask)

    assert pi.shape == A.shape
    assert np.isfinite(pi).all()


def test_clipping_is_optional_and_applied_after_prediction():
    rng = np.random.default_rng(10)
    z = np.r_[rng.normal(-3, 0.2, 100), rng.normal(3, 0.2, 100)]
    A = np.r_[np.zeros(100), np.ones(100)]
    X = np.c_[np.ones(200), z]

    raw = estimate_propensity_scores(A, X, clip=None, C=100)
    clipped = estimate_propensity_scores(A, X, clip=(0.1, 0.9), C=100)

    assert raw.min() < 0.1 and raw.max() > 0.9
    assert clipped.min() >= 0.1 and clipped.max() <= 0.9


def test_cross_fitting_can_return_raw_and_clipped_scores():
    rng = np.random.default_rng(3)
    n, p = 120, 4
    z = rng.standard_normal(n)
    A = rng.binomial(1, 1 / (1 + np.exp(-3 * z)))[:, None]
    X = np.c_[np.ones(n), z]
    Y = rng.poisson(2, (n, p)).astype(float)
    Y_hat = np.ones((n, p, 1, 2))

    _, clipped, raw = cross_fitting(
        Y, A, X, X, K=3, Y_hat=Y_hat, ps_clip=(0.1, 0.9),
        return_raw_pi=True,
    )

    assert raw.min() < clipped.min() or raw.max() > clipped.max()
    assert clipped.min() >= 0.1 and clipped.max() <= 0.9


def test_cross_fitting_defaults_to_calibrated_and_allows_balanced_scores():
    rng = np.random.default_rng(31)
    n, p = 300, 3
    z = rng.standard_normal(n)
    A = rng.binomial(1, 1 / (1 + np.exp(-(-2 + z))))[:, None]
    X = np.c_[np.ones(n), z]
    Y = rng.poisson(2, (n, p)).astype(float)
    Y_hat = np.ones((n, p, 1, 2))

    _, _, calibrated = cross_fitting(
        Y, A, X, X, Y_hat=Y_hat, return_raw_pi=True,
    )
    _, _, legacy = cross_fitting(
        Y, A, X, X, Y_hat=Y_hat, return_raw_pi=True,
        ps_class_weight='balanced',
    )
    expected_calibrated = estimate_propensity_scores(A, X)
    expected_legacy = estimate_propensity_scores(A, X, class_weight='balanced')

    np.testing.assert_allclose(calibrated, expected_calibrated)
    np.testing.assert_allclose(legacy, expected_legacy)
    assert not np.allclose(calibrated, legacy)


def test_deprecated_class_weight_overrides_default():
    rng = np.random.default_rng(32)
    n, p = 200, 2
    z = rng.standard_normal(n)
    A = rng.binomial(1, 1 / (1 + np.exp(-(-2 + z))))[:, None]
    X = np.c_[np.ones(n), z]
    Y = rng.poisson(2, (n, p)).astype(float)
    Y_hat = np.ones((n, p, 1, 2))

    with pytest.warns(FutureWarning, match='ps_class_weight'):
        _, _, raw = cross_fitting(
            Y, A, X, X, Y_hat=Y_hat, return_raw_pi=True,
            class_weight='balanced',
        )

    np.testing.assert_allclose(
        raw, estimate_propensity_scores(A, X, class_weight='balanced'))


def test_propensity_summary_and_plot_for_named_treatments():
    rng = np.random.default_rng(14)
    A = pd.DataFrame(np.zeros((180, 2)), columns=['pert_a', 'pert_b'])
    A.loc[60:99, 'pert_a'] = 1
    A.loc[100:139, 'pert_b'] = 1
    pi = np.clip(0.2 + 0.1 * rng.standard_normal(A.shape), 0.01, 0.99)

    summary = summarize_propensity_scores(A, pi)
    fig, axes, plotted = plot_propensity_scores(A, pi, treatments=['pert_b'])

    expected = {
        'treatment', 'overlap_ratio', 'outside_overlap_fraction',
        'ess_control_fraction', 'ess_treated_fraction', 'auc', 'brier_score',
    }
    assert expected.issubset(summary.columns)
    assert summary['overlap_ratio'].between(0, 1).all()
    assert summary['ess_treated_fraction'].between(0, 1).all()
    assert plotted.equals(summary)
    assert axes.size >= 1
    plt.close(fig)


def test_treatment_associations_use_shared_controls_and_global_bh():
    A = pd.DataFrame(np.zeros((18, 2)), columns=['pert_a', 'pert_b'])
    A.loc[6:11, 'pert_a'] = 1
    A.loc[12:17, 'pert_b'] = 1
    Z = pd.DataFrame({
        'observed': np.r_[np.arange(6), np.arange(6) + 2, np.arange(6) + 100],
        'constant': np.ones(18),
    })

    summary = summarize_treatment_associations(
        A, Z, covariate_types=['observed', 'latent'],
    )
    row_a = summary.query("treatment == 'pert_a' and covariate == 'observed'").iloc[0]
    eligible_a = np.r_[np.zeros(6), np.ones(6)]
    expected_rho, expected_p = stats.spearmanr(Z['observed'][:12], eligible_a)
    finite_p = summary['pvalue'].dropna().to_numpy()
    expected_adjusted = multipletests(finite_p, method='fdr_bh')[1]

    assert len(summary) == 4
    assert row_a['n_control'] == 6
    assert row_a['n_treated'] == 6
    assert row_a['spearman_rho'] == pytest.approx(expected_rho)
    assert row_a['pvalue'] == pytest.approx(expected_p)
    np.testing.assert_allclose(summary['padj'].dropna(), expected_adjusted)
    assert summary.loc[summary['covariate'] == 'constant', 'constant'].all()
    assert summary.loc[summary['covariate'] == 'constant', 'padj'].isna().all()


def test_plot_treatment_associations_supports_subsets_and_annotations():
    A = pd.DataFrame(np.zeros((24, 2)), columns=['pert_a', 'pert_b'])
    A.loc[8:15, 'pert_a'] = 1
    A.loc[16:23, 'pert_b'] = 1
    Z = pd.DataFrame({
        'x': np.r_[np.zeros(8), np.ones(8), np.arange(8)],
        'u1': np.linspace(-1, 1, 24),
    })
    summary = summarize_treatment_associations(
        A, Z, covariate_types=['observed', 'latent'],
    )

    fig, ax = plot_treatment_associations(
        summary, value='standardized_mean_difference',
        treatments=['pert_a'], alpha=0.05,
    )

    assert [tick.get_text() for tick in ax.get_yticklabels()] == ['pert_a']
    assert len(ax.get_xticklabels()) == 2
    plt.close(fig)


def test_refit_propensity_scores_updates_only_requested_treatment():
    rng = np.random.default_rng(42)
    A = pd.DataFrame(np.zeros((180, 2)), columns=['pert_a', 'pert_b'])
    A.loc[60:119, 'pert_a'] = 1
    A.loc[120:179, 'pert_b'] = 1
    X_A = pd.DataFrame({
        'intercept': np.ones(180),
        'x_a': np.r_[rng.normal(-2, .2, 60), rng.normal(2, .2, 60),
                     rng.normal(0, 1, 60)],
        'x_b': rng.normal(size=180),
    })
    base = np.c_[np.full(180, 0.2), np.full(180, 0.7)]

    updated, report = refit_propensity_scores(
        A, X_A, {'pert_a': ['x_a']}, pi_hat=base, K=3,
        random_state=5,
    )

    assert not np.array_equal(updated[:, 0], base[:, 0])
    np.testing.assert_array_equal(updated[:, 1], base[:, 1])
    assert report.loc[0, 'treatment'] == 'pert_a'
    assert report.loc[0, 'dropped_covariates'] == ['x_a']
    assert report.loc[0, 'retained_covariates'] == ['intercept', 'x_b']


def test_refit_propensity_scores_without_base_fits_every_treatment():
    rng = np.random.default_rng(43)
    A = np.zeros((150, 2))
    A[50:100, 0] = 1
    A[100:, 1] = 1
    X_A = np.c_[np.ones(150), rng.normal(size=(150, 2))]

    updated, report = refit_propensity_scores(
        A, X_A, {0: [1]}, covariate_names=['intercept', 'x1', 'x2'], K=3,
    )
    eligible = (A.sum(axis=1) == 0) | (A[:, 0] == 1)
    expected = estimate_propensity_scores(
        A[:, [0]], X_A[:, [0, 2]], K=3, mask=eligible[:, None],
    )

    assert updated.shape == A.shape
    assert len(report) == 2
    np.testing.assert_allclose(updated[:, 0], expected[:, 0])


def test_filtered_propensity_scores_reuse_cached_outcome_predictions():
    rng = np.random.default_rng(44)
    n, p = 90, 3
    A = np.zeros((n, 1))
    A[45:, 0] = 1
    W = np.c_[np.ones(n), rng.normal(size=(n, 2))]
    Y = rng.poisson(2, size=(n, p)).astype(float)
    Y_hat = np.full((n, p, 1, 2), 2.0)
    base_pi = np.full((n, 1), 0.5)
    updated, _ = refit_propensity_scores(
        A, W, {0: [2]}, pi_hat=base_pi,
        covariate_names=['intercept', 'x1', 'u1'],
    )

    with patch('causarray.DR_estimation._gcate_glm.fit_glm_auto') as outcome_fit:
        result, estimation = LFC(
            Y, W, A, W, family='poisson', Y_hat=Y_hat, pi_hat=updated,
        )

    outcome_fit.assert_not_called()
    assert len(result) == p
    np.testing.assert_allclose(estimation['Y_hat'], Y_hat)


def test_new_diagnostics_reject_invalid_names_and_drop_lists():
    A = pd.DataFrame({'pert': [0, 0, 1, 1]})
    Z = np.c_[np.ones(4), np.arange(4)]

    with pytest.raises(ValueError, match='unique'):
        summarize_treatment_associations(
            A, Z, covariate_names=['duplicate', 'duplicate'],
        )
    with pytest.raises(ValueError, match='Unknown covariate'):
        refit_propensity_scores(
            A, Z, {'pert': ['missing']}, pi_hat=np.full((4, 1), 0.5),
            covariate_names=['intercept', 'x'],
        )
    with pytest.raises(ValueError, match='every covariate'):
        refit_propensity_scores(
            A, Z, {'pert': ['intercept', 'x']},
            pi_hat=np.full((4, 1), 0.5),
            covariate_names=['intercept', 'x'],
        )


def test_refit_propensity_scores_applies_feature_specific_l2_penalty():
    rng = np.random.default_rng(45)
    A = pd.DataFrame(np.zeros((180, 2)), columns=['pert_a', 'pert_b'])
    A.loc[60:119, 'pert_a'] = 1
    A.loc[120:179, 'pert_b'] = 1
    X_A = pd.DataFrame({
        'intercept': np.ones(180),
        'library_size': rng.normal(size=180),
        'u1': rng.normal(size=180),
    })
    base = np.c_[np.full(180, 0.2), np.full(180, 0.7)]

    updated, report = refit_propensity_scores(
        A, X_A, pi_hat=base,
        penalty_factors_by_treatment={'pert_a': {'library_size': 9}},
        K=3, random_state=6,
    )
    eligible = (A.to_numpy().sum(axis=1) == 0) | (A['pert_a'].to_numpy() == 1)
    X_manual = X_A.to_numpy().copy()
    X_manual[:, 1] /= 3
    expected = estimate_propensity_scores(
        A[['pert_a']], X_manual, K=3, mask=eligible[:, None], random_state=6,
    )

    np.testing.assert_allclose(updated[:, 0], expected[:, 0])
    np.testing.assert_array_equal(updated[:, 1], base[:, 1])
    assert report.loc[0, 'penalty_factors'] == {'library_size': 9.0}


def test_feature_specific_penalties_validate_model_factor_and_conflicts():
    A = pd.DataFrame({'pert': [0, 0, 1, 1]})
    X_A = pd.DataFrame({'intercept': np.ones(4), 'library_size': np.arange(4)})
    base = np.full((4, 1), 0.5)

    with pytest.raises(ValueError, match='at least 1'):
        refit_propensity_scores(
            A, X_A, pi_hat=base,
            penalty_factors_by_treatment={'pert': {'library_size': 0.5}},
        )
    with pytest.raises(ValueError, match='only for logistic'):
        refit_propensity_scores(
            A, X_A, pi_hat=base, ps_model='ensemble',
            penalty_factors_by_treatment={'pert': {'library_size': 2}},
        )
    with pytest.raises(ValueError, match='both dropped and penalized'):
        refit_propensity_scores(
            A, X_A, {'pert': ['library_size']}, pi_hat=base,
            penalty_factors_by_treatment={'pert': {'library_size': 2}},
        )


def _two_treatment_design(n=180, seed=11):
    """Balanced two-treatment design with genuine all-zero controls."""
    rng = np.random.default_rng(seed)
    A = pd.DataFrame(np.zeros((n, 2)), columns=['pert_a', 'pert_b'])
    A.iloc[n // 3:2 * n // 3, 0] = 1
    A.iloc[2 * n // 3:, 1] = 1
    X_A = pd.DataFrame({
        'intercept': np.ones(n),
        'library_size': rng.normal(size=n),
        'u1': rng.normal(size=n),
    })
    return A, X_A


@pytest.mark.parametrize('clip', [0.01, (0.5,), (0.1, 0.2, 0.3), 'wide'])
def test_clip_rejects_values_that_are_not_bound_pairs(clip):
    A, X_A = _two_treatment_design()

    with pytest.raises(ValueError, match='clip must be None or a pair'):
        estimate_propensity_scores(A, X_A, clip=clip)
    with pytest.raises(ValueError, match='clip must be None or a pair'):
        refit_propensity_scores(A, X_A, clip=clip)


def test_refit_clip_also_bounds_carried_over_columns():
    """The documented contract: clip covers refit and carried-over columns."""
    A, X_A = _two_treatment_design()
    base = np.full((len(A), 2), 0.001)

    unclipped, _ = refit_propensity_scores(
        A, X_A, pi_hat=base, drop_by_treatment={'pert_a': ['u1']}, clip=None,
    )
    clipped, _ = refit_propensity_scores(
        A, X_A, pi_hat=base, drop_by_treatment={'pert_a': ['u1']},
        clip=(0.01, 0.99),
    )

    # 'pert_b' is never refitted: preserved exactly without clip, bounded with it.
    np.testing.assert_array_equal(unclipped[:, 1], base[:, 1])
    np.testing.assert_array_equal(clipped[:, 1], np.full(len(A), 0.01))
    assert clipped[:, 0].min() >= 0.01 and clipped[:, 0].max() <= 0.99


def test_refit_warns_and_reports_degenerate_design():
    A, X_A = _two_treatment_design()

    with pytest.warns(RuntimeWarning, match='constant after filtering'):
        scores, report = refit_propensity_scores(
            A, X_A, drop_by_treatment={'pert_a': ['library_size', 'u1']},
        )

    indexed = report.set_index('treatment')
    assert bool(indexed.loc['pert_a', 'degenerate_design'])
    assert not bool(indexed.loc['pert_b', 'degenerate_design'])
    assert indexed.loc['pert_a', 'n_retained'] == 1
    assert indexed.loc['pert_a', 'score_std'] == pytest.approx(0.0)
    # class_weight='balanced' on a constant design falls back to 0.5.
    np.testing.assert_allclose(scores[:, 0], 0.5)


def test_extreme_penalty_collapse_is_visible_without_being_degenerate():
    A, X_A = _two_treatment_design()

    _, mild = refit_propensity_scores(
        A, X_A, penalty_factors_by_treatment={'pert_a': {'library_size': 1}},
    )
    _, harsh = refit_propensity_scores(
        A, X_A, penalty_factors_by_treatment={
            'pert_a': {'library_size': 1e6, 'u1': 1e6},
        },
    )

    assert not harsh.loc[0, 'degenerate_design']
    assert harsh.loc[0, 'score_std'] < 1e-3 < mild.loc[0, 'score_std']


def test_refit_supports_cross_fitting_with_an_explicit_mask():
    A, X_A = _two_treatment_design(n=240, seed=3)
    keep = np.ones((len(A), 2), dtype=bool)
    keep[:20] = False  # drop some controls from the fitting sample

    scores, report = refit_propensity_scores(
        A, X_A, drop_by_treatment={'pert_a': ['u1']}, K=3, mask=keep,
        random_state=4,
    )

    eligible = (A.to_numpy().sum(axis=1) == 0) | (A['pert_a'].to_numpy() == 1)
    expected = estimate_propensity_scores(
        A[['pert_a']], X_A[['intercept', 'library_size']], K=3,
        mask=(eligible & keep[:, 0])[:, None], random_state=4,
    )
    np.testing.assert_allclose(scores[:, 0], expected[:, 0])
    assert not report.loc[0, 'degenerate_design']


def test_association_heatmap_uses_fixed_limits_for_bounded_correlations():
    A, X_A = _two_treatment_design()
    Z = 0.01 * X_A[['library_size', 'u1']].to_numpy()
    summary = summarize_treatment_associations(
        A, Z, covariate_names=['library_size', 'u1'],
    )

    _, rho_ax = plot_treatment_associations(summary)
    _, override_ax = plot_treatment_associations(summary, vmax=0.25)
    _, smd_ax = plot_treatment_associations(
        summary, value='standardized_mean_difference',
    )

    assert rho_ax.images[0].get_clim() == (-1.0, 1.0)
    assert override_ax.images[0].get_clim() == (-0.25, 0.25)
    smd_limit = smd_ax.images[0].get_clim()[1]
    assert smd_limit == pytest.approx(
        np.nanmax(np.abs(summary['standardized_mean_difference']))
    )
    with pytest.raises(ValueError, match='vmax must be None'):
        plot_treatment_associations(summary, vmax=0)
    plt.close('all')


def test_summary_reports_unknown_clipping_for_raw_scores():
    A, X_A = _two_treatment_design()
    raw = estimate_propensity_scores(A, X_A, clip=None)

    assumed = summarize_propensity_scores(A, raw)
    unknown = summarize_propensity_scores(A, raw, clip_bounds=None)

    assert assumed['clipped_fraction'].notna().all()
    assert unknown['clipped_fraction'].isna().all()

    _, _, plotted = plot_propensity_scores(A, raw, clip_bounds=None)
    assert plotted['clipped_fraction'].isna().all()
    plt.close('all')


def test_treatment_associations_support_per_treatment_bh():
    A, X_A = _two_treatment_design(n=300, seed=8)
    Z = X_A[['library_size', 'u1']].to_numpy()

    glob = summarize_treatment_associations(
        A, Z, covariate_names=['library_size', 'u1'],
    )
    per = summarize_treatment_associations(
        A, Z, covariate_names=['library_size', 'u1'], bh_scope='per_treatment',
    )

    assert (glob['n_tests_in_family'] == 4).all()
    assert (per['n_tests_in_family'] == 2).all()
    np.testing.assert_allclose(per['pvalue'], glob['pvalue'])
    # Each scope is exactly BH over its own family. Note that a smaller family
    # is not uniformly less conservative: BH is a step-up procedure, so an
    # individual padj can move either way.
    np.testing.assert_allclose(
        glob['padj'], multipletests(glob['pvalue'], method='fdr_bh')[1],
    )
    for treatment in ('pert_a', 'pert_b'):
        block = per[per['treatment'] == treatment]
        np.testing.assert_allclose(
            block['padj'], multipletests(block['pvalue'], method='fdr_bh')[1],
        )

    with pytest.raises(ValueError, match="bh_scope must be"):
        summarize_treatment_associations(A, Z, bh_scope='per_covariate')


# ---------------------------------------------------------------------------
# tune_penalty_factor
# ---------------------------------------------------------------------------

def _separating_design(n=400, seed=0):
    """One arm separated by a single covariate, one arm overlapping."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, 2))
    A[:40, 0] = 1        # separated arm
    A[40:80, 1] = 1      # benign arm
    ctrl = A.sum(axis=1) == 0
    sep = rng.normal(0, 1, n)
    sep[A[:, 0] == 1] += 6.0          # drives near-perfect separation
    noise = rng.normal(0, 1, n)
    X_A = np.column_stack([np.ones(n), sep, noise])
    names = ['sep_arm', 'benign_arm']
    cov = ['intercept', 'driver', 'noise']
    return A, X_A, names, cov, ctrl


def test_tune_penalty_factor_recovers_overlap_for_the_separated_arm():
    A, X_A, names, cov, _ = _separating_design()
    factors, report = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.9}, target={'auc_lt': 0.9}, random_state=0)
    row = report.set_index('treatment').loc['sep_arm']
    assert row['feasible']
    assert row['auc_unpenalized'] > 0.9
    assert row['auc_chosen'] < 0.9
    assert factors['sep_arm']['driver'] > 1.0
    # the benign arm is never triggered, so it gets no penalty
    assert 'benign_arm' not in factors
    assert 'benign_arm' not in report['treatment'].tolist()


def test_tune_penalty_factor_leaves_untouched_arms_identical():
    A, X_A, names, cov, _ = _separating_design()
    base = estimate_propensity_scores(A, X_A, K=1, clip=None, random_state=0)
    factors, _ = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.9}, target={'auc_lt': 0.9}, random_state=0)
    updated, _ = refit_propensity_scores(
        A, X_A, pi_hat=base.copy(), treatment_names=names, covariate_names=cov,
        penalty_factors_by_treatment=factors, K=1, clip=None, random_state=0)
    j = names.index('benign_arm')
    np.testing.assert_array_equal(base[:, j], updated[:, j])


def test_tune_penalty_factor_reports_infeasible_without_searching():
    """Dropping the covariate is the infinite-penalty limit, so a target the
    dropped fit misses cannot be reached by any finite factor."""
    A, X_A, names, cov, _ = _separating_design()
    factors, report = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.9}, target={'auc_lt': 0.0},   # unreachable
        bracket=(1.0, 500.0), random_state=0)
    row = report.set_index('treatment').loc['sep_arm']
    assert not row['feasible']
    assert not row['target_met']
    assert row['n_fits'] == 3          # baseline, dropped endpoint, chosen factor
    # default on_infeasible='best' gives the arm the closest attainable support
    assert row['penalty_factor'] == 500.0
    assert factors['sep_arm']['driver'] == 500.0


def test_tune_penalty_factor_infeasible_none_leaves_arm_unpenalized():
    A, X_A, names, cov, _ = _separating_design()
    factors, report = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.9}, target={'auc_lt': 0.0},
        on_infeasible='none', random_state=0)
    row = report.set_index('treatment').loc['sep_arm']
    assert not row['feasible']
    assert row['penalty_factor'] == 1.0
    assert factors == {}
    with pytest.raises(ValueError, match="on_infeasible"):
        tune_penalty_factor(A, X_A, 'driver', treatment_names=names,
                            covariate_names=cov, trigger={'auc_gt': 0.9},
                            target={'auc_lt': 0.0}, on_infeasible='nope')


def test_tune_penalty_factor_returns_no_penalty_when_target_already_met():
    A, X_A, names, cov, _ = _separating_design()
    factors, report = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.5},        # triggers the benign arm too
        target={'auc_lt': 1.01},        # already satisfied everywhere
        random_state=0)
    assert factors == {}
    assert (report['penalty_factor'] == 1.0).all()


def test_tune_penalty_factor_rejects_unknown_metric_and_covariate():
    A, X_A, names, cov, _ = _separating_design()
    with pytest.raises(ValueError, match='names no metric'):
        tune_penalty_factor(A, X_A, 'driver', treatment_names=names,
                            covariate_names=cov, target={'ess_treated_lt': 0.5})
    with pytest.raises(ValueError, match='not in covariate_names'):
        tune_penalty_factor(A, X_A, 'nope', treatment_names=names,
                            covariate_names=cov)


def test_tune_penalty_factor_tolerance_controls_fit_count():
    A, X_A, names, cov, _ = _separating_design()
    _, coarse = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.9}, target={'auc_lt': 0.9}, tol=1.5, random_state=0)
    _, fine = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.9}, target={'auc_lt': 0.9}, tol=0.05, random_state=0)
    assert coarse['n_fits'].sum() < fine['n_fits'].sum()
    # a finer search cannot need a larger factor than a coarser one
    assert fine['penalty_factor'].iloc[0] <= coarse['penalty_factor'].iloc[0] * 1.5


def test_tune_penalty_factor_scores_the_factor_it_reports():
    """When no finite factor in the bracket meets the target, the chosen
    metrics come from that factor's fit, not the dropped-covariate fit."""
    A, X_A, names, cov, _ = _separating_design()
    _, probe = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.9}, target={'auc_lt': 1.01}, random_state=0)
    auc_dropped = probe.set_index('treatment').loc['sep_arm', 'auc_dropped']
    factors, report = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.9}, target={'auc_lt': auc_dropped + 1e-9},
        bracket=(1.0, 2.0), random_state=0)
    row = report.set_index('treatment').loc['sep_arm']
    assert row['feasible']
    assert not row['target_met']
    assert row['penalty_factor'] == 2.0
    assert row['auc_chosen'] > auc_dropped


def test_tune_penalty_factor_scores_only_masked_cells():
    A, X_A, names, cov, ctrl = _separating_design()
    mask = np.ones(A.shape, dtype=bool)
    mask[np.flatnonzero(ctrl)[:100], :] = False
    _, report = tune_penalty_factor(
        A, X_A, 'driver', treatment_names=names, covariate_names=cov,
        trigger={'auc_gt': 0.0}, mask=mask, random_state=0)
    pi = estimate_propensity_scores(A, X_A, mask=mask, clip=None, random_state=0)
    j = names.index('benign_arm')
    keep = (ctrl | (A[:, j] == 1)) & mask[:, j]
    from sklearn.metrics import roc_auc_score
    expected = roc_auc_score(A[keep, j], pi[keep, j])
    got = report.set_index('treatment').loc['benign_arm', 'auc_unpenalized']
    assert got == pytest.approx(expected)

