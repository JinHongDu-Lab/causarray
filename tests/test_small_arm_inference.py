"""
Stage 0 tests for the 0.1.0 inference fix.

They pin the behaviour that the SCARF investigation showed was wrong before
0.1.0:

(a) the reported standard error matches the estimator's true sampling SD
    under an oracle outcome model, for rare and common treatments;
(b) a sparse gene whose small perturbed arm reads all-zero by chance is not
    called significant;
(c) a genuine complete knockout of an expressed gene is still called;
(d) type-I error is controlled under an NB null with 100 vs. 5,000 cells and
    genes spanning 0.02-5 counts per cell;
(e) balanced designs give the same result under calibrated and 'balanced'
    propensity weighting;
(f) the prevalence-aware clip does not clip calibrated scores of a rare
    treatment wholesale, and the support columns report raw counts.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import causarray.gcate_glm as gcate_glm
from causarray.DR_learner import LFC, _resolve_ps_clip
from causarray.DR_estimation import estimate_propensity_scores


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nb_counts(rng, mu, r):
    """NB counts with mean ``mu`` (array) and size ``r``."""
    lam = mu * rng.gamma(r, 1.0 / r, size=mu.shape)
    return rng.poisson(lam).astype(float)


def _oracle_lfc(rng, n0, n1, mu0, p, pi_val, r=3.0):
    """One draw of LFC under an oracle outcome model and fixed propensity."""
    n = n0 + n1
    A = np.r_[np.zeros(n0), np.ones(n1)]
    mu = np.full((n, p), mu0)
    Y = _nb_counts(rng, mu, r)
    Y_hat = np.empty((n, p, 1, 2))
    Y_hat[..., 0] = mu0
    Y_hat[..., 1] = mu0
    pi_hat = np.full((n, 1), pi_val)
    W = np.ones((n, 1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df, _ = LFC(Y, W, A, Y_hat=Y_hat, pi_hat=pi_hat, family='poisson',
                    thres_min=0, thres_diff=0, ps_clip=None)
    return df


# ---------------------------------------------------------------------------
# (a) reported SE matches the true sampling SD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n1,mu0', [(25, 0.5), (250, 0.5), (2500, 2.0)])
def test_pooled_se_matches_sampling_sd_with_calibrated_pi(n1, mu0):
    """Default (pooled IF variance, calibrated pi): median reported SE within
    15% of the empirical SD of tau across repeats, for prevalence 0.5%, 5%, 50%."""
    rng = np.random.default_rng(0)
    n0, p, reps = 5000 - n1 if n1 < 2500 else 2500, 20, 120
    pi_val = n1 / (n0 + n1)
    taus, ses = [], []
    for _ in range(reps):
        df = _oracle_lfc(rng, n0, n1, mu0, p, pi_val)
        taus.append(df['tau'].to_numpy())
        ses.append(df['std'].to_numpy())
    emp_sd = np.std(np.array(taus), axis=0, ddof=1)
    rep_se = np.median(np.array(ses), axis=0)
    ratio = np.median(rep_se / emp_sd)
    assert 0.85 <= ratio <= 1.15, f'reported SE / empirical SD = {ratio:.2f}'


# ---------------------------------------------------------------------------
# (b) chance all-zero arm is not called, (c) real knockout is called
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def zero_arm_data():
    """5,000 controls, 100 treated; 60 sparse null genes forced all-zero in the
    treated arm, 30 expressed null genes, and 10 complete knockouts of
    expressed genes."""
    rng = np.random.default_rng(2)
    n0, n1 = 5000, 100
    n = n0 + n1
    A = np.r_[np.zeros(n0), np.ones(n1)]
    p_sparse, p_expr, p_ko = 60, 30, 10
    mu = np.empty((n, p_sparse + p_expr + p_ko))
    mu[:, :p_sparse] = 0.03                       # ~3% detection
    mu[:, p_sparse:p_sparse + p_expr] = rng.uniform(0.5, 3, p_expr)
    mu[:, p_sparse + p_expr:] = 2.0
    Y = _nb_counts(rng, mu, 4.0)
    Y[n0:, :p_sparse] = 0.0                       # chance all-zero arms
    Y[n0:, p_sparse + p_expr:] = 0.0              # genuine complete knockouts
    W = np.c_[np.ones(n), rng.standard_normal(n)]
    return Y, W, A, p_sparse, p_expr, p_ko


def test_chance_zero_arm_not_called_and_flagged(zero_arm_data):
    Y, W, A, p_sparse, p_expr, p_ko = zero_arm_data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df, est = LFC(Y, W, A, family='nb', offset=False, backend='fast')
    sparse = df.iloc[:p_sparse]
    assert (sparse['count_treated'] == 0).all()
    assert (sparse['n_treated'] == 100).all() and (sparse['n_control'] == 5000).all()
    tested = sparse[np.isfinite(sparse['std'])]
    # padj may be NaN for genes removed by thres_min='auto' (control mean 0.03
    # < 5/100 = 0.05); whatever survives must not be significant and floored.
    assert not (tested['padj'] < 0.05).any(), 'chance all-zero arm was called significant'
    assert tested['var_floored'].all()
    assert (tested['std'] >= 0.9).all(), 'SE of an all-zero 100-cell arm must be >= ~1'


def test_complete_knockout_is_still_called(zero_arm_data):
    Y, W, A, p_sparse, p_expr, p_ko = zero_arm_data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df, _ = LFC(Y, W, A, family='nb', offset=False, backend='fast')
    ko = df.iloc[p_sparse + p_expr:]
    assert (ko['count_treated'] == 0).all()
    assert (ko['tau'] < -3).all()
    assert (ko['padj'] < 0.05).all(), 'genuine complete knockout must remain significant'


def test_expressed_null_genes_not_called(zero_arm_data):
    Y, W, A, p_sparse, p_expr, p_ko = zero_arm_data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df, _ = LFC(Y, W, A, family='nb', offset=False, backend='fast')
    expr = df.iloc[p_sparse:p_sparse + p_expr]
    assert (expr['padj'] < 0.05).sum() <= 1


# ---------------------------------------------------------------------------
# (d) type-I error, 100 vs 5,000 cells, wide expression range
# ---------------------------------------------------------------------------

def test_type1_rare_treatment_nb_null():
    rng = np.random.default_rng(3)
    n0, n1, p = 5000, 100, 300
    n = n0 + n1
    A = np.r_[np.zeros(n0), np.ones(n1)]
    x = rng.standard_normal(n)
    base = np.exp(rng.uniform(np.log(0.02), np.log(5.0), p))
    mu = base[None, :] * np.exp(0.2 * x[:, None])
    Y = _nb_counts(rng, mu, 4.0)
    W = np.c_[np.ones(n), x]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df, _ = LFC(Y, W, A, family='nb', offset=False, backend='fast')
    tested = df[np.isfinite(df['std'])]
    assert (tested['padj'] < 0.05).mean() <= 0.02, 'false discoveries under the null'
    expressed = tested[tested['mean_control'] >= 0.2]
    frac = (expressed['stat'].abs() > 1.96).mean()
    assert 0.02 <= frac <= 0.10, f'nominal 5% two-sided rate off: {frac:.3f}'
    assert 0.8 <= expressed['stat'].std() <= 1.25


# ---------------------------------------------------------------------------
# (e) balanced arms: calibrated == 'balanced'
# ---------------------------------------------------------------------------

def test_balanced_design_is_invariant_to_class_weight():
    rng = np.random.default_rng(4)
    n0, n1, p = 100, 100, 60
    n = n0 + n1
    A = np.r_[np.zeros(n0), np.ones(n1)]
    x = rng.standard_normal(n)
    mu = np.exp(rng.uniform(np.log(0.5), np.log(5.0), p))[None, :] * np.exp(0.3 * x[:, None])
    mu[n1:, :10] *= 2.0
    Y = _nb_counts(rng, mu, 4.0)
    W = np.c_[np.ones(n), x]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df_cal, est_cal = LFC(Y, W, A, family='nb', offset=False, backend='fast')
        df_bal, est_bal = LFC(Y, W, A, family='nb', offset=False, backend='fast',
                              ps_class_weight='balanced')
    np.testing.assert_allclose(est_cal['pi_hat_raw'], est_bal['pi_hat_raw'], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(df_cal['tau'], df_bal['tau'], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(df_cal['std'], df_bal['std'], rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# (f) prevalence-aware clip and support columns
# ---------------------------------------------------------------------------

def test_auto_clip_is_prevalence_aware():
    A = np.zeros((10000, 2))
    A[:50, 0] = 1          # prevalence 0.5%
    A[50:5050, 1] = 1      # prevalence ~50%
    lower, upper = _resolve_ps_clip('auto', A)
    # prevalence is computed on the cells eligible for that treatment: its 50
    # cases plus the 4,950 shared controls -> 1%, so the bound is 0.1%.
    assert lower[0] == pytest.approx(0.001)
    assert lower[0] < 0.01
    assert lower[1] == pytest.approx(0.01)
    assert upper[1] == pytest.approx(0.99)
    lo, hi = _resolve_ps_clip((0.05, 0.95), A)
    assert np.all(lo == 0.05) and np.all(hi == 0.95)
    assert _resolve_ps_clip(None, A) is None
    with pytest.raises(ValueError):
        _resolve_ps_clip('bogus', A)


def test_rare_treatment_scores_are_not_clipped_wholesale():
    rng = np.random.default_rng(5)
    n0, n1 = 5000, 30
    n = n0 + n1
    A = np.r_[np.zeros(n0), np.ones(n1)]
    x = rng.standard_normal(n)
    X = np.c_[np.ones(n), x]
    pi_cal = estimate_propensity_scores(A, X)              # default is calibrated
    assert np.median(pi_cal) < 0.02
    pi_bal = estimate_propensity_scores(A, X, class_weight='balanced')
    assert 0.3 < np.median(pi_bal) < 0.7
    Y = _nb_counts(rng, np.full((n, 50), 1.0), 4.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, est = LFC(Y, X, A, family='poisson', offset=False, backend='fast')
    lower, upper = est['ps_clip_bounds']
    assert lower[0] < 0.001
    # with the old fixed bound every score would sit at 0.01
    assert (est['pi_hat'] < 0.01).mean() > 0.5


def test_fast_backend_warns_when_crispyx_missing(monkeypatch):
    monkeypatch.setattr(gcate_glm, '_CRISPYX_AVAILABLE', False)
    with pytest.warns(RuntimeWarning, match='crispyx is not importable'):
        with gcate_glm._backend_override('fast'):
            pass


def test_eps_var_is_deprecated():
    rng = np.random.default_rng(6)
    n = 200
    A = np.r_[np.zeros(100), np.ones(100)]
    Y = rng.poisson(2.0, (n, 5)).astype(float)
    with pytest.warns(FutureWarning, match='eps_var'):
        LFC(Y, np.ones((n, 1)), A, family='poisson', offset=False, eps_var=1e-4)


def test_small_sample_t_reference_and_hc1_scaling():
    """In-sample fits use a t reference with n - d degrees of freedom and the
    n/(n-d) variance rescaling; cross-fitted (K=2) fits are not rescaled."""
    from scipy import stats
    rng = np.random.default_rng(7)
    n0 = n1 = 30
    n = n0 + n1
    A = np.r_[np.zeros(n0), np.ones(n1)]
    x = rng.standard_normal(n)
    W = np.c_[np.ones(n), x]                     # d = 2 + 1 treatment = 3
    Y = _nb_counts(rng, np.exp(1.0 + 0.2 * x)[:, None] * np.ones((1, 40)), 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df, _ = LFC(Y, W, A, family='nb', offset=False, backend='fast')
    ok = np.isfinite(df['stat'])
    expected_p = 2 * stats.t.sf(np.abs(df.loc[ok, 'stat']), df=n - 3)
    np.testing.assert_allclose(df.loc[ok, 'pvalue'], expected_p, rtol=1e-8)
    assert not np.allclose(df.loc[ok, 'pvalue'], 2 * stats.norm.sf(np.abs(df.loc[ok, 'stat'])))


def test_auto_expression_threshold_scales_with_smaller_arm():
    """thres_min='auto' requires ~5 expected counts in the smaller arm: a gene
    at 0.02 counts/cell is untestable with 100 treated cells but testable with
    700."""
    rng = np.random.default_rng(8)
    for n1, expect_tested in [(100, False), (700, True)]:
        n0 = 3000
        n = n0 + n1
        A = np.r_[np.zeros(n0), np.ones(n1)]
        Y = _nb_counts(rng, np.full((n, 60), 0.02), 4.0)
        Y[:, :10] = _nb_counts(rng, np.full((n, 10), 2.0), 4.0)     # anchor genes
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df, _ = LFC(Y, np.ones((n, 1)), A, family='poisson', offset=False, backend='fast',
                        thres_diff=0)          # isolate thres_min from the mean-difference filter
        sparse = df.iloc[10:]
        tested = np.isfinite(sparse['std']).mean()
        assert (tested > 0.5) == expect_tested, f'n1={n1}: tested fraction {tested:.2f}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df_fixed, _ = LFC(Y, np.ones((n, 1)), A, family='poisson', offset=False, backend='fast',
                          thres_min=0.05, thres_diff=0)
    assert np.isinf(df_fixed.iloc[10:]['std']).all()


def test_threshold_uses_observed_support_not_model_means():
    """A sparse gene with zero counts in a 100-cell arm is not tested even if
    the supplied outcome model predicts an inflated treated mean."""
    rng = np.random.default_rng(9)
    n0, n1 = 5000, 100
    n = n0 + n1
    A = np.r_[np.zeros(n0), np.ones(n1)]
    Y = np.zeros((n, 4))
    Y[:n0, 0] = rng.poisson(0.01, n0)            # sparse gene, all-zero treated arm
    Y[:, 1] = rng.poisson(2.0, n)                # expressed null gene
    Y[:, 2] = rng.poisson(1.0, n); Y[n0:, 2] = 0  # expressed gene, complete knockout (AIPW mean exactly 0)
    Y[:, 3] = rng.poisson(1.0, n); Y[n0:, 3] = rng.poisson(0.02, n1)   # strong partial knockdown
    Y_hat = np.empty((n, 4, 1, 2))
    Y_hat[:, 0, 0, 0] = 0.01; Y_hat[:, 0, 0, 1] = 0.2      # inflated treated prediction
    Y_hat[:, 1, 0, :] = 2.0
    Y_hat[:, 2, 0, 0] = 1.0; Y_hat[:, 2, 0, 1] = 1e-4
    Y_hat[:, 3, 0, 0] = 1.0; Y_hat[:, 3, 0, 1] = 0.02
    pi_hat = np.full((n, 1), n1 / n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df, _ = LFC(Y, np.ones((n, 1)), A, Y_hat=Y_hat, pi_hat=pi_hat, family='poisson', ps_clip=None)
    assert np.isinf(df.loc[0, 'std']) and np.isnan(df.loc[0, 'padj']), 'sparse zero-arm gene must be filtered'
    assert np.isfinite(df.loc[1, 'std'])
    assert df.loc[2, 'estimable'] and df.loc[2, 'var_floored'], 'complete knockout must stay estimable at the floor'
    assert df.loc[2, 'padj'] < 0.05 and df.loc[2, 'tau'] < -3
    assert df.loc[3, 'padj'] < 0.05 and df.loc[3, 'tau'] < -3
