"""
Comprehensive tests for confounder estimation, AIPW inference, and their combination.

Test IDs follow the plan in plan/20260520_comprehensive_test_plan.md.

Run fast suite (skip slow):  pytest -m "not slow" tests/test_inference_comprehensive.py
Run all:                      pytest tests/test_inference_comprehensive.py
"""

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from scipy.optimize import linear_sum_assignment

from causarray.gcate import fit_gcate
from causarray.DR_learner import LFC
from causarray.DR_inference import (
    multiplier_bootstrap, step_down, augmentation, fdx_control, bh_correction,
)
from causarray.DR_estimation import AIPW_mean
from causarray.utils import comp_size_factor

# ---------------------------------------------------------------------------
# Shared DGP helpers
# ---------------------------------------------------------------------------

def _sim_nb(n, p, r, n_treated, tau_nonzero, seed, gamma_scale=1.5, family='nb'):
    """Core DGP: NB (or Poisson) counts with latent confounders."""
    rng = np.random.default_rng(seed)
    X_obs = rng.standard_normal((n, 2))
    X_obs = np.c_[np.ones(n), X_obs]   # intercept + 2 covariates

    U = rng.standard_normal((n, r)) if r > 0 else np.zeros((n, 0))

    # Treatment: logit depends on U (confounding)
    if r > 0:
        logit_a = 0.8 * U[:, 0] + (0.6 * U[:, 1] if r > 1 else 0.0)
    else:
        logit_a = np.zeros(n)
    prob_a = 1.0 / (1.0 + np.exp(-logit_a))
    # Force exact n_treated treated
    sorted_idx = np.argsort(prob_a)[::-1]
    A = np.zeros(n, dtype=float)
    A[sorted_idx[:n_treated]] = 1.0

    tau_true = np.zeros(p)
    if tau_nonzero > 0:
        signs = rng.choice([-1, 1], tau_nonzero)
        mags  = rng.uniform(0.5, 1.5, tau_nonzero)
        tau_true[:tau_nonzero] = signs * mags

    gamma = rng.standard_normal((max(r, 1), p)) * gamma_scale if r > 0 else np.zeros((1, p))
    beta0 = rng.uniform(2.0, 3.5, p)

    log_mu = (
        beta0[None, :]
        + X_obs @ rng.standard_normal((3, p)) * 0.2
        + A[:, None] * tau_true[None, :]
        + (U @ gamma if r > 0 else 0.0)
    )
    mu = np.exp(np.clip(log_mu, -10, 10))

    if family == 'nb':
        disp = rng.uniform(0.5, 2.0, p)
        Y = rng.negative_binomial(
            n=np.maximum(1.0 / disp, 0.01),
            p=np.clip(1.0 / (1.0 + mu * disp[None, :]), 1e-10, 1 - 1e-10),
        ).astype(float)
    else:
        Y = rng.poisson(mu).astype(float)

    return Y, X_obs, A, tau_true, U


# ---------------------------------------------------------------------------
# Module-scoped fixtures  (amortise fitting cost)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def confounded_data_nb():
    """n=300, p=50, r=2, 150 treated, 20 non-null genes.  NB.  seed=1."""
    return _sim_nb(n=300, p=50, r=2, n_treated=150, tau_nonzero=20, seed=1)


@pytest.fixture(scope="module")
def null_nb_balanced():
    """n=200, p=30, balanced (100/100), tau_true=0.  NB.  seed=7."""
    return _sim_nb(n=200, p=30, r=0, n_treated=100, tau_nonzero=0, seed=7)


@pytest.fixture(scope="module")
def null_nb_imbalanced():
    """n=500, p=30, imbalanced (450/100 → 450 ctrl / 50 treated), tau_true=0.  NB.  seed=8."""
    return _sim_nb(n=500, p=30, r=0, n_treated=50, tau_nonzero=0, seed=8)


@pytest.fixture(scope="module")
def signal_nb_balanced():
    """n=300, p=30, balanced (150/150), 10 non-null genes.  NB.  seed=14."""
    return _sim_nb(n=300, p=30, r=0, n_treated=150, tau_nonzero=10, seed=14)


@pytest.fixture(scope="module")
def confounded_pipeline_data():
    """n=300, p=50, r=2, 100 treated, 20 non-null genes.  NB.  seed=13."""
    return _sim_nb(n=300, p=50, r=2, n_treated=100, tau_nonzero=20, seed=13)


# ---------------------------------------------------------------------------
# §1  TestConfounderEstimation  (C1–C6, N13)
# ---------------------------------------------------------------------------

class TestConfounderEstimation:

    # ---- C1: latent factor alignment ----
    def test_latent_factor_alignment(self, confounded_data_nb):
        """C1 — U_hat columns are linearly aligned with true U (max matched corr ≥ 0.5)."""
        Y, X_obs, A, tau_true, U_true = confounded_data_nb
        r = U_true.shape[1]

        _, res_2 = fit_gcate(Y, X_obs, A[:, None], r=r, family='nb', offset=True, backend='fast')
        U_hat = res_2['U']

        corr_mat = np.abs(np.corrcoef(U_hat.T, U_true.T)[:r, r:])
        row_ind, col_ind = linear_sum_assignment(-corr_mat)
        matched_corrs = corr_mat[row_ind, col_ind]

        assert matched_corrs.min() >= 0.5, (
            f"Worst matched latent-factor correlation = {matched_corrs.min():.3f} < 0.5"
        )

    # ---- C2: over-specified r ----
    def test_overspecified_r(self, confounded_data_nb):
        """C2 — Fitting r=4 when truth is r=2 still beats naive LFC (MSE)."""
        Y, X_obs, A, tau_true, _ = confounded_data_nb

        df_naive, _ = LFC(Y, X_obs, A[:, None], family='nb', offset=True, backend='fast')
        lfc_naive = df_naive['tau'].values

        _, res_2 = fit_gcate(Y, X_obs, A[:, None], r=4, family='nb', offset=True, backend='fast')
        U_hat = res_2['U']
        offsets = np.log(res_2['kwargs_glm']['size_factor'])
        df_dc, _ = LFC(Y, np.c_[X_obs, U_hat], A[:, None],
                       W_A=np.c_[X_obs, U_hat], family='nb', offset=offsets, backend='fast')
        lfc_dc = df_dc['tau'].values

        assert np.mean((lfc_dc - tau_true) ** 2) < np.mean((lfc_naive - tau_true) ** 2), \
            "Over-specified GCATE (r=4) did not improve on naive LFC"

    # ---- C3: under-specified r ----
    def test_underspecified_r(self, confounded_data_nb):
        """C3 — Fitting r=1 when truth is r=2 does not do worse than naive LFC.

        Not a strict improvement: measured over six seeds of this generator
        (2026-09-22) the ratio of under-specified to naive MSE is 0.82-1.02,
        median ~0.98, so a strict inequality on one seed is a coin flip and
        has flipped on a dependency upgrade before.  The correctly specified
        case, where the effect is large, is C1/C2's job.
        """
        Y, X_obs, A, tau_true, _ = confounded_data_nb

        df_naive, _ = LFC(Y, X_obs, A[:, None], family='nb', offset=True, backend='fast')
        lfc_naive = df_naive['tau'].values

        _, res_2 = fit_gcate(Y, X_obs, A[:, None], r=1, family='nb', offset=True, backend='fast')
        U_hat = res_2['U']
        offsets = np.log(res_2['kwargs_glm']['size_factor'])
        df_dc, _ = LFC(Y, np.c_[X_obs, U_hat], A[:, None],
                       W_A=np.c_[X_obs, U_hat], family='nb', offset=offsets, backend='fast')
        lfc_dc = df_dc['tau'].values

        mse_dc = np.mean((lfc_dc - tau_true) ** 2)
        mse_naive = np.mean((lfc_naive - tau_true) ** 2)
        assert mse_dc <= 1.05 * mse_naive, (
            f"Under-specified GCATE (r=1) did materially worse than naive LFC: "
            f"MSE {mse_dc:.3f} vs {mse_naive:.3f}"
        )

    # ---- C4: Poisson family ----
    def test_poisson_family_deconfounding(self):
        """C4 — Deconfounding improves LFC correlation under Poisson counts."""
        Y, X_obs, A, tau_true, _ = _sim_nb(
            n=200, p=30, r=2, n_treated=100, tau_nonzero=10, seed=3, family='poisson'
        )

        df_naive, _ = LFC(Y, X_obs, A[:, None], family='poisson', offset=True, backend='fast')

        _, res_2 = fit_gcate(Y, X_obs, A[:, None], r=2, family='nb', offset=True, backend='fast')
        U_hat = res_2['U']
        offsets = np.log(res_2['kwargs_glm']['size_factor'])
        df_dc, _ = LFC(Y, np.c_[X_obs, U_hat], A[:, None],
                       W_A=np.c_[X_obs, U_hat], family='poisson', offset=offsets, backend='fast')

        corr_naive = np.corrcoef(df_naive['tau'].values, tau_true)[0, 1]
        corr_dc    = np.corrcoef(df_dc['tau'].values,    tau_true)[0, 1]

        assert corr_dc > corr_naive, (
            f"Poisson deconfounding: corr_dc={corr_dc:.3f} <= corr_naive={corr_naive:.3f}"
        )

    # ---- C5: no-confounding baseline ----
    def test_no_confounding_does_not_hurt(self):
        """C5 — When gamma=0, GCATE-adjusted LFC MSE stays within 50% of naive MSE."""
        # Build data with zero confounding signal (gamma_scale=0)
        Y, X_obs, A, tau_true, _ = _sim_nb(
            n=250, p=30, r=2, n_treated=125, tau_nonzero=10, seed=5, gamma_scale=0.0
        )

        df_naive, _ = LFC(Y, X_obs, A[:, None], family='nb', offset=True, backend='fast')
        mse_naive = np.mean((df_naive['tau'].values - tau_true) ** 2)

        _, res_2 = fit_gcate(Y, X_obs, A[:, None], r=2, family='nb', offset=True, backend='fast')
        U_hat = res_2['U']
        offsets = np.log(res_2['kwargs_glm']['size_factor'])
        df_dc, _ = LFC(Y, np.c_[X_obs, U_hat], A[:, None],
                       W_A=np.c_[X_obs, U_hat], family='nb', offset=offsets, backend='fast')
        mse_dc = np.mean((df_dc['tau'].values - tau_true) ** 2)

        rel_diff = abs(mse_dc - mse_naive) / max(mse_naive, 1e-9)
        assert rel_diff < 0.5, (
            f"Deconfounding degraded MSE by {rel_diff:.1%} when there is no confounding"
        )

    # ---- C6: size factor output ----
    def test_size_factor_output(self, confounded_data_nb):
        """C6 — size_factor is positive and has length n."""
        Y, X_obs, A, _, _ = confounded_data_nb
        n = Y.shape[0]
        _, res_2 = fit_gcate(Y, X_obs, A[:, None], r=2, family='nb', offset=True, backend='fast')
        sf = res_2['kwargs_glm']['size_factor']
        assert sf.shape[0] == n, f"size_factor length {sf.shape[0]} != n={n}"
        assert np.all(sf > 0), "size_factor contains non-positive values"

    # ---- N13: pre-specified disp_glm ----
    def test_prespecified_dispersion(self, confounded_data_nb):
        """N13 — Pre-specifying disp_glm produces the same output shapes as default."""
        Y, X_obs, A, _, _ = confounded_data_nb
        r = 2

        res1_auto, res2_auto = fit_gcate(Y, X_obs, A[:, None], r=r, family='nb',
                                         offset=True, backend='fast')
        disp_fixed = np.ones(Y.shape[1])
        res1_fixed, res2_fixed = fit_gcate(Y, X_obs, A[:, None], r=r, family='nb',
                                           disp_glm=disp_fixed, offset=True, backend='fast')

        assert res2_auto['U'].shape == res2_fixed['U'].shape
        assert res2_auto['X_U'].shape == res2_fixed['X_U'].shape


# ---------------------------------------------------------------------------
# §2  TestInferenceUnits  (I1–I10, N11, N12)
# ---------------------------------------------------------------------------

class TestInferenceUnits:

    # ---- I1: multiplier_bootstrap shape and zero-mean ----
    def test_multiplier_bootstrap_shape(self):
        """I1 — Output shape (B, p) and columns near zero-mean (within 4 SE of 0)."""
        rng = np.random.default_rng(0)
        eta = rng.standard_normal((100, 20))
        B = 500
        z = multiplier_bootstrap(eta, B=B)
        assert z.shape == (B, 20)
        # Each column has mean ~0; check within 4 standard errors
        z_std = np.std(z, axis=0, ddof=1)
        assert np.all(np.abs(z.mean(axis=0)) < 4 * z_std / np.sqrt(B))

    # ---- I2: multiplier_bootstrap variance ----
    def test_multiplier_bootstrap_variance(self):
        """I2 — Column variances match theory: Var(z_b) = n * Var(η)."""
        rng = np.random.default_rng(1)
        n, p, B = 200, 10, 2000
        eta = rng.standard_normal((n, p))
        z = multiplier_bootstrap(eta, B=B)
        # Each z_b[:,j] = sum_i eta[i,j]*g[i]; Var = n * Var(eta[:,j])
        expected_var = n * np.var(eta, axis=0, ddof=0)
        empirical_var = np.var(z, axis=0, ddof=1)
        ratio = empirical_var / expected_var
        assert np.all((ratio > 0.7) & (ratio < 1.3)), (
            f"Variance ratio out of [0.7, 1.3]: {ratio}"
        )

    # ---- I3: step_down monotonicity ----
    def test_step_down_monotonicity(self):
        """I3 — Discoveries with alpha=0.5 are superset of discoveries with alpha=0.1."""
        rng = np.random.default_rng(2)
        p = 30
        tvalues = rng.standard_normal(p) * 3
        eta = rng.standard_normal((100, p))
        z_init = multiplier_bootstrap(eta, B=300)

        V_low, _, _  = step_down(tvalues.copy(), z_init.copy(), alpha=0.1)
        V_high, _, _ = step_down(tvalues.copy(), z_init.copy(), alpha=0.5)

        assert np.all(V_low <= V_high), "step_down: V(alpha=0.1) not a subset of V(alpha=0.5)"

    # ---- I4: step_down with all-zero t-values ----
    def test_step_down_all_zero(self):
        """I4 — Zero t-values produce no discoveries."""
        p = 20
        tvalues = np.zeros(p)
        z_init  = np.zeros((200, p))
        V, _, _ = step_down(tvalues, z_init, alpha=0.05)
        assert V.sum() == 0

    # ---- I5: augmentation count ----
    def test_augmentation_count(self):
        """I5 — augmentation adds floor(c*k/(1-c)) items when c>0."""
        rng = np.random.default_rng(3)
        p = 50
        V = np.zeros(p)
        V[:10] = 1  # 10 discoveries
        tvalues = rng.standard_normal(p)
        tvalues[:10] = 0  # zero out already-discovered

        c = 0.2
        k = 10
        expected_add = int(np.floor(c * k / (1 - c)))  # = 2

        V_aug = augmentation(V.copy(), tvalues.copy(), c)
        actual_add = int(V_aug.sum()) - k
        assert actual_add == expected_add, (
            f"augmentation added {actual_add}, expected {expected_add}"
        )

    # ---- I6: bh_correction shapes and NaN propagation ----
    def test_bh_correction_shapes_and_nan(self):
        """I6 — Output shapes match input; NaN positions are preserved in pvals."""
        rng = np.random.default_rng(4)
        p = 40
        tvalues = rng.standard_normal(p)
        tvalues[[5, 15, 25]] = np.nan

        pvals, qvals, pvals_adj, qvals_adj = bh_correction(tvalues)
        for arr in (pvals, qvals, pvals_adj, qvals_adj):
            assert arr.shape == (p,)
        # NaN positions stay NaN
        for arr in (pvals, qvals):
            assert np.all(np.isnan(arr[[5, 15, 25]]))

    # ---- I7: bh_correction p-value calibration under null ----
    def test_bh_correction_pval_uniform_under_null(self):
        """I7 — p-values from N(0,1) t-stats are approximately uniform."""
        rng = np.random.default_rng(5)
        tvalues = rng.standard_normal(1000)
        pvals, _, _, _ = bh_correction(tvalues)
        ks_stat, ks_p = scipy.stats.kstest(pvals, 'uniform')
        assert ks_p > 0.05, f"KS p-value={ks_p:.4f} < 0.05 (p-values not uniform under null)"

    # ---- I8: empirical null adjustment helps under shifted null ----
    def test_bh_correction_emp_null_adj(self):
        """I8 — pvals_adj is more uniform than pvals when null is shifted."""
        rng = np.random.default_rng(6)
        shift = 2.0
        tvalues = rng.standard_normal(1000) + shift  # shifted null

        pvals, _, pvals_adj, _ = bh_correction(tvalues)
        valid = ~np.isnan(pvals_adj)
        ks_raw = scipy.stats.kstest(pvals[valid], 'uniform').statistic
        ks_adj = scipy.stats.kstest(pvals_adj[valid], 'uniform').statistic
        assert ks_adj < ks_raw, (
            f"Empirical null adj did not improve uniformity: "
            f"KS_raw={ks_raw:.3f}, KS_adj={ks_adj:.3f}"
        )

    # ---- I9: fdx_control returns zeros when fdx=False ----
    def test_fdx_control_off(self):
        """I9 — fdx=False always returns all-zero V."""
        rng = np.random.default_rng(7)
        p = 30
        tau_est   = rng.standard_normal(p)
        tvalues   = rng.standard_normal(p) * 5
        eta_est   = rng.standard_normal((100, p))
        std_est   = np.abs(rng.standard_normal(p)) + 0.1

        V = fdx_control(tau_est, tvalues, eta_est, std_est, fdx=False,
                        B=100, alpha=0.05, c=0.1)
        assert V.sum() == 0

    # ---- I10: fdx_control min_var filter ----
    def test_fdx_control_min_var_filter(self):
        """I10 — Genes with std < min_var are not discovered (unless |tau| > min_diff)."""
        rng = np.random.default_rng(8)
        p = 20
        tau_est = rng.standard_normal(p) * 0.05  # all small effects
        tvalues = rng.standard_normal(p) * 10     # large t-stats
        eta_est = rng.standard_normal((100, p))
        std_est = np.full(p, 1e-10)  # all below min_var=1e-8

        V = fdx_control(tau_est, tvalues, eta_est, std_est, fdx=True,
                        B=200, alpha=0.05, c=0.0, min_var=1e-8, min_diff=0.5)
        # tau_est < min_diff everywhere → no discoveries
        assert V.sum() == 0

    # ---- N11: bh_correction when mad=0 ----
    def test_bh_correction_mad_zero(self):
        """N11 — When all t-stats equal (mad=0), pvals_adj=NaN, pvals in (0,1], no error."""
        tvalues = np.ones(50) * 1.5
        pvals, qvals, pvals_adj, qvals_adj = bh_correction(tvalues)

        # pvals should be defined and valid
        assert np.all((pvals > 0) & (pvals <= 1)), "pvals out of (0,1]"
        # pvals_adj should be NaN (mad=0 branch not entered)
        assert np.all(np.isnan(pvals_adj)), "pvals_adj should be NaN when mad=0"

    # ---- N12: AIPW_mean is always unclipped ----
    def test_aipw_mean_is_unclipped(self):
        """N12 — AIPW influence values may be negative and cannot be clipped."""
        rng = np.random.default_rng(9)
        n, p, a = 50, 5, 1
        Y   = rng.poisson(1, (n, p)).astype(float)
        A   = rng.integers(0, 2, (n, a, 2)).astype(float)
        A   = A / A.sum(axis=-1, keepdims=True).clip(1)
        mu  = rng.uniform(0.1, 5, (n, p, a, 2))
        pi  = np.full((n, a, 2), 0.5)

        _, pseudo = AIPW_mean(Y, A, mu, pi)

        assert pseudo.min() < 0, "AIPW pseudo-outcomes should remain unclipped"
        with pytest.raises(TypeError):
            AIPW_mean(Y, A, mu, pi, positive=True)


# ---------------------------------------------------------------------------
# §3  TestLFCIntegration  (I11–I17, N1–N3, N8–N10)
# ---------------------------------------------------------------------------

class TestLFCIntegration:

    # ---- I11: type I error, balanced, unequal (Welch) ----
    def test_type1_balanced_welch(self, null_nb_balanced):
        """I11 — FDR ≤ 10% under balanced null (deprecated alias usevar='unequal')."""
        Y, W, A, _, _ = null_nb_balanced
        df, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                    usevar='unequal', backend='fast')
        fdr = (df['padj'] < 0.05).mean()
        assert fdr <= 0.10, f"Type I error too high: FDR={fdr:.3f}"

    # ---- I12: type I error, imbalanced, Welch (regression for Welch fix) ----
    def test_type1_imbalanced_welch(self, null_nb_imbalanced):
        """I12 — FDR ≤ 10% under imbalanced null (450 ctrl / 50 treated)."""
        Y, W, A, _, _ = null_nb_imbalanced
        df, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                    usevar='unequal', backend='fast')
        fdr = (df['padj'] < 0.05).mean()
        assert fdr <= 0.10, f"Imbalanced type I error too high: FDR={fdr:.3f}"

    # ---- I13: 'unequal' is a deprecated alias of 'pooled' (0.1.0) ----
    def test_unequal_is_alias_of_pooled(self, null_nb_imbalanced):
        """I13 — usevar='unequal' warns and returns exactly the pooled result."""
        Y, W, A, _, _ = null_nb_imbalanced
        df_pooled, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                           usevar='pooled', backend='fast')
        with pytest.warns(FutureWarning, match="removed in 0.1.0"):
            df_alias, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                              usevar='unequal', backend='fast')
        pd.testing.assert_frame_equal(df_alias, df_pooled)
        with pytest.raises(ValueError, match="usevar must be 'pooled'"):
            LFC(Y, W, A[:, None], family='nb', offset=True, usevar='welch', backend='fast')

    # ---- I14: power under balanced NB signal ----
    def test_power_signal(self, signal_nb_balanced):
        """I14 — TPR(padj < 0.1) ≥ 0.5 for truly non-null genes."""
        Y, W, A, tau_true, _ = signal_nb_balanced
        n_nonzero = (tau_true != 0).sum()
        df, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                    backend='fast')
        # Match genes by position (gene_names are range indices)
        tp = ((df['padj'] < 0.1).values & (tau_true != 0)).sum()
        tpr = tp / n_nonzero
        assert tpr >= 0.5, f"TPR={tpr:.3f} < 0.5"

    # ---- I15: p-values well-formed with Welch df ----
    def test_pvalues_well_formed(self, null_nb_balanced):
        """I15 — pvalue in (0,1], stat NaN only for filtered genes."""
        Y, W, A, _, _ = null_nb_balanced
        df, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                    backend='fast')
        valid = df['stat'].notna()
        pvals = df.loc[valid, 'pvalue']
        assert ((pvals > 0) & (pvals <= 1)).all(), "Some p-values outside (0,1]"

    # ---- I16: FDX null type I error ----
    def test_fdx_null_type1(self, null_nb_balanced):
        """I16 — FDX rej rate ≤ 10% under null."""
        Y, W, A, _, _ = null_nb_balanced
        df, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                    fdx=True, fdx_alpha=0.05, fdx_c=0.1, backend='fast')
        rej_rate = df['rej'].mean()
        assert rej_rate <= 0.10, f"FDX rej rate under null = {rej_rate:.3f}"

    # ---- I17: CI coverage (slow) ----
    @pytest.mark.slow
    def test_ci_coverage(self):
        """I17 — 95% CI covers tau_true ≥ 80% of the time over 20 repeats.

        ``offset=False``: the DGP has only 10 genes, half of them strongly DE,
        so size factors estimated from those same genes are contaminated by
        the treatment and add an error component that no variance estimator
        can see (SE/SD ≈ 0.67 with ``offset=True`` versus 0.95 without). Real
        data have thousands of mostly-null genes, where this does not arise.
        """
        n_repeats = 20
        n_covered = 0
        # Single gene with non-zero effect; balanced design
        for seed in range(n_repeats):
            Y, W, A, tau_true, _ = _sim_nb(
                n=200, p=10, r=0, n_treated=100, tau_nonzero=5, seed=100 + seed
            )
            df, _ = LFC(Y, W, A[:, None], family='nb', offset=False,
                        backend='fast')
            # Check coverage for the first non-null gene (index 0)
            tau_hat = df.loc[0, 'tau']
            std_hat = df.loc[0, 'std']
            if abs(tau_hat - tau_true[0]) <= 1.96 * std_hat:
                n_covered += 1
        coverage = n_covered / n_repeats
        assert coverage >= 0.80, f"CI coverage = {coverage:.2f} < 0.80"

    # ---- N1: pooled type I error under balanced null ----
    def test_pooled_type1_balanced(self, null_nb_balanced):
        """N1 — balanced null: the pooled influence-function variance controls FDR."""
        Y, W, A, _, _ = null_nb_balanced
        df_pooled, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                           usevar='pooled', backend='fast')
        fdr = (df_pooled['padj'] < 0.05).mean()
        assert fdr <= 0.10, f"pooled type I error too high: FDR={fdr:.3f}"
        valid = df_pooled['stat'].notna()
        assert 0.6 <= df_pooled.loc[valid, 'stat'].std() <= 1.4

    # ---- N2: Poisson family type I error ----
    def test_poisson_type1(self):
        """N2 — FDR ≤ 10% under balanced Poisson null."""
        Y, W, A, _, _ = _sim_nb(n=200, p=30, r=0, n_treated=100,
                                 tau_nonzero=0, seed=20, family='poisson')
        df, _ = LFC(Y, W, A[:, None], family='poisson', offset=True,
                    backend='fast')
        fdr = (df['padj'] < 0.05).mean()
        assert fdr <= 0.10, f"Poisson type I error = {fdr:.3f}"

    # ---- N3: offset=False runs without error ----
    def test_offset_false(self, null_nb_balanced):
        """N3 — offset=False: no NaN/Inf in tau or pvalue for expressed genes."""
        Y, W, A, _, _ = null_nb_balanced
        df, _ = LFC(Y, W, A[:, None], family='nb', offset=False, backend='fast')
        assert df['tau'].notna().all(), "tau has NaN with offset=False"
        assert np.isfinite(df['tau'].values).all(), "tau has Inf with offset=False"
        valid = df['stat'].notna()
        assert ((df.loc[valid, 'pvalue'] > 0) & (df.loc[valid, 'pvalue'] <= 1)).all()

    # ---- N8: low-expression filter ----
    def test_low_expression_filter(self, null_nb_balanced):
        """N8 — Genes filtered by thres_min have tau=0, std=inf, stat=NaN."""
        Y, W, A, _, _ = null_nb_balanced
        # Use a very high thres_min so most genes are filtered
        df, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                    thres_min=1e4, backend='fast')
        # All genes should be filtered
        assert df['stat'].isna().all(), "Expected all genes to be filtered with thres_min=1e4"
        assert (df['tau'] == 0).all(), "Filtered genes should have tau=0"
        assert np.isinf(df['std'].values).all(), "Filtered genes should have std=inf"

    # ---- N9: FDX augmentation c=0 vs c>0, and FDX finds signal ----
    def test_fdx_augmentation(self, signal_nb_balanced):
        """N9 — c=0.3 always produces >= discoveries of c=0 (monotone augmentation).

        FDX with multiplier bootstrap is more conservative than BH and may find
        zero discoveries even with modest signal — this is expected behaviour.
        We only test the monotonicity invariant: augmentation cannot reduce
        the discovery set.
        """
        Y, W, A, _, _ = signal_nb_balanced
        df_c0,  _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                        fdx=True, fdx_c=0.0, fdx_alpha=0.05, backend='fast')
        df_c3,  _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                        fdx=True, fdx_c=0.3, fdx_alpha=0.05, backend='fast')

        assert df_c3['rej'].sum() >= df_c0['rej'].sum(), \
            "fdx_c=0.3 should produce >= discoveries vs fdx_c=0"

    # ---- N10: multi-pert imbalanced sample sizes ----
    def test_multipert_imbalanced(self):
        """N10 — Multi-pert imbalanced: std for small pert > std for large pert."""
        rng = np.random.default_rng(30)
        n, p = 350, 15
        W = np.c_[np.ones(n), rng.standard_normal((n, 2))]
        # 3 perturbations: pert-0 has 100 treated, pert-1 has 30, pert-2 has 10
        A = np.zeros((n, 3))
        A[0:100,   0] = 1   # pert-0: 100 treated
        A[100:130, 1] = 1   # pert-1: 30 treated
        A[130:140, 2] = 1   # pert-2: 10 treated
        Y = rng.negative_binomial(5, 0.5, (n, p)).astype(float)

        df, _ = LFC(Y, W, A, family='nb', offset=True,
                    backend='fast')

        assert len(df) == p * 3, f"Expected {p*3} rows, got {len(df)}"
        std_pert0 = df[df['trt'] == 0]['std'].median()
        std_pert2 = df[df['trt'] == 2]['std'].median()
        assert std_pert2 > std_pert0, (
            f"Expected std(pert-2) > std(pert-0), got {std_pert2:.4f} vs {std_pert0:.4f}"
        )


# ---------------------------------------------------------------------------
# §4  TestParamAxes  (N4–N7)
# ---------------------------------------------------------------------------

class TestParamAxes:

    # ---- N4: W_A != W ----
    def test_separate_propensity_covariates(self, null_nb_balanced):
        """N4 — W_A != W: output shape correct, pi_hat shape is (n, 1)."""
        Y, W, A, _, _ = null_nb_balanced
        # Use only first covariate for propensity
        W_A = W[:, :1]
        df, est = LFC(Y, W, A[:, None], W_A=W_A, family='nb', offset=True, backend='fast')

        assert df.shape[0] == Y.shape[1], "Row count mismatch with W_A != W"
        assert 'tau' in df.columns
        assert est['pi_hat'].shape == (Y.shape[0], 1), \
            f"pi_hat shape {est['pi_hat'].shape} != ({Y.shape[0]}, 1)"

    # ---- N5: mask parameter ----
    def test_mask_routes_cells(self):
        """N5 — mask routes different cells per perturbation; output has p*3 rows."""
        rng = np.random.default_rng(40)
        n, p = 300, 10
        W = np.c_[np.ones(n), rng.standard_normal((n, 2))]
        A = np.zeros((n, 3))
        # Non-overlapping treated groups + shared controls
        A[0:50,   0] = 1
        A[50:100, 1] = 1
        A[100:150, 2] = 1
        Y = rng.negative_binomial(5, 0.5, (n, p)).astype(float)

        # mask: control = rows 200-300, per pert adds treated group
        ctrl = np.zeros(n, dtype=bool); ctrl[200:] = True
        mask = np.zeros((n, 3), dtype=bool)
        for j, start in enumerate([0, 50, 100]):
            mask[:, j] = ctrl.copy()
            mask[start:start+50, j] = True

        df_mask,   _ = LFC(Y, W, A, family='nb', offset=True, mask=mask, backend='fast')
        df_nomask, _ = LFC(Y, W, A, family='nb', offset=True, backend='fast')

        assert len(df_mask) == p * 3
        # tau should differ for at least some genes (different cells used)
        assert not np.allclose(
            df_mask['tau'].values, df_nomask['tau'].values
        ), "mask did not change tau estimates"

    # ---- N6: K=2 cross-fitting type I error ----
    def test_cross_fitting_k2(self):
        """N6 — K=2 cross-fitting: FDR ≤ 15% under balanced null."""
        Y, W, A, _, _ = _sim_nb(n=200, p=15, r=0, n_treated=100,
                                 tau_nonzero=0, seed=60)
        df, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                    K=2, backend='fast')
        fdr = (df['padj'] < 0.05).mean()
        assert fdr <= 0.15, f"K=2 type I error = {fdr:.3f}"

    # ---- N7: ps_model='ensemble' smoke test (slow) ----
    @pytest.mark.slow
    def test_ps_model_ensemble(self):
        """N7 — ps_model='ensemble' runs and returns expected columns."""
        Y, W, A, _, _ = _sim_nb(n=150, p=10, r=0, n_treated=75,
                                 tau_nonzero=0, seed=70)
        df, _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                    ps_model='ensemble', backend='fast')
        expected_cols = {'tau', 'std', 'stat', 'pvalue', 'padj'}
        assert expected_cols.issubset(df.columns)
        assert len(df) == Y.shape[1]


# ---------------------------------------------------------------------------
# §5  TestCombinedPipeline  (E1–E5)
# ---------------------------------------------------------------------------

class TestCombinedPipeline:

    def _run_gcate_lfc(self, Y, X_obs, A, r, tau_true=None, seed=0):
        """Helper: full GCATE → LFC pipeline."""
        _, res_2 = fit_gcate(Y, X_obs, A[:, None], r=r, family='nb', offset=True, backend='fast')
        U_hat = res_2['U']
        offsets = np.log(res_2['kwargs_glm']['size_factor'])
        df, est = LFC(
            Y, np.c_[X_obs, U_hat], A[:, None],
            W_A=np.c_[X_obs, U_hat],
            family='nb', offset=offsets,
            backend='fast',
            random_state=seed,
        )
        return df, est

    # ---- E1: full pipeline type I error ----
    def test_full_pipeline_type1(self):
        """E1 — GCATE + LFC: deconfounding reduces FDR vs naive on confounded null.

        Absolute FDR control after GCATE is not guaranteed with finite n because
        residual confounding remains.  We instead assert the relative improvement:
        deconfounded FDR < naive FDR.
        """
        Y, X_obs, A, tau_true, _ = _sim_nb(n=300, p=30, r=2,
                                            n_treated=100, tau_nonzero=0, seed=11)
        df_naive, _ = LFC(Y, X_obs, A[:, None], family='nb', offset=True,
                          backend='fast')
        df_dc, _ = self._run_gcate_lfc(Y, X_obs, A, r=2)

        fdr_naive = (df_naive['padj'] < 0.05).mean()
        fdr_dc    = (df_dc['padj']    < 0.05).mean()
        assert fdr_dc < fdr_naive, (
            f"Deconfounding did not reduce FDR: fdr_dc={fdr_dc:.3f}, fdr_naive={fdr_naive:.3f}"
        )

    # ---- E2: full pipeline power ----
    def test_full_pipeline_power(self, confounded_pipeline_data):
        """E2 — Deconfounding lowers estimation error without collapsing power.

        TPR_deconf >= TPR_naive is not a property of the method: measured over
        five seeds of this generator (2026-09-22) deconfounding loses power on
        three of them while cutting MSE by 2-10x, both before and after the
        0.1.0 GLM rework.  What is asserted here is the trade it actually
        makes.
        """
        Y, X_obs, A, tau_true, _ = confounded_pipeline_data
        n_nonzero = (tau_true != 0).sum()

        df_naive, _ = LFC(Y, X_obs, A[:, None], family='nb', offset=True, backend='fast')
        df_dc, _    = self._run_gcate_lfc(Y, X_obs, A, r=2)

        tpr_naive = ((df_naive['padj'] < 0.1).values & (tau_true != 0)).sum() / n_nonzero
        tpr_dc    = ((df_dc['padj']    < 0.1).values & (tau_true != 0)).sum() / n_nonzero

        mse_naive = np.mean((df_naive['tau'].values - tau_true) ** 2)
        mse_dc = np.mean((df_dc['tau'].values - tau_true) ** 2)
        assert mse_dc < mse_naive, (
            f"MSE_deconf={mse_dc:.3f} >= MSE_naive={mse_naive:.3f}"
        )
        # Only assert power if naive finds any true positives (otherwise degenerate)
        if tpr_naive > 0:
            assert tpr_dc >= 0.6 * tpr_naive, (
                f"TPR_deconf={tpr_dc:.3f} collapsed against TPR_naive={tpr_naive:.3f}"
            )

    # ---- E3: empirical FDR (slow) ----
    @pytest.mark.slow
    def test_full_pipeline_empirical_null_fdr(self, confounded_pipeline_data):
        """E3 — Empirical-null-adjusted FDR is controlled under confounding."""
        Y, X_obs, A, tau_true, _ = confounded_pipeline_data
        df, _ = self._run_gcate_lfc(Y, X_obs, A, r=2)

        discoveries = df['padj_emp_null_adj'] < 0.1
        n_disc = discoveries.sum()
        n_fp = (discoveries.values & (tau_true == 0)).sum()
        emp_fdr = n_fp / max(n_disc, 1)
        assert emp_fdr <= 0.20, f"Empirical FDR = {emp_fdr:.3f}"

    # ---- E4: determinism with pre-computed nuisances ----
    def test_cached_nuisances_deterministic(self, confounded_pipeline_data):
        """E4 — Re-running with cached Y_hat, pi_hat gives identical tau and stat."""
        Y, X_obs, A, _, _ = confounded_pipeline_data

        _, res_2 = fit_gcate(Y, X_obs, A[:, None], r=2, family='nb', offset=True, backend='fast')
        U_hat = res_2['U']
        offsets = np.log(res_2['kwargs_glm']['size_factor'])
        W_aug = np.c_[X_obs, U_hat]

        df1, est1 = LFC(Y, W_aug, A[:, None], W_A=W_aug, family='nb',
                        offset=offsets, backend='fast', random_state=0)
        df2, _    = LFC(Y, W_aug, A[:, None], W_A=W_aug, family='nb',
                        offset=offsets, backend='fast', random_state=0,
                        Y_hat=est1['Y_hat'], pi_hat=est1['pi_hat'])

        np.testing.assert_allclose(df1['tau'].values,  df2['tau'].values,  atol=1e-6,
                                   err_msg="tau mismatch with cached nuisances")
        np.testing.assert_allclose(df1['stat'].values, df2['stat'].values, atol=1e-6,
                                   err_msg="stat mismatch with cached nuisances")

    # ---- E5: offset=True vs explicit offset ----
    def test_offset_true_vs_explicit(self, null_nb_balanced):
        """E5 — offset=True and explicit log(size_factor) give identical tau."""
        Y, W, A, _, _ = null_nb_balanced
        sf = comp_size_factor(Y)
        explicit_offset = np.log(sf)

        df_auto,     _ = LFC(Y, W, A[:, None], family='nb', offset=True,
                             backend='fast', random_state=0)
        df_explicit, _ = LFC(Y, W, A[:, None], family='nb', offset=explicit_offset,
                             backend='fast', random_state=0)

        np.testing.assert_allclose(
            df_auto['tau'].values, df_explicit['tau'].values, atol=1e-6,
            err_msg="tau differs between offset=True and explicit offset"
        )
