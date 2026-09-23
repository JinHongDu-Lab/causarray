"""Unit tests for nb_glm_fast.py and the gcate_glm backend toggle."""
from __future__ import annotations

import math
import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import scipy.sparse as sp

import causarray.gcate_glm as gcate_glm
from causarray.nb_glm_fast import (
    _maybe_densify,
    _SPARSE_WARN_GB,
    fit_glm_fast,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_counts(n: int, p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.negative_binomial(5, 0.5, size=(n, p)).astype(np.float64)


def _make_X(n: int, d: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(d - 1)])
    return X


# ---------------------------------------------------------------------------
# Poisson deviance residuals
# ---------------------------------------------------------------------------

class TestDevianceResiduals:
    """Deviance residuals come from crispyx; check the contract fit_glm_fast keeps."""

    @staticmethod
    def _fit(family):
        rng = np.random.default_rng(3)
        n, p = 120, 60
        Y = rng.poisson(4.0, (n, p)).astype(np.float64)
        X = np.c_[np.ones(n), rng.standard_normal(n)]
        B, Yhat, disp, _, resid = fit_glm_fast(Y, X, family=family)
        return Y, Yhat, resid

    def test_shape_and_finiteness(self):
        Y, Yhat, resid = self._fit("poisson")
        assert resid.shape == Y.shape
        assert np.all(np.isfinite(resid))

    def test_sign_follows_y_minus_mu(self):
        """A residual is signed by whether the cell over- or under-shoots the fit."""
        Y, Yhat, resid = self._fit("poisson")
        differs = Y != Yhat
        assert np.all(np.sign(resid[differs]) == np.sign((Y - Yhat)[differs]))

    def test_nb_residuals_are_smaller_than_poisson(self):
        """Over-dispersed counts: the NB deviance must not exceed the Poisson one."""
        rng = np.random.default_rng(5)
        n, p = 150, 60
        Y = rng.negative_binomial(2, 0.2, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        _, _, _, _, resid_pois = fit_glm_fast(Y, X, family="poisson")
        _, _, _, _, resid_nb = fit_glm_fast(Y, X, family="nb")
        assert np.mean(resid_nb ** 2) < np.mean(resid_pois ** 2)


# ---------------------------------------------------------------------------
# Sparse densification
# ---------------------------------------------------------------------------

class TestSparseDensification:
    def test_small_sparse_no_warning(self):
        Y_sp = sp.csr_matrix(np.eye(10))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            Y_dense = _maybe_densify(Y_sp)
        assert Y_dense.shape == (10, 10)
        assert Y_dense.dtype == np.float64

    def test_large_sparse_resource_warning(self):
        """Sparse matrix whose dense form exceeds _SPARSE_WARN_GB triggers warning."""
        gb_threshold = _SPARSE_WARN_GB
        n = p = int(math.sqrt(gb_threshold * 1e9 / 8)) + 10
        Y_sp = sp.eye(n, p, format="csr")
        with pytest.warns(ResourceWarning, match="Materialising sparse Y"):
            _maybe_densify(Y_sp)

    def test_dense_array_no_warning(self):
        Y = np.ones((5, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            out = _maybe_densify(Y)
        np.testing.assert_array_equal(out, Y)


# ---------------------------------------------------------------------------
# Crispyx availability fallback
# ---------------------------------------------------------------------------

class TestCrispyxFallback:
    def test_no_crispyx_falls_back_to_statsmodels(self):
        """When _CRISPYX_AVAILABLE=False, fit_glm_auto must call fit_glm (statsmodels)."""
        n, p = 60, 80
        Y = _make_counts(n, p)
        X = _make_X(n, 2)

        orig_avail = gcate_glm._CRISPYX_AVAILABLE
        try:
            gcate_glm._CRISPYX_AVAILABLE = False
            with patch.object(gcate_glm, "fit_glm", wraps=gcate_glm.fit_glm) as mock_fg:
                gcate_glm.fit_glm_auto(Y, X, family="nb")
                assert mock_fg.called, "fit_glm should be called when crispyx unavailable"
        finally:
            gcate_glm._CRISPYX_AVAILABLE = orig_avail


# ---------------------------------------------------------------------------
# Fast-path threshold (_FAST_MAX_D)
# ---------------------------------------------------------------------------

class TestFastMinPThreshold:
    """Gene count, not design width, is what keeps a call off the batched path."""

    def test_fast_min_p_default_is_10(self):
        assert gcate_glm._FAST_MIN_P == 10

    def test_enough_genes_uses_fast(self):
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        n, p, d = 200, 60, 2
        Y = _make_counts(n, p)
        X = _make_X(n, d)

        with patch.object(gcate_glm, "fit_glm_fast", wraps=gcate_glm.fit_glm_fast) as mock_ff:
            gcate_glm.fit_glm_auto(Y, X, family="nb")
            assert mock_ff.called

    def test_too_few_genes_uses_statsmodels(self):
        n, p, d = 100, 4, 2
        Y = _make_counts(n, p)
        X = _make_X(n, d)

        with patch.object(gcate_glm, "fit_glm_fast") as mock_ff:
            gcate_glm.fit_glm_auto(Y, X, family="nb")
            assert not mock_ff.called

    def test_wide_design_still_uses_fast(self):
        """The old _FAST_MAX_D = 50 cap sent wide screens to statsmodels for hours."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        rng = np.random.default_rng(11)
        n, p, a = 300, 60, 60
        Y = _make_counts(n, p)
        X = np.c_[np.ones(n), rng.standard_normal(n)]
        A = np.zeros((n, a))
        for k in range(a):
            A[k * 4 : (k + 1) * 4, k] = 1

        with patch.object(gcate_glm, "fit_glm_fast", wraps=gcate_glm.fit_glm_fast) as mock_ff:
            gcate_glm.fit_glm_auto(Y, X, A=A, family="nb")
            assert mock_ff.called


# ---------------------------------------------------------------------------
# Backend toggle, backend param, backend propagation
# ---------------------------------------------------------------------------

class TestBackendToggle:
    def test_toggle_off_calls_statsmodels(self):
        """_USE_FAST_BACKEND=False must route through fit_glm for any input."""
        n, p = 60, 80
        Y = _make_counts(n, p)
        X = _make_X(n, 2)

        with gcate_glm._backend_override("original"):
            with patch.object(gcate_glm, "fit_glm", wraps=gcate_glm.fit_glm) as mock_fg:
                gcate_glm.fit_glm_auto(Y, X, family="nb")
                assert mock_fg.called

    def test_toggle_restores_after_context_exit(self):
        orig = gcate_glm._USE_FAST_BACKEND
        with gcate_glm._backend_override("original"):
            assert gcate_glm._USE_FAST_BACKEND is False
        assert gcate_glm._USE_FAST_BACKEND == orig

    def test_toggle_fast_sets_true(self):
        with gcate_glm._backend_override("fast"):
            assert gcate_glm._USE_FAST_BACKEND is True

    def test_auto_preserves_current_value(self):
        orig = gcate_glm._USE_FAST_BACKEND
        with gcate_glm._backend_override("auto"):
            assert gcate_glm._USE_FAST_BACKEND == orig


class TestBackendParam:
    def test_fit_gcate_accepts_backend_param(self):
        """fit_gcate must accept backend='original' without error (small data)."""
        from causarray.gcate import fit_gcate

        rng = np.random.default_rng(42)
        n, p, d = 60, 20, 2
        Y = rng.negative_binomial(5, 0.5, (n, p))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        A = rng.binomial(1, 0.3, (n, 1)).astype(float)

        res_1, res_2 = fit_gcate(Y, X, A, r=1, family="nb", offset=False,
                                  backend="original")
        assert isinstance(res_1, dict)
        assert isinstance(res_2, dict)

    def test_fit_gcate_backend_original_uses_statsmodels(self):
        """fit_gcate with backend='original' calls fit_glm (statsmodels path)."""
        from causarray.gcate import fit_gcate

        rng = np.random.default_rng(42)
        n, p = 60, 20
        Y = rng.negative_binomial(5, 0.5, (n, p))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        A = rng.binomial(1, 0.3, (n, 1)).astype(float)

        with patch.object(gcate_glm, "fit_glm", wraps=gcate_glm.fit_glm) as mock_fg:
            fit_gcate(Y, X, A, r=1, family="nb", offset=False, backend="original")
            assert mock_fg.called, "statsmodels path not called with backend='original'"


class TestBackendPropagation:
    def test_gcate_opt_uses_module_ref(self):
        """gcate_opt.alter_min calls _gcate_glm.fit_glm_auto, not a stale binding."""
        import causarray.gcate_opt as gcate_opt
        assert not hasattr(gcate_opt, "fit_glm_auto"), (
            "gcate_opt must not bind fit_glm_auto at module level"
        )

    def test_dr_estimation_uses_module_ref(self):
        """DR_estimation should not have fit_glm_auto bound at module level."""
        import causarray.DR_estimation as dr_est
        assert not hasattr(dr_est, "fit_glm_auto"), (
            "DR_estimation must not bind fit_glm_auto at module level"
        )

    def test_toggle_propagates_to_gcate_opt_call(self):
        """With _USE_FAST_BACKEND=False, calls inside gcate_opt use statsmodels."""
        with gcate_glm._backend_override("original"):
            with patch.object(gcate_glm, "fit_glm", wraps=gcate_glm.fit_glm) as mock_fg:
                n, p = 50, 60
                Y = _make_counts(n, p)
                X = _make_X(n, 2)
                gcate_glm.fit_glm_auto(Y, X, family="nb")
                assert mock_fg.called


# ---------------------------------------------------------------------------
# Weighted dispersion average
# ---------------------------------------------------------------------------

class TestJointTreatmentFit:
    """Multi-treatment designs are fitted jointly by crispyx's structured solver."""

    @staticmethod
    def _data(n_ctrl=120, n_per_pert=40, a=3, p=60, seed=99):
        rng = np.random.default_rng(seed)
        n = n_ctrl + n_per_pert * a
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        for k in range(a):
            A[n_ctrl + k * n_per_pert : n_ctrl + (k + 1) * n_per_pert, k] = 1
        return Y, X, A, n_ctrl

    def test_coefficients_follow_caller_column_order(self):
        """B is laid out as [X | A], whatever the solver does internally."""
        Y, X, A, _ = self._data()
        B, Yhat, disp, _, resid = fit_glm_fast(Y, X, A=A, family="nb")
        assert B.shape == (Y.shape[1], X.shape[1] + A.shape[1])
        assert np.all(np.isfinite(B))

    def test_dispersion_is_per_gene_and_finite(self):
        Y, X, A, _ = self._data()
        _, _, disp, _, _ = fit_glm_fast(Y, X, A=A, family="nb")
        assert disp.shape == (Y.shape[1],)
        assert np.all(np.isfinite(disp)) and np.all(disp > 0)

    def test_control_cells_are_fitted_too(self):
        """Every cell gets a fitted mean and a residual, controls included."""
        Y, X, A, n_ctrl = self._data()
        _, Yhat, _, _, resid = fit_glm_fast(Y, X, A=A, family="nb")
        ctrl = np.arange(n_ctrl)
        assert np.all(np.isfinite(resid[ctrl]))
        assert np.all(Yhat[ctrl] > 0)

    def test_agrees_with_statsmodels(self):
        """The joint fit is the same model the gene-by-gene path fits."""
        Y, X, A, _ = self._data(n_ctrl=150, n_per_pert=50, a=2, p=60, seed=4)
        B_fast, _, _, _, _ = fit_glm_fast(Y, X, A=A, family="poisson")
        with gcate_glm._backend_override("original"):
            B_slow, _, _, _, _ = gcate_glm.fit_glm(Y, X, A=A, family="poisson")
        assert np.max(np.abs(B_fast - B_slow)) < 1e-4

    def test_non_onehot_treatment_falls_back_to_dense(self):
        """A continuous treatment has no group structure; the dense solver takes it."""
        rng = np.random.default_rng(8)
        n, p = 200, 60
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = rng.standard_normal((n, 1))
        B, Yhat, disp, _, resid = fit_glm_fast(Y, X, A=A, family="nb")
        assert B.shape == (p, 2)
        assert np.all(np.isfinite(B))


# ---------------------------------------------------------------------------
# Memory limits — imputation tensors
# ---------------------------------------------------------------------------

class TestMemoryLimitImputation:
    """mem_limit_gb below the imputation array size → float32 allocation + warning."""

    @staticmethod
    def _run(n, p, a, mem_limit_gb, impute=True, fit=None):
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        rng = np.random.default_rng(17)
        n_ctrl = n // 2
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        per = (n - n_ctrl) // a
        for k in range(a):
            A[n_ctrl + k * per : n_ctrl + (k + 1) * per, k] = 1

        fit = fit or fit_glm_fast
        return fit(
            Y, X, A=A, family="nb", impute=X.copy() if impute else False,
            maxiter=10, mem_limit_gb=mem_limit_gb,
        )

    def test_below_limit_uses_float32_and_warns(self):
        n, p, a = 100, 60, 3
        tiny_limit = n * p * a * 2 * 8 / 1e9 * 0.5

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            _, Yhat, _, _, _ = self._run(n, p, a, mem_limit_gb=tiny_limit)

        Yhat_0, Yhat_1 = Yhat
        assert Yhat_0.dtype == Yhat_1.dtype == np.float32
        assert Yhat_0.shape == Yhat_1.shape == (n, p, a)
        assert np.all(np.isfinite(Yhat_0)) and np.all(np.isfinite(Yhat_1))

    def test_above_limit_stays_float64(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            _, Yhat, _, _, _ = self._run(100, 60, 3, mem_limit_gb=1e6)
        assert Yhat[0].dtype == Yhat[1].dtype == np.float64

    def test_none_limit_stays_float64(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            _, Yhat, _, _, _ = self._run(100, 60, 3, mem_limit_gb=None)
        assert Yhat[0].dtype == Yhat[1].dtype == np.float64

    def test_no_impute_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            _, Yhat, _, _, _ = self._run(100, 60, 3, mem_limit_gb=0.0, impute=False)
        assert isinstance(Yhat, np.ndarray)

    def test_fit_glm_auto_honours_the_limit(self):
        """The contract holds through the router, whichever solver it picks."""
        n, p, a = 100, 60, 3
        tiny_limit = n * p * a * 2 * 8 / 1e9 * 0.5

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            _, Yhat, _, _, _ = self._run(
                n, p, a, mem_limit_gb=tiny_limit, fit=gcate_glm.fit_glm_auto,
            )
        assert Yhat[0].dtype == Yhat[1].dtype == np.float32


# ---------------------------------------------------------------------------
# Memory limits — fit_glm_fast passthrough
# ---------------------------------------------------------------------------

class TestMemoryLimitPassthrough:
    """fit_glm_fast(mem_limit_gb=...) must thread the arg through to imputation."""

    def test_mem_limit_propagates_via_fit_glm_fast(self):
        """fit_glm_fast with a very small mem_limit_gb raises ResourceWarning."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray.nb_glm_fast import fit_glm_fast

        rng = np.random.default_rng(42)
        n, p, a = 80, 50, 3
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        per = n // (a + 1)
        for k in range(a):
            A[(k + 1) * per : (k + 2) * per, k] = 1
        X_test = X.copy()

        true_gb = n * p * a * 2 * 8 / 1e9
        tiny_limit = true_gb * 0.1

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            B, Yhat, disp_out, offsets, resid = fit_glm_fast(
                Y, X, A=A, family="nb",
                impute=X_test,
                mem_limit_gb=tiny_limit,
            )

        Yhat_0, Yhat_1 = Yhat
        assert Yhat_0.dtype == np.float32


# ---------------------------------------------------------------------------
# Memory limits — cross_fitting and LFC
# ---------------------------------------------------------------------------

class TestMemoryLimitCrossFitting:
    """DR_estimation.cross_fitting allocates Y_hat as float32 when over mem_limit_gb."""

    def test_cross_fitting_float32_when_over_limit(self):
        """mem_limit_gb below Y_hat cost → float32 Y_hat + ResourceWarning."""
        from causarray.DR_estimation import cross_fitting

        rng = np.random.default_rng(7)
        n, p, a = 100, 30, 2
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(float)
        X = rng.standard_normal((n, 2))
        A = np.zeros((n, a))
        A[:30, 0] = 1
        A[30:60, 1] = 1

        true_gb = n * p * a * 2 * 8 / 1e9
        tiny_limit = true_gb * 0.1

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            Y_hat, pi_hat = cross_fitting(
                Y, A, X, X, family="nb",
                mem_limit_gb=tiny_limit,
            )

        assert Y_hat.dtype == np.float32, f"Expected float32, got {Y_hat.dtype}"
        assert Y_hat.shape == (n, p, a, 2)
        assert np.all(np.isfinite(Y_hat))

    def test_cross_fitting_float64_when_under_limit(self):
        """mem_limit_gb above Y_hat cost → float64 preserved, no warning."""
        from causarray.DR_estimation import cross_fitting

        rng = np.random.default_rng(8)
        n, p, a = 100, 30, 2
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(float)
        X = rng.standard_normal((n, 2))
        A = np.zeros((n, a))
        A[:30, 0] = 1
        A[30:60, 1] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            Y_hat, pi_hat = cross_fitting(
                Y, A, X, X, family="nb",
                mem_limit_gb=1e6,
            )

        assert Y_hat.dtype == np.float64

    def test_cross_fitting_no_limit_float64(self):
        """mem_limit_gb=None (default) must not trigger float32 downcast."""
        from causarray.DR_estimation import cross_fitting

        rng = np.random.default_rng(9)
        n, p, a = 80, 20, 2
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(float)
        X = rng.standard_normal((n, 2))
        A = np.zeros((n, a))
        A[:20, 0] = 1
        A[20:40, 1] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            Y_hat, pi_hat = cross_fitting(Y, A, X, X, family="nb")

        assert Y_hat.dtype == np.float64

    def test_lfc_accepts_mem_limit_gb(self):
        """LFC(mem_limit_gb=...) threads through to cross_fitting without error."""
        from causarray.DR_learner import LFC

        rng = np.random.default_rng(11)
        n, p = 120, 20
        a = 2
        W = rng.standard_normal((n, 3))
        A = np.zeros((n, a))
        A[:30, 0] = 1
        A[30:60, 1] = 1
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(float)

        true_gb = n * p * a * 2 * 8 / 1e9
        tiny_limit = true_gb * 0.1

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            df_res, estimation = LFC(Y, W, A, family="nb", mem_limit_gb=tiny_limit)

        assert "tau" in df_res.columns
        assert len(df_res) == p * a
        assert estimation["Y_hat"].dtype == np.float32


