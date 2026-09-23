"""
Tests for NB-GLM integration between causarray and crispyx.

These tests ensure that the new crispyx-backed NB-GLM fitting produces
results comparable to the original statsmodels-based implementation.
"""

import numpy as np
import pytest
import os
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_nb_data():
    """Simulate a small NB count dataset for testing."""
    np.random.seed(42)
    n, p, d = 200, 50, 3
    X_cov = np.random.randn(n, d)
    A = np.random.binomial(1, 0.3, (n, 1)).astype(float)
    X = np.column_stack([np.ones(n), X_cov, A])

    beta_true = np.random.randn(X.shape[1], p) * 0.3
    beta_true[0, :] = np.random.uniform(1.0, 3.0, p)  # intercept
    beta_true[-1, :] = np.random.uniform(-0.5, 0.5, p)  # treatment effect

    log_sf = np.random.normal(0, 0.3, n)
    eta = X @ beta_true + log_sf[:, None]
    mu = np.exp(np.clip(eta, -10, 10))
    disp = np.random.uniform(0.1, 2.0, p)
    Y = np.random.negative_binomial(
        n=np.maximum(1.0 / disp[None, :], 0.01),
        p=np.clip(1.0 / (1.0 + mu * disp[None, :]), 1e-10, 1 - 1e-10),
    ).astype(float)

    return Y, X_cov, A, log_sf, beta_true, disp


@pytest.fixture
def sim_small_data():
    """Very small dataset for quick sanity checks."""
    np.random.seed(0)
    n, p = 100, 10
    X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
    mu = np.exp(X @ np.random.randn(3, p) * 0.2 + 1.5)
    Y = np.random.poisson(mu).astype(float)
    return Y, X


# ---------------------------------------------------------------------------
# 1. Test dispersion estimation: old vs new
# ---------------------------------------------------------------------------

class TestDispersionEstimation:
    """Compare dispersion estimates from original and new methods."""

    def test_estimate_disp_consistency(self, sim_nb_data):
        """Original estimate_disp should produce finite positive values."""
        from causarray.gcate_glm import estimate_disp

        Y, X_cov, A, log_sf, _, _ = sim_nb_data
        X = np.column_stack([np.ones(Y.shape[0]), X_cov, A])
        disp = estimate_disp(Y, X, offset=log_sf, disp_family='poisson')

        assert disp.shape == (Y.shape[1],)
        assert np.all(np.isfinite(disp))
        assert np.all(disp > 0)

    def test_estimate_disp_fast_consistency(self, sim_nb_data):
        """New fast dispersion estimation (via crispyx) should be close to original."""
        from causarray.gcate_glm import estimate_disp
        from causarray.nb_glm_fast import estimate_disp_fast

        Y, X_cov, A, log_sf, _, _ = sim_nb_data
        X = np.column_stack([np.ones(Y.shape[0]), X_cov, A])

        disp_old = estimate_disp(Y, X, offset=log_sf, disp_family='poisson')
        disp_new = estimate_disp_fast(Y, X, offset=log_sf)

        # Both should be finite and positive
        assert np.all(np.isfinite(disp_new))
        assert np.all(disp_new > 0)

        # Correlation should be high (dispersion estimates may differ in scale
        # but should rank genes similarly)
        corr = np.corrcoef(disp_old, disp_new)[0, 1]
        assert corr > 0.4, f"Dispersion correlation too low: {corr:.3f}"


# ---------------------------------------------------------------------------
# 2. Test fit_glm: old vs new
# ---------------------------------------------------------------------------

class TestFitGLM:
    """Compare coefficient estimates from original and new GLM fitting."""

    def test_fit_glm_poisson_consistency(self, sim_small_data):
        """Poisson GLM results should be close between old and new."""
        from causarray.gcate_glm import fit_glm
        from causarray.nb_glm_fast import fit_glm_fast

        Y, X = sim_small_data

        B_old, Yhat_old, _, _, _ = fit_glm(Y, X, family='poisson')
        B_new, Yhat_new, _, _, _ = fit_glm_fast(Y, X, family='poisson')

        # Coefficients should be close
        # B shape is (p, d) — each row is a gene, columns are features
        # Compare feature-by-feature across genes (correlation over p genes)
        for k in range(X.shape[1]):
            corr = np.corrcoef(B_old[:, k], B_new[:, k])[0, 1]
            if np.std(B_old[:, k]) > 1e-6:
                assert corr > 0.8, f"Feature {k}: coefficient correlation too low: {corr:.3f}"

        # Predicted values should be highly correlated
        corr_yhat = np.corrcoef(Yhat_old.ravel(), Yhat_new.ravel())[0, 1]
        assert corr_yhat > 0.9, f"Yhat correlation too low: {corr_yhat:.3f}"

    def test_fit_glm_nb_consistency(self, sim_nb_data):
        """NB GLM results should be comparable between old and new."""
        from causarray.gcate_glm import fit_glm
        from causarray.nb_glm_fast import fit_glm_fast

        Y, X_cov, A, log_sf, _, _ = sim_nb_data
        X = np.column_stack([np.ones(Y.shape[0]), X_cov])

        B_old, Yhat_old, disp_old, _, _ = fit_glm(
            Y, X, family='nb', offset=log_sf
        )
        B_new, Yhat_new, disp_new, _, _ = fit_glm_fast(
            Y, X, family='nb', offset=log_sf
        )

        # Coefficients should be correlated (B is p x d)
        # Compare feature-by-feature across genes
        for k in range(X.shape[1]):
            if np.std(B_old[:, k]) > 1e-6:
                corr = np.corrcoef(B_old[:, k], B_new[:, k])[0, 1]
                assert corr > 0.5, f"Feature {k}: NB coefficient correlation too low: {corr:.3f}"

    def test_fit_glm_fast_output_shapes(self, sim_nb_data):
        """New fit_glm_fast should return same shapes as original."""
        from causarray.nb_glm_fast import fit_glm_fast

        Y, X_cov, A, log_sf, _, _ = sim_nb_data
        X = np.column_stack([np.ones(Y.shape[0]), X_cov])

        B, Yhat, disp, offsets, resid_dev = fit_glm_fast(
            Y, X, family='nb', offset=log_sf
        )

        assert B.shape == (Y.shape[1], X.shape[1])
        assert Yhat.shape == (Y.shape[0], Y.shape[1])
        assert disp.shape == (Y.shape[1],)
        assert resid_dev.shape == (Y.shape[0], Y.shape[1])

    def test_fit_glm_fast_with_treatment(self, sim_nb_data):
        """fit_glm_fast with treatment indicator (impute mode)."""
        from causarray.nb_glm_fast import fit_glm_fast

        Y, X_cov, A, log_sf, _, _ = sim_nb_data
        X = np.column_stack([np.ones(Y.shape[0]), X_cov])

        B, Yhat, disp, offsets, resid_dev = fit_glm_fast(
            Y, X, A=A, family='nb', offset=log_sf, impute=True
        )

        # B shape is (p, d_full) where d_full = n_covariates + n_treatments
        assert B.shape[1] == X.shape[1] + A.shape[1]
        # Yhat should be a tuple (Yhat_0, Yhat_1) in impute mode
        if isinstance(Yhat, tuple):
            assert Yhat[0].shape == (Y.shape[0], Y.shape[1], A.shape[1])
            assert Yhat[1].shape == (Y.shape[0], Y.shape[1], A.shape[1])

    def test_poisson_coefficients_match_statsmodels(self):
        """fit_glm_fast (Poisson) coefficients agree with statsmodels within 20%."""
        from causarray.nb_glm_fast import fit_glm_fast

        rng = np.random.default_rng(0)
        n, p = 120, 8
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        B_true = rng.standard_normal((p, 2)) * 0.3
        Y = rng.poisson(np.exp(X @ B_true.T)).astype(float)

        B, Yhat, disp_glm, offsets, resid_dev = fit_glm_fast(Y, X, family="poisson")

        assert B.shape == (p, 2)
        assert Yhat.shape == (n, p)
        assert disp_glm is None
        assert resid_dev.shape == (n, p)
        assert np.all(np.isfinite(B))

        for j in range(p):
            mod = sm.GLM(Y[:, j], X, family=sm.families.Poisson()).fit()
            np.testing.assert_allclose(B[j], mod.params, rtol=0.2,
                                       err_msg=f"gene {j} coefficient mismatch")


# ---------------------------------------------------------------------------
# 3. Test GCATE pipeline with new GLM
# ---------------------------------------------------------------------------

class TestGCATEIntegration:
    """Test that GCATE works with the new fast GLM backend."""

    @pytest.mark.skip(reason="GCATE Numba/OMP segfault in test env; test separately")
    def test_gcate_with_fast_glm(self):
        """fit_gcate should work with the new backend and produce same structure."""
        from causarray.gcate import fit_gcate

        np.random.seed(42)
        Y = np.random.poisson(5, (100, 20))
        X = np.random.randn(100, 3)
        A = np.random.randn(100, 1)
        r = 2
        offset = np.ones(100)

        res_1, res_2 = fit_gcate(
            Y, X, A, r, family='nb', offset=offset
        )

        assert isinstance(res_1, dict)
        assert isinstance(res_2, dict)
        assert res_1['X_U'].shape[1] == X.shape[1] + A.shape[1] + r


# ---------------------------------------------------------------------------
# 4. Test on-disk NB-GLM fitting
# ---------------------------------------------------------------------------

class TestOnDiskNBGLM:
    """Test on-disk NB-GLM fitting using crispyx streaming functions."""

    @pytest.fixture
    def adamson(self):
        """An Adamson h5ad with its perturbation column and control label.

        Repo-relative by default (the tutorial subset); point
        ``CAUSARRAY_TEST_H5AD`` at another file to use that instead.  No
        absolute path belongs in a test -- it only runs on one machine.
        """
        import anndata as ad

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.environ.get('CAUSARRAY_TEST_H5AD') or os.path.join(
            repo, 'docs', 'source', 'tutorial', 'adamson', 'adamson_subset.h5ad')
        if not os.path.exists(path):
            pytest.skip(f"{os.path.relpath(path, repo)} not found; "
                        "set CAUSARRAY_TEST_H5AD to another h5ad")
        adata = ad.read_h5ad(path, backed='r')
        pert_col = adata.uns.get('pert_col', 'perturbation')
        ctrl = adata.uns.get('ctrl_label', 'control')
        labels = adata.obs[pert_col].astype(str).values
        adata.file.close()
        target = next((p for p in np.unique(labels) if p != ctrl), None)
        if target is None:
            pytest.skip("no non-control perturbation in the test h5ad")
        return path, pert_col, ctrl, target

    def test_fit_glm_ondisk_vs_inmemory(self, adamson):
        """On-disk fitting should produce results comparable to in-memory."""
        import anndata as ad
        from causarray.nb_glm_fast import fit_glm_fast, fit_glm_ondisk

        path, pert_col, ctrl, target = adamson
        adata = ad.read_h5ad(path)
        # Use a small subset of genes for speed
        Y = np.asarray(adata.X[:, :100].toarray() if hasattr(adata.X, 'toarray') else adata.X[:, :100], dtype=float)
        n = Y.shape[0]

        # Binary treatment: control vs the first perturbation
        perturbations = adata.obs[pert_col].astype(str).values
        mask = np.isin(perturbations, [ctrl, target])
        Y_sub = Y[mask]
        A_sub = (perturbations[mask] != ctrl).astype(float)[:, None]
        # Size factors come from the genes read, so cells with no counts among
        # them have no offset; fit_glm_ondisk drops them, so do the same here.
        nonempty = Y_sub.sum(axis=1) > 0
        Y_sub, A_sub = Y_sub[nonempty], A_sub[nonempty]
        X_sub = np.ones((Y_sub.shape[0], 1))

        # In-memory fit
        B_mem, Yhat_mem, disp_mem, _, _ = fit_glm_fast(
            Y_sub, X_sub, A=A_sub, family='nb', offset=True
        )

        # On-disk fit (reads from h5ad)
        B_disk, Yhat_disk, disp_disk, _, _ = fit_glm_ondisk(
            path,
            perturbation_col=pert_col,
            control_label=ctrl,
            target_label=target,
            gene_indices=np.arange(100),
        )

        # Results should be correlated
        # On-disk may use different dispersion estimation, so we check correlation
        # B is (p, d), last column is treatment effect
        corr = np.corrcoef(B_mem[:, -1], B_disk[:, -1])[0, 1]
        assert corr > 0.7, f"On-disk vs in-memory LFC correlation: {corr:.3f}"

    def test_fit_glm_ondisk_produces_valid_output(self, adamson):
        """On-disk fitting should produce finite, well-shaped output."""
        from causarray.nb_glm_fast import fit_glm_ondisk

        path, pert_col, ctrl, target = adamson
        B, Yhat, disp, offsets, resid_dev = fit_glm_ondisk(
            path,
            perturbation_col=pert_col,
            control_label=ctrl,
            target_label=target,
            gene_indices=np.arange(50),
        )

        assert np.all(np.isfinite(B))
        assert np.all(np.isfinite(disp))
        assert np.all(disp > 0)
        assert B.shape[0] == 50  # 50 genes (B is p x d)


# ---------------------------------------------------------------------------
# 5. Performance smoke test
# ---------------------------------------------------------------------------

class TestPerformance:
    """Basic timing comparison between old and new implementations."""

    def test_fast_is_not_slower(self, sim_nb_data):
        """New implementation should not be dramatically slower than old."""
        import time
        from causarray.gcate_glm import fit_glm
        from causarray.nb_glm_fast import fit_glm_fast

        Y, X_cov, A, log_sf, _, _ = sim_nb_data
        X = np.column_stack([np.ones(Y.shape[0]), X_cov])

        # Warm up
        fit_glm_fast(Y[:10], X[:10], family='poisson')

        t0 = time.time()
        fit_glm(Y, X, family='poisson')
        t_old = time.time() - t0

        t0 = time.time()
        fit_glm_fast(Y, X, family='poisson')
        t_new = time.time() - t0

        # New should not be more than 5x slower (it should typically be faster)
        assert t_new < t_old * 5, (
            f"New impl too slow: {t_new:.2f}s vs old {t_old:.2f}s"
        )

