import numpy as np
import pandas as pd
import pytest
from causarray.DR_learner import LFC
from causarray.DR_estimation import AIPW_mean


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(0)
    Y = rng.poisson(5, (100, 10)).astype(float)
    W = rng.standard_normal((100, 5))
    A = rng.binomial(1, 0.5, 100)
    Y += rng.poisson(1, (100, 10)) * A[:, None]
    return Y, W, A


class TestLFCOutputSchema:
    def test_output_columns(self, sample_data):
        Y, W, A = sample_data
        result, estimation = LFC(Y, W, A)
        assert isinstance(result, pd.DataFrame)
        for col in ('tau', 'std', 'log2fc', 'log2fc_se', 'stat', 'rej', 'pvalue', 'padj',
                    'pvalue_emp_null_adj', 'padj_emp_null_adj',
                    'mean_control', 'mean_treated', 'estimable'):
            assert col in result.columns
        assert result.columns.get_loc('log2fc') == result.columns.get_loc('std') + 1
        assert result.columns.get_loc('log2fc_se') == result.columns.get_loc('log2fc') + 1

    @pytest.mark.parametrize('usevar', ['pooled', 'unequal'])
    def test_log2fc_columns_are_exact_scale_conversions(self, sample_data, usevar):
        Y, W, A = sample_data
        result, _ = LFC(Y, W, A, usevar=usevar)
        np.testing.assert_allclose(result['log2fc'], result['tau'] / np.log(2.0))
        np.testing.assert_allclose(result['log2fc_se'], result['std'] / np.log(2.0))

        finite = np.isfinite(result['stat']) & np.isfinite(result['log2fc_se'])
        np.testing.assert_allclose(
            result.loc[finite, 'log2fc'] / result.loc[finite, 'log2fc_se'],
            result.loc[finite, 'stat'],
        )

    def test_float32_aipw_mean_uses_float64_accumulation(self):
        Y = np.ones((8, 2), dtype=np.float32)
        A = np.zeros((8, 1, 2), dtype=np.float32)
        A[:4, 0, 0] = 1
        A[4:, 0, 1] = 1
        mu = np.ones((8, 2, 1, 2), dtype=np.float32)
        pi = np.full((8, 1, 2), 0.5, dtype=np.float32)

        means, _ = AIPW_mean(Y, A, mu, pi)

        assert means.dtype == np.float64

    def test_nonpositive_arm_mean_with_observed_counts_is_nonestimable(self):
        # Control arm has observed counts but a negative AIPW mean (the
        # pseudo-outcome correction overshoots): that stays non-estimable.
        # (Since 0.0.10 an arm with *no* observed counts is instead kept
        # estimable at the floor; see test_small_arm_inference.)
        # With calibrated scores the AIPW arm mean equals the observed arm
        # mean, so a negative value needs miscalibrated scores: pi = 0.5 for a
        # 1-in-4 treatment gives mean_control = (2*sum(Y_ctrl) - 2*mu)/4 = -0.5.
        Y = np.zeros((4, 1), dtype=float)
        Y[0, 0] = 1.0
        W = np.ones((4, 1), dtype=float)
        A = np.array([0, 0, 0, 1], dtype=float)
        Y_hat = np.full((4, 1, 1, 2), 2.0)
        pi_hat = np.full((4, 1), 0.5)

        with pytest.warns(RuntimeWarning, match='non-estimable'):
            result, _ = LFC(
                Y, W, A, Y_hat=Y_hat, pi_hat=pi_hat, family='poisson',
                ps_clip=None,
            )

        assert result.loc[0, 'mean_control'] == pytest.approx(-0.5)
        assert result.loc[0, 'mean_treated'] == pytest.approx(1.0)
        assert not bool(result.loc[0, 'estimable'])
        assert result.loc[0, 'tau'] == 0
        assert np.isinf(result.loc[0, 'std'])
        assert result.loc[0, 'log2fc'] == 0
        assert np.isinf(result.loc[0, 'log2fc_se'])
        assert np.isnan(result.loc[0, 'pvalue'])

    def test_aipw_pseudo_outcomes_are_always_unclipped(self):
        Y = np.array([[0.0], [1.0], [3.0], [4.0]])
        W = np.ones((4, 1), dtype=float)
        A = np.array([0, 0, 1, 1], dtype=float)
        Y_hat = np.full((4, 1, 1, 2), 2.0)
        pi_hat = np.full((4, 1), 0.5)
        stacked_A = np.stack([1 - A[:, None], A[:, None]], axis=-1)
        stacked_pi = np.stack([1 - pi_hat, pi_hat], axis=-1)
        _, expected = AIPW_mean(Y, stacked_A, Y_hat, stacked_pi)
        assert expected.min() < 0

        result, estimation = LFC(
            Y, W, A, Y_hat=Y_hat, pi_hat=pi_hat, family='poisson',
            ps_class_weight='balanced', thres_diff=0,
        )

        np.testing.assert_allclose(result['mean_control'], expected[..., 0].mean(axis=0).ravel())
        np.testing.assert_allclose(result['mean_treated'], expected[..., 1].mean(axis=0).ravel())
        assert 'clip_pseudo_outcomes' not in estimation
        assert estimation['ps_class_weight'] == 'balanced'

        with pytest.raises(TypeError, match='has been removed'):
            LFC(
                Y, W, A, Y_hat=Y_hat, pi_hat=pi_hat,
                clip_pseudo_outcomes=True,
            )

    def test_aggregate_mean_is_floored_only_for_log_ratio(self):
        Y = np.array([[0.001], [0.001], [0.02], [0.02]])
        W = np.ones((4, 1), dtype=float)
        A = np.array([0, 0, 1, 1], dtype=float)
        Y_hat = np.empty((4, 1, 1, 2), dtype=float)
        Y_hat[..., 0] = 0.001
        Y_hat[..., 1] = 0.02
        pi_hat = np.full((4, 1), 0.5)

        result, _ = LFC(
            Y, W, A, Y_hat=Y_hat, pi_hat=pi_hat,
            family='poisson', thres_min=0, thres_diff=0.01,
            usevar='pooled',
        )

        # Diagnostics retain the raw aggregate AIPW means, while the LFC uses
        # max(mean, thres_diff): log(0.02 / 0.01) = log(2).
        assert result.loc[0, 'mean_control'] == pytest.approx(0.001)
        assert result.loc[0, 'mean_treated'] == pytest.approx(0.02)
        assert result.loc[0, 'tau'] == pytest.approx(np.log(2))

    def test_with_offset(self, sample_data):
        Y, W, A = sample_data
        rng = np.random.default_rng(1)
        offset = np.log(rng.poisson(5, 100).clip(1))
        result, estimation = LFC(Y, W, A, offset=offset)
        assert isinstance(result, pd.DataFrame)
        assert 'tau' in result.columns

    def test_with_fdx(self, sample_data):
        Y, W, A = sample_data
        result, estimation = LFC(Y, W, A, fdx=True)
        assert isinstance(result, pd.DataFrame)
        assert 'tau' in result.columns

    def test_with_custom_family(self, sample_data):
        Y, W, A = sample_data
        result, estimation = LFC(Y, W, A, family='poisson')
        assert isinstance(result, pd.DataFrame)
        assert 'tau' in result.columns

    def test_cross_est_uses_two_folds_and_returns_raw_scores(self, sample_data):
        Y, W, A = sample_data
        result, estimation = LFC(Y, W, A, cross_est=True, family='poisson')

        assert isinstance(result, pd.DataFrame)
        assert estimation['pi_hat_raw'].shape == (Y.shape[0], 1)
        assert estimation['pi_hat'].shape == (Y.shape[0], 1)

    def test_explicit_k_overrides_cross_est_default(self, sample_data, monkeypatch):
        import causarray.DR_learner as learner

        Y, W, A = sample_data
        seen = {}

        def fake_cross_fitting(Y, A, X, X_A, **kwargs):
            seen['K'] = kwargs['K']
            n, p = Y.shape
            a = A.shape[1]
            y_hat = np.ones((n, p, a, 2))
            pi = np.full((n, a), 0.5)
            return y_hat, pi, pi.copy()

        monkeypatch.setattr(learner, 'cross_fitting', fake_cross_fitting)
        learner.LFC(Y, W, A, cross_est=True, K=3, family='poisson')

        assert seen['K'] == 3


class TestLFCMultiTreatment:
    def test_multi_treatment(self):
        """LFC with multiple perturbations exercises the per-perturbation fast path."""
        np.random.seed(42)
        n, p, a = 200, 50, 3
        W = np.random.normal(0, 1, (n, 3))
        A = np.zeros((n, a))
        A[50:100, 0] = 1
        A[100:150, 1] = 1
        A[150:200, 2] = 1
        Y = np.random.poisson(5, (n, p))
        Y[50:100] += np.random.poisson(2, (50, p))

        result, estimation = LFC(Y, W, A, family='nb')
        assert isinstance(result, pd.DataFrame)
        assert 'trt' in result.columns
        assert len(result) == p * a
        assert result['tau'].notna().all()
        assert result['padj'].notna().all()
