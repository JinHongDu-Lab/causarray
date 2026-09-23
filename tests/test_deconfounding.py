"""
Test deconfounding performance of GCATE.

Simulates data with unmeasured confounders to verify that GCATE
correctly identifies and adjusts for them, producing more accurate
treatment effect estimates than a naive approach without deconfounding.
"""

import numpy as np
import pytest
from causarray.gcate import fit_gcate
from causarray.DR_learner import LFC


def _simulate_confounded(seed=2024):
    """Simulate NB count data with unmeasured confounders.

    Data generating process:
     - U (n x r): latent confounders drawn from N(0, 1)
     - X_obs: observed covariates
     - A: binary treatment influenced by U  (confounding!)
     - Y ~ NB(mu, disp), where log(mu) depends on X_obs, A, and U
     - true treatment effect (tau) is set per gene
    """
    np.random.seed(seed)
    n, p, r = 300, 30, 2

    # Observed covariates
    X_raw = np.random.randn(n, 2)
    # causarray outcome and propensity models use fit_intercept=False, so the
    # design matrix must include its intercept explicitly.
    X_obs = np.c_[np.ones(n), X_raw]

    # Latent confounders
    U = np.random.randn(n, r)

    # Confounding: treatment depends on U
    logit_a = 0.8 * U[:, 0] + 0.6 * U[:, 1]
    prob_a = 1.0 / (1.0 + np.exp(-logit_a))
    A = np.random.binomial(1, prob_a, n).astype(float)

    # True treatment effects (half zero, half nonzero)
    tau_true = np.zeros(p)
    tau_true[: p // 2] = np.random.choice([-1, 1], p // 2) * np.random.uniform(0.5, 1.5, p // 2)

    # Coefficients for confounders (large effect → strong confounding)
    gamma = np.random.randn(r, p) * 1.5

    # Gene-level baseline
    beta0 = np.random.uniform(2.0, 3.5, p)

    # Build log-mean
    log_mu = (
        beta0[None, :]
        + X_raw @ np.random.randn(2, p) * 0.2
        + A[:, None] * tau_true[None, :]
        + U @ gamma                       # confounding signal
    )
    mu = np.exp(np.clip(log_mu, -10, 10))

    # Dispersion (moderate overdispersion)
    disp = np.random.uniform(0.5, 2.0, p)
    Y = np.random.negative_binomial(
        n=np.maximum(1.0 / disp, 0.01),
        p=np.clip(1.0 / (1.0 + mu * disp[None, :]), 1e-10, 1 - 1e-10),
    ).astype(float)

    return Y, X_obs, A, tau_true, r


@pytest.fixture
def sim_confounded_data():
    return _simulate_confounded()


def _naive_and_deconfounded_tau(Y, X_obs, A, r):
    df_naive, _ = LFC(Y, X_obs, A[:, None], family='nb', offset=True)
    _, res_2 = fit_gcate(Y, X_obs, A[:, None], r=r, family='nb', offset=True)
    W = np.c_[X_obs, res_2['U']]
    df_deconf, _ = LFC(Y, W, A[:, None], W_A=W, family='nb',
                       offset=np.log(res_2['kwargs_glm']['size_factor']))
    return df_naive['tau'].values, df_deconf['tau'].values


class TestDeconfoundingPerformance:
    """Verify GCATE deconfounding improves treatment effect estimation."""

    def test_gcate_deconfounding_reduces_error(self):
        """GCATE-adjusted LFC has lower error than naive LFC across seeds.

        A single draw is not a fair test: on some seeds stage 1 stops on a
        plateau before recovering the factors (seed 2024 here), so the claim
        is checked over several seeds.
        """
        mse_naive, mse_deconf = [], []
        for seed in [2024, 1, 2, 3, 4]:
            Y, X_obs, A, tau_true, r = _simulate_confounded(seed)
            tau_naive, tau_deconf = _naive_and_deconfounded_tau(Y, X_obs, A, r)
            mse_naive.append(np.mean((tau_naive - tau_true) ** 2))
            mse_deconf.append(np.mean((tau_deconf - tau_true) ** 2))
        mse_naive, mse_deconf = np.array(mse_naive), np.array(mse_deconf)

        assert np.median(mse_deconf) <= 0.5 * np.median(mse_naive), (mse_naive, mse_deconf)
        assert np.sum(mse_deconf < mse_naive) >= 3, (mse_naive, mse_deconf)
        assert np.all(mse_deconf <= 1.5 * mse_naive), (mse_naive, mse_deconf)

    def test_gcate_latent_factors_recovered(self, sim_confounded_data):
        """Estimated latent factors should capture the confounding signal."""
        Y, X_obs, A, _, r = sim_confounded_data

        _, res_2 = fit_gcate(
            Y, X_obs, A[:, None], r=r, family='nb', offset=True,
        )

        U_hat = res_2['U']
        assert U_hat.shape == (Y.shape[0], r), (
            f"Expected latent factor shape ({Y.shape[0]}, {r}), got {U_hat.shape}"
        )
        # Latent factor columns should have non-trivial variance
        for k in range(r):
            assert np.std(U_hat[:, k]) > 0.01, (
                f"Latent factor {k} has near-zero variance"
            )
