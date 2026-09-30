"""fit_gcate with an explicit offset array.

The fit reads its offset from ``kwargs_glm['size_factor']``. That entry used to
be set only for ``offset=True``, so a supplied offset array was silently
replaced by ones everywhere except the dispersion estimate.
"""

import numpy as np
import pytest

from causarray import comp_size_factor, fit_gcate


@pytest.fixture(scope='module')
def depth_varying_counts():
    rng = np.random.default_rng(0)
    n, p = 120, 300
    sf = np.exp(rng.normal(0, 0.5, n))
    Y = rng.poisson(sf[:, None] * np.exp(rng.normal(1, 0.5, p))[None, :]).astype(float)
    X = np.ones((n, 1))
    A = (rng.random((n, 1)) < 0.5).astype(float)
    return Y, X, A


def test_offset_array_is_used_like_offset_true(depth_varying_counts):
    Y, X, A = depth_varying_counts
    _, res_true = fit_gcate(Y, X, A, 2, offset=True, verbose=False)
    _, res_array = fit_gcate(Y, X, A, 2, offset=np.log(comp_size_factor(Y)), verbose=False)
    np.testing.assert_allclose(res_array['kwargs_glm']['size_factor'],
                               res_true['kwargs_glm']['size_factor'])
    np.testing.assert_allclose(res_array['U'], res_true['U'], atol=1e-6)


def test_offset_array_must_match_rows(depth_varying_counts):
    Y, X, A = depth_varying_counts
    with pytest.raises(ValueError, match='one value per row'):
        fit_gcate(Y, X, A, 2, offset=np.zeros(len(Y) - 1), verbose=False)
