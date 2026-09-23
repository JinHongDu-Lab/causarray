"""Tests for fit_gcate_batch and gcate_lfc_batch."""
import gc
import numpy as np
import pandas as pd
import pytest

from causarray.utils import subsample_ctrl_cells, subsample_pert_cells
from causarray.gcate import fit_gcate_batch
from causarray.DR_learner import gcate_lfc_batch, LFC_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=200, p=40, a=6, intercept=True, seed=0, family='nb'):
    """Generate a tiny synthetic dataset for testing."""
    rng = np.random.default_rng(seed)

    # Simple NB data: mean = exp(X @ beta)
    d = 3
    X = rng.standard_normal((n, d))
    if intercept:
        X = np.c_[np.ones(n), X]

    # One-hot treatment assignment (each cell assigned to at most one pert)
    A_np = np.zeros((n, a))
    cells_per_pert = n // (a + 1)
    for j in range(a):
        start = j * cells_per_pert
        end = start + cells_per_pert
        A_np[start:end, j] = 1.0

    if family == 'nb':
        mean = np.exp(X @ rng.standard_normal((X.shape[1], p)) * 0.3 + 3)
        disp = rng.uniform(0.1, 0.5, size=p)
        var = mean + disp * mean ** 2
        p_nb = mean / var
        n_nb = 1 / disp
        Y = rng.negative_binomial(n_nb, p_nb).astype(float)
    else:
        lam = np.exp(X @ rng.standard_normal((X.shape[1], p)) * 0.3 + 3)
        Y = rng.poisson(lam).astype(float)

    # Ensure no all-zero genes
    Y = np.maximum(Y, 1)
    return Y, X, A_np


# ---------------------------------------------------------------------------
# Tests: subsample utilities
# ---------------------------------------------------------------------------

class TestSubsampleUtils:
    def test_ctrl_cells_smaller_pool(self):
        ctrl_idx = np.arange(500)
        result = subsample_ctrl_cells(ctrl_idx, n_ctrl=2000, random_state=0)
        np.testing.assert_array_equal(result, ctrl_idx)

    def test_ctrl_cells_larger_pool(self):
        ctrl_idx = np.arange(5000)
        result = subsample_ctrl_cells(ctrl_idx, n_ctrl=2000, random_state=0)
        assert result.shape == (2000,)
        assert np.all(result[:-1] < result[1:]), "Result should be sorted"
        assert np.all(np.isin(result, ctrl_idx)), "All indices should be in ctrl_idx"

    def test_ctrl_cells_reproducible(self):
        ctrl_idx = np.arange(5000)
        r1 = subsample_ctrl_cells(ctrl_idx, n_ctrl=1000, random_state=42)
        r2 = subsample_ctrl_cells(ctrl_idx, n_ctrl=1000, random_state=42)
        np.testing.assert_array_equal(r1, r2)

    def test_pert_cells_within_budget(self):
        pert_idx = np.arange(100)
        result = subsample_pert_cells(pert_idx, max_cells=2000)
        np.testing.assert_array_equal(result, pert_idx)

    def test_pert_cells_over_budget(self):
        pert_idx = np.arange(4000)
        result = subsample_pert_cells(pert_idx, max_cells=2000)
        assert len(result) == 2000
        assert np.all(result[:-1] < result[1:])

    def test_pert_cells_no_cap(self):
        """max_cells=None keeps all pert cells."""
        pert_idx = np.arange(5000)
        result = subsample_pert_cells(pert_idx, max_cells=None)
        np.testing.assert_array_equal(result, pert_idx)


# ---------------------------------------------------------------------------
# Tests: fit_gcate_batch
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def small_data():
    return _make_data(n=200, p=40, a=6, seed=0)


def test_fit_gcate_batch_returns_list(small_data):
    Y, X, A = small_data
    results = fit_gcate_batch(
        Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb',
        kwargs_es_1=dict(max_iters=2, warmup=0, patience=2),
        kwargs_es_2=dict(max_iters=2, warmup=0, patience=2),
    )
    n_batches = int(np.ceil(A.shape[1] / 3))
    assert isinstance(results, list)
    assert len(results) == n_batches


def test_fit_gcate_batch_keys(small_data):
    Y, X, A = small_data
    results = fit_gcate_batch(
        Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb',
        kwargs_es_1=dict(max_iters=2, warmup=0, patience=2),
        kwargs_es_2=dict(max_iters=2, warmup=0, patience=2),
    )
    expected_keys = {'batch_i', 'pert_names', 'ctrl_idx', 'pert_idx',
                     'cell_idx', 'res_1', 'res_2', 'disp_glm', 't_batch'}
    for br in results:
        assert expected_keys.issubset(br.keys())


def test_fit_gcate_batch_shared_ctrl(small_data):
    """The same ctrl cell indices are used in every batch."""
    Y, X, A = small_data
    results = fit_gcate_batch(
        Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb',
        kwargs_es_1=dict(max_iters=2, warmup=0, patience=2),
        kwargs_es_2=dict(max_iters=2, warmup=0, patience=2),
    )
    ctrl_first = results[0]['ctrl_idx']
    for br in results[1:]:
        np.testing.assert_array_equal(br['ctrl_idx'], ctrl_first)


def test_fit_gcate_batch_with_dataframe(small_data):
    """Accepts DataFrame A with named columns."""
    Y, X, A_np = small_data
    A_df = pd.DataFrame(A_np, columns=[f'pert_{j}' for j in range(A_np.shape[1])])
    results = fit_gcate_batch(
        Y, X, A_df, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb',
        kwargs_es_1=dict(max_iters=2, warmup=0, patience=2),
        kwargs_es_2=dict(max_iters=2, warmup=0, patience=2),
    )
    for i, br in enumerate(results):
        expected = [f'pert_{j}' for j in range(i * 3, min((i + 1) * 3, A_np.shape[1]))]
        assert br['pert_names'] == expected


def test_fit_gcate_batch_max_cells_respected(small_data):
    """Each batch must have at most max_cells pert cells (ctrl added on top)."""
    Y, X, A = small_data
    max_cells = 20  # tight cap: each batch has ~28 pert cells → should be capped
    n_ctrl = 30
    results = fit_gcate_batch(
        Y, X, A, r=2, batch_size=3, max_cells=max_cells, n_ctrl=n_ctrl,
        family='nb',
        kwargs_es_1=dict(max_iters=2, warmup=0, patience=2),
        kwargs_es_2=dict(max_iters=2, warmup=0, patience=2),
    )
    for br in results:
        n_pert = len(br['pert_idx'])
        assert n_pert <= max_cells, f"Pert cells {n_pert} > max_cells {max_cells}"


def test_fit_gcate_batch_shared_dispersion(small_data):
    """All batches share the same pre-estimated dispersion value (or all None)."""
    Y, X, A = small_data
    results = fit_gcate_batch(
        Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb',
        kwargs_es_1=dict(max_iters=2, warmup=0, patience=2),
        kwargs_es_2=dict(max_iters=2, warmup=0, patience=2),
    )
    disp_first = results[0]['disp_glm']
    for br in results[1:]:
        if disp_first is None:
            assert br['disp_glm'] is None
        else:
            assert br['disp_glm'].shape == (Y.shape[1],)
            np.testing.assert_array_equal(br['disp_glm'], disp_first)


def test_fit_gcate_batch_n_batches(small_data):
    """n_batches overrides batch_size; batches are as even as possible."""
    Y, X, A = small_data
    a_total = A.shape[1]  # number of perturbations
    n_b = 2
    results = fit_gcate_batch(
        Y, X, A, r=2, n_batches=n_b, max_cells=200, n_ctrl=30,
        family='nb',
        kwargs_es_1=dict(max_iters=2, warmup=0, patience=2),
        kwargs_es_2=dict(max_iters=2, warmup=0, patience=2),
    )
    assert len(results) == n_b
    # Batches should be balanced: sizes differ by at most 1
    sizes = [len(br['pert_names']) for br in results]
    assert max(sizes) - min(sizes) <= 1
    # All perturbations covered
    assert sum(sizes) == a_total


def test_fit_gcate_batch_warm_start_U_changes_init(small_data):
    """warm_start_U=True must actually seed ctrl-cell U from the prior batch.

    Regression test for Finding 3 + Finding 23 in the v0.0.6 review:
    previously the warm-start matrix was buried in ``kwargs_ls_1['A']``
    where ``alter_min`` never read it, so this code path was a silent
    no-op despite the docstring promising the opposite.

    We exercise it by running the second batch (a) without and (b) with
    a warm-started ``A_init`` and asserting that the resulting X_U
    matrices DIFFER on the ctrl rows.  If the warm-start were dropped
    on the floor again, both runs would produce bit-identical results
    because ``alter_min`` re-initialises via SVD.
    """
    Y, X, A = small_data
    a_total = A.shape[1]

    # Cold start: batch 0 → batch 1, both initialised from scratch.
    cold = fit_gcate_batch(
        Y, X, A, r=2, n_batches=2, max_cells=200, n_ctrl=30,
        family='nb',
        warm_start_U=False,
        random_state=0,
        kwargs_es_1=dict(max_iters=3, warmup=0, patience=2),
        kwargs_es_2=dict(max_iters=3, warmup=0, patience=2),
    )

    # Warm start: batch 1's U_ctrl rows should be seeded from batch 0.
    warm = fit_gcate_batch(
        Y, X, A, r=2, n_batches=2, max_cells=200, n_ctrl=30,
        family='nb',
        warm_start_U=True,
        random_state=0,
        kwargs_es_1=dict(max_iters=3, warmup=0, patience=2),
        kwargs_es_2=dict(max_iters=3, warmup=0, patience=2),
    )

    # Sanity: same batch layout under the same random_state.
    assert len(cold) == len(warm) == 2
    np.testing.assert_array_equal(cold[1]['ctrl_idx'], warm[1]['ctrl_idx'])

    # Compare the latent block of batch 1's res_1 X_U.  d_b columns of
    # X_U correspond to [X | A_b], and the remaining r columns are U.
    cell_idx = warm[1]['cell_idx']
    ctrl_local = np.searchsorted(cell_idx, warm[1]['ctrl_idx'])
    d_b = np.asarray(X).shape[1] + len(warm[1]['pert_names'])

    U_cold = cold[1]['res_1']['X_U'][ctrl_local, d_b:]
    U_warm = warm[1]['res_1']['X_U'][ctrl_local, d_b:]

    # If warm_start_U were a silent no-op (the pre-fix behaviour),
    # U_cold and U_warm would match to bit precision — they share
    # random_state and every other input.  After the fix they must
    # differ at least somewhere because warm seeds those ctrl rows
    # from batch 0's U_ctrl_prev before stage 1 begins.
    assert not np.allclose(U_cold, U_warm, atol=1e-10), (
        "warm_start_U=True produced an identical X_U to warm_start_U=False; "
        "the warm-start matrix may not be reaching alter_min — the fix for "
        "Finding 3 has regressed."
    )


def test_gcate_lfc_batch_n_batches(small_data):
    """n_batches parameter works end-to-end in gcate_lfc_batch."""
    Y, X, A = small_data
    a_total = A.shape[1]
    n_b = 2
    df = gcate_lfc_batch(
        Y, X, A, r=2, n_batches=n_b, max_cells=200, n_ctrl=30,
        family='nb', gcate_kwargs=_GCATE_KW,
    )
    assert isinstance(df, pd.DataFrame)
    assert df['batch'].nunique() == n_b


# ---------------------------------------------------------------------------
# Tests: gcate_lfc_batch
# ---------------------------------------------------------------------------

_GCATE_KW = dict(
    kwargs_es_1=dict(max_iters=2, warmup=0, patience=2),
    kwargs_es_2=dict(max_iters=2, warmup=0, patience=2),
)


def test_lfc_batch_returns_dataframe(small_data):
    Y, X, A = small_data
    df = gcate_lfc_batch(
        Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb', gcate_kwargs=_GCATE_KW,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_lfc_batch_covers_all_perts(small_data):
    """Result DataFrame should contain rows for all perturbations."""
    Y, X, A_np = small_data
    A_df = pd.DataFrame(A_np, columns=[f'pert_{j}' for j in range(A_np.shape[1])])
    df = gcate_lfc_batch(
        Y, X, A_df, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb', gcate_kwargs=_GCATE_KW,
    )
    assert 'trt' in df.columns
    found_perts = set(df['trt'].unique())
    expected = {f'pert_{j}' for j in range(A_np.shape[1])}
    assert expected == found_perts


def test_lfc_batch_result_columns(small_data):
    """Result DataFrame should have standard LFC output columns."""
    Y, X, A = small_data
    df = gcate_lfc_batch(
        Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb', gcate_kwargs=_GCATE_KW,
    )
    for col in ('tau', 'std', 'log2fc', 'log2fc_se', 'stat', 'pvalue', 'padj', 'batch'):
        assert col in df.columns, f"Missing column: {col}"


def test_lfc_batch_memory_stable(small_data):
    """Peak memory after all batches should not grow linearly with # batches."""
    import tracemalloc
    Y, X, A = small_data

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    gcate_lfc_batch(
        Y, X, A, r=2, batch_size=2, max_cells=200, n_ctrl=30,
        family='nb', gcate_kwargs=_GCATE_KW,
    )
    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    net_mb = sum(stat.size_diff for stat in top_stats) / 1024 ** 2
    assert net_mb < 50, (
        f"Excessive memory net allocation after gcate_lfc_batch: {net_mb:.1f} MB. "
        "Check that large intermediate arrays are freed after each batch."
    )


def test_lfc_batch_cache_path(small_data, tmp_path):
    """cache_path: results are written to HDF5 and resumed correctly."""
    Y, X, A = small_data
    cache = str(tmp_path / 'cache.h5')

    df_full = gcate_lfc_batch(
        Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb', gcate_kwargs=_GCATE_KW, cache_path=cache,
    )

    # Cache file should contain one key per batch plus the schema /meta key
    # (added in the cache-rework fix for Findings 18/19).
    with pd.HDFStore(cache, mode='r') as store:
        keys = store.keys()
    batch_keys = [k for k in keys if k.startswith('/batch_')]
    n_batches = int(np.ceil(A.shape[1] / 3))
    assert len(batch_keys) == n_batches
    assert '/meta' in keys

    # Simulate a pre-0.0.8 cache whose batch frames contain only the original
    # natural-log result columns. Resuming must derive the new aliases without
    # rerunning the completed GCATE batches.
    with pd.HDFStore(cache, mode='a') as store:
        for key in batch_keys:
            legacy = store[key].drop(columns=['log2fc', 'log2fc_se'])
            store.put(key, legacy, format='fixed')
        meta = store['/meta']
        meta.loc[0, 'columns'] = ','.join(legacy.columns.astype(str))
        store.put('/meta', meta, format='fixed')

    # Re-running with same cache_path should skip all batches and return same result
    df_resumed = gcate_lfc_batch(
        Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb', gcate_kwargs=_GCATE_KW, cache_path=cache,
    )
    assert len(df_resumed) == len(df_full)
    assert set(df_resumed.columns) == set(df_full.columns)
    np.testing.assert_allclose(df_resumed['log2fc'], df_resumed['tau'] / np.log(2.0))
    np.testing.assert_allclose(df_resumed['log2fc_se'], df_resumed['std'] / np.log(2.0))



def test_lfc_batch_save_nuisances(small_data, tmp_path):
    """save_nuisances: outcome predictions persist and support re-estimation.

    The outcome model does not depend on the propensity design, so the stored
    ``Y_hat`` must let :func:`LFC` re-estimate under a different propensity
    specification without refitting it.
    """
    import h5py

    from causarray.DR_learner import LFC, _nuisance_path_for

    Y, X, A = small_data
    cache = str(tmp_path / 'cache.h5')

    df_full = gcate_lfc_batch(
        Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
        family='nb', gcate_kwargs=_GCATE_KW, cache_path=cache,
        save_nuisances=True,
    )

    nuisance = _nuisance_path_for(cache)
    n_batches = int(np.ceil(A.shape[1] / 3))
    with h5py.File(nuisance, 'r') as handle:
        assert sorted(handle.keys()) == [f'batch_{i:04d}' for i in range(n_batches)]
        group = handle['batch_0000']
        assert {'Y_hat', 'pi_hat', 'cell_idx', 'offset', 'U'} <= set(group.keys())
        Y_hat = group['Y_hat'][:]
        cell_idx = group['cell_idx'][:]
        offset = group['offset'][:]
        U = group['U'][:]
        pert_names = [n.decode() if isinstance(n, bytes) else n
                      for n in group['pert_names'][:]]

    # Y_hat holds counterfactual predictions per cell, gene, treatment and arm.
    assert Y_hat.shape == (len(cell_idx), Y.shape[1], len(pert_names), 2)
    assert U.shape == (len(cell_idx), 2)

    # The result cache must keep exactly the keys that drive resumption.
    with pd.HDFStore(cache, mode='r') as store:
        assert sorted(k for k in store.keys() if k.startswith('/batch_')) == [
            f'/batch_{i:04d}' for i in range(n_batches)]

    # Re-estimating batch 0 from the stored predictions reproduces its rows.
    # A is a plain array here, so the stored names are its column indices.
    cols = [int(name) for name in pert_names]
    Y_b = np.asarray(Y)[cell_idx]
    A_b = pd.DataFrame(np.asarray(A)[np.ix_(cell_idx, cols)], columns=pert_names)
    W_b = np.c_[np.asarray(X)[cell_idx], U]
    df_reused, _ = LFC(Y_b, W_b, A_b, W_b, family='nb', offset=offset, Y_hat=Y_hat)

    expected = df_full[df_full['batch'] == 0].copy()
    expected['trt'] = expected['trt'].astype(str)
    df_reused['trt'] = df_reused['trt'].astype(str)
    merged = expected.merge(df_reused, on=['gene_names', 'trt'], suffixes=('', '_reused'))
    assert len(merged) == len(expected)
    np.testing.assert_allclose(merged['tau'], merged['tau_reused'], rtol=1e-5, atol=1e-7)


def test_nuisance_store_bytes_counts_every_axis():
    """Y_hat is (n_cells, n_genes, n_treatments, 2) in float32."""
    from causarray.DR_learner import _nuisance_store_bytes

    batches = [{'cell_idx': list(range(100)), 'pert_names': list(range(3))},
               {'cell_idx': list(range(50)), 'pert_names': list(range(2))}]
    assert _nuisance_store_bytes(batches, 10) == 8 * (100 * 10 * 3 + 50 * 10 * 2)
    batches[1]['skipped'] = True          # a skipped batch writes nothing
    assert _nuisance_store_bytes(batches, 10) == 8 * 100 * 10 * 3


def test_warn_nuisance_store_size_only_for_large_stores(tmp_path, recwarn):
    from causarray.DR_learner import NUISANCE_WARN_BYTES, _warn_nuisance_store_size

    path = str(tmp_path / 'cache.nuisances.h5')
    small = [{'cell_idx': list(range(10)), 'pert_names': [0]}]
    _warn_nuisance_store_size(small, 10, path)
    assert not [w for w in recwarn if issubclass(w.category, ResourceWarning)]

    n_cells = int(NUISANCE_WARN_BYTES // (8 * 1000 * 2)) + 1000
    large = [{'cell_idx': list(range(n_cells)), 'pert_names': [0, 1]}]
    with pytest.warns(ResourceWarning, match='save_nuisances will write about'):
        total = _warn_nuisance_store_size(large, 1000, path)
    assert total > NUISANCE_WARN_BYTES


def test_lfc_batch_save_nuisances_requires_cache_path(small_data, monkeypatch):
    """save_nuisances without cache_path is rejected before any fitting."""
    import causarray.gcate as gcate_mod

    def fail(*args, **kwargs):
        raise AssertionError('GCATE ran before the argument check')

    monkeypatch.setattr(gcate_mod, 'fit_gcate_batch', fail)
    Y, X, A = small_data
    with pytest.raises(ValueError, match='requires cache_path'):
        gcate_lfc_batch(
            Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
            family='nb', gcate_kwargs=_GCATE_KW, save_nuisances=True,
        )


def test_lfc_batch_resume_refits_batches_missing_nuisances(small_data, tmp_path):
    """A cached batch without saved nuisances is refitted on resume."""
    import h5py

    from causarray.DR_learner import _nuisance_path_for

    Y, X, A = small_data
    cache = str(tmp_path / 'cache.h5')
    kw = dict(r=2, batch_size=3, max_cells=200, n_ctrl=30, family='nb',
              gcate_kwargs=_GCATE_KW, cache_path=cache)
    df_first = gcate_lfc_batch(Y, X, A, **kw)
    df_resumed = gcate_lfc_batch(Y, X, A, save_nuisances=True, **kw)

    n_batches = int(np.ceil(A.shape[1] / 3))
    with h5py.File(_nuisance_path_for(cache), 'r') as handle:
        assert sorted(handle.keys()) == [f'batch_{i:04d}' for i in range(n_batches)]
    np.testing.assert_allclose(df_first['tau'], df_resumed['tau'])


def test_lfc_batch_deprecation_warning(small_data):
    """LFC_batch should emit DeprecationWarning."""
    import warnings
    Y, X, A = small_data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        LFC_batch(
            Y, X, A, r=2, batch_size=3, max_cells=200, n_ctrl=30,
            family='nb', gcate_kwargs=_GCATE_KW,
        )
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w), \
        "LFC_batch should emit a DeprecationWarning"
