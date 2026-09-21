"""Stage 3b: Replogle tutorial batched GCATE + LFC on the raw-count subset with
the 0.0.10 defaults. ``r`` is the JIC choice from ``run_estimate_r_0.0.10.py``
(``replogle-r-0.0.10.csv``; r = 10 on the raw counts, 2026-09-21) unless given
on the command line. Writes ``replogle_results_r{r}_0.0.10.h5`` (resumable
cache); copy it to ``replogle_results.h5`` for the notebook."""
import sys, time
sys.path.insert(0, '../../../..')
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from causarray import prep_causarray_data, gcate_lfc_batch
import causarray; assert causarray.__version__ == '0.0.10'
if len(sys.argv) > 1:
    R = int(sys.argv[1])
else:
    df_r = pd.read_csv('replogle-r-0.0.10.csv')
    R = int(df_r.loc[df_r['JIC'].idxmin(), 'r'])
print(f'r = {R}', flush=True)
t0 = time.perf_counter()
adata = ad.read_h5ad('replogle_subset.h5ad')
Y = pd.DataFrame(adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X), columns=adata.var_names.tolist())
assert np.allclose(Y.to_numpy()[:2000], np.round(Y.to_numpy()[:2000])), 'subset must hold integer counts'
A = pd.get_dummies(adata.obs[adata.uns['pert_col']].astype(str), drop_first=False).drop(columns=[adata.uns['ctrl_label']])
Y, A, X, X_A = prep_causarray_data(Y, A)
df_res = gcate_lfc_batch(
    Y, X, A, R, W_A=X_A, batch_size=15, max_cells=2000, n_ctrl=2000, family='nb',
    cache_path=f'replogle_results_r{R}_0.0.10.h5', random_state=0, verbose=True,
    gcate_kwargs=dict(kwargs_es_1=dict(rel_tol=2e-4, max_iters=30), kwargs_es_2=dict(rel_tol=2e-4, max_iters=30)),
)
sig = df_res[df_res.padj < 0.05]
print(f'DONE in {(time.perf_counter()-t0)/60:.1f} min: {len(df_res):,} rows, {len(sig):,} discoveries, '
      f'zero-arm={int((sig.count_treated == 0).sum())}, neg frac={(sig.tau < 0).mean():.2f}', flush=True)
