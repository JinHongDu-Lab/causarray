"""Re-select the number of latent factors for the Replogle tutorial on the
raw-count subset (the cached replogle-r.csv was computed on log-normalised
values). Same grid as the tutorial; writes replogle-r-0.0.10.csv."""
import sys, time
sys.path.insert(0, '../../../..')
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from causarray import prep_causarray_data, estimate_r
t0 = time.perf_counter()
adata = ad.read_h5ad('replogle_subset.h5ad')
Y = pd.DataFrame(adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X), columns=adata.var_names.tolist())
A = pd.get_dummies(adata.obs[adata.uns['pert_col']].astype(str), drop_first=False).drop(columns=[adata.uns['ctrl_label']])
Y, A, X, X_A = prep_causarray_data(Y, A)
df_r = estimate_r(Y, X, A, [0, 5, 10, 15, 20, 25, 30], family='nb', max_cells=6000, backend='fast',
                  kwargs_es_1=dict(rel_tol=2e-4, max_iters=30), kwargs_es_2=dict(rel_tol=2e-4, max_iters=30))
df_r.to_csv('replogle-r-0.0.10.csv', index=False)
print(df_r.to_string(index=False))
print(f"DONE in {(time.perf_counter()-t0)/60:.1f} min; selected r = {int(df_r.loc[df_r['JIC'].idxmin(), 'r'])}", flush=True)
