"""Re-estimate LFC with a propensity model chosen per arm.

Guides are assigned at random, so the propensity model only needs to absorb
chance imbalance. Log library size is left out: knocking down transcription
machinery shrinks a cell's total RNA, so library size mostly records the
perturbation's own effect. The model starts from the intercept and all latent
factors; ``select_propensity_factors`` then drops, for each arm whose
weights concentrate (treated ESS < 0.5, overlap < 0.3 or AUC > 0.9), the
factors most imbalanced between that arm and the controls until its support
recovers.

The outcome model does not depend on the propensity design, so this reuses
the saved ``Y_hat`` from ``3_run_batch.py`` and refits nothing else.

    python 4_refit_propensity.py
"""
import os, sys, time
sys.path.insert(0, '../../../..')
import h5py
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from causarray import (prep_causarray_data, LFC, estimate_propensity_scores,
                       refit_propensity_scores, select_propensity_factors,
                       summarize_propensity_scores)
import causarray

R = 10
NUISANCE = f'results/replogle_results_r{R}.nuisances.h5'
OUT = 'results/replogle_results_selected.h5'
REPORT = 'results/replogle_propensity_selection.csv'
t0 = time.perf_counter()
log = lambda *a: print(f'[{(time.perf_counter()-t0)/60:6.1f} min]', *a, flush=True)

if not os.path.exists(NUISANCE):
    raise SystemExit(f'{NUISANCE} not found; run 3_run_batch.py first')
for path in (OUT, REPORT):
    if os.path.exists(path):
        os.remove(path)

adata = ad.read_h5ad('data/replogle_subset.h5ad')
pert_col, ctrl = adata.uns['pert_col'], adata.uns['ctrl_label']
Y = pd.DataFrame(adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X),
                 columns=adata.var_names.tolist())
A = pd.get_dummies(adata.obs[pert_col].astype(str), drop_first=False).drop(columns=[ctrl])
Y, A, X, X_A = prep_causarray_data(Y, A)
del adata
col_names = list(A.columns)
log(f'{Y.shape[0]:,} cells x {Y.shape[1]:,} genes, {A.shape[1]} perturbations')

with h5py.File(NUISANCE, 'r') as handle:
    batch_keys = sorted(handle.keys())

reports = []
for batch_i, key in enumerate(batch_keys):
    with h5py.File(NUISANCE, 'r') as handle:
        g = handle[key]
        Y_hat = g['Y_hat'][:]; cell_idx = g['cell_idx'][:]
        offset = g['offset'][:]; U = g['U'][:]
        names = [n.decode() if isinstance(n, bytes) else n for n in g['pert_names'][:]]
    cols = [col_names.index(n) for n in names]
    Y_b = Y.iloc[cell_idx].reset_index(drop=True)
    A_b = A.iloc[cell_idx, cols].reset_index(drop=True)
    W_b = np.c_[X[cell_idx], U]
    W_A_b = np.c_[X_A[cell_idx, :1], U]          # intercept + latent factors; no library size
    covariates = ['intercept'] + [f'U{k + 1}' for k in range(U.shape[1])]
    A_np = A_b.to_numpy(dtype=float)
    fit = dict(K=1, C=1.0, class_weight=None, clip=None, random_state=0)
    pi_all = estimate_propensity_scores(A_np, W_A_b, **fit)
    drops, _ = select_propensity_factors(A_np, W_A_b, treatment_names=names,
                                         covariate_names=covariates)
    pi, _ = refit_propensity_scores(A_np, W_A_b, pi_hat=pi_all.copy(), treatment_names=names,
                                    covariate_names=covariates, drop_by_treatment=drops, **fit)
    before, after = (summarize_propensity_scores(A_np, p, treatment_names=names, clip_bounds=None)
                     .set_index('treatment') for p in (pi_all, pi))
    report = pd.DataFrame({
        'batch': key, 'flagged': [n in drops for n in names],
        'dropped': [','.join(drops.get(n, [])) for n in names],
        'auc_before': before.loc[names, 'auc'].to_numpy(),
        'auc_after': after.loc[names, 'auc'].to_numpy(),
        'overlap_before': before.loc[names, 'overlap_ratio'].to_numpy(),
        'overlap_after': after.loc[names, 'overlap_ratio'].to_numpy(),
        'ess_treated_before': before.loc[names, 'ess_treated_fraction'].to_numpy(),
        'ess_treated_after': after.loc[names, 'ess_treated_fraction'].to_numpy(),
    }, index=pd.Index(names, name='treatment'))
    reports.append(report)

    df_b, _ = LFC(Y_b, W_b, A_b, W_A_b, family='nb', offset=offset,
                  Y_hat=Y_hat, pi_hat=pi, verbose=False)
    df_b['batch'] = batch_i
    with pd.HDFStore(OUT, mode='a') as store:
        if '/meta' not in store.keys():
            store.put('/meta', pd.DataFrame({
                'causarray_version': [causarray.__version__], 'family': ['nb'],
                'a_total': [int(A.shape[1])],
                'columns': [','.join(df_b.columns.astype(str))]}), format='fixed')
        store.put(f'batch_{batch_i:04d}', df_b, format='fixed')
    log(f'{key}: {int(report.flagged.sum())}/{len(names)} arms flagged, '
        f'{int((df_b.padj < 0.05).sum()):,} discoveries')
    del Y_hat, Y_b, A_b, df_b

rep = pd.concat(reports)
rep.to_csv(REPORT)
with pd.HDFStore(OUT, mode='r') as store:
    total = pd.concat([store[k] for k in store.keys() if k.startswith('/batch_')])
sig = total[total.padj < 0.05]
log(f'DONE: {len(total):,} rows, {len(sig):,} discoveries, '
    f'zero-arm={int((sig.count_treated == 0).sum())}, neg frac={(sig.tau < 0).mean():.2f}')
log(f'{int(rep.flagged.sum())}/{len(rep)} arms flagged -> {OUT}, {REPORT}')
