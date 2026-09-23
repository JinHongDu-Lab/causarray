"""Stage 3c: re-estimate LFC with a targeted penalty on log library size.

``prep_causarray_data`` appends standardized log library size to ``X_A``.  For
some perturbations it differs so strongly between the perturbed and control
cells that the propensity model separates them almost perfectly, leaving the
weights on a handful of cells.  Measured library size mixes capture depth with
total RNA content and these data do not identify which drives a given arm's
shift, so rather than dropping the covariate everywhere this keeps it and
penalizes *only that coefficient*, and only for arms that fail a pre-specified
support check:

    trigger   treated ESS < 0.5 or AUC > 0.9
    factor    ``tune_penalty_factor``: the smallest factor bringing the arm's
              histogram overlap above 0.3, found by bisection between the
              unpenalized fit and the dropped-covariate limit.  An arm whose
              dropped fit cannot reach the target is marked infeasible and given
              the largest factor, the closest support the covariate allows.

``penalty_factors_by_treatment`` leaves every other coefficient, and every
unflagged arm, exactly as fitted.  The outcome model does not depend on the
propensity design, so this reuses the cached ``Y_hat`` from ``3_run_batch.py``
(``save_nuisances=True``) and refits nothing.

    python 5_refit_propensity.py
"""
import os, sys, time
sys.path.insert(0, '../../../..')
import h5py
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from causarray import (prep_causarray_data, estimate_propensity_scores,
                       refit_propensity_scores, summarize_propensity_scores,
                       tune_penalty_factor, LFC)
import causarray

R = 10
TRIGGER = {'auc_gt': 0.9, 'ess_treated_fraction_lt': 0.5}
TARGET = {'overlap_ratio_gt': 0.3}   # histogram overlap, the quantity being repaired
FOCUS = ('SUPT5H', 'SUPT6H', 'SRRT', 'TSR2')   # arms whose before/after scores are saved
NUISANCE = f'results/replogle_results_r{R}.nuisances.h5'
OUT = 'results/replogle_results_penalized.h5'
REPORT = 'results/replogle_propensity_penalties.csv'
SCORES = 'results/replogle_penalized_scores.npz'
t0 = time.perf_counter()
log = lambda *a: print(f'[{(time.perf_counter()-t0)/60:6.1f} min]', *a, flush=True)

if not os.path.exists(NUISANCE):
    raise SystemExit(f'{NUISANCE} not found; run 3_run_batch.py with save_nuisances=True')
for path in (OUT, REPORT, SCORES):
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

records = []
focus_scores = {}
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
    W_A_b = np.c_[X_A[cell_idx], U]          # library size stays in the design
    cov = ['intercept', 'log_library_size'] + [f'U{k+1}' for k in range(U.shape[1])]

    A_np = A_b.to_numpy(dtype=float)
    pi = estimate_propensity_scores(A_np, W_A_b, K=1, C=1.0, class_weight=None,
                                    clip=None, random_state=0)
    before = summarize_propensity_scores(A_np, pi, treatment_names=names,
                                         clip_bounds=None).set_index('treatment')
    pi_base_saved = pi.copy()
    chosen, tuning = tune_penalty_factor(
        A_np, W_A_b, 'log_library_size', treatment_names=names, covariate_names=cov,
        trigger=TRIGGER, target=TARGET, K=1, C=1.0, class_weight=None, random_state=0)
    if chosen:
        pi, _ = refit_propensity_scores(
            A_np, W_A_b, pi_hat=pi.copy(), treatment_names=names, covariate_names=cov,
            penalty_factors_by_treatment=chosen,
            K=1, C=1.0, class_weight=None, clip=None, random_state=0)
    after = summarize_propensity_scores(A_np, pi, treatment_names=names,
                                        clip_bounds=None).set_index('treatment')
    for name in (set(names) & set(FOCUS)):
        j = names.index(name)
        focus_scores[f'{name}__before'] = pi_base_saved[:, j].astype(np.float32)
        focus_scores[f'{name}__after'] = pi[:, j].astype(np.float32)
        focus_scores[f'{name}__treated'] = A_np[:, j].astype(np.uint8)
        focus_scores[f'{name}__control'] = (A_np.sum(axis=1) == 0).astype(np.uint8)
    tuned = tuning.set_index('treatment') if len(tuning) else None
    for name in names:
        rec = {'batch': key, 'treatment': name,
               'penalty_factor': chosen.get(name, {}).get('log_library_size', 1.0),
               'triggered': tuned is not None and name in tuned.index,
               'feasible': bool(tuned.loc[name, 'feasible']) if tuned is not None and name in tuned.index else True,
               'auc_before': before.loc[name, 'auc'], 'auc_after': after.loc[name, 'auc'],
               'overlap_before': before.loc[name, 'overlap_ratio'],
               'overlap_after': after.loc[name, 'overlap_ratio'],
               'ess_treated_before': before.loc[name, 'ess_treated_fraction'],
               'ess_treated_after': after.loc[name, 'ess_treated_fraction']}
        if tuned is not None and name in tuned.index:
            rec['auc_dropped'] = tuned.loc[name, 'auc_dropped']
            rec['n_fits'] = int(tuned.loc[name, 'n_fits'])
        records.append(rec)

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
    log(f'{key}: {len(chosen)}/{len(names)} penalized, '
        f'{int((df_b.padj < 0.05).sum()):,} discoveries')
    del Y_hat, Y_b, A_b, df_b

pd.DataFrame(records).to_csv(REPORT, index=False)
np.savez_compressed(SCORES, **focus_scores)
with pd.HDFStore(OUT, mode='r') as store:
    total = pd.concat([store[k] for k in store.keys() if k.startswith('/batch_')])
sig = total[total.padj < 0.05]
rep = pd.DataFrame(records)
log(f'DONE: {len(total):,} rows, {len(sig):,} discoveries, '
    f'zero-arm={int((sig.count_treated == 0).sum())}, neg frac={(sig.tau < 0).mean():.2f}')
log(f'{int((rep.penalty_factor > 1).sum())}/{len(rep)} arms penalized -> {OUT}, {REPORT}')
