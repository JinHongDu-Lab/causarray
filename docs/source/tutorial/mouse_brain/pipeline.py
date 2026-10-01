"""Per-cell-type causarray + Wilcoxon pipeline for the mouse-brain Perturb-seq screen.

Every step caches its output under ``results/<tag>/`` and is skipped when the
cache exists, so a crashed or interrupted run resumes where it stopped. The
notebook ``L45Glut-py.ipynb`` walks through these steps for one cell type; the
same calls run unchanged for every other cell type.

    results/<tag>/gcate.pkl           latent factors (one GCATE fit per cell type)
    results/<tag>/lfc_batches/*.csv   causarray, ~10 perturbations per batch
    results/<tag>/lfc_batches/*_propensity.csv   arms the propensity check adjusted (if any)
    results/<tag>/lfc.csv             causarray, all perturbations
    results/<tag>/wilcoxon.h5ad       crispyx Wilcoxon, all perturbations
    results/<tag>/nbglm.h5ad, ttest.h5ad, deseq2.csv   other methods for comparison
    results/<tag>/null/seed*.csv      causarray on fake perturbations of controls
    results/<tag>/null/wilcoxon_seed*.h5ad   Wilcoxon on the same fake labels

Command line (run from this folder)::

    python pipeline.py subset --src <combined.h5ad> --cell-type "005 L4-5 IT CTX Glut" --tag L45Glut
    python pipeline.py lfc  --tag L45Glut [--batches 0,1,2]
    python pipeline.py null --tag L45Glut [--seeds 0,1,2]

When several jobs run at once, cap each one's threads so that together they
do not exceed the cores (numba and BLAS otherwise start one thread per core in
every job, and the jobs slow each other down about tenfold)::

    export NUMBA_NUM_THREADS=10 OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import crispyx
from causarray import (LFC, fit_gcate, prep_causarray_data, estimate_propensity_scores,
                       refit_propensity_scores, select_propensity_factors)

HERE = Path(__file__).parent.resolve()
DATA_DIR = HERE / 'data'
RESULTS_DIR = HERE / 'results'

PERT_COL = 'gene_target'
CTRL_LABEL = 'Non_target'
CELL_TYPE_COL = 'predicted_group'
MIN_CELLS = 50       # keep genes detected in at least this many cells
RANK = 8             # latent factors; chosen by JIC on L4-5 IT Glut, fixed for all cell types
BATCH_SIZE = 10      # perturbations per LFC call; peak memory ~ n_cells * n_genes * BATCH_SIZE * 16 bytes
Q = 0.05             # FDR level for both methods
MIN_COUNTS = 20      # LFC tests a gene in an arm only if (higher of the two arm means, in
                     # counts per cell) x (cells in the smaller arm) >= 20, i.e. about 20 UMI
                     # counts expected in the smaller arm (package default 5); chosen from
                     # the null check in the notebook


def paths(tag, variant=''):
    """File layout for one cell type.

    ``variant`` puts the causarray outputs of a sensitivity run (e.g. another
    ``min_counts``) in a subfolder; latent factors and Wilcoxon are shared.
    """
    out = RESULTS_DIR / tag
    ca = out / variant if variant else out
    ca.mkdir(parents=True, exist_ok=True)
    return {
        'counts': DATA_DIR / f'{tag}.h5ad',
        'norm': DATA_DIR / f'{tag}_norm.h5ad',
        'gcate': out / 'gcate.pkl',
        'batches': ca / 'lfc_batches',
        'lfc': ca / 'lfc.csv',
        'wilcoxon': out / 'wilcoxon.h5ad',
        'nbglm': out / 'nbglm.h5ad',
        'ttest': out / 'ttest.h5ad',
        'deseq2': out / 'deseq2.csv',
        'null': ca / 'null',
        'null_wilcoxon': out / 'null',
    }


# ── Step 1: one cell type ────────────────────────────────────────────────────
def subset_cell_type(src, cell_type, tag):
    """Write the cells of one cell type from ``src`` to ``data/<tag>.h5ad``."""
    p = paths(tag)
    if not p['counts'].exists():
        obs = crispyx.load_obs(src)
        cell_mask = (obs[CELL_TYPE_COL].astype(str) == cell_type).to_numpy()
        gene_mask = np.ones(ad.read_h5ad(src, backed='r').n_vars, dtype=bool)
        crispyx.write_filtered_subset(src, cell_mask=cell_mask, gene_mask=gene_mask,
                                      output_path=p['counts'])
    return p['counts']


def load_inputs(tag):
    """Counts and designs for causarray, restricted to genes seen in >= MIN_CELLS cells."""
    adata = ad.read_h5ad(paths(tag)['counts'])
    counts = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    keep = np.asarray((counts > 0).sum(0)).ravel() >= MIN_CELLS
    adata = adata[:, keep].copy()
    X_raw = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    Y = pd.DataFrame(X_raw, columns=adata.var_names.tolist())
    A = pd.get_dummies(adata.obs[PERT_COL].astype(str)).drop(columns=[CTRL_LABEL])
    Y, A, X, X_A = prep_causarray_data(Y, A)
    return Y, A, X, X_A


# ── Step 2: latent factors ───────────────────────────────────────────────────
def fit_factors(tag, Y, X, A, r=RANK):
    """One GCATE fit per cell type, shared by every perturbation batch."""
    path = paths(tag)['gcate']
    if path.exists():
        with open(path, 'rb') as f:
            res_2 = pickle.load(f)['res_2']
        if res_2['U'].shape[1] == r:
            return res_2
    res_1, res_2 = fit_gcate(Y, X, A, r, backend='fast', verbose=False,
                             kwargs_es_1=dict(rel_tol=2e-4, max_iters=10),
                             kwargs_es_2=dict(rel_tol=2e-4, max_iters=10))
    with open(path, 'wb') as f:
        pickle.dump({'res_1': res_1, 'res_2': res_2}, f)
    return res_2


def _propensity(A, W_A):
    """Propensity support check; returns (pi_hat or None, report).

    The design holds the intercept, log library size and the latent factors.
    Library size is a technical factor; the outcome model adjusts every arm for
    it through the size-factor offset, whatever the propensity design.
    ``select_propensity_factors`` flags an arm whose treated ESS falls below
    0.5, overlap below 0.3 or AUC above 0.9, and drops from that arm's
    propensity model only the covariate most imbalanced between it and the
    controls (library size included), one at a time, until support recovers.
    With nothing flagged LFC fits its own scores (None is returned).
    """
    names = list(A.columns)
    cov = ['intercept', 'log_library_size'] + [f'U{k + 1}' for k in range(W_A.shape[1] - 2)]
    A_np = A.to_numpy(dtype=float)
    drops, report = select_propensity_factors(A_np, W_A, treatment_names=names,
                                              covariate_names=cov)
    if not drops:
        return None, report
    pi = estimate_propensity_scores(A_np, W_A, K=1, random_state=0)
    pi, _ = refit_propensity_scores(A_np, W_A, pi_hat=pi, treatment_names=names,
                                    covariate_names=cov, drop_by_treatment=drops,
                                    K=1, random_state=0)
    return pi, report


def _lfc(Y, W, A, W_A, offset, report_path=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pi, report = _propensity(A, W_A)
        if report_path is not None and len(report):
            report.to_csv(report_path, index=False)
        df, _ = LFC(Y, W, A, W_A, offset=offset, pi_hat=pi, min_counts=MIN_COUNTS,
                    backend='fast', verbose=False)
    return df


# ── Step 3: causarray LFC in perturbation batches ────────────────────────────
def run_lfc(tag, Y, A, X, X_A, res_2, batches=None, batch_size=BATCH_SIZE, variant=''):
    """Run LFC on all controls + ``batch_size`` perturbations at a time.

    ``batches`` restricts the run to some batch indices (to spread one cell
    type over several processes); the combined table is written once every
    batch is on disk.
    """
    p = paths(tag, variant)
    if p['lfc'].exists():
        return pd.read_csv(p['lfc'])
    p['batches'].mkdir(exist_ok=True)
    U = res_2['U']
    offset = np.log(res_2['kwargs_glm']['size_factor'])
    W, W_A = np.c_[X, U], np.c_[X_A, U]
    A_np, Y_np = A.to_numpy(dtype=float), Y.to_numpy()
    names = list(A.columns)
    is_ctrl = A_np.sum(1) == 0
    splits = np.array_split(np.arange(len(names)), int(np.ceil(len(names) / batch_size)))
    todo = range(len(splits)) if batches is None else batches
    for b in todo:
        out = p['batches'] / f'batch_{b:03d}.csv'
        if out.exists():
            continue
        cols = splits[b]
        cells = is_ctrl | (A_np[:, cols].sum(1) > 0)
        df_b = _lfc(pd.DataFrame(Y_np[cells], columns=Y.columns), W[cells],
                    pd.DataFrame(A_np[np.ix_(cells, cols)], columns=[names[c] for c in cols]),
                    W_A[cells], offset[cells],
                    report_path=p['batches'] / f'batch_{b:03d}_propensity.csv')
        df_b.to_csv(out, index=False)
        del df_b
        gc.collect()
    files = [p['batches'] / f'batch_{b:03d}.csv' for b in range(len(splits))]  # results only
    if not all(f.exists() for f in files):
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(p['lfc'], index=False)
    return df


# ── Step 4: Wilcoxon ────────────────────────────────────────────────────────
def wilcoxon_table(h5ad):
    """Long table (trt, gene_names, wilcox_logfc, wilcox_pvalue, wilcox_padj) from a crispyx result.

    The log-fold change is the ``logfoldchanges`` layer (log2). The result's
    ``effect_size`` field is a rank statistic, not a fold change.
    """
    res = ad.read_h5ad(h5ad)
    out = []
    for layer, name in (('logfoldchanges', 'wilcox_logfc'), ('pvalue', 'wilcox_pvalue'),
                        ('pvalue_adj', 'wilcox_padj')):
        m = pd.DataFrame(res.layers[layer], index=res.obs_names, columns=res.var_names)
        out.append(m.stack().rename(name))
    df = pd.concat(out, axis=1).reset_index()
    return df.rename(columns={df.columns[0]: 'trt', df.columns[1]: 'gene_names'})


def wilcoxon_on(df, df_wc):
    """Wilcoxon results on the gene-perturbation pairs causarray tests.

    Keeps the pairs of ``df`` with a finite statistic, attaches the Wilcoxon
    columns and recomputes ``wilcox_padj`` by BH within each perturbation over
    those pairs, the same family causarray's ``padj`` uses. crispyx's own
    adjusted p-values are taken over every gene in the file, including genes
    seen in one or two cells. For those the tie-corrected normal approximation
    breaks down: a gene in one perturbed cell and no control gets
    z = sqrt(n_control / n_treated), about 15 here (p ~ 1e-46), where an exact
    rank test gives p of order n_treated / n_cells. Those genes pass BH and
    loosen the cutoff for the rest of the arm.
    """
    from statsmodels.stats.multitest import multipletests

    m = df[np.isfinite(df['stat'])].merge(df_wc, on=['trt', 'gene_names'], how='left')
    m['wilcox_padj'] = m.groupby('trt')['wilcox_pvalue'].transform(
        lambda p: pd.Series(multipletests(p.fillna(1.0), method='fdr_bh')[1], index=p.index))
    return m


def run_wilcoxon(tag):
    p = paths(tag)
    if not p['wilcoxon'].exists():
        if not p['norm'].exists():
            crispyx.normalize_total_log1p(p['counts'], output_path=p['norm'], verbose=False)
        crispyx.wilcoxon_test(p['norm'], perturbation_column=PERT_COL, control_label=CTRL_LABEL,
                              verbose=False, output_path=p['wilcoxon'])
    return wilcoxon_table(p['wilcoxon'])


# ── Step 4b: other methods for comparison ───────────────────────────────────
# Each runs once per cell type on the genes load_inputs keeps (seen in >= MIN_CELLS
# cells) and, like Wilcoxon, is shared by causarray runs with any min_counts.
def _kept_input(src, dst, counts_path):
    """Write ``src`` restricted to the genes load_inputs keeps; crispyx reads from file."""
    counts = ad.read_h5ad(counts_path)
    keep = np.asarray((counts.X > 0).sum(0)).ravel() >= MIN_CELLS
    ad.read_h5ad(src)[:, keep].copy().write_h5ad(dst)
    return keep


def _de_table(h5ad, name, lfc_layer='logfoldchanges', lfc_scale=1.0):
    """Long table (trt, gene_names, <name>_logfc in log2, <name>_pvalue) from a crispyx result."""
    res = ad.read_h5ad(h5ad)
    cols = [pd.DataFrame(np.asarray(res.layers[lfc_layer]) * lfc_scale, index=res.obs_names,
                         columns=res.var_names).stack().rename(f'{name}_logfc'),
            pd.DataFrame(np.asarray(res.layers['pvalue']), index=res.obs_names,
                         columns=res.var_names).stack().rename(f'{name}_pvalue')]
    df = pd.concat(cols, axis=1).reset_index()
    return df.rename(columns={df.columns[0]: 'trt', df.columns[1]: 'gene_names'})


def run_nb_glm(tag, res_2):
    """Negative-binomial GLM per perturbation (crispyx), with causarray's size factors and
    no latent factors or weights: causarray's count model without its adjustment."""
    p = paths(tag)
    if not p['nbglm'].exists():
        tmp = p['nbglm'].with_name('nbglm_input.h5ad')
        _kept_input(p['counts'], tmp, p['counts'])
        sf = np.asarray(res_2['kwargs_glm']['size_factor']).ravel()
        crispyx.nb_glm_test(str(tmp), perturbation_column=PERT_COL, control_label=CTRL_LABEL,
                            size_factors=sf, min_mu=1e-8, min_pct_ctrl=0, min_pct_pert=0,
                            min_mean_ctrl=0, min_mean_pert=0, min_total_count=0,
                            min_cells_ctrl=0, min_cells_pert=0, output_path=p['nbglm'],
                            verbose=False, n_jobs=8)
        tmp.unlink()
    return _de_table(p['nbglm'], 'nbglm', 'logfoldchange_raw_ln', 1 / np.log(2))


def run_ttest(tag):
    """Welch t-test per perturbation on log-normalized counts (crispyx)."""
    p = paths(tag)
    if not p['ttest'].exists():
        if not p['norm'].exists():
            crispyx.normalize_total_log1p(p['counts'], output_path=p['norm'], verbose=False)
        tmp = p['ttest'].with_name('ttest_input.h5ad')
        _kept_input(p['norm'], tmp, p['counts'])
        crispyx.t_test(str(tmp), perturbation_column=PERT_COL, control_label=CTRL_LABEL,
                       min_pct_ctrl=0, min_pct_pert=0, min_mean_ctrl=0, min_mean_pert=0,
                       output_path=p['ttest'], verbose=False, n_jobs=8)
        tmp.unlink()
    return _de_table(p['ttest'], 'ttest')


def run_deseq2(tag):
    """PyDESeq2: one model ~ perturbation on all cells, controls as reference, defaults except
    poscounts size factors (the default ratio method needs genes without zeros). About 25
    minutes on L4-5 IT Glut; needs ``pip install pydeseq2`` to recompute."""
    p = paths(tag)
    if not p['deseq2'].exists():
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats
        except ImportError as e:
            raise ImportError('run_deseq2 needs pydeseq2 (pip install pydeseq2) to compute '
                              f"{p['deseq2'].name}") from e
        a = ad.read_h5ad(p['counts'])
        keep = np.asarray((a.X > 0).sum(0)).ravel() >= MIN_CELLS
        a = a[:, keep]
        X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        counts = pd.DataFrame(np.rint(X).astype(int), index=a.obs_names.astype(str),
                              columns=a.var_names.astype(str))
        # factor levels without underscores; 'control' is the reference
        meta = pd.DataFrame({'perturbation': a.obs[PERT_COL].astype(str).str.replace('_', '-')
                             .replace({CTRL_LABEL.replace('_', '-'): 'control'}).to_numpy()},
                            index=counts.index)
        dds = DeseqDataSet(counts=counts, metadata=meta, design='~perturbation',
                           ref_level=['perturbation', 'control'], size_factors_fit_type='poscounts',
                           n_cpus=16, quiet=True)
        dds.deseq2()
        rows = []
        for trt in sorted(set(meta['perturbation']) - {'control'}):
            st = DeseqStats(dds, contrast=['perturbation', trt, 'control'], n_cpus=16, quiet=True)
            st.summary()
            r = st.results_df[['baseMean', 'log2FoldChange', 'pvalue', 'padj']].copy()
            r['trt'] = trt; r.index.name = 'gene_names'
            rows.append(r.reset_index())
        pd.concat(rows, ignore_index=True).to_csv(p['deseq2'], index=False)
    d = pd.read_csv(p['deseq2'])
    return d.rename(columns={'log2FoldChange': 'deseq2_logfc', 'pvalue': 'deseq2_pvalue'})[
        ['trt', 'gene_names', 'deseq2_logfc', 'deseq2_pvalue']]


# Method name -> (log fold change column, adjusted p-value column) in the methods_on table.
METHODS = {
    'causarray': ('tau', 'padj'),
    'NB GLM, no latent factors': ('nbglm_logfc', 'nbglm_padj'),
    'DESeq2': ('deseq2_logfc', 'deseq2_padj'),
    'Wilcoxon': ('wilcox_logfc', 'wilcox_padj'),
    't-test, log-normalized': ('ttest_logfc', 'ttest_padj'),
}


def methods_on(df, tag, res_2):
    """causarray's tested pairs with every method's log fold change and p-value attached.

    As in ``wilcoxon_on``, each method's p-values are BH-adjusted within each
    perturbation over the pairs causarray tests, so all methods share one family.
    """
    from statsmodels.stats.multitest import multipletests

    m = wilcoxon_on(df, run_wilcoxon(tag))
    for name, table in (('nbglm', run_nb_glm(tag, res_2)), ('deseq2', run_deseq2(tag)),
                        ('ttest', run_ttest(tag))):
        m = m.merge(table, on=['trt', 'gene_names'], how='left')
        m[f'{name}_padj'] = m.groupby('trt')[f'{name}_pvalue'].transform(
            lambda p: pd.Series(multipletests(p.fillna(1.0), method='fdr_bh')[1], index=p.index))
    return m


# ── Step 5: null calibration on controls ─────────────────────────────────────
def null_labels(A, seed, n_arms=10):
    """Fake perturbation labels for control cells only.

    Arm sizes are drawn from the real perturbation sizes, so the fake arms are
    as small as the real ones. Returns a label per control cell (the control
    label for cells left in the control pool).
    """
    A_np = A.to_numpy(dtype=float)
    n_ctrl = int((A_np.sum(1) == 0).sum())
    rng = np.random.default_rng(seed)
    sizes = rng.choice(A_np.sum(0).astype(int), size=n_arms, replace=False)
    labels = np.full(n_ctrl, CTRL_LABEL, dtype=object)
    perm = rng.permutation(n_ctrl)
    start = 0
    for k, n in enumerate(sizes):
        labels[perm[start:start + n]] = f'fake{seed}_{k:02d}'
        start += n
    return labels


def run_null(tag, Y, A, X, X_A, res_2, seed, variant=''):
    """causarray and Wilcoxon on fake perturbations of control cells.

    Nothing was perturbed, so every discovery is a false one.
    """
    p = paths(tag, variant)
    p['null'].mkdir(exist_ok=True)
    p['null_wilcoxon'].mkdir(exist_ok=True)
    out_ca = p['null'] / f'seed{seed}.csv'
    out_wc = p['null_wilcoxon'] / f'wilcoxon_seed{seed}.h5ad'
    ctrl = (A.to_numpy().sum(1) == 0)
    labels = null_labels(A, seed)
    if not out_ca.exists():
        U = res_2['U'][ctrl]
        offset = np.log(res_2['kwargs_glm']['size_factor'])[ctrl]
        A_null = pd.get_dummies(pd.Series(labels)).drop(columns=[CTRL_LABEL]).astype(float)
        df = _lfc(Y.loc[ctrl].reset_index(drop=True), np.c_[X[ctrl], U], A_null,
                  np.c_[X_A[ctrl], U], offset,
                  report_path=p['null'] / f'seed{seed}_propensity.csv')
        df.to_csv(out_ca, index=False)
    if not out_wc.exists():
        run_wilcoxon(tag)  # makes sure the normalized file exists
        norm = ad.read_h5ad(paths(tag)['norm'])
        norm = norm[(norm.obs[PERT_COL].astype(str) == CTRL_LABEL).to_numpy()].copy()
        norm.obs['null_label'] = pd.Categorical(labels)
        tmp = p['null_wilcoxon'] / f'norm_seed{seed}.h5ad'
        norm.write_h5ad(tmp)
        crispyx.wilcoxon_test(tmp, perturbation_column='null_label', control_label=CTRL_LABEL,
                              verbose=False, output_path=out_wc)
        tmp.unlink()
    return pd.read_csv(out_ca), wilcoxon_table(out_wc)


# ── Step 6: one-line QC per cell type ────────────────────────────────────────
def summarize(tag, df, df_wc, nulls=()):
    """Numbers to scan across cell types; see the notebook for expected values."""
    m = wilcoxon_on(df, df_wc)
    ca = m['padj'] < Q
    wc = m['wilcox_padj'] < Q
    row = {
        'cell type': tag,
        'perturbations': df['trt'].nunique(),
        'causarray hits': int(ca.sum()),
        'Wilcoxon hits': int(wc.sum()),
        'Wilcoxon hits recovered': round((ca & wc).sum() / max(wc.sum(), 1), 2),
        'fraction down': round((m.loc[ca, 'tau'] < 0).mean(), 2),
        'hits with zero perturbed counts': int((m.loc[ca, 'count_treated'] == 0).sum()),
    }
    if len(nulls):
        # Both methods are scored, and BH-adjusted, on the genes causarray tests.
        n = pd.concat([wilcoxon_on(d, w) for d, w in nulls])
        row['fake perturbations: causarray hits'] = int((n['padj'] < Q).sum())
        row['fake perturbations: Wilcoxon hits'] = int((n['wilcox_padj'] < Q).sum())
        row['fake perturbations: SD of z (target 1)'] = round(n['stat'].std(), 2)
    return row


def main():
    global MIN_COUNTS
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['subset', 'lfc', 'null'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--src')
    ap.add_argument('--cell-type')
    ap.add_argument('--batches', help='comma-separated batch indices (default: all)')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--min-counts', type=float, default=MIN_COUNTS)
    ap.add_argument('--variant', default='', help='subfolder for a sensitivity run')
    args = ap.parse_args()
    MIN_COUNTS = args.min_counts
    if args.step == 'subset':
        subset_cell_type(args.src, args.cell_type, args.tag)
        return
    Y, A, X, X_A = load_inputs(args.tag)
    res_2 = fit_factors(args.tag, Y, X, A)
    if args.step == 'lfc':
        batches = None if args.batches is None else [int(b) for b in args.batches.split(',')]
        run_lfc(args.tag, Y, A, X, X_A, res_2, batches=batches, variant=args.variant)
    else:
        for s in args.seeds.split(','):
            run_null(args.tag, Y, A, X, X_A, res_2, int(s), variant=args.variant)


if __name__ == '__main__':
    main()
