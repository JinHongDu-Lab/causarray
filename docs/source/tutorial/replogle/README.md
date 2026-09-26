# Replogle tutorial directory

Layout convention, shared by the other tutorial folders:

- `N_*.py` — scripts, numbered in the order you run them.
- `*.ipynb` — the tutorial itself, unnumbered; run it after the scripts.
- `data/` — the raw download and the prepared inputs.
- `results/` — everything generated: fits, caches, tables, logs.

Only the notebook, the four scripts, this file and the two small JIC tables in
`results/` are tracked in git; the rest of `data/`
and `results/` is local and regenerated.

## Pipeline

    1_prep_tutorial_data.py        Zenodo -> data/ReplogleWeissman2022_K562_essential.h5ad
                                          -> data/replogle_subset.h5ad

    2_estimate_r.py                data/replogle_subset.h5ad
                                          -> results/replogle-r.csv          (JIC table)

    3_run_batch.py                 data/replogle_subset.h5ad, results/replogle-r.csv
                                          -> results/replogle_results_r{r}.h5   (resumable)
                                             results/replogle_results_r{r}.nuisances.h5

    4_refit_propensity.py          results/replogle_results_r{r}.nuisances.h5
                                          -> results/replogle_results_selected.h5
                                             results/replogle_propensity_selection.csv
                                          copy the .h5 to results/replogle_results.h5 for the notebook

    replogle-py.ipynb              reads the files marked "notebook input" below

## Files

| File | Size | Role |
|---|---|---|
| `replogle-py.ipynb` | 1.1 MB | the tutorial |
| `1_prep_tutorial_data.py` | 8 KB | downloads the screen and builds the subset (200 largest perturbations + 2,000 non-targeting controls, raw counts) |
| `2_estimate_r.py` | 1 KB | JIC rank-selection table on a control-heavy subsample |
| `3_run_batch.py` | 2 KB | batch GCATE + LFC fit, saving the outcome model; resumable |
| `4_refit_propensity.py` | 4 KB | re-estimates LFC with latent factors chosen per arm by `select_propensity_factors`, reusing the saved outcome model |
| `data/ReplogleWeissman2022_K562_essential.h5ad` | 1.4 GB | raw Zenodo download |
| `data/replogle_subset.h5ad` | 2.2 GB | **notebook input** — raw-count subset; source for every other artifact |
| `data/replogle_subset_norm.h5ad` | 2.2 GB | notebook cache — log1p-normalized counts; rebuilt from the subset in one `crispyx` call if deleted |
| `results/replogle_results.h5` | 283 MB | **notebook input** — copy of `results/replogle_results_selected.h5` |
| `results/replogle_results_r10.nuisances.h5` | 50 GB | **notebook input** — outcome model and latent factors per batch; the notebook reads one batch's factors |
| `results/replogle_propensity_selection.csv` | 30 KB | **notebook input** — per-arm propensity report from `4_refit_propensity.py` |
| `results/replogle-r.csv` | 0.6 KB | **notebook input** — JIC table on raw counts (tracked) |
| `results/replogle-r-legacy.csv` | 0.4 KB | **notebook input** — JIC table from the log-normalized era (tracked); the notebook falls back to it only if the current table is absent, and its preprocessing differs |
| `results/replogle_subset_norm_cx_wilcoxon.h5ad` | 92 MB | notebook cache — Wilcoxon comparison results |
| `results/replogle_supt5h_go_results.csv` | 2.6 MB | notebook cache — SUPT5H GO enrichment; regenerate when the input gene lists change |
| `results/run_batch.log` | 11 KB | run record for `3_run_batch.py`; the last line reports rows, discoveries, zero-count arms and runtime |

## Rebuilding from the tracked files alone

Run `1_`, `3_` and `4_`, copy the refit results into place, then execute the
notebook, which rebuilds its own normalization, Wilcoxon and GO caches.
`2_estimate_r.py` only needs re-running if the rank choice is revisited.
