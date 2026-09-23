# Replogle tutorial directory

Layout convention, shared by the other tutorial folders:

- `N_*.py` — scripts, numbered in the order you run them.
- `*.ipynb` — the tutorial itself, unnumbered; run it after the scripts.
- `data/` — the raw download and the prepared inputs.
- `results/` — everything generated: fits, caches, tables, logs.

Only the notebook, the four scripts, this file and the two small JIC tables in
`results/` are tracked in git; the rest of `data/` and `results/` is local and
regenerated.

## Pipeline

    1_prep_tutorial_data.py        Zenodo -> data/ReplogleWeissman2022_K562_essential.h5ad
                                          -> data/replogle_subset.h5ad

    2_estimate_r.py                data/replogle_subset.h5ad
                                          -> results/replogle-r.csv          (JIC table)

    3_run_batch.py                 data/replogle_subset.h5ad, results/replogle-r.csv
                                          -> results/replogle_results_r{r}.h5   (resumable)
                                          copy to results/replogle_results.h5 for the notebook

    4_cache_propensity_batch.py    data/replogle_subset.h5ad, results/replogle-r.csv
                                   stage 1 -> results/replogle_propensity_batch12.npz
                                              results/replogle_propensity_batch12_baseline.csv.gz
                                   stage 2 -> results/replogle_propensity_batch12_summary.csv
                                              results/replogle_propensity_batch12_tuning.csv
                                              results/replogle_propensity_batch12_selected_scores.npz

    replogle-py.ipynb              reads the files marked "notebook input" below

## Files

| File | Size | Role |
|---|---|---|
| `replogle-py.ipynb` | 1.1 MB | the tutorial |
| `1_prep_tutorial_data.py` | 8 KB | downloads the screen and builds the subset (200 largest perturbations + 2,000 non-targeting controls, raw counts) |
| `2_estimate_r.py` | 1 KB | JIC rank-selection table on a control-heavy subsample |
| `3_run_batch.py` | 2 KB | batch GCATE + LFC fit; resumable |
| `4_cache_propensity_batch.py` | 14 KB | propensity-score sensitivity sweep, two stages |
| `data/ReplogleWeissman2022_K562_essential.h5ad` | 1.4 GB | raw Zenodo download |
| `data/replogle_subset.h5ad` | 2.2 GB | **notebook input** — raw-count subset; source for every other artifact |
| `data/replogle_subset_norm.h5ad` | 2.2 GB | notebook cache — log1p-normalized counts; rebuilt from the subset in one `crispyx` call if deleted |
| `results/replogle_results.h5` | 283 MB | **notebook input** — batch fit results |
| `results/replogle-r.csv` | 0.6 KB | **notebook input** — JIC table on raw counts (tracked) |
| `results/replogle-r-legacy.csv` | 0.4 KB | **notebook input** — JIC table from the log-normalized era (tracked); the notebook falls back to it only if the current table is absent, and its preprocessing differs |
| `results/replogle_subset_norm_cx_wilcoxon.h5ad` | 92 MB | notebook cache — Wilcoxon comparison results |
| `results/replogle_supt5h_go_results.csv` | 2.6 MB | notebook cache — SUPT5H GO enrichment; regenerate when the input gene lists change |
| `results/replogle_propensity_batch12_summary.csv` | 23 KB | **notebook input** — propensity sweep summary |
| `results/replogle_propensity_batch12_tuning.csv` | 11 KB | **notebook input** — propensity tuning grid |
| `results/replogle_propensity_batch12_selected_scores.npz` | 281 KB | **notebook input** — the selected propensity scores |
| `results/replogle_propensity_batch12.npz` | 1.9 GB | stage-1 intermediate of `4_cache_propensity_batch.py`; **not read by the notebook**. Keep it only to re-run the stage-2 sweep without redoing stage 1 |
| `results/replogle_propensity_batch12_baseline.csv.gz` | 12.6 MB | stage-1 intermediate, as above; not read by the notebook |
| `results/run_batch.log` | 11 KB | run record for `results/replogle_results.h5` (progress bars stripped); the last line reports rows, discoveries, zero-count arms and runtime |

## Rebuilding from the tracked files alone

Run `1_`, then `3_` and `4_` (hours each), then execute the notebook, which
rebuilds its own normalization, Wilcoxon and GO caches. `2_estimate_r.py` only
needs re-running if the rank choice is revisited.
