# Mouse-brain Perturb-seq (L4-5 IT Glut)

A causarray 0.1.1 tutorial on one cell type of a mouse-brain CRISPR screen.
Only the two notebooks, `pipeline.py` and this file are tracked in git; the
notebooks are rendered in the documentation but not listed in its index. `data/`
holds inputs and `results/` everything generated; both are local.

    L45Glut-py.ipynb              the tutorial: pipeline, propensity and calibration
                                  checks, biological readouts, Wilcoxon reference, and
                                  the loop for all cell types (min_counts = 20)
    L45Glut-mc40-py.ipynb         the same analysis with min_counts = 40
    pipeline.py                   the per-cell-type pipeline the notebook calls; also a CLI:
                                    python pipeline.py lfc  --tag L45Glut --batches 3
                                    python pipeline.py null --tag L45Glut --seeds 3,4,5

The input `data/L45Glut.h5ad` is a subset of collaborator-provided pilot data
and is not distributed with the repository.

Results for one cell type live in `results/<tag>/`:

    gcate.pkl, lfc.csv, lfc_batches/, wilcoxon.h5ad, null/     (pipeline outputs)
    lfc_batches/*_propensity.csv  arms the propensity check adjusted (L45Glut: Tpr)
    min_counts5/                  the same run with the package default min_counts=5,
                                  used in the notebook's section 4
    min_counts40/                 the run with min_counts=40, read by L45Glut-mc40-py.ipynb:
                                    python pipeline.py lfc  --tag L45Glut --min-counts 40 --variant min_counts40
                                    python pipeline.py null --tag L45Glut --min-counts 40 --variant min_counts40 --seeds 3,4,5
    enrichr/                      cached Enrichr queries

Memory: one LFC batch (15k controls + 10 perturbations, 14k genes) peaks near
90 GB. Three batches at a time fit on a 512 GB machine; nine did not.

Threads: when running several jobs at once, cap each so that together they do
not exceed the cores (3 jobs on 32 cores: `export NUMBA_NUM_THREADS=10
OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10`). Uncapped, the jobs slow each other
down about tenfold. With the cap, a batch takes about 5 minutes and GCATE about
34 minutes (in a native arm64 Python).
