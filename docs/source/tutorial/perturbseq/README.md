# Perturb-seq tutorial (Jin et al. 2020 subset)

Layout: `data/` holds prepared inputs, `results/` holds everything generated.
There is no prep script — the subset is committed directly.

    data/perturbseq-exneu.h5ad   excitatory-neuron count subset (Python tutorial)
    data/perturbseq-exneu.rds    the same subset for the R tutorial
    results/perturbseq-r.csv     JIC rank-selection table, read by both tutorials

    perturbseq-py.ipynb          Python tutorial
    perturbseq-r.Rmd             R tutorial (same workflow via reticulate); knit to perturbseq-r.md
    perturbseq-r_files/          figures produced by knitting the Rmd

Everything here is tracked in git. Both tutorials are published in
`docs/source/index.rst`.
