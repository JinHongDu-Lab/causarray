[![Documentation Status](https://readthedocs.org/projects/causarray/badge/?version=latest)](https://causarray.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/causarray?label=pypi)](https://pypi.org/project/causarray)
[![PyPI-Downloads](https://img.shields.io/pepy/dt/causarray)](https://pepy.tech/project/causarray)


# causarray

Advances in single-cell sequencing and CRISPR technologies have enabled detailed case-control comparisons and experimental perturbations at single-cell resolution. However, uncovering causal relationships in observational genomic data remains challenging due to selection bias and inadequate adjustment for unmeasured confounders, particularly in heterogeneous datasets. To address these challenges, we introduce `causarray` [Du26], a doubly robust causal inference framework for analyzing array-based genomic data at both bulk-cell and single-cell levels. `causarray` integrates a generalized confounder adjustment method to account for unmeasured confounders and employs semiparametric inference with ﬂexible machine learning techniques to ensure robust statistical estimation of treatment effects.


## Usage

We recommend using `causarray` in a conda environment:
```cmd
# create a new conda environment and install the necessary packages
conda create -n causarray python=3.12 -y

# activate the environment
conda activate causarray
```

On an Apple Silicon Mac, make sure the environment is native: `python -c "import
platform; print(platform.machine())"` should print `arm64`. A conda started from
a terminal running under Rosetta creates Intel (`x86_64`) environments, in which
causarray runs several times slower; `causarray` warns about this at import.
Create a native one with `CONDA_SUBDIR=osx-arm64 conda create -n causarray
python=3.12 -y`, then `conda config --env --set subdir osx-arm64` inside it.

The module can be installed via PyPI:
```cmd
pip install causarray
```

For optimal parallel performance, we recommend installing `llvm-openmp` if using conda:
```cmd
conda install -c conda-forge llvm-openmp
```

For `R` users, `reticulate` can be used to call `causarray` from R while
keeping NumPy >2 in the Python environment. Create the separate R environment
with a current NumPy-2-compatible reticulate build:

```bash
conda env create -f environment-r.yaml
```

The R tutorial runs from `causarray-r` and connects to the Python package in
the `causarray` environment.
The documentation and tutorials using both `Python` and `R` are available at [causarray.readthedocs.io](https://causarray.readthedocs.io/en/latest/).



## Tutorials

| Tutorial | Language | Description | Link |
|----------|----------|-------------|------|
| Perturb-seq [Jin20] | Python | CRISPR screen analysis on excitatory neurons | [Notebook](https://causarray.readthedocs.io/en/latest/tutorial/perturbseq/perturbseq-py.html) |
| Perturb-seq [Jin20] | R | Same analysis using `reticulate` | [Notebook](https://causarray.readthedocs.io/en/latest/tutorial/perturbseq/perturbseq-r.html) |
| Genome-wide CRISPRi screen [Replogle22] | Python | Batch fitting on 200 perturbations from a K562 genome-wide CRISPRi screen | [Notebook](https://causarray.readthedocs.io/en/latest/tutorial/replogle/replogle-py.html) |
| Case-control: SEA-AD [Gabitto24] | Python | Causal inference on observational single-cell data (Alzheimer's disease) | [Notebook](https://causarray.readthedocs.io/en/latest/tutorial/case_control/sea_ad_case_control.html) |

### Batch fitting API

For screens with hundreds to thousands of perturbations, use `gcate_lfc_batch` so
that peak memory is bounded by one batch at a time:

```python
from causarray import gcate_lfc_batch

df_res = gcate_lfc_batch(
    Y, X, A, r,
    batch_size=10,    # perturbations per batch (or use n_batches= for a fixed count)
    max_cells=2000,   # max pert cells per batch (ctrl added on top)
    n_ctrl=2000,      # fixed ctrl subsample shared across batches
    cache_path='results.h5',   # resume if interrupted
    verbose=True,
)
```

See the [Replogle-E-K562 tutorial](https://causarray.readthedocs.io/en/latest/tutorial/replogle/replogle-py.html)
for a demonstration on 200 perturbations from a genome-wide CRISPRi screen.

### Diagnostic masks without refitting effects

Treatment-by-gene support or quality-control rules can be aligned to an
existing causarray result table by label, even when its rows are reordered:

```python
from causarray import align_test_mask

keep = align_test_mask(
    df_res,
    support_mask,                 # treatments × genes, Boolean
    treatment_names=perturbations,
    gene_names=genes,
)
df_res_flagged = df_res.assign(support_keep=keep)
```

This operation only annotates or subsets existing results. It does not refit
the causarray LFC or change standard errors and p-values. If a diagnostic rule
was selected after inspecting the outcomes, retain the original adjusted
p-values rather than redefining the multiple-testing family post hoc. The
[Replogle tutorial](https://causarray.readthedocs.io/en/latest/tutorial/replogle/replogle-py.html)
compares several expression-support rules with marginal Wilcoxon results.

## Changelog

See [CHANGELOG](https://causarray.readthedocs.io/en/latest/changelog.html) for a full version history.


<!-- 
# Development

The dependencies for running `causarray` method are listed in `environment.yml` and can be installed by running

```cmd
PIP_NO_DEPS=1 conda env create -f environment.yml
```


## Build
```cmd
git tag 0.0.0
git tag --delete 1.0.0
python -m pip install .
```

## Testing
```cmd
python -m pytest tests/test_gcate.py
python -m pytest tests/test_DR_learner.py
```

## Documentation

```cmd
mkdir docs
cd docs
sphinx-quickstart

make html # sphinx-build source build


rmarkdown::render("perturbseq.Rmd", rmarkdown::md_document(variant = "markdown_github"))
```
-->


## References
[Du26] Jin-Hong Du, Maya Shen, Hansruedi Mathys, and Kathryn Roeder. "Uncovering causal relationships in single cell omic studies with causarray". In: Briefings in Bioinformatics (2026).

[Gabitto24] Mariano I. Gabitto et al. "Integrated multimodal cell atlas of Alzheimer's disease". In: Nature Neuroscience (2024).

[Jin20] Xin Jin et al. "In vivo Perturb-seq reveals neuronal and glial abnormalities associated with autism risk genes". In: Nature Neuroscience (2020).

[Replogle22] Joseph M. Replogle et al. "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq". In: Cell (2022).
