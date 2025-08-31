![xpot-logo](images/xpot-logo.png)

# XPOT: Cross-Platform Hyperparameter Optimizer for Machine Learning Potentials

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15853808.svg)](https://doi.org/10.5281/zenodo.15853808)

This software package provides an interface to machine learning (ML) potential fitting methods and allows for the automated optimization of relevant hyperparameters.

**XPOT** was originally developed by [Daniel Thomas du Toit](https://github.com/dft-dutoit) during his DPhil in the Deringer group (2021-2025).  
Continued development will be carried out by the wider Deringer group!

## 🚀 Quick Start

Only compatible with `python >= 3.10`. Older python version may work, but have not been tested.

```bash
git clone https://github.com/dft-dutoit/XPOT.git
cd xpot
pip install --upgrade .
```

After installation you must install your desired fitting software yourself for fitting to work.

There are example scripts available in the `examples` and `data` directories!


## 📖 How to Cite XPOT

If you use XPOT in your research, please cite the following works:  

- **XPOT-ACE**: [https://doi.org/10.1021/acs.jctc.4c01012](https://doi.org/10.1021/acs.jctc.4c01012)  
- **Original XPOT paper**: [https://doi.org/10.1063/5.0166765](https://doi.org/10.1063/5.0166765)  

<details>
<summary>📑 Show BibTeX entries</summary>

```bibtex
@article{ThomasduToit-JCTC-24,
  author    = {Thomas du Toit, Daniel F. and Zhou, Yuxing and Deringer, Volker L.},
  title     = {Cross-platform hyperparameter optimization for transferable atomistic machine learning potentials},
  journal   = {Journal of Chemical Theory and Computation},
  year      = {2024},
  volume    = {20},
  issue    = {22},
  pages     = {10103-10113},
  doi       = {10.1021/acs.jctc.4c01012}
}

@article{ThomasduToit-JCP-23,
  author    = {Thomas du Toit, Daniel F. and Deringer, Volker L.},
  title     = {Cross-platform hyperparameter optimization for transferable atomistic machine learning potentials},
  journal   = {Journal of Chemical Physics},
  year      = {2023},
  volume    = {159},
  number    = {2},
  pages     = {024803},
  doi       = {10.1063/5.0166765}
}
```
</details>

## Features

- ACE, SNAP, and GAP hyperparameter optimisation
- Summary tables for all hyperparameters tested
- Resumeable optimisation runs using previous files
- New architectures and integrations under development!

