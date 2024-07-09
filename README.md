![xpot-logo](images/xpot-logo.png)

# XPOT: Cross-Platform Hyperparameter Optimizer for Machine Learning Potentials

This software package provides an interface to machine learning (ML) potential fitting methods and allows for the automated optimization of relevant hyperparameters.

The software is described in a [paper in the Special Machine learning edition of JCP](https://pubs.aip.org/aip/jcp/article/159/2/024803/2901815). Please cite this paper when using XPOT in your research.

### Installation Instructions

Only compatible with `python >= 3.10`. Older python version may work, but have not been tested.

```
git clone https://github.com/dft-dutoit/XPOT.git
cd xpot
pip install --upgrade .
```

After this - you must install pacemaker, gap_fit, and/or fitSNAP yourself, for fitting to work properly.

Required Python Packages:
- LAMMPS python interface
- scikit-optimize
- numpy
- hjson
- pandas
- matplotlib
- ase
- tabulate
- quippy-ase >=0.9.0 (GAP potentials only)
- FitSNAP (SNAP potentials only) https://fitsnap.github.io/
- pacemaker (ACE potentials only) https://pacemaker.readthedocs.io/

### Usage

To use XPOT, please check out the example notebooks & python files included in the examples folder. These examples give walkthroughs for using the code. The documentation can be compiled and viewed with sphinx. In the near future, the documentation will be hosted online and linked here. 

### Original Data for JCP 2023 Paper

The current files are updated to be compatible with the latest version of XPOT. The original files for the JCP 2023 paper can be found in the v1.0.0 XPOT-2023 release.
