# XPOT - Cross-platform optimizer for machine-learning interatomic potentials

This software package provides an interface to machine learning (ML) potential fitting methods and allows for the automated optimization of relevant hyperparameters.

### Installation Instructions

Required Software:
- LAMMPS (compiled with packages required for potential styles)
- Python >= 3.8
- cmake >= 3

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

1. Create a virtual environment, and insure that cmake is installed.

2. pip install the required python packages, before adding FitSNAP and LAMMPS. Specific instructions for these packages can be found in their respective documentation pages, listed above. A requirements.txt file is provided for all other packages for ease of installation.

3. Clone this github repository to a location of your choice.

4. place the following into your .bashrc. `export PYTHONPATH="XPOT_FOLDER_PATH":$PYTHONPATH`.

5. Try `import xpot` in python. 

### Usage

To import all modules required, please run the following code at the top of your python script. This will make sure that you can call optimization of any potential style.
```
from xpot.transfer_funcs.optimizers import *
from xpot.transfer_funcs.general import *
```

In running xpot, input files are specified as follows:
```
python your-script.py input_file.hjson
```
Make sure that the input script is accessible on the node that you submit to if using xpot on a cluster.

There are three examples to ensure that your xpot installation is working correctly. These are located in the examples folder. To run these examples, please check the README.md files in each folder.

## FAQ

### **What fitting methods are included in xpot?**

Currently SNAP (including qSNAP and pytorch potentials with SNAP descriptors) and GAP are available. ACE potentials via pacemaker are in active development, and will be included in the next release.

Development for expansion to other methods will follow.

### __What are the best practices for using xpot?__

xpot users are recommended to start with pilot runs, to check the hardware limitations of thier system, before starting an optimization sweep. xpot works best with robust testing sets, to accurately measure the extrapolative performance of the potentials fitted.

Users are required to make sure that the parameter space that they optimize over is within the limitations of their hardware. Optimizing convergence parameters over large ranges often results in the most expensive parameters being chosen during optimization. 

The more constrained the search space is, the fewer iterations are *generally* required to reach a stage with diminishing returns.

Several functions are provided for ease of use (conversion between dataset types, extraction of the best potential etc.)

### **How parallelisable is xpot?**

Currently, a single optimization run occurs "serially". Each fit can be parallelised dependent on the fitting method, but xpot itself does not support asynchronous optimization. This is a priority and currently in development for future versions.

### **Is xpot compatible with multi-objective optimization?**

Currently, xpot does not have multi-objective optimization included. In general, we suggest that users select convergence hyperparameters manually, and then optimize the hyperparameters that do not significantly affect the resource cost.

In the future, we aim to include multi-objective optimization in XPOT.

### **What $\alpha$ value should I use?**

We recommend either performing a sweep of $\alpha$ values, or selecting the alpha value based on desired energy and force loss values. By comparing these values, you can find a good $\alpha$ for your system.
