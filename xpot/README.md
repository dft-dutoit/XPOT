# Fitting Potentials with XPOT

To fit potentials with XPOT, input files for your dataset and parameters are required. In this document, we walk through the required input files, and how to optimize hyperparameters using XPOT.

## Dataset 
Your dataset must be in a file format that is supported by the fitting method that you are using. For GAP & SNAP, this is possible using `.xyz` files. SNAP .xyz files must use the keywords "forces" and "energy" for the forces and cell energy respectively.

For GAP, the training data must be merged into a single file, for SNAP potentials we recommend splitting the data based on structure type, to allow easy optimization of structure weights in training. 

## Input File (.hjson)

The parameter input file is in `hjson` format. This file includes all hyperparameters for XPOT to parse when optimizing/fitting potentials. 

Using scikit's notation, in order to provide optimization for a parameter, you must use one of the following methods:

```
skopt.space.Integer(low, high)
skopt.space.Real(low, high)
skopt.space.Categorical([list])
```

These tell XPOT that you want to optimize this parameter. All other parameters should be entered as desired for fitting, as with a normal single fit in GAP/SNAP. 

The .hjson file provides improved exception handling for missing commas and other formatting oversights, to reduce the likelihood of user error resulting in failure to run, as well as providing the ability to add comments to each line. 

## Python Script

There are examples of python run scripts in the [examples](examples) section of this directory. These should be used as described in the main documentation to fit potentials with:

```
python run_script.py input.hjson
```

To submit this job to a node on HPCs, simply run the python file within your .sh script.

## Parsing Previous Results

To parse previous run results there is a function included in XPOT. The following script works well:

```
from xpot.transfer_funcs.general import parse_previous_sweep
from xpot.transfer_funcs.optimizers import sweep_init_gp

# Parse previous results
results = parse_previous_sweep('previous_sweep_folder')

# Initialize new sweep
kwargs = {
    "n_calls" : 41,
    "n_initial_points" : 6,
    "initial_point_generator" : "hammersly",
    "n_jobs" : 1,
    "x0":results[0], # Parameters from the previous sweep.
    "y0":results[1] # Loss values from the previous sweep.
}
sweep_init_gp(method, **kwargs)
```