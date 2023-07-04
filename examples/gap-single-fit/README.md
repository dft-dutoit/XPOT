# GAP Fitting with XPOT

XPOT can be used to either undertake single GAP fitting, or optimize parameters of the GAP fitting process. 

As included in [ref] we recommend fixing the convergence parameters when undertaking significant optimization. XPOT can also be used to explore the effects of increasing convergence parameters, but we recommend doing this through a series of single fits, or by specifying `x0` vales to `scikit-optimize` via the `kwargs` dictionary in the python script. 

These input scripts are based on the GAP-17 dataset from Deringer et al. <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.094203>

