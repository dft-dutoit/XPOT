FAQ
====

**What fitting methods are included in XPOT?**

ACE, SNAP, and GAP potentials are currently implemented in XPOT. Development of inclusion for GNNs is underway.

**What are the best practices for using XPOT?**

XPOT users are recommended to start with pilot runs, to check the hardware limitations of their system, before starting an optimization sweep. XPOT works best with robust testing sets, to accurately measure the extrapolative performance of the potentials fitted.

Users are required to make sure that the parameter space that they optimize over is within the limitations of their hardware. Optimizing convergence parameters over large ranges often results in the most expensive parameters being chosen during optimization. 

The more constrained the search space is, the fewer iterations are *normally* required to reach a stage with diminishing returns.

Several generalised functions are provided for ease of use (conversion between dataset types, extraction of the best potential etc.).

**Generating Datasets for Optimization**

Some ML potentials benefit from different things in datasets. For example, GAP potentials are often fit with dimer data in the dataset, while other methods (such as ACE) may perform better when this data is excluded. It is worth noting that dimer data can skew the `cutoff` hyperparameter if it is optimized. This is because the dimer data will go out to a distance that is often significantly longer than the interactions in the rest of the (bulk) dataset. As such, we recommend not including dimer data in fitting unless you are aware of this effect, and alter the dimer data included *for optimization* to end at the minimum cutoff distance allowed in optimization. 

Additionally, we recommend that users can optimize their potentials on smaller datasets which are representative of their large final datasets. This can significantly reduce the time required for optimization. However, the complexity of the potential will often need to be reduced for smaller datasets, and this should be taken into account when optimization is undertaken.

**How parallelisable is XPOT?**

Currently, a single optimization run occurs "serially". Each fit can be parallelised dependent on the fitting method, but XPOT itself does not support asynchronous optimization. This is a priority and currently in development for future versions.

**Is XPOT compatible with multi-objective optimization?**

Currently, XPOT does not have multi-objective optimization included. In general, we suggest that users select convergence hyperparameters manually, and then optimize the hyperparameters that do not significantly affect the resource cost.

In the future, we aim to include multi-objective optimization in XPOT.

**What** :math:`{\alpha}` **value should I use?**

We recommend either performing a sweep of :math:`{\alpha}` values, or selecting the alpha value based on desired energy and force loss values. By comparing these values, you can find a good :math:`{\alpha}` for your system.