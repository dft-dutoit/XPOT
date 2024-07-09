FAQ
====

**What fitting methods are included in XPOT?**

ACE, SNAP, and GAP potentials are currently implemented in XPOT. Development for expansion to other methods will follow.

**What are the best practices for using XPOT?**

XPOT users are recommended to start with pilot runs, to check the hardware limitations of their system, before starting an optimization sweep. XPOT works best with robust testing sets, to accurately measure the extrapolative performance of the potentials fitted.

Users are required to make sure that the parameter space that they optimize over is within the limitations of their hardware. Optimizing convergence parameters over large ranges often results in the most expensive parameters being chosen during optimization. 

The more constrained the search space is, the fewer iterations are *normally* required to reach a stage with diminishing returns.

Several generalised functions are provided for ease of use (conversion between dataset types, extraction of the best potential etc.).

**How parallelisable is XPOT?**

Currently, a single optimization run occurs "serially". Each fit can be parallelised dependent on the fitting method, but XPOT itself does not support asynchronous optimization. This is a priority and currently in development for future versions.

**Is XPOT compatible with multi-objective optimization?**

Currently, XPOT does not have multi-objective optimization included. In general, we suggest that users select convergence hyperparameters manually, and then optimize the hyperparameters that do not significantly affect the resource cost.

In the future, we aim to include multi-objective optimization in XPOT.

**What** :math:`{\alpha}` **value should I use?**

We recommend either performing a sweep of :math:`{\alpha}` values, or selecting the alpha value based on desired energy and force loss values. By comparing these values, you can find a good :math:`{\alpha}` for your system.