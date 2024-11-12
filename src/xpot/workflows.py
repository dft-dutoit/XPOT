from __future__ import annotations

import math
import time
from collections.abc import Callable

import xpot.loaders as load
from xpot.models import *
from xpot.optimiser import NamedOptimiser


def optimise(
    mlip: Callable,
    input_file: str,
    kwargs: dict[str, str | int | float],
    max_iter: int = 32,
    min_loss: float = 0,
    time_limit: float = math.inf,
) -> None:
    """
    Single function to run full optimization of an MLIP. This function contains
    the full workflow detailed in the XPOT documentation notebooks for
    optimizing MLIPs.

    Parameters
    ----------
    mlip : object
        The MLIP class to optimize
    input_file : str
        Path to the input file containing hyperparameters for XPOT and MLIP
    kwargs : dict
        Dictionary of scikit-optimize keyword arguments to pass to the
        NamedOptimiser class
    max_iter : int, optional
        Maximum number of iterations to run, by default 32
    min_loss : float, optional
        Minimum loss to stop optimization, by default 0 (disabled)
    time_limit : float, optional
        Time limit for optimization in minutes, by default 0 (disabled)

    Returns
    -------
    None
    """
    mlip_obj = mlip(input_file)

    opt = NamedOptimiser(
        mlip_obj.optimisation_space,
        mlip_obj.sweep_path,
        kwargs,
    )
    start = time.time()

    # convert time to seconds
    time_limit = time_limit * 60
    loss = math.inf
    elapsed = 0

    while opt.iter <= max_iter or loss >= min_loss or elapsed <= time_limit:
        # run optimization iteration and return loss
        loss = opt.run_optimisation(mlip_obj.fit, path=mlip_obj.sweep_path)

        # update elapsed time
        elapsed = time.time() - start

    opt.tabulate_final_results(mlip_obj.sweep_path)

    print(
        "Optimization Finalized. Results saved in"
        + mlip_obj.sweep_path
        + ". Exiting..."
    )


def single_fit(
    mlip: Callable, input_file: str, autoplex: bool = False
) -> None | tuple[float, float, float]:
    """
    Single function to run a single fit of an MLIP. This function contains
    the full workflow detailed in the XPOT documentation notebooks for
    optimizing MLIPs. This workflow is also autoplex compatible, enabling
    the PACE fitting process.

    Parameters
    ----------
    mlip : object
        The MLIP class to optimize
    input_file : str
        Path to the input file containing hyperparameters for XPOT and MLIP

    Returns
    -------
    None | tuple
        If autoplex is True, returns a tuple containing the loss, train E error,
        and test E error. These errors are returned depending on the metric
        requested during teh fitting. If autoplex is False, returns None.
    """
    mlip_obj = mlip(input_file)
    load.initialise_csvs(
        mlip_obj.sweep_path, mlip_obj.optimisation_space.keys()
    )
    path = mlip_obj.sweep_path
    loss = mlip_obj.fit(mlip_obj.optimisation_space, 0)

    print(
        "Optimization Finalized. Results saved in"
        + mlip_obj.sweep_path
        + ". Exiting..."
    )
    if autoplex:
        output = load.autoplex_return(path, single=True, single_loss=loss)
        return output
