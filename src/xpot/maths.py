from __future__ import annotations

import numpy as np


def get_rmse(
    list_in: list[float | int],
) -> float:
    """
    Get the root mean squared error of a list of floats or integers. Used in
    producing error metrics from ML potential fitting method.

    Parameters
    ----------
    list_in : list
        List of floats or integers.

    Returns
    -------
    float
        The root mean squared error, as a float.
    """
    return np.sqrt(np.mean([i**2 for i in list_in]))


def get_mae(
    list_in: list[float | int],
) -> float:
    """
    Get the mean absolute error of a list of floats or integers. Used in
    producing error metrics from ML potential fitting method.

    Parameters
    ----------
    list_in : list
        List of floats or integers.

    Returns
    -------
    float
        The mean absolute error, as a float.
    """
    return np.mean(np.abs(list_in))


def scale_list_values(
    list_in: list[float | int],
    scale_list: list[float | int],
    n_exponent: float = 1,
) -> list[float]:
    """
    Scale a list of floats or integers by a given factor.

    Parameters
    ----------
    list_in : list
        List of floats or integers.
    scale_list : list
        Number of atoms for each list element.
    n_exponent : float, optional
        Exponent for scaling by number of atoms, by default 1.

    Returns
    -------
    list
        The scaled list.
    """
    return [i / (j**n_exponent) for i, j in zip(list_in, scale_list)]
