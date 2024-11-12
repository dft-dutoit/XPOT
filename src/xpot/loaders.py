from __future__ import annotations

import csv
import json
import os
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Union  # noqa: UP035

import hjson
import numpy as np
import skopt
from ase import Atoms
from ase.io import iread

# DictValue = str | list[str | int] | int | float
# NestedDict = dict[str, "NestedDict" | DictValue]

DictValue = str | list[str | int | float] | int | float
NestedDict = dict[str, Union["NestedDict", Any]]


def get_defaults(path: str) -> NestedDict:
    """
    Get the default parameters from a json file.

    Parameters
    ----------
    path : str
        Path to the json file.

    Returns
    -------
    dict
        The default parameters in dictionary format.
    """
    with open(path) as f:
        return json.load(f)


def get_input_hypers(path: str) -> NestedDict:
    """
    Get the input hyperparameters from a hjson file.

    Parameters
    ----------
    path : str
        Path to the json file.

    Returns
    -------
    dict
        The input hyperparameters in dictionary format.
    """
    with open(path) as f:
        # Used to convert from OrderedDict to dict.
        return json.loads(json.dumps(hjson.load(f)))


def merge_hypers(
    defaults: NestedDict,
    inputs: NestedDict,
) -> NestedDict:
    """
    Merge the default parameters and input hyperparameters.

    Parameters
    ----------
    defaults : dict
        The default parameters in dictionary format.
    inputs : dict
        The input hyperparameters in dictionary format.

    Returns
    -------
    dict
        The merged parameters in dictionary format, with input hyperparameters
        overriding the defaults initially loaded.
    """
    return {**defaults, **inputs}


def recursive_items(
    d: dict[str, dict | str],
    previous_keys: list[str] | None = None,
) -> Iterator[tuple[tuple[str, ...], str]]:
    """
    Iterate over a dictionary recursively, yielding a
    tuple of nested keys and a value.
    """

    if previous_keys is None:
        previous_keys = []

    for key, value in d.items():
        all_keys = previous_keys + [key]
        if isinstance(value, dict):
            yield from recursive_items(value, all_keys)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                yield (tuple(all_keys + [str(i)]), v)
        else:
            yield (tuple(all_keys), value)


def validate_subdict(
    parent: NestedDict,
    child: NestedDict,
) -> bool:
    """
    Validate that the child dictionary is a subset of the parent dictionary.
    Used to validate that the input hyperparameters have been correctly loaded
    over the default parameters.

    Parameters
    ----------
    parent : dict
        The parent dictionary.
    child : dict
        The child dictionary.

    Returns
    -------
    bool
        True if the child dictionary is a subset of the parent dictionary.
    """
    return all([parent[k] == child[k] for k in child])


def validate_hypers(
    merged_hypers: NestedDict,
    input_hypers: NestedDict,
) -> bool:
    """
    Validate the hyperparameters.

    Parameters
    ----------
    merged_hypers : dict
        The merged dictionary of defaults and hypers.
    input_hypers : dict
        A dictionary of a subset of values (normally input hypers).

    Raises
    ------
    ValueError
        If the input hyperparameters are not a subset of the merged parameters.
    """
    if validate_subdict(merged_hypers, input_hypers):
        return True
    else:
        raise ValueError(
            "Input hyperparameters are not a subset of the merged parameters."
        )


def get_optimisable_params(
    hypers: NestedDict,
) -> dict[tuple[str, ...], skopt.space.Dimension]:
    """
    Extract the optimisable parameters from the hyperparameters.

    Parameters
    ----------
    hypers : dict
        The hyperparameters in dictionary format.

    Returns
    -------
    dict
        The optimisable parameters in dictionary format.
    """
    # list_values = []
    opt_values = {}
    for kv_pair in recursive_items(hypers):
        # if isinstance(kv_pair[-1], list):
        #     list_values.append(kv_pair[-1])
        if isinstance(kv_pair[-1], str) and "skopt" in kv_pair[-1]:
            # v_trimmed = kv_pair[-1].replace("skopt.space.", "")
            opt_values[kv_pair[:-1]] = eval(kv_pair[-1])
        else:
            continue
    return opt_values


def trim_empty_values(
    dict_in: NestedDict,
) -> dict[str, str | int | float | list]:
    """
    Remove empty values from a dictionary.

    Parameters
    ----------
    dict_in : dict
        The input dictionary.

    Returns
    -------
    dict
        The dictionary with empty values removed.
    """
    if not isinstance(dict_in, dict):
        return dict_in
    return {
        key: v
        for key, value in dict_in.items()
        if (v := trim_empty_values(value)) not in ("", {})
    }  # type: ignore


def _create_dir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str
        The path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.exists(path) and not os.listdir(path):
        warnings.warn(
            f"{path} already exists, but is empty. Continuing...", stacklevel=2
        )
    else:
        raise FileExistsError(
            f"{path} already exists and contains files, please "
            + "remove all files before continuing."
        )


def directory_setup(path: str) -> None:
    """
    Set up the directory for the potential fitting.

    Parameters
    ----------
    path : str
        The path to the directory.
    """
    _create_dir(path)
    os.chdir(path)
    print(f"Directory setup for {path} complete. Continuing...")


def reconstitute_lists(hypers: NestedDict, opt_values: dict) -> NestedDict:
    """
    Reconstitute lists from separated keys sorted by index in the
    hyperparameter dictionary.

    Parameters
    ----------
    hypers : dict
        The hyperparameters in dictionary format, with lists separated into
        key per index.
    opt_values : dict
        The optimiser values in dictionary format, with tuple keys representing
        the nested keys of the hyperparameters in the dictionary.

    Returns
    -------
    dict
        The hyperparameters dictionary with lists reconstituted into single
        key: value pairs.
    """
    opt_keys = {x[0]: x for x in opt_values}
    a = list(recursive_items(hypers))
    for tuple_kv_pair in a:
        if tuple_kv_pair[0] in opt_keys:
            b = ""
            for i in range(len(tuple_kv_pair[0])):
                if tuple_kv_pair[0][i].isdigit():
                    b += "[" + tuple_kv_pair[0][i] + "]"
                else:
                    b += "['" + tuple_kv_pair[0][i] + "']"
            exec(f"hypers{b} = opt_values[opt_keys[tuple_kv_pair[0]]]")
        else:
            continue

    return hypers


def prep_dict_for_dump(d: dict) -> dict:
    """
    Prepare a dictionary for dumping to a file by removing numpy int and float
    types.

    Parameters
    ----------
    d : dict
        The nested dictionary to be dumped.

    Returns
    -------
    dict
        The dictionary with numpy types converted to native types.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            prep_dict_for_dump(v)
        elif isinstance(v, np.int64):  # type: ignore
            d[k] = int(v)
        elif isinstance(v, np.float64):  # type: ignore
            d[k] = float(v)
    return d


def list_to_string(d: NestedDict) -> NestedDict:
    """
    Convert lists to strings in a dictionary. Specifically used for SNAP
    :code:`[GROUPS]` section.

    Parameters
    ----------
    d : dict
        The nested dictionary to be converted.

    Returns
    -------
    dict
        The dictionary with lists converted to strings.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            list_to_string(v)
        elif isinstance(v, list):
            d[k] = " ".join(str(x) for x in v)
    return d


def return_best_potential(
    run_path: str,
) -> str:
    """
    Return the path to the potential with the lowest loss from an optimization
    sweep.

    Parameters
    ----------
    run_path : str
        The path to the run directory.

    Returns
    -------
    str
        The path to the best potential.
    """
    with open(f"{run_path}/parameters.csv") as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)

    loss = [float(i[1]) for i in data]
    iteration = [int(i[0]) for i in data]
    best_iteration = iteration[loss.index(min(loss))]
    return f"{run_path}/{best_iteration}/"


def autoplex_return(
    run_path: str,
    single: bool = False,
    single_loss: float = 1,
) -> tuple[float, float, float]:
    """
    Return the autoplex-compatible loss, train error, and test error of the best
    potential from an optimization sweep.

    Parameters
    ----------
    run_path : str
        The path to the run directory.
    single : bool, optional
        Whether the run is a single fit, by default False.
    single_loss : float, optional
        The loss value for a single fit, this should be set manually,
        but is by default 1.

    Returns
    -------
    tuple
        The loss, train error, and test error of the potential with the lowest
        loss.
    """
    # ml_path = return_best_potential(run_path)
    if single:
        a = single_autoplex_return(run_path, single_loss)
        return a

    with open(f"{run_path}/parameters.csv") as f:
        reader = csv.reader(f)
        next(reader)
        loss = list(reader)

    with open(f"{run_path}/atomistic_errors.csv") as f:
        reader = csv.reader(f)
        next(reader)
        errors = list(reader)
    loss = [i[1] for i in loss]
    loss = [float(i) for i in loss]
    iter = loss.index(min(loss))
    best_loss = min(loss)
    train_error = errors[iter][1]
    test_error = errors[iter][2]

    return (best_loss, float(train_error), float(test_error))


def single_autoplex_return(
    run_path: str,
    loss: float,
) -> tuple[float, float, float]:
    """
    Return the autoplex-compatible loss, train error, and test error of a single
    potential from a single fit.
    Parameters
    ----------
    run_path : str
        The path to the run directory.

    Returns
    -------
    tuple
        The loss, train error, and test error of the potential with the lowest
        loss.
    """
    with open(f"{run_path}/atomistic_errors.csv") as f:
        reader = csv.reader(f)
        next(reader)
        errors = list(reader)
    best_loss = loss
    train_error = errors[0][1]
    test_error = errors[0][2]

    return (best_loss, float(train_error), float(test_error))


def convert_numpy_types(data):
    """
    Convert numpy string to native string type. Used to avoid errors when
    writing yaml files.

    Parameters
    ----------
    data : any
        The variable to be converted.

    Returns
    -------
    any
        The variable with numpy strings converted to native strings.
    """
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(element) for element in data]
    elif isinstance(data, np.str_):
        return str(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    else:
        return data


def load_structures(data_file: str | Path) -> Iterable[Atoms]:
    structures = iread(data_file, index=":")
    if isinstance(structures, Atoms):
        return [structures]
    return structures


def write_error_file(
    iteration: int,
    errors: list[float],
    filename: str,
):
    """
    Write the error values to a file.

    Parameters
    ----------
    e_train : float
        The energy training error.
    f_train : float
        The force training error.
    e_test : float
        The energy validation error.
    f_test : float
        The force validation error.
    filename : str
        The file to write to.
    """
    if len(errors) != 4:
        raise ValueError(
            "Error values must be a list of length 4, made up of "
            "the training and testing energy and force errors."
        )
    output_data = [iteration, *errors]
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_data)


def initialise_csvs(path: str, params: list) -> None:
    keys = params
    with open(f"{path}/parameters.csv", "w+") as f:
        f.write("iteration,loss," + ",".join(map(str, keys)) + "\n")
    with open(f"{path}/atomistic_errors.csv", "w+") as f:
        f.write(
            "Iteration,"
            + "Train Δ Energy,"
            + "Test Δ Energy,"
            + "Train Δ Force,"
            + "Test Δ Force"
            + "\n"
        )
    with open(f"{path}/loss_function_errors.csv", "w+") as f:
        f.write(
            "Iteration,"
            + "Train Δ Energy,"
            + "Test Δ Energy,"
            + "Train Δ Force,"
            + "Test Δ Force"
            + "\n"
        )
    print("Initialised CSV Files")
