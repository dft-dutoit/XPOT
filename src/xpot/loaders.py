from __future__ import annotations

import csv
import json
import os
import warnings
from collections.abc import Iterator
from typing import Union

import hjson
import numpy as np
import pandas as pd
import skopt

# DictValue = str | list[str | int] | int | float
# NestedDict = dict[str, "NestedDict" | DictValue]

DictValue = str | list[str, int] | int | float
NestedDict = dict[str, Union["NestedDict", DictValue]]


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
        return hjson.load(f)


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
    return all(
        pair in recursive_items(child) for pair in recursive_items(parent)
    )


def validate_hypers(
    merged_hypers: NestedDict,
    input_hypers: NestedDict,
) -> None:
    """
    Validate the hyperparameters.

    Parameters
    ----------
    hypers : dict
        The hyperparameters in dictionary format.
    defaults : dict
        The default parameters in dictionary format.

    Raises
    ------
    ValueError
        If the input hyperparameters are not a subset of the default parameters.
    """
    if validate_subdict(merged_hypers, input_hypers):
        print("Hyperparameters validated")
    else:
        raise ValueError(
            "Input hyperparameters are not a subset of the default parameters."
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
    list_values = []
    opt_values = {}
    for kv_pair in recursive_items(hypers):
        if isinstance(kv_pair[-1], list):
            list_values.append(kv_pair[-1])
        elif isinstance(kv_pair[-1], str) and "skopt" in kv_pair[-1]:
            # v_trimmed = kv_pair[-1].replace("skopt.space.", "")
            opt_values[kv_pair[:-1]] = eval(kv_pair[-1])
        else:
            continue
    return opt_values


def trim_empty_values(dict_in: dict) -> dict[str, str | int | float | list]:
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
    }


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
    elif os.path.exists(path) and not os.path.listdir(path):
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
        elif isinstance(v, np.int64):
            d[k] = int(v)
        elif isinstance(v, np.float64):
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


def drop_col_with_text(
    df: pd.DataFrame,
    column_name: str,
    text: str,
    reverse: bool = False,
) -> pd.DataFrame:
    """
    Drop a column from a dataframe if it contains a specific string. This
    function is case sensitive.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be modified.
    text : str
        The text to search for in the column names.

    Returns
    -------
    pd.DataFrame
        The modified dataframe.
    """
    if column_name not in df.columns:
        raise ValueError(f"{column_name} not in dataframe columns, exiting.")

    mask = df["column_name"].apply(lambda x: any(text in str(e) for e in x))

    return df.loc[:, mask] if reverse else df.loc[:, ~mask]


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

    best_iteration = loss.index(min(loss))

    return f"{run_path}/{best_iteration}/"


def autoplex_return(
    run_path: str,
) -> tuple[float, float, float]:
    """
    Return the autoplex-compatible loss, train error, and test error of the best
    potential from an optimization sweep.

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
    ml_path = return_best_potential(run_path)

    with open(f"{ml_path}/parameters.csv") as f:
        reader = csv.reader(f)
        next(reader)
        loss = list(reader)

    with open(f"{ml_path}/atomistic_errors.csv") as f:
        reader = csv.reader(f)
        next(reader)
        errors = list(reader)
    loss = [float(i[1]) for i in loss]
    iter = loss.index(min(loss))
    best_loss = min([float(i[1]) for i in loss])

    return (best_loss, errors[iter][1], errors[iter][2])


def convert_numpy_str(data):
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
        return {k: convert_numpy_str(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_str(element) for element in data]
    elif isinstance(data, np.str_):
        return str(data)
    else:
        return data
